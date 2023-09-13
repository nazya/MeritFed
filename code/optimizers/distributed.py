import torch
import torch.distributed as dist
from .base import _OptimizerBase
from copy import deepcopy

class SGD(_OptimizerBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)

    def step(self) -> None:
        self.problem.sample()
        gradients = self.problem.grad()
        with torch.no_grad():
            # server
            if self.problem.rank == self.problem.master_node:
                grads = []
                for i, g in enumerate(gradients):
                    p_grads = [torch.empty_like(g) for _ in range(self.n_peers)]
                    dist.gather(gradients[i], gather_list=p_grads)
                    grads.append(p_grads)

                for i, p in enumerate(self.problem.model.parameters()):
                    p_grad = torch.mean(torch.stack(grads[i]), dim=0)
                    p.data -= self.lr * p_grad
            # node
            else:
                for i, _ in enumerate(gradients):
                    dist.gather(tensor=gradients[i], dst=self.problem.master_node)

        # broadcast new point
        for p in self.problem.model.parameters():
            dist.broadcast(p.data, src=self.problem.master_node)
        self.i += 1


class SGDMD(_OptimizerBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.weights = torch.ones(self.n_peers) / self.n_peers
        self.config = config

    def step(self) -> None:
        self.problem.sample()
        gradients = self.problem.grad()
        # with torch.no_grad():

        if self.problem.rank == self.problem.master_node:  # server
            parameters_save = deepcopy(self.problem.model.state_dict())
            grads = []
            for i, g in enumerate(gradients):
                p_grads = [torch.empty_like(g) for _ in range(self.n_peers)]
                dist.gather(gradients[i], gather_list=p_grads)
                grads.append(p_grads)

            # mirror_prox
            self.problem.sample(full=self.config.md_full)
            # print(self.weights.numpy(), ' ', torch.argmax(self.weights).item())
            for t in range(self.config.md_iter):
                self.problem.model.load_state_dict(parameters_save)
                for i, p in enumerate(self.problem.model.parameters()):
                    p_grad = 0
                    for j, g in enumerate(grads[i]):
                        p_grad += self.weights[j] * g
                    p.data -= self.lr * p_grad

                gradients = self.problem.grad()
                grad_weight = torch.zeros_like(self.weights)
                for j in range(self.n_peers):
                    for i, g in enumerate(gradients):
                        grad_weight[j] += torch.sum(grads[i][j]*gradients[i])

                step = self.config.md_lr * self.lr * grad_weight
                vec = self.weights * torch.exp(step)
                self.weights = vec / torch.sum(vec)
                # print(self.weights.numpy(), ' ', torch.argmax(self.weights).item())

            for i, p in enumerate(self.problem.model.parameters()):
                p_grad = 0
                for j, g in enumerate(grads[i]):
                    p_grad += self.weights[j] * g
                p.data -= self.lr * p_grad

            self.problem.metrics_dict["best_node"] = torch.argmax(self.weights).item()

            # self.problem.model.load_state_dict(parameters_save)
            # for i, p in enumerate(self.problem.model.parameters()):
            #     p_grad = torch.mean(torch.stack(grads[i]), dim=0)
            #     p.data -= self.lr * p_grad
        else:
            for i, _ in enumerate(gradients):  # nodes
                dist.gather(tensor=gradients[i], dst=self.problem.master_node)

        # broadcast new point
        for p in self.problem.model.parameters():
            dist.broadcast(p.data, src=self.problem.master_node)
        self.i += 1
