import torch
import torch.distributed as dist
from .base import _OptimizerBase
from copy import deepcopy


class SGD(_OptimizerBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.config = config
        
        if self.config.true_weights:
            self.weights = torch.zeros(self.n_peers)
            for i, w in enumerate(self.weights):
                if i % 10 == 0:
                    self.weights[i] = 1
            self.weights /= self.weights.sum()
            
            # self.weights = torch.zeros(self.n_peers)
            # for i, w in enumerate(self.weights):
            #     if i % 2 == 0:
            #         self.weights[i] = 1
            # self.weights /= self.weights.sum()
            

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
                
                if self.config.true_weights:
                    for i, p in enumerate(self.problem.model.parameters()):
                        p_grad = 0
                        for j, g in enumerate(grads[i]):
                            p_grad += self.weights[j] * g
                        p.data -= self.lr * p_grad
                else:
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

    def metrics(self) -> float:
        return self.metrics_dict


class SGDMD(_OptimizerBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.config = config
        self.weights = torch.ones(self.n_peers) / self.n_peers

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
            self.problem.sample(full=self.config.md_full_)
            # print(self.weights.numpy(), ' ', torch.argmax(self.weights).item())
            for t in range(self.config.md_n_iters_):
                if not self.config.md_full_:
                    self.problem.sample()
                
                self.problem.model.load_state_dict(parameters_save)
                for i, p in enumerate(self.problem.model.parameters()):
                    p_grad = 0
                    for j, g in enumerate(grads[i]):
                        p_grad += self.weights[j] * g
                    p.data -= self.lr * p_grad

                gradients = self.problem.grad()
                weights_grad = torch.zeros_like(self.weights)
                for j in range(self.n_peers):
                    for i, g in enumerate(gradients):
                        weights_grad[j] += torch.sum(grads[i][j]*gradients[i])

                step = self.config.md_lr_ * self.lr * weights_grad
                vec = self.weights * torch.exp(step)
                self.weights = vec / torch.sum(vec)
                # print(self.weights.numpy(), ' ', torch.argmax(self.weights).item())

            # self.problem.metrics_dict["loss"] = torch.argmax(self.weights).item()
            self.problem.model.load_state_dict(parameters_save)
            for i, p in enumerate(self.problem.model.parameters()):
                p_grad = 0
                for j, g in enumerate(grads[i]):
                    p_grad += self.weights[j] * g
                p.data -= self.lr * p_grad


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

    def metrics(self) -> float:
        for i in range(len(self.weights)):
            key = 'weights_%s' % (str(i))
            self.problem.metrics_dict[key] = self.weights[i].item()
        return self.metrics_dict