import torch
import numpy as np
import torch.distributed as dist
from .base import _OptimizerBase, Optimizer
from copy import deepcopy



def load_distributed_optimizer(config, rank):
    if config.optimizer == Optimizer.SGD:
        return SGD(config, rank)
    if config.optimizer == Optimizer.MeritFed:
        return MeritFed(config, rank)
    else:
        raise NotImplementedError()



# class SGD(_OptimizerBase):
#     def __init__(self, config, rank):
#         super().__init__(config, rank)
#         self.config = config

#         if self.config.true_weights:
#             self.weights = self.problem.dataset.true_weights

#     def step(self) -> None:
#         self.problem.sample()
#         grad = self.problem.grad()
#         # with torch.no_grad():

#         if self.problem.rank == self.problem.master_node:  # server
#             grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
#             dist.gather(grad, gather_list=grads)
#             grads = torch.stack(grads, 0)
#             # print(grads.shape)

#             if self.config.true_weights:
#                 # print(grads.shape, self.weights.shape)
#                 step = torch.einsum('i...,i->...', grads, self.weights)
#                 for i, p in enumerate(self.problem.model.parameters()):
#                     p.data -= self.lr * step[i]
#             else:
#                 step = torch.mean(grads, dim=0)
#                 for i, p in enumerate(self.problem.model.parameters()):
#                     # print(p.shape)
#                     p_grad = step[i]
#                     p.data -= self.lr * p_grad
#                 # raise RuntimeError("Non-master client accessed test dataset")
#         else:  # node
#             dist.gather(tensor=grad, dst=self.problem.master_node)

#         # broadcast new point
#         for p in self.problem.model.parameters():
#             dist.broadcast(p.data, src=self.problem.master_node)
#         self.i += 1

#     def metrics(self) -> float:
#         return self.metrics_dict


# class MeritFed(_OptimizerBase):
#     def __init__(self, config, rank):
#         super().__init__(config, rank)
#         self.config = config
#         self.weights = torch.ones(self.n_peers) / self.n_peers

#     def step(self) -> None:
#         self.problem.sample()
#         grad = self.problem.grad()
#         # with torch.no_grad():

#         if self.problem.rank == self.problem.master_node:  # server
#             parameters_save = deepcopy(self.problem.model.state_dict())
#             grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
#             dist.gather(grad, gather_list=grads)
#             grads = torch.stack(grads, 0)

#             # mirror_prox
#             # self.problem.sample(full=self.config.md_full_)
#             self.problem.sample_test(full=self.config.md_full_)
#             for t in range(self.config.md_n_iters_):
#             # for t in range(self.config.md_n_iters_ + self.i // 10):
#                 if not self.config.md_full_:
#                     self.problem.sample_test(full=self.config.md_full_)

#                 self.problem.model.load_state_dict(parameters_save)
#                 step = torch.einsum('i...,i->...', grads, self.weights)
#                 for i, p in enumerate(self.problem.model.parameters()):
#                     p.data -= self.lr * step[i]

#                 grad = self.problem.grad()

#                 w_grad = torch.einsum('j...,...->j', grads, grad)

#                 # lr = self.config.md_lr_ // int(self.i / 300 + 1)
#                 # print(lr)
#                 # step = self.config.md_lr_ * self.lr * w_grad #  * 1/np.sqrt(t+1) 
#                 # step = lr * self.lr * w_grad #  * 1/np.sqrt(t+1) 
#                 step = self.config.md_lr_ * self.lr * w_grad
#                 vec = self.weights * torch.exp(step)
#                 # self.weights = self.weights * t / (t+1) + vec / torch.sum(vec) / (t+1)
#                 self.weights = vec / torch.sum(vec)

#             # self.weights = self.problem.dataset.true_weights
#             self.problem.model.load_state_dict(parameters_save)
#             step = torch.einsum('i...,i->...', grads, self.weights)
#             for i, p in enumerate(self.problem.model.parameters()):
#                 p.data -= self.lr * step[i]

#         else:
#             dist.gather(tensor=grad, dst=self.problem.master_node)

#         # broadcast new point
#         for p in self.problem.model.parameters():
#             dist.broadcast(p.data, src=self.problem.master_node)
#         self.i += 1

#     def metrics(self) -> float:
#         for i in range(len(self.weights)):
#             key = 'weights_%s' % (str(i))
#             self.problem.metrics_dict[key] = self.weights[i].item()
#         return self.metrics_dict


class SGD(_OptimizerBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.config = config

        if self.config.true_weights:
            self.weights = self.problem.dataset.true_weights
        else:
            self.weights = torch.ones(self.n_peers) / self.n_peers

    def step(self) -> None:
        self.problem.sample()
        gradients = self.problem.grad()
        # with torch.no_grad():

        if self.problem.rank == self.problem.master_node:  # server
            grads = []
            for i, g in enumerate(gradients):
                p_grads = [torch.empty_like(g) for _ in range(self.n_peers)]
                dist.gather(gradients[i], gather_list=p_grads)
                grads.append(p_grads)

            for i, p in enumerate(self.problem.model.parameters()):
                p_grad = 0
                for j, g in enumerate(grads[i]):
                    p_grad += self.weights[j] * g
                p.data -= self.lr * p_grad

        else:  # node
            for i, _ in enumerate(gradients):  # nodes
                dist.gather(tensor=gradients[i], dst=self.problem.master_node)

        # broadcast new point
        for p in self.problem.model.parameters():
            dist.broadcast(p.data, src=self.problem.master_node)
        self.i += 1

    def metrics(self) -> float:
        return self.metrics_dict


class MeritFed(_OptimizerBase):
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
            self.problem.sample_test(full=self.config.md_full_)
            for t in range(self.config.md_n_iters_):
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

                step = self.config.md_lr_ * self.lr * grad_weight
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

    def metrics(self) -> float:
        for i in range(len(self.weights)):
            key = 'weights_%s' % (str(i))
            self.problem.metrics_dict[key] = self.weights[i].item()
        return self.metrics_dict


