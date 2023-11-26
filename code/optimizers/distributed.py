import torch
import torch.distributed as dist
from .base import _OptimizerBase
from copy import deepcopy


class SGD(_OptimizerBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.config = config

        if self.config.trueweights:
            self.weights = self.problem.dataset.true_weights
        else:
            self.weights = torch.ones(self.npeers) / self.npeers
        self.weights.to(self.problem.device)

    def step(self) -> None:
        self.problem.sample()
        gradients = self.problem.grad()

        with torch.no_grad():
            if self.problem.rank == self.problem.master_node:  # server
                grads = []
                for i, g in enumerate(gradients):
                    p_grads = [torch.empty_like(g) for _ in range(self.npeers)]
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

    @torch.no_grad()
    def metrics(self) -> float:
        return self.metrics_dict


class MeritFed(_OptimizerBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.config = config
        if self.problem.rank == self.problem.master_node:
            self.weights = torch.ones(self.npeers) / self.npeers
            self.weights = self.weights.to(self.problem.device)
            # self.grad_weight = torch.zeros_like(self.weights)
            # self.grad_weight = self.grad_weight.to(self.problem.device)

    def step(self) -> None:
        self.problem.sample()
        gradients = self.problem.grad()

        if self.problem.rank == self.problem.master_node:  # server
            self.problem.model.eval()
            parameters_save = deepcopy(self.problem.model.state_dict())
            grads = []
            for i, g in enumerate(gradients):
                p_grads = [torch.empty_like(g) for _ in range(self.npeers)]
                dist.gather(gradients[i], gather_list=p_grads)
                grads.append(p_grads)

            # mirror_prox
            self.problem.sample_test(full=self.config.mdfull_)
            for t in range(self.config.mdniters_):
                for i, p in enumerate(self.problem.model.parameters()):
                    p_grad = 0
                    for j, g in enumerate(grads[i]):
                        p_grad += self.weights[j] * g
                    p.data -= self.lr * p_grad

                if not self.config.mdfull_:
                    self.problem.sample_test(full=False)
                gradients = self.problem.grad()

                # self.grad_weight.mul(0)
                self.grad_weight = torch.zeros_like(self.weights, device=self.problem.device)
                # self.grad_weight = self.grad_weight.to(self.problem.device)
                for j in range(self.npeers):
                    for i, g in enumerate(gradients):
                        self.grad_weight[j] = self.grad_weight[j].add(torch.sum(grads[i][j]*gradients[i]))

                step = self.config.mdlr_ * self.lr * self.grad_weight
                step = torch.exp(step)
                vec = self.weights * step
                self.weights = vec / torch.sum(vec)
                self.problem.model.load_state_dict(parameters_save)

            for i, p in enumerate(self.problem.model.parameters()):
                p_grad = 0
                for j, g in enumerate(grads[i]):
                    p_grad += self.weights[j] * g
                p.data -= self.lr * p_grad

            self.problem.model.train()            
        else:
            for i, _ in enumerate(gradients):  # nodes
                dist.gather(tensor=gradients[i], dst=self.problem.master_node)

        # broadcast new point
        for p in self.problem.model.parameters():
            dist.broadcast(p.data, src=self.problem.master_node)
        self.i += 1

    @torch.no_grad()
    def metrics(self) -> float:
        for i in range(len(self.weights)):
            key = 'weights_%s' % (str(i))
            self.problem.metrics_dict[key] = self.weights[i].item()
        return self.metrics_dict



# class SGD(_OptimizerBase):
#     def __init__(self, config, rank):
#         super().__init__(config, rank)
#         self.config = config

#         if self.config.trueweights:
#             self.weights = self.problem.dataset.trueweights

#     def step(self) -> None:
#         self.problem.sample()
#         grad = self.problem.grad()
#         # with torch.no_grad():

#         if self.problem.rank == self.problem.master_node:  # server
#             grads = [torch.empty_like(grad) for _ in range(self.npeers)]
#             dist.gather(grad, gather_list=grads)
#             grads = torch.stack(grads, 0)
#             # print(grads.shape)

#             if self.config.trueweights:
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
#         self.weights = torch.ones(self.npeers) / self.npeers

#     def step(self) -> None:
#         self.problem.sample()
#         grad = self.problem.grad()
#         # with torch.no_grad():

#         if self.problem.rank == self.problem.master_node:  # server
#             parameters_save = deepcopy(self.problem.model.state_dict())
#             grads = [torch.empty_like(grad) for _ in range(self.npeers)]
#             dist.gather(grad, gather_list=grads)
#             grads = torch.stack(grads, 0)

#             # mirror_prox
#             # self.problem.sample(full=self.config.mdfull_)
#             self.problem.sample_test(full=self.config.mdfull_)
#             for t in range(self.config.mdniters_):
#             # for t in range(self.config.mdniters_ + self.i // 10):
#                 if not self.config.mdfull_:
#                     self.problem.sample_test(full=self.config.mdfull_)

#                 self.problem.model.load_state_dict(parameters_save)
#                 step = torch.einsum('i...,i->...', grads, self.weights)
#                 for i, p in enumerate(self.problem.model.parameters()):
#                     p.data -= self.lr * step[i]

#                 grad = self.problem.grad()

#                 w_grad = torch.einsum('j...,...->j', grads, grad)

#                 # lr = self.config.mdlr_ // int(self.i / 300 + 1)
#                 # print(lr)
#                 # step = self.config.mdlr_ * self.lr * w_grad #  * 1/np.sqrt(t+1) 
#                 # step = lr * self.lr * w_grad #  * 1/np.sqrt(t+1) 
#                 step = self.config.mdlr_ * self.lr * w_grad
#                 vec = self.weights * torch.exp(step)
#                 # self.weights = self.weights * t / (t+1) + vec / torch.sum(vec) / (t+1)
#                 self.weights = vec / torch.sum(vec)

#             # self.weights = self.problem.dataset.trueweights
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
