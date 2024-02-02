from copy import deepcopy
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, rank, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        defaults = dict()
        super().__init__(params, defaults)

        self.rank = rank
        self.cfg = cfg
        self.i = 0

        if rank == 0:
            self.metrics_dict = defaultdict(float)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, shared_grads, model, md_loader, criterion, device):
        dist.barrier()
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                shared_grads[self.rank][i].data.copy_(p.grad.data)
                i += 1
        dist.barrier()

        if self.rank == 0:
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(shared_grads):
                        if self.cfg.trueweights:
                            p_step += self.cfg.weights[j] * g[i]
                        else:
                            p_step += g[i] / self.cfg.npeers
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)
            self.i += 1

        # for group in self.param_groups:
        #     for p in group['params']:
        # # for p in self.problem.model.parameters():
        #         dist.broadcast(p.data, src=0)


    @torch.no_grad()
    def metrics(self) -> float:
        return self.metrics_dict


class MeritFed(Optimizer):
    def __init__(self, params, rank, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        defaults = dict()
        super().__init__(params, defaults)

        if rank == 0:
            self.metrics_dict = defaultdict(float)

        self.rank = rank
        self.cfg = cfg
        self.weights = torch.ones(cfg.npeers, device=device) / cfg.npeers
        self.i = 0

    def __setstate__(self, state):
        super().__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)

    def step(self, shared_grads, model, md_loader, criterion, device):
        # if self.rank == 0:
        #     i = 0
        #     for group in self.param_groups:
        #         for p in group['params']:
        #             shared_grads[self.rank][i].data.copy_(p.grad.data)# - shared_grads[self.rank][i]
        #             i += 1
        #     print(f"{self.rank, shared_grads[0][0]=}")
        # if self.rank == 1:
        #     time.sleep(3)
        #     print(f"{self.rank, shared_grads[0][0]=}")

        # time.sleep(30)
        # raise ValueError('-08/1635')    
        dist.barrier()
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                shared_grads[self.rank][i].data.copy_(p.grad.data)
                i += 1
        dist.barrier()

        # if not self.rank and self.i%100==0:
        #     print(f"{self.i=}")
        #     for j, g in enumerate(shared_grads):
        #         print(f"{g[0]=}")
        #     print()

        if self.rank == 0:  # server
            # mirror_prox
            parameters_save = deepcopy(model.state_dict())
            md_iter = iter(md_loader)

            for t in range(self.cfg.mdniters_):

                i = 0
                for group in self.param_groups:
                    for p in group['params']:
                        p_step = torch.zeros_like(p, device=device)
                        for j, g in enumerate(shared_grads):
                            p_step += self.weights[j] * g[i]
                        i += 1
                        p.data.add_(p_step, alpha=-self.cfg.lr)

                self.zero_grad()
                try:
                    data, labels = next(md_iter)
                except StopIteration:
                    md_iter = iter(md_loader)
                    data, labels = next(md_iter)

                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()

                self.weights_grad = torch.zeros_like(self.weights, device=device)
                i = 0
                for group in self.param_groups:
                    for p in group['params']:
                        for j, g in enumerate(shared_grads):
                            # print(f"{g[1]=}")
                            self.weights_grad[j] = self.weights_grad[j].add(torch.sum(p.grad.data * g[i]))
                        i += 1

                step = self.cfg.mdlr_ * self.cfg.lr * self.weights_grad
                if self.i > 400:
                    step /= 10.
                step = torch.exp(step)
                vec = self.weights * step
                self.weights = vec / torch.sum(vec)
                model.load_state_dict(parameters_save)

            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(shared_grads):
                        p_step += self.weights[j] * g[i]
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)
            self.i += 1


    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict

# class SGD(_OptimizerBase):
#     def __init__(self, config, rank):
#         super().__init__(config, rank)
#         self.config = config

#         if self.config.trueweights:
#             self.weights = self.problem.dataset.true_weights
#         else:
#             self.weights = torch.ones(self.npeers) / self.npeers
#         self.weights.to(self.problem.device)

#     def step(self) -> None:
#         self.problem.sample()
#         gradients = self.problem.grad()

#         with torch.no_grad():
#             if self.problem.rank == self.problem.master_node:  # server
#                 grads = []
#                 for i, g in enumerate(gradients):
#                     p_grads = [torch.empty_like(g) for _ in range(self.npeers)]
#                     dist.gather(gradients[i], gather_list=p_grads)
#                     grads.append(p_grads)

#                 for i, p in enumerate(self.problem.model.parameters()):
#                     p_grad = 0
#                     for j, g in enumerate(grads[i]):
#                         p_grad += self.weights[j] * g
#                     p.data -= self.lr * p_grad

#             else:  # node
#                 for i, _ in enumerate(gradients):  # nodes
#                     dist.gather(tensor=gradients[i], dst=self.problem.master_node)

#             # broadcast new point
#             for p in self.problem.model.parameters():
#                 dist.broadcast(p.data, src=self.problem.master_node)
#             self.i += 1

#     @torch.no_grad()
#     def metrics(self) -> float:
#         return self.metrics_dict


# class MeritFed(_OptimizerBase):
#     def __init__(self, config, rank):
#         super().__init__(config, rank)
#         self.config = config
#         if self.problem.rank == self.problem.master_node:
#             self.weights = torch.ones(self.npeers) / self.npeers
#             self.weights = self.weights.to(self.problem.device)
#             # self.grad_weight = torch.zeros_like(self.weights)
#             # self.grad_weight = self.grad_weight.to(self.problem.device)

#     def step(self) -> None:
#         self.problem.sample()
#         gradients = self.problem.grad()

#         if self.problem.rank == self.problem.master_node:  # server
#             self.problem.model.eval()
#             parameters_save = deepcopy(self.problem.model.state_dict())
#             grads = []
#             for i, g in enumerate(gradients):
#                 p_grads = [torch.empty_like(g) for _ in range(self.npeers)]
#                 dist.gather(gradients[i], gather_list=p_grads)
#                 grads.append(p_grads)

#             # mirror_prox
#             self.problem.sample_test(full=self.config.mdfull_)
#             for t in range(self.config.mdniters_):
#                 for i, p in enumerate(self.problem.model.parameters()):
#                     p_grad = 0
#                     for j, g in enumerate(grads[i]):
#                         p_grad += self.weights[j] * g
#                     p.data -= self.lr * p_grad

#                 if not self.config.mdfull_:
#                     self.problem.sample_test(full=False)
#                 gradients = self.problem.grad()

#                 # self.grad_weight.mul(0)
#                 self.grad_weight = torch.zeros_like(self.weights, device=self.problem.device)
#                 # self.grad_weight = self.grad_weight.to(self.problem.device)
#                 for j in range(self.npeers):
#                     for i, g in enumerate(gradients):
#                         self.grad_weight[j] = self.grad_weight[j].add(torch.sum(grads[i][j]*gradients[i]))

#                 step = self.config.mdlr_ * self.lr * self.grad_weight
#                 step = torch.exp(step)
#                 vec = self.weights * step
#                 self.weights = vec / torch.sum(vec)
#                 self.problem.model.load_state_dict(parameters_save)

#             for i, p in enumerate(self.problem.model.parameters()):
#                 p_grad = 0
#                 for j, g in enumerate(grads[i]):
#                     p_grad += self.weights[j] * g
#                 p.data -= self.lr * p_grad

#             self.problem.model.train()            
#         else:
#             for i, _ in enumerate(gradients):  # nodes
#                 dist.gather(tensor=gradients[i], dst=self.problem.master_node)

#         # broadcast new point
#         for p in self.problem.model.parameters():
#             dist.broadcast(p.data, src=self.problem.master_node)
#         self.i += 1

#     @torch.no_grad()
#     def metrics(self) -> float:
#         for i in range(len(self.weights)):
#             key = 'weights_%s' % (str(i))
#             self.problem.metrics_dict[key] = self.weights[i].item()
#         return self.metrics_dict



class TAWT(Optimizer):
    def __init__(self, params, rank, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        defaults = dict()
        super().__init__(params, defaults)

        if rank == 0:
            self.metrics_dict = defaultdict(float)

        self.rank = rank
        self.cfg = cfg
        self.weights = torch.ones(cfg.npeers, device=device) / cfg.npeers
        self.i = 0

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, shared_grads, model, md_loader, criterion, device):
        dist.barrier()
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                shared_grads[self.rank][i].data.copy_(p.grad.data)
                i += 1
        dist.barrier()

        if self.rank == 0:  # server
            # mirror_prox
            parameters_save = deepcopy(model.state_dict())
            md_iter = iter(md_loader)

            self.i += 1
            # for t in range(self.cfg.mdniters_):
            if self.i % 10 == 0:
                i = 0
                for group in self.param_groups:
                    for p in group['params']:
                        p_step = torch.zeros_like(p, device=device)
                        for j, g in enumerate(shared_grads):
                            p_step += self.weights[j] * g[i]
                        i += 1
                        p.data.add_(p_step, alpha=-self.cfg.lr)

                self.zero_grad()
                try:
                    data, labels = next(md_iter)
                except StopIteration:
                    md_iter = iter(md_loader)
                    data, labels = next(md_iter)

                data, labels = data.to(device), labels.to(device)
                output = model(data)
                loss = criterion(output, labels)
                # loss.backward()

#                 grad = torch.autograd.grad(outputs=loss, inputs=model.parameters(), create_graph=True)
#                 grad = grad[0].squeeze()

#                 for i, g in enumerate(grad):
#                     model.zero_grad()
#                     second_order_grad = torch.autograd.grad(outputs=g, inputs=model.parameters(), retain_graph=True)
#                     second_order_grad = second_order_grad[0].squeeze()

#                 self.weights_grad = torch.zeros_like(self.weights, device=device)
#                 i = 0
#                 for group in self.param_groups:
#                     for p in group['params']:
#                         for j, g in enumerate(shared_grads):
#                             # print(f"{g[1]=}")
#                             self.weights_grad[j] = self.weights_grad[j].add(torch.sum(p.grad.data * g[i]))
#                         i += 1

#                 step = self.cfg.mdlr_ * self.cfg.lr * self.weights_grad
#                 if self.i > 400:
#                     step /= 10.
#                 step = torch.exp(step)
#                 vec = self.weights * step
#                 self.weights = vec / torch.sum(vec)
                model.load_state_dict(parameters_save)

            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(shared_grads):
                        p_step += self.weights[j] * g[i]
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)


    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict
