import torch
from copy import deepcopy
from collections import defaultdict
from torch.optim.optimizer import Optimizer

from torch.autograd.functional import hessian


def gompertz(alpha, x):
    x = -alpha*(x-1)
    x = -torch.exp(x)
    x = 1 - torch.exp(x)
    return alpha*x



class SGD(Optimizer):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        defaults = dict()
        super().__init__(params, defaults)

        self.cfg = cfg
        self.i = 0

        # self.n_workers = n_workers
        self.grads_received = 0

        self.metrics_dict = defaultdict(float)

        self.grads = list()
        for i in range(cfg.npeers):
            p_grads = list()
            for group in self.param_groups:
                for p in group['params']:
                    p_grads.append(torch.zeros_like(p, device=device))
            self.grads.append(p_grads)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, w_id, model, md_loader, criterion, device):
        # print(f"{w_id=}")
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                self.grads[w_id][i].data.copy_(p.grad.data)
                i += 1

        self.grads_received += 1

        if self.grads_received == self.cfg.npeers:
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(self.grads):
                        if self.cfg.trueweights:
                            p_step += self.cfg.weights[j] * g[i]
                        else:
                            w = 1 / self.cfg.npeers
                            # print(f"{w=}")
                            p_step += w * g[i] 
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)

            self.grads_received = 0
            self.i += 1

    @torch.no_grad()
    def metrics(self) -> float:
        return self.metrics_dict


class FedAdp(Optimizer):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        defaults = dict()
        super().__init__(params, defaults)

        self.cfg = cfg
        self.weights = torch.ones(cfg.npeers, device=device) / cfg.npeers

        self.i = 0

        # self.n_workers = n_workers
        self.grads_received = 0

        self.metrics_dict = defaultdict(float)

        self.grads = list()
        for i in range(cfg.npeers):
            p_grads = list()
            for group in self.param_groups:
                for p in group['params']:
                    p_grads.append(torch.zeros_like(p, device=device))
            self.grads.append(p_grads)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, w_id, model, md_loader, criterion, device):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                self.grads[w_id][i].data.copy_(p.grad.data)
                i += 1

        self.grads_received += 1

        if self.grads_received == self.cfg.npeers:
            weights = torch.zeros(self.cfg.npeers, device=device)
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    for j, g in enumerate(self.grads):
                        weights[j] = weights[j].add(torch.sum(self.grads[0][i] * g[i]))
                    i += 1

            norms = torch.zeros_like(weights, device=device)
            for i, g in enumerate(self.grads):
                for j, gj in enumerate(g):
                    norms[i] += torch.sum(gj*gj)

            for i, _ in enumerate(weights):
                weights[i] /= torch.sqrt(norms[i])*torch.sqrt(norms[0])

            weights = torch.clamp(weights, min=-1, max=1)
            weights = torch.arccos(weights)
            # print(f"")
            # print(f"firstn{weights=}")
            # print(f"{gompertz(1., weights)=}")
            self.weights = torch.nn.functional.softmax(gompertz(5., weights), dim=0)

            # print(f"{self.weights=}")
            # print(f"")
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(self.grads):
                        p_step += self.weights[j] * g[i]
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)

            self.grads_received = 0
            self.i += 1

    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict


class MeritFed(Optimizer):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        defaults = dict()
        super().__init__(params, defaults)

        self.cfg = cfg
        self.weights = torch.ones(cfg.npeers, device=device) / cfg.npeers
        self.i = 0

        self.grads_received = 0

        self.metrics_dict = defaultdict(float)

        self.grads = list()
        for i in range(cfg.npeers):
            p_grads = list()
            for group in self.param_groups:
                for p in group['params']:
                    p_grads.append(torch.zeros_like(p, device=device))
            self.grads.append(p_grads)

    def __setstate__(self, state):
        super().__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)

    def step(self, w_id, model, md_loader, criterion, device):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                self.grads[w_id][i].data.copy_(p.grad.data)
                i += 1

        self.grads_received += 1

        if self.grads_received == self.cfg.npeers:
            # mirror_prox
            parameters_save = deepcopy(model.state_dict())
            md_iter = iter(md_loader)

            for t in range(self.cfg.mdniters_):
            # for t, batch in enumerate(md_loader):

                i = 0
                for group in self.param_groups:
                    for p in group['params']:
                        p_step = torch.zeros_like(p, device=device)
                        for j, g in enumerate(self.grads):
                            p_step += self.weights[j] * g[i]
                        i += 1
                        p.data.add_(p_step, alpha=-self.cfg.lr)

                self.zero_grad()
                try:
                    batch = next(md_iter)
                except StopIteration:
                    md_iter = iter(md_loader)
                    batch = next(md_iter)
                # print(f"{len(batch[0])=}")

                batch = [data.to(device) for data in batch]

                output = model(*batch[:-1])
                if hasattr(output, "logits"):
                    output = output.logits
                loss = criterion(output, batch[-1])

                # data, labels = data.to(device), labels.to(device)
                # output = model(data)
                # loss = criterion(output, labels)
                loss.backward()

                self.weights_grad = torch.zeros_like(self.weights, device=device)
                i = 0
                for group in self.param_groups:
                    for p in group['params']:
                        for j, g in enumerate(self.grads):
                            self.weights_grad[j] = self.weights_grad[j].add(torch.sum(p.grad.data * g[i]))
                        i += 1

                step = self.cfg.mdlr_ * self.cfg.lr * self.weights_grad
                # if self.i > 400:
                if self.i > 400 and self.cfg.dataset['name'] != 'Normal':
                    step /= 10.
                step = torch.exp(step)
                vec = self.weights * step
                if self.cfg.dataset['name'] == 'Normal':
                    self.weights = vec / torch.sum(vec)
                else:
                    self.weights = self.weights * t / (t+1) + vec / torch.sum(vec) / (t+1)
                # self.weights = torch.nn.functional.softmax(self.weights, dim=0)
                model.load_state_dict(parameters_save)

            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(self.grads):
                        p_step += self.weights[j] * g[i]
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)
            self.grads_received = 0
            self.i += 1


    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict

    
# data, labels = next(self.md_iter)

# data, labels = data.to(device), labels.to(device)
# output = model(data)
# # loss = criterion(output, labels)

# names = list(n for n, _ in model.named_parameters())
# def loss(*params):
#     out: torch.Tensor = torch.func.functional_call(model, {n: p.detach() for n, p in zip(names, params)}, data)
#     return criterion (out, labels)

# hessian(loss, tuple(model.parameters()))
# print()
# print(f"{self.i=}")

# for n, p in model.named_parameters():
#     hessian(loss, tuple(p))
#     print(f"{self.i, n=}")


# if (self.i + 1) % 10 == 0:
# if self.md_iter is None:
#     self.md_iter = iter(md_loader)

# try:
#     data, labels = next(self.md_iter)
# except StopIteration:
#     self.md_iter = iter(md_loader)
#     data, labels = next(self.md_iter)

# data, labels = data.to(device), labels.to(device)
# output = model(data)
# loss = criterion(output, labels)


class TAWT(Optimizer):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        defaults = dict()
        super().__init__(params, defaults)

        self.cfg = cfg
        self.weights = torch.ones(cfg.npeers, device=device) / cfg.npeers
        self.i = 0

        # self.md_iter = None

        self.grads_received = 0

        self.metrics_dict = defaultdict(float)

        self.grads = list()
        for i in range(cfg.npeers):
            p_grads = list()
            for group in self.param_groups:
                for p in group['params']:
                    p_grads.append(torch.zeros_like(p, device=device))
            self.grads.append(p_grads)
        # self.hess = None

    def __setstate__(self, state):
        super().__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault('nesterov', False)

    def step(self, w_id, model, md_loader, criterion, device):
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                self.grads[w_id][i].data.copy_(p.grad.data)
                i += 1

        self.grads_received += 1

        if self.grads_received == self.cfg.npeers:
            if (self.i + 1) % 2 == 0:
                # mirror_prox
#                 parameters_save = deepcopy(model.state_dict())
#                 md_iter = iter(md_loader)

#                 for t in range(self.cfg.mdniters_):

#                     i = 0
#                     for group in self.param_groups:
#                         for p in group['params']:
#                             p_step = torch.zeros_like(p, device=device)
#                             for j, g in enumerate(self.grads):
#                                 p_step += self.weights[j] * g[i]
#                             i += 1
#                             p.data.add_(p_step, alpha=-self.cfg.lr)

#                     self.zero_grad()
#                     try:
#                         data, labels = next(md_iter)
#                     except StopIteration:
#                         md_iter = iter(md_loader)
#                         data, labels = next(md_iter)

#                     data, labels = data.to(device), labels.to(device)
#                     output = model(data)
#                     loss = criterion(output, labels)
#                     loss.backward()

                    self.weights_grad = torch.zeros_like(self.weights, device=device)
                    i = 0
                    for group in self.param_groups:
                        for p in group['params']:
                            for j, g in enumerate(self.grads):
                                self.weights_grad[j] = self.weights_grad[j].add(torch.sum(self.grads[0][i] * g[i]))
                            i += 1

                    norms = torch.zeros_like(self.weights, device=device)                        
                    for i, g in enumerate(self.grads):
                        for j, gj in enumerate(g):
                            norms[i] += torch.sum(gj*gj)

                    for i, _ in enumerate(self.weights_grad):
                        self.weights_grad[i] /= torch.sqrt(norms[i])*torch.sqrt(norms[0])

                    step = 100 * self.cfg.lr * self.weights_grad
                    step = torch.exp(step)
                    vec = self.weights * step
                    self.weights = vec / torch.sum(vec)
                    # model.load_state_dict(parameters_save)

            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(self.grads):
                        p_step += self.weights[j] * g[i]
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)
            self.grads_received = 0
            self.i += 1


    @torch.no_grad()
    def metrics(self) -> float:
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict