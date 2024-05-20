import torch
import random
from copy import deepcopy
from collections import defaultdict
from torch.optim.optimizer import Optimizer
import codes.attacks as attacks

def gompertz(alpha, x):
    x = -alpha*(x-1)
    x = -torch.exp(x)
    x = 1 - torch.exp(x)
    return alpha*x

class OptimizerBase(Optimizer):
    def __init__(self, params, cfg, device):
        defaults = dict()
        super().__init__(params, defaults)
        self.attack = None
        if cfg.attack is not None:
            self.attack = getattr(attacks, cfg.attack['name'])(cfg)


class FedFomo(OptimizerBase):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        super().__init__(params, cfg, device)

        self.cfg = cfg
        self.weights = torch.diag(torch.ones(self.cfg.npeers, device=device))

        self.i = 0

        self.loaders = list()
        self.device = device
        # self.n_workers = n_workers
        self.grads_received = 0

        self.metrics_dict = defaultdict(float)

        self.old_loss = torch.diag(torch.zeros(self.cfg.npeers, device=self.device))
        self.cur_loss = torch.diag(torch.zeros(self.cfg.npeers, device=self.device))
        
            
        for group in self.param_groups:
            for p in group['params']:
                # grad = p.grad.data
                state = self.state[p]
                for w_i in range(cfg.npeers):
                    state['cur'+str(w_i)] = torch.zeros_like(p.data)
                    state['cur'+str(w_i)].data.copy_(p.data)
                    state['old'+str(w_i)] = torch.zeros_like(p.data)
                    state['tmp'+str(w_i)] = torch.zeros_like(p.data)
                    


    def __setstate__(self, state):
        super().__setstate__(state)
        
    @torch.no_grad()
    def init_losses(self, model, loaders, criterion):
        for w_n in range(self.cfg.npeers):
            for w_i in range(self.cfg.npeers):
                self.old_loss[w_n][w_i] = self.cur_loss[w_n][w_i]
                self.copy_cur_params_to_model_params(model, w_n)
                #loader[w_i] on cur_w_n
                self.cur_loss[w_n][w_i] = self.calc_loss(model, loaders[w_i], criterion)

    @torch.no_grad()
    def calc_loss(self, model, loader, criterion):
        total_loss = 0
        model.eval()
        # print(len(loader))
        for i, batch in enumerate(loader):
            # print(len(batch), end='\r')
            # print(i, batch)
            batch = [data.to(self.device) for data in batch]
            output = model(*batch[:-1])
            loss = criterion(output, batch[-1])
            total_loss += loss.item()
            if i == 1:
                break
        model.train()
        return total_loss / i / self.cfg.batchsize

    @torch.no_grad()
    def copy_cur_params_to_model_params(self, model, w_i):
        for group in self.param_groups:
            for p in group['params']:
                # grad = p.grad.data
                state = self.state[p]
                p.data.copy_(state['cur'+str(w_i)].data)
                
    @torch.no_grad()
    def before_backward(self, w_i, model):
        self.copy_cur_params_to_model_params(model, w_i)
        
    @torch.no_grad()
    def step(self, w_i, model, loader, criterion, device):
        self.loaders.append(loader)
        # print(w_i, len(loader))
        ######################################
        # update curreht params and old params
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]
                state['old'+str(w_i)].data.copy_(state['cur'+str(w_i)].data)
                
                p.data.add_(grad, alpha=-self.cfg.lr)
                state['cur'+str(w_i)].data.copy_(p.data)
                
               
        ######################################
        

        self.grads_received += 1

        # if False:
        if self.grads_received == self.cfg.npeers:
            
            #calc losses
            for w_n in range(self.cfg.npeers):
                for w_i in range(self.cfg.npeers):
                    self.old_loss[w_n][w_i] = self.cur_loss[w_n][w_i]
                    self.copy_cur_params_to_model_params(model, w_n)
                    #loader[w_i] on cur_w_n
                    self.cur_loss[w_n][w_i] = self.calc_loss(model, self.loaders[w_i], criterion)
                    # print(self.cur_loss[w_n][w_i])
                    
            # print(f"{self.old_loss=}")
            # print(f"")
            # print(f"{self.cur_loss=}")
            # print(f"")
            # print(f"")
            
            #calc weights for each client
            for w_n in range(self.cfg.npeers):
                # s = 1e-5
                s = 0.
                for w_i in range(self.cfg.npeers):
                    denom = 0
                    for group in self.param_groups:
                        for p in group['params']:
                            state = self.state[p]
                            tmp = state['old'+str(w_i)] - state['cur'+str(w_n)]
                            denom += (tmp*tmp).sum()
                    denom = torch.sqrt(denom) + 1e-5


                    self.weights[w_n][w_i] = self.old_loss[w_i][w_i] - self.cur_loss[w_n][w_i]
                    self.weights[w_n][w_i] = max(0., self.weights[w_n][w_i])
                    
                    self.weights[w_n][w_i] /= denom
                    s += self.weights[w_n][w_i]
                # if s == 0.:
                #     self.weights = torch.ones(self.cfg.npeers, device=self.device) / self.cfg.npeers
                # else:
                if s > 0.:
                    self.weights[w_n] /= s
            # raise RuntimeError("Too many classes for that split")
            
            # print(f"{self.old_loss[w_i][w_i] - self.cur_loss[w_n][w_i]=}")
            # print(f"{denom=}")
            # print(f"{self.weights=}")
            # print(f"")
            #cals new current params for each client
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    for w_n in range(self.cfg.npeers):
                        state['tmp'+str(w_n)].data.copy_(torch.zeros_like(p.data, device=self.device))
                    
                    for w_i in range(self.cfg.npeers):
                        state['tmp'+str(w_i)].data.copy_(state['old'+str(w_i)].data)
                        for w_n in range(self.cfg.npeers):
                            state['tmp'+str(w_i)].data.add_(self.weights[w_i][w_n]*(state['cur'+str(w_n)] - state['old'+str(w_i)]))
                            # state['tmp'+str(w_i)].data.add_(state['old'+str(w_i)] + self.weights[w_i][w_n]*(state['cur'+str(w_n)] - state['old'+str(w_i)]))
                        # print(f"{state['tmp'+str(w_i)]=}")
                            
            #move client current params to old params 
            #move new current to current
            for group in self.param_groups:
                for p in group['params']:
                    for w_i in range(self.cfg.npeers):
                        state = self.state[p]
                        state['old'+str(w_i)].data.copy_(state['cur'+str(w_i)].data)
                        state['cur'+str(w_i)].data.copy_(state['tmp'+str(w_i)].data)
            
            ######################################
            
            ######################################
            self.grads_received = 0
            self.loaders = list()
            self.i += 1

    @torch.no_grad()
    def metrics(self):
        for i, w in enumerate(self.weights[0]):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict



class FedAvg(OptimizerBase):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        super().__init__(params, cfg, device)

        self.cfg = cfg
        self.i = 0

        self.K = 10
        self.weights = [1]*self.K + [0]*(self.cfg.npeers-self.K)

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
            if self.attack is not None:
                self.attack(self.grads)
            random.shuffle(self.weights)
            i = 0
            for group in self.param_groups:
                for p in group['params']:
                    p_step = torch.zeros_like(p, device=device)
                    for j, g in enumerate(self.grads):
                        p_step += self.weights[j] * g[i] / self.K
                    i += 1
                    p.data.add_(p_step, alpha=-self.cfg.lr)

            self.grads_received = 0
            self.i += 1

    @torch.no_grad()
    def metrics(self):
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict

class SGD(OptimizerBase):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        super().__init__(params, cfg, device)

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
            if self.attack is not None:
                self.attack(self.grads)
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
    def metrics(self):
        return self.metrics_dict

class FedAdp(OptimizerBase):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        super().__init__(params, cfg, device)

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
            if self.attack is not None:
                self.attack(self.grads)
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
    def metrics(self):
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict


class MeritFed(OptimizerBase):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        super().__init__(params, cfg, device)

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
            if self.attack is not None:
                self.attack(self.grads)
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
    def metrics(self):
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict


class TAWT(OptimizerBase):
    def __init__(self, params, cfg, device):
        if cfg.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(cfg.lr))

        super().__init__(params, cfg, device)

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
            if self.attack is not None:
                self.attack(self.grads)
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
    def metrics(self):
        for i, w in enumerate(self.weights):
            key = 'weights_%s' % (str(i))
            self.metrics_dict[key] = w.item()
        return self.metrics_dict
