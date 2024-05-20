import torch
import random
from copy import deepcopy
from collections import defaultdict
from torch.optim.optimizer import Optimizer
from enum import auto
from codes import DictEnum


# from torch.autograd.functional import hessian


class Attack(DictEnum):
    BitFlipping = auto()
    IPM = auto()
    RandomNoise = auto()
    ALIE = auto()


class BitFlipping():
    def __init__(self, config) -> None:
        self.n_peers = config.npeers
        self.n_byzan = 50
        self.n_attacking = self.n_byzan

    def __call__(self, grads):
        for i in range(self.n_peers):
            if i >= self.n_peers - self.n_byzan:
                for g in grads[i]:
                    g.data.copy_(-g)

    def __str__(self):
        return "BitFlipping"


class IPM():
    def __init__(self, config) -> None:
        self.n_peers = config.npeers
        self.n_byzan = 50
        self.ipm_epsilon = 1e-1

    def __call__(self, grads):
        
        sum_good_grads = [torch.zeros_like(g) for g in grads[0]] 
        for i in range(self.n_peers):
            for j, g in enumerate(grads[i]):
                sum_good_grads[j] += g
        len_good_grads = self.n_peers
        
        for i in range(self.n_peers):
            if i >= self.n_peers - self.n_byzan:
                for j, g in enumerate(grads[i]):
                    g.data.copy_(-self.ipm_epsilon * sum_good_grads[j] / len_good_grads)
        
    def __str__(self):
        return "InnerProductManipulation"


class RandomNoise():
    def __init__(self, config) -> None:
        self.rn_sigma = 1e1
        self.n_peers = config.npeers
        self.n_byzan = 50

    def __call__(self, grads):
        for i in range(self.n_peers):
            if i >= self.n_peers - self.n_byzan:
                for g in grads[i]:
                    g.data.copy_(self.rn_sigma*torch.randn_like(g))

    def __str__(self):
        return "RandomNoise"


class ALIE():
    def __init__(self, config) -> None:
        self.alie_z = 1e2
        
        self.n_peers = config.npeers
        self.n_byzan = 50

    def __call__(self, grads):
        n_byzan = self.n_byzan
        n_peers = self.n_peers

        if self.alie_z is None:
            s = np.floor(n_peers / 2 + 1) - n_byzan
            cdf_value = (n_peers - n_byzan - s) / (n_peers - n_byzan)
            z_max = norm.ppf(cdf_value)
            # print(z_max, n_peers, n_byzan, s, (n_peers - n_byzan - s) / (n_peers - n_byzan))
        else:
            z_max = self.alie_z

        good_grads = [list() for g in grads[0]] 
        for i in range(self.n_peers):
            for j, g in enumerate(grads[i]):
                good_grads[j].append(g)

        stacked_gradients = [torch.stack(good_grads[i], 1) for i, g in enumerate(grads[0])]
        mu = [torch.mean(stacked_gradients[i], 1) for i, g in enumerate(grads[0])]
        std = [torch.std(stacked_gradients[i], 1) for i, g in enumerate(grads[0])]
        
        for i in range(self.n_peers):
            if i >= self.n_peers - self.n_byzan:
                for j, g in enumerate(grads[i]):
                    g.data.copy_(mu[j] - std[j]* z_max)
                    

    def __str__(self):
        return "ALittleIsEnough"