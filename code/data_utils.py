from collections import defaultdict
from torch.utils.data import DataLoader, Subset
import torch
import random as random
# import numpy as np


def split(dset, npeers, hratio):

    nclasses = 3
    d = defaultdict(list)
    m = len(dset)
    for i, c in enumerate(dset.classes):
        indices = [j for j, x in enumerate(dset.targets) if x == i]
        random.shuffle(indices)
        d[i] = indices
        m = min(m, len(indices))

    for i, _ in enumerate(dset.targets):
        dset.targets[i] %= nclasses

    
    target_rank_below, near_target_rank_below = 1, 11

    trueweights = torch.zeros(npeers)
    trueweights[:target_rank_below] = 1 / target_rank_below
    
    indices_split = [list() for _ in range(npeers)]

    m //= 2
    for rank, _ in enumerate(indices_split):
        if rank < target_rank_below:
            for i in range(nclasses):
                indices_split[rank] += d[i][:m]
        elif target_rank_below <= rank and rank < near_target_rank_below:
            for i in range(nclasses):
                n = int(m * hratio)
                indices_split[rank] += d[i][m:m+n]
                indices_split[rank] += d[i+nclasses][:m-n]
        else:
            for i in range(nclasses):
                indices_split[rank] += d[i+2*nclasses][:m]
    
    return [Subset(dset, inds) for inds in indices_split]
