from abc import ABC, abstractmethod
from enum import auto
from code import DictEnum
from collections import defaultdict
import torch
from code.models import load_model

import torch
import math
import numpy as np
from collections import defaultdict
from torch.utils import data
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

class Dataset(DictEnum):
    Normal = auto()
    MNIST = auto()


def load_dataset(config, rank, train=True):
    if config.dataset == Dataset.Normal:
        return Normal(config, rank, train=train)
    if config.dataset == Dataset.MNIST:
        return MNIST(config, rank, train=train)
    else:
        raise ValueError()


class Normal(data.Dataset):
    def __init__(self, config, rank, train=True):
        dim = 10
        self.dim = dim
        self.n_samples = config.n_samples
        if train is False:
            if rank:
                raise RuntimeError("Non-master client accessed test dataset")
            self.n_samples *= 1
            mean = 0.0 * torch.ones(dim)
            std = torch.eye(dim)
            
            dist = MultivariateNormal(mean, std)
            self.dset = dist.sample((self.n_samples,))
            print(self.n_samples, len(self.dset))
            print(self.dset.mean(0))
        elif train is True:
            target_rank_below = 5
            self.true_weights = torch.zeros(config.n_peers)
            self.true_weights[:target_rank_below] = 1 / target_rank_below
            if rank < target_rank_below:
                mean = 0.0 * torch.ones(dim)
            elif target_rank_below <= rank and rank < 100:
                mean = torch.ones(dim)
                mean /= torch.norm(mean)
                mean *= config.mu_normal
            else:
                s = slice(0, dim, 2)
                mean = torch.ones(dim)
                mean[s] = 0
                mean /= torch.norm(mean)
                # mean /= 2

            std = torch.eye(dim)
            dist = MultivariateNormal(mean, std)
            self.dset = dist.sample((self.n_samples,))
        # print(self.mean())
    

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if hasattr(idx, '__len__'):
            return torch.zeros((len(idx), 1)), self.dset[idx]
        else:
            return torch.zeros((1, 1)), self.dset[idx]

    def model_args(self):
        return (self.dim,)

    def loss_star(self, full_batch, criterion):
        mean = self.dset.mean(dim=0)
        return criterion(mean.expand((len(full_batch[0]), *mean.shape)), full_batch[1]).data


class MNIST(datasets.MNIST):
# class MNIST(datasets.FashionMNIST):
    def __init__(self, config, rank, train=True):
        root = '/tmp'
        self.root = root
        if config.n_peers and not rank and self.download:
            self.download()

        self.n_classes = 8

        if train is None or train is False:
            super().__init__(root=root, train=False, transform=transforms.ToTensor(), download=False)
        else:
            super().__init__(root=root, train=train, transform=transforms.ToTensor(), download=False)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()
        self.targets = self.targets.reshape((len(self.targets), 1)).long()
        self.data = self.data.reshape((len(self.data), -1)) / 255.
        self.output_dim = self.n_classes+1
        # self.output_dim = len(self.classes)
        self.input_dim = len(self.data[0].view(-1))

        if train is None:
            if rank:
                raise RuntimeError("Non-master client accessed test dataset")
            mask = self.targets == 1
            for i in range(2, self.n_classes+1):
                mask = torch.logical_or(mask, self.targets == i)
            self.targets[self.targets == self.n_classes] = 0
            indices = torch.nonzero(mask.squeeze()).squeeze()
            # indices = indices[:config.n_samples]
            self.targets = self.targets[indices]
            self.data = self.data[indices]
            print(len(self.data))
            print('full test len ', len(self.data))
            return

        if train is False:
            if rank:
                raise RuntimeError("Non-master client accessed test dataset")
            mask = self.targets == 1
            for i in range(2, self.n_classes+1):
                mask = torch.logical_or(mask, self.targets == i)
            self.targets[self.targets == self.n_classes] = 0
            indices = torch.nonzero(mask.squeeze()).squeeze()
            indices = indices[:config.n_samples]
            self.targets = self.targets[indices]
            self.data = self.data[indices]
            print('test len ', len(self.data))
            return

        target_rank_below = 1
        self.true_weights = torch.zeros(config.n_peers)
        self.true_weights[:target_rank_below] = 1 / target_rank_below
        near_target_rank_below = 10

        target_ratio = config.h_ratio

        # mask = self.targets == 0
        # more_indices = torch.nonzero(mask.squeeze()).squeeze()
        # for i in range(self.n_classes+1, len(self.classes)):
        #     mask = torch.logical_or(mask, self.targets == i)
        #     more_indices = torch.cat((more_indices, torch.nonzero(mask.squeeze()).squeeze()), 0)

        mask = self.targets == 1
        for i in range(2, self.n_classes):
            mask = torch.logical_or(mask, self.targets == i)
        
        indices = torch.nonzero(mask.squeeze()).squeeze()
        # print('skldjfklasjdf', len(indices))
        if rank < target_rank_below:
            n = config.n_samples - config.n_samples//self.n_classes
            if target_rank_below*n > len(indices):
                raise ValueError('target_rank_below*n_samples too big')
            per_worker = n
            beg = rank * per_worker
            end = beg + per_worker
            if end > len(indices) - 1:
                raise ValueError('invalid partitioning')
            if rank == target_rank_below - 1:
                print('actual1', end)
            indices = indices[beg:end]

            mask = self.targets == self.n_classes
            more_indices = torch.nonzero(mask.squeeze()).squeeze()
            per_worker = config.n_samples - n
            beg = (rank) * per_worker
            end = beg + per_worker
            if end > len(more_indices) - 1:
                raise ValueError('invalid rounding')
            if rank == target_rank_below - 1:
                print('actual2', end)
            self.targets[more_indices[beg:end]] *= 0
            indices = torch.cat((indices, more_indices[beg:end]), 0)

        elif target_rank_below <= rank and rank < near_target_rank_below:
            n = config.n_samples - config.n_samples//self.n_classes # n1
            if rank == target_rank_below:
                print('calc1', target_rank_below*n)
            indices = indices[target_rank_below*n:]
            per_worker = n
            beg = (rank-target_rank_below) * per_worker
            end = beg + per_worker
            if end > len(indices) - 1:
                raise ValueError('invalid partitioning')
            indices = indices[beg:end]

            # add different
            mask = self.targets == self.n_classes
            more_indices = torch.nonzero(mask.squeeze()).squeeze()
            n = (config.n_samples - n)*target_rank_below
            if rank == target_rank_below:
                print('calc2', n)
            more_indices = more_indices[n:]
            per_worker = int((target_ratio)*config.n_samples//self.n_classes) # n2
            beg = (rank-target_rank_below) * per_worker
            end = beg + per_worker
            if end > len(more_indices) - 1:
                raise ValueError('invalid rounding')
            self.targets[more_indices[beg:end]] *= 0
            indices = torch.cat((indices, more_indices[beg:end]), 0)

            # add others
            mask = self.targets == 0
            more_indices = torch.nonzero(mask.squeeze()).squeeze()
            for i in range(self.n_classes+1, len(self.classes)):
                mask = torch.logical_or(mask, self.targets == i)
                more_indices = torch.cat((more_indices, torch.nonzero(mask.squeeze()).squeeze()), 0)
            per_worker = config.n_samples - len(indices)
            if per_worker < 0:
                raise ValueError('too low ratio')
            beg = (rank-target_rank_below) * per_worker
            end = beg + per_worker
            if rank == near_target_rank_below - 1:
                print('actual3', end)
            if end > len(more_indices) - 1:
                raise ValueError('invalid rounding')
            self.targets[more_indices[beg:end]] *= 0
            indices = torch.cat((indices, more_indices[beg:end]), 0)

        # per_worker = config.n_samples - per_worker
            # beg = (rank-target_rank_below) * per_worker
            # end = beg + per_worker
            # if end > len(more_indices) - 1:
            #     raise ValueError('invalid rounding')
            # # more_indices = more_indices[beg:end]
            # self.targets[more_indices[beg:end]] *= 0
            # # self.targets[more_indices] += 1
            # indices = torch.cat((indices, more_indices[beg:end]), 0)
            # # self.targets[indices] += 1
        else:
            mask = self.targets == 0
            more_indices = torch.nonzero(mask.squeeze()).squeeze()
            for i in range(self.n_classes+1, len(self.classes)):
                mask = torch.logical_or(mask, self.targets == i)
                more_indices = torch.cat((more_indices, torch.nonzero(mask.squeeze()).squeeze()), 0)

            n = target_rank_below*(n1)
            n = (config.n_samples - n2 - n)*(near_target_rank_below-target_rank_below)

            if rank == near_target_rank_below:
                print('calc3', n)
            more_indices = more_indices[n:]
            # mask = self.targets > self.n_classes
            # indices = torch.nonzero(mask.squeeze()).squeeze()
            per_worker = config.n_samples
            beg = (rank-target_rank_below) * per_worker
            end = min(beg + per_worker, len(more_indices) - 1)
            if end > len(more_indices) - 1:
                raise ValueError('invalid partitioning')
            indices = more_indices[beg:end]
            self.targets[indices] *= 0

        self.targets = self.targets[indices]
        self.data = self.data[indices]
        if len(self.targets) != config.n_samples:
            raise ValueError(f'config failed {len(self.targets)} mismatch {config.n_samples}')

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]  # .float()
        return img, target.squeeze()

    def model_args(self):
        return self.input_dim, self.output_dim

    def loss_star(self, full_batch, criterion):
        return 0.
        return criterion(full_batch[1].float(), full_batch[1]).data



# class _Normal(data.Dataset):
#     def __init__(self, config, rank):
#         dim = 10
#         self.dim = dim
#         self.n_samples = config.n_samples
#         mean = torch.ones(dim) * (rank % 2)
#         std = torch.eye(dim)
#         dist = MultivariateNormal(mean, std)
#         self.dset = dist.sample((self.n_samples,))
#         # print(self.mean())

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, idx):
#         return torch.zeros((len(idx), self.dim)), self.dset[idx]
#         # print(idx)
#         # idx = torch.IntTensor(idx)
#         # return torch.zeros(len(idx)), torch.index_select(self.dset[1], 0, idx)

#     def model_args(self):
#         return (self.dim,)

#     def loss_star(self, full_batch, criterion):
#         mean = self.dset.mean(dim=0)
#         return criterion(mean.expand((len(full_batch[0]), *mean.shape)), full_batch[1]).data



# class MNIST(datasets.MNIST):
#     def __init__(self, config, rank):
#         root = '/tmp'
#         self.root = root
#         if config.n_peers and not rank and self.download:
#             self.download()

#         torch.distributed.barrier()
#         super().__init__(root=root, train=True, transform=transforms.ToTensor(), download=False)

#         if not self._check_exists():
#             raise RuntimeError("Dataset not found. You can use download=True to download it")

#         self.data, self.targets = self._load_data() 
#         target_hratio = 0.1
#         mask = self.targets == 0
#         g_indices = torch.nonzero(mask).squeeze()
#         # if rank == 0:
            

#         # self.output_dim = len(self.classes)
#         # self.input_dim = len(self.data[0].view(-1))

#         mask = self.targets != 0
#         indices = torch.nonzero(mask).squeeze()
#         per_worker = int(math.ceil(len(indices) / float(config.n_peers)))

#         beg = rank * per_worker
#         end = min(beg + per_worker, len(indices) - 1)
#         indices = indices[beg:end]

#         if rank % self.output_dim == 0:
#             mask = self.targets == 0
#             g_indices = torch.nonzero(mask).squeeze()
#             n_g = int(config.n_peers / self.output_dim + config.n_peers % self.output_dim)
#             print(n_g)

#             per_worker = int(math.ceil(len(g_indices) / float(n_g)))
#             for i in range(n_g):
#                 if i == int(rank / self.output_dim):
#                     beg = i * per_worker
#                     end = min(beg + per_worker, len(g_indices) - 1)
#                     g_indices = g_indices[beg:end]
#                     # print('skdjf', len(g_indices))
#                     ratio = config.h_ratio
#                     n = math.ceil(config.n_samples * (1 - ratio))
#                     indices[:n] = g_indices[:n]

#         indices = indices[:config.n_samples]
#         if len(indices) < config.n_samples:
#             print("n_samples actual: ", len(indices))

#         mask = torch.zeros_like(mask).scatter_(0, indices, 1)
#         self.targets = self.targets[mask].float()
#         self.data = self.data[mask]
#         print('rank: ', rank, ' targets ', sum(self.targets == 0))

#     def model_args(self):
#         return self.input_dim, self.output_dim
    
#     def loss_star(self, full_batch, criterion):
#         return criterion(full_batch[1], full_batch[1]).data