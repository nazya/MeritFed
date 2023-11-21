from abc import ABC, abstractmethod
from enum import auto
from code import DictEnum
from collections import defaultdict
import torch

import torch
import math
import numpy as np
from collections import defaultdict
from torch.utils import data
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import time as time


class Dataset(DictEnum):
    Normal = auto()
    MNIST = auto()
    CIFAR10 = auto()


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
            # print(self.n_samples, len(self.dset))
            # print(self.dset.mean(0))
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
        elif train is None:
            self.dset = list()
            self.n_samples = 0

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
    def __init__(self, config, rank, train=True):
        root = '/tmp'
        self.root = root
        if config.n_peers and not rank and self.download and not self._check_exists():
            self.download()

        while not self._check_exists():
            time.sleep(3)

        self.n_classes = 4

        if train is None or train is False:
            super().__init__(root=root, train=False, transform=transforms.ToTensor(), download=False)
        else:
            super().__init__(root=root, train=train, transform=transforms.ToTensor(), download=False)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

        self.targets = self.targets.reshape((len(self.targets), 1)).long()
        self.data = self.data.reshape((len(self.data), 1, 28, 28)) / 255.
        self.output_dim = self.n_classes
        self.input_dim = len(self.data[0].view(-1))

        mask = self.targets == 0
        for i in range(1, self.n_classes):
            mask = torch.logical_or(mask, self.targets == i)
        indices = torch.nonzero(mask.squeeze()).squeeze()

        if train is None:
            if rank:
                raise RuntimeError("Non-master client accessed test dataset")

            indices = torch.nonzero(mask.squeeze()).squeeze()
            indices = indices[config.n_samples:]
            # print(len(self.data))
            # print('full test len ', len(self.data))

        elif train is False:
            if rank:
                raise RuntimeError("Non-master client accessed test dataset")
            indices = torch.nonzero(mask.squeeze()).squeeze()
            indices = indices[:config.n_samples]
            # print('test len ', len(self.data))

        else:
            target_rank_below = 1
            self.true_weights = torch.zeros(config.n_peers)
            self.true_weights[:target_rank_below] = 1 / target_rank_below
            near_target_rank_below = 11

            target_ratio = config.h_ratio
            if target_ratio is None:
                target_ratio = 0.5  # any

            if rank < target_rank_below:
                n = config.n_samples
                if target_rank_below*n > len(indices):
                    raise ValueError('target_rank_below*n_samples too big')

                per_worker = n
                beg = rank * per_worker
                end = beg + per_worker
                if end > len(indices) - 1:
                    raise ValueError('invalid partitioning')
                # if rank == target_rank_below - 1:
                #     print('actual1', end)
                #     print(f"{indices[end]=}")
                indices = indices[beg:end]

            elif target_rank_below <= rank and rank < near_target_rank_below:
                n = target_rank_below*config.n_samples
                if rank == target_rank_below:
                    print('calc1', n)
                indices = indices[n:]
                per_worker = int(target_ratio*config.n_samples)  # n1
                beg = (rank-target_rank_below) * per_worker
                end = beg + per_worker
                if end > len(indices) - 1:
                    raise ValueError('invalid partitioning')
                indices = indices[beg:end]

                # add others
                mask = self.targets > len(self.classes)  # False mask
                for i in range(self.n_classes):
                    m = self.targets == i + self.n_classes
                    self.targets[m] = i
                    mask = torch.logical_or(mask, m)
                more_indices = torch.nonzero(mask.squeeze()).squeeze()

                per_worker = config.n_samples - int(target_ratio*config.n_samples)
                beg = (rank-target_rank_below) * per_worker
                end = beg + per_worker
                if rank == near_target_rank_below - 1:
                    print('actual3', end)
                if end > len(more_indices) - 1:
                    raise ValueError('invalid rounding')
                # self.targets[more_indices[beg:end]] = 0
                indices = torch.cat((indices, more_indices[beg:end]), 0)

            else:
                mask = self.targets > len(self.classes)
                if 2*self.n_classes+1 > len(self.classes):
                    raise ValueError('too many classes')
                for i in range(self.n_classes):
                    if i + 2*self.n_classes > len(self.classes):
                        break
                    m = self.targets == i + 2*self.n_classes
                    self.targets[m] = i
                    mask = torch.logical_or(mask, m)
                indices = torch.nonzero(mask.squeeze()).squeeze()

                n = config.n_samples - int(target_ratio*config.n_samples)
                n *= (near_target_rank_below-target_rank_below)
                if rank == near_target_rank_below:
                    print('calc3', n)
                mask = self.targets > len(self.classes)  # False mask
                for i in range(self.n_classes):
                    m = self.targets == i + self.n_classes
                    self.targets[m] = i
                    mask = torch.logical_or(mask, m)
                more_indices = torch.nonzero(mask.squeeze()).squeeze()
                more_indices = more_indices[n:]
                indices = torch.cat((more_indices, indices), 0)

                per_worker = config.n_samples
                beg = (rank-target_rank_below) * per_worker
                end = min(beg + per_worker, len(indices) - 1)
                if end > len(indices) - 1:
                    raise ValueError('invalid partitioning')
                indices = indices[beg:end]

        self.targets = self.targets[indices]
        self.data = self.data[indices]

    def __getitem__(self, index):
        # print(index)
        img, target = self.data[index], self.targets[index]  # .float()
        return img, target.squeeze()

    def model_args(self):
        return self.input_dim, self.output_dim

    def loss_star(self, full_batch, criterion):
        return 0.
        # return criterion(full_batch[1].float(), full_batch[1]).data


class CIFAR10(datasets.CIFAR10):
    def __init__(self, config, rank, train=True):
        root = '/tmp'
        self.root = root
        if config.n_peers and not rank and self.download and not self._check_integrity():
            self.download()

        while not self._check_integrity():
            time.sleep(3)

        self.n_classes = 4

        if train is None or train is False:
            super().__init__(root=root, train=False, transform=transforms.ToTensor(), download=False)
        else:
            super().__init__(root=root, train=train, transform=transforms.ToTensor(), download=False)

        if not self._check_integrity():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.targets = torch.Tensor(self.targets)
        self.targets = self.targets.reshape((len(self.targets), 1)).long()

        mask = self.targets == 0
        for i in range(1, self.n_classes):
            mask = torch.logical_or(mask, self.targets == i)
        indices = torch.nonzero(mask.squeeze()).squeeze()

        if train is None:
            if rank:
                raise RuntimeError("Non-master client accessed test dataset")

            indices = indices[config.n_samples:]
            # print(f"{len(indices)=}")
            # print(len(self.data))
            # print('full test len ', len(self.data))

        elif train is False:
            if rank:
                raise RuntimeError("Non-master client accessed test dataset")

            indices = indices[:config.n_samples]
            # print('test len ', len(self.data))

        else:
            target_rank_below = 1
            self.true_weights = torch.zeros(config.n_peers)
            self.true_weights[:target_rank_below] = 1 / target_rank_below
            near_target_rank_below = 11

            target_ratio = config.h_ratio
            if config.h_ratio is None:
                target_ratio = 0.5  # any

            if rank < target_rank_below:
                n = config.n_samples
                if target_rank_below*n > len(indices):
                    raise ValueError('target_rank_below*n_samples too big')

                per_worker = n
                beg = rank * per_worker
                end = beg + per_worker
                if end > len(indices) - 1:
                    raise ValueError('invalid partitioning')
                # if rank == target_rank_below - 1:
                #     print('actual1', end)
                #     print(f"{indices[end]=}")
                indices = indices[beg:end]

            elif target_rank_below <= rank and rank < near_target_rank_below:
                n = target_rank_below*config.n_samples
                # if rank == target_rank_below:
                #     print('calc1', n)
                indices = indices[n:]
                per_worker = int(target_ratio*config.n_samples)  # n1
                beg = (rank-target_rank_below) * per_worker
                end = beg + per_worker
                if end > len(indices) - 1:
                    raise ValueError('invalid partitioning')
                indices = indices[beg:end]

                # add others
                mask = self.targets > len(self.classes)  # False mask
                for i in range(self.n_classes):
                    m = self.targets == i + self.n_classes
                    self.targets[m] = i
                    mask = torch.logical_or(mask, m)
                more_indices = torch.nonzero(mask.squeeze()).squeeze()

                per_worker = config.n_samples - int(target_ratio*config.n_samples)
                beg = (rank-target_rank_below) * per_worker
                end = beg + per_worker
                # if rank == near_target_rank_below - 1:
                #     print('actual3', end)
                if end > len(more_indices) - 1:
                    raise ValueError('invalid rounding')
                # self.targets[more_indices[beg:end]] = 0
                indices = torch.cat((indices, more_indices[beg:end]), 0)

            else:
                mask = self.targets > len(self.classes)
                if 2*self.n_classes+1 > len(self.classes):
                    raise ValueError('too many classes')
                for i in range(self.n_classes):
                    if i + 2*self.n_classes > len(self.classes):
                        break
                    m = self.targets == i + 2*self.n_classes
                    self.targets[m] = i
                    mask = torch.logical_or(mask, m)
                indices = torch.nonzero(mask.squeeze()).squeeze()

                n = config.n_samples - int(target_ratio*config.n_samples)
                n *= (near_target_rank_below-target_rank_below)
                # if rank == near_target_rank_below:
                #     print('calc3', n)
                mask = self.targets > len(self.classes)  # False mask
                for i in range(self.n_classes):
                    m = self.targets == i + self.n_classes
                    self.targets[m] = i
                    mask = torch.logical_or(mask, m)
                more_indices = torch.nonzero(mask.squeeze()).squeeze()
                more_indices = more_indices[n:]
                indices = torch.cat((more_indices, indices), 0)

                per_worker = config.n_samples
                beg = (rank-target_rank_below) * per_worker
                end = min(beg + per_worker, len(indices) - 1)
                if end > len(indices) - 1:
                    raise ValueError('invalid partitioning')
                indices = indices[beg:end]

        data = list()
        targets = list()
        self.targets = self.targets[indices]
        self.data = self.data[indices]
        for index, _ in enumerate(self.data):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            data.append(img)
            targets.append(target)

        self.data = torch.stack(data, dim=0)

        # self.targets = torch.tensor(targets)
        self.output_dim = self.n_classes
        self.input_dim = len(self.data[0].view(-1))

    def __getitem__(self, index):
        # print(index)
        img, target = self.data[index], self.targets[index]  # .float()
        return img, target.squeeze()

    def model_args(self):
        return self.input_dim, self.output_dim

    def loss_star(self, full_batch, criterion):
        return 0.
        # return criterion(full_batch[1].float(), full_batch[1]).data
