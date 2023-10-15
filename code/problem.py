from abc import ABC, abstractmethod
from enum import auto
from code import DictEnum
from collections import defaultdict
import torch
from code.models import load_model
from code.datasets import load_dataset
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy


class Loss(DictEnum):
    MSE = auto()
    CrossEntropy = auto()


def load_loss(config):
    if config.loss == Loss.MSE:
        return torch.nn.MSELoss(reduction='mean')
    if config.loss == Loss.CrossEntropy:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError()


class _ProblemBase(ABC):
    def __init__(self, config, rank) -> None:
        self.master_node = 0
        self.rank = rank
        self.size = config.n_peers

        if rank is None:
            torch.manual_seed(config.seed)
        else:
            torch.manual_seed(config.seed + rank)

        if rank == self.master_node:
            self.metrics_dict = defaultdict(float)

    @abstractmethod
    def metrics(self):
        pass


class Problem(_ProblemBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.config = config
        self.criterion = load_loss(config)
        
        self.rank = rank
        if rank == 0:
            test_dataset = load_dataset(config, rank, train=False)
            test_sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(test_dataset),
                batch_size=config.batch_size,
                drop_last=False)
            self.test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=None, num_workers=0)

            test_sampler_full = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(test_dataset),
                batch_size=len(test_dataset),
                drop_last=False)

            test_loader_full = DataLoader(test_dataset, sampler=test_sampler_full, batch_size=None, num_workers=0)

            self.test_full_batch = next(iter(test_loader_full))
            self.loss_star = 0.  # test_dataset.loss_star(self.test_full_batch, self.criterion)

            test = load_dataset(config, rank, train=None)
            test_full = torch.utils.data.sampler.BatchSampler(
                    torch.utils.data.sampler.RandomSampler(test),
                    batch_size=len(test),
                    drop_last=False)
            test_full = DataLoader(test, sampler=test_full, batch_size=None, num_workers=0)
            self.full_test = next(iter(test_full))
            print('full test len ', len(self.full_test[0]))

        dataset = load_dataset(config, rank)
        self.accuracy = Accuracy(task="multiclass", num_classes=dataset.n_classes+1, top_k=1)
        self.dataset = dataset
        sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(dataset),
                batch_size=config.batch_size,
                drop_last=False)
        self.loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=0)
        self.full_loader = DataLoader(dataset=dataset,
                                      sampler=torch.utils.data.sampler.BatchSampler(
                                          torch.utils.data.sampler.RandomSampler(dataset),
                                          batch_size=config.n_samples, drop_last=False), batch_size=None, num_workers=0)
        self.full_batch = next(iter(self.full_loader))
        
        self.model = load_model(config, dataset.model_args())

    def sample(self, full=False):
        if full:
            self.batch = self.full_batch
        else:
            self.batch = next(iter(self.loader))

    def sample_test(self, full=False):
        if full:
            self.batch = self.test_full_batch
        else:
            self.batch = next(iter(self.test_loader))

    def loss(self):
        inputs = self.batch[0]
        outputs = self.model(inputs)
        # if self.rank == 0:
        #     print(outputs.dtype)
        #     print(outputs.shape)
        #     print(self.batch[1].dtype)
        #     print(self.batch[1].shape)
        #     print()
        loss = self.criterion(outputs, self.batch[1])
        return loss.data

    def grad(self):
        self.model.zero_grad()

        inputs = self.batch[0]
        outputs = self.model(inputs)

        loss = self.criterion(outputs, self.batch[1])
        loss.backward()
        grads = []
        for p in self.model.parameters():
            # print('grad s ', p.grad.shape)
            grads.append(p.grad.detach())
        return torch.stack(grads, 0)
        return grads

    def metrics(self) -> float:
        # self.sample_test(full=self.config.md_full_)
        self.sample_test(full=True)
        self.metrics_dict["loss"] = self.loss().item() - self.loss_star
        self.batch = self.full_test
        self.metrics_dict["loss-full"] = self.loss().item() - self.loss_star
        if self.accuracy is not None:
            inputs = self.batch[0]
            outputs = self.model(inputs)
            self.metrics_dict["accuracy"] = self.accuracy(outputs, self.batch[1])
        # x = list()
        # for p in self.model.parameters():
        #     x.append(p.data)
        # x = torch.stack(x, 0)
        # x = x.detach()
        # self.metrics_dict["loss"] = torch.norm(x)
        # self.metrics_dict["loss"] = self.loss().item() - torch.norm(x)
        # self.metrics_dict["loss"] = self.problem.model
        # self.metrics_dict["loss"] = self.loss().item() - self.loss_star
        # # print(self.metrics_dict["loss"].item() - self.loss_star)
        return self.metrics_dict