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

        dataset = load_dataset(config, rank)
        if hasattr(dataset, 'n_classes'):
            self.accuracy = Accuracy(task="multiclass", num_classes=dataset.n_classes, top_k=1)
        else:
            self.accuracy = None

        self.dataset = dataset
        # sampler = torch.utils.data.sampler.BatchSampler(
        #         torch.utils.data.sampler.RandomSampler(dataset),
        #         batch_size=config.batch_size,
        #         drop_last=False)
        # self.loader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=0)
        self.loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=0)
        # self.full_loader = DataLoader(dataset=dataset,
        #                               sampler=torch.utils.data.sampler.BatchSampler(
        #                                   torch.utils.data.sampler.RandomSampler(dataset),
        #                                   batch_size=config.n_samples, drop_last=False), batch_size=None, num_workers=0)
        self.full_loader = DataLoader(dataset=dataset, batch_size=config.n_samples, num_workers=0)
        self.full_batch = next(iter(self.full_loader))

        self.model = load_model(config, dataset.model_args())
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:4")
        self.model.to(self.device)

        if rank == 0:
            md_test_dataset = load_dataset(config, rank, train=False)
            # test_sampler = torch.utils.data.sampler.BatchSampler(
            #     torch.utils.data.sampler.RandomSampler(md_test_dataset),
            #     batch_size=config.batch_size,
            #     drop_last=False)
            # self.md_test_loader = DataLoader(md_test_dataset, sampler=test_sampler, batch_size=None, num_workers=0)
            self.md_test_loader = DataLoader(md_test_dataset, batch_size=config.batch_size, num_workers=0)
            # test_sampler_full = torch.utils.data.sampler.BatchSampler(
            #     torch.utils.data.sampler.RandomSampler(md_test_dataset),
            #     batch_size=len(md_test_dataset),
            #     drop_last=False)
            # md_test_loader_full = DataLoader(md_test_dataset, sampler=test_sampler_full, batch_size=None, num_workers=0)
            md_test_loader_full = DataLoader(md_test_dataset, batch_size=config.n_samples, num_workers=0)
            self.md_test_full = next(iter(md_test_loader_full))
            self.loss_star = 0  # md_test_dataset.loss_star(self.md_test_full, self.criterion)

        if rank == 0:
            test_dataset = load_dataset(config, rank, train=None)
            if len(test_dataset):
                # test_full = torch.utils.data.sampler.BatchSampler(
                #         torch.utils.data.sampler.RandomSampler(test_dataset),
                #         batch_size=len(test_dataset),
                #         drop_last=False)
                # test_full = DataLoader(test_dataset, sampler=test_full, batch_size=None, num_workers=0)
                test_full = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0)
                self.test_full = next(iter(test_full))
            else:
                self.test_full = None
                
    def batch_to_device(self):
        self.batch[0] = self.batch[0].to(self.device)
        self.batch[1] = self.batch[1].to(self.device)

    def sample(self, full=False):
        if full:
            self.batch = self.full_batch
        else:
            self.batch = next(iter(self.loader))
        self.batch_to_device()

    def sample_test(self, full=False):
        if full:
            self.batch = self.md_test_full
        else:
            self.batch = next(iter(self.md_test_loader))
        self.batch_to_device()

    def loss(self):
        inputs = self.batch[0]
        outputs = self.model(inputs)
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
            grads.append(p.grad.detach())
        return grads

    def metrics(self) -> float:
        # self.sample_test(full=self.config.md_full_)
        # self.sample_test(full=True)
        # self.metrics_dict["loss"] = self.loss().item() - self.loss_star

        if self.test_full is not None:
            self.batch = self.test_full
            self.batch_to_device()
            self.metrics_dict["loss"] = self.loss().item() - self.loss_star
            
            # if self.accuracy is not None:
            #     inputs = self.batch[0]
            #     outputs = self.model(inputs)
            #     self.metrics_dict["accuracy"] = self.accuracy(outputs, self.batch[1])

        else:
            loss = 0.
            for p in self.model.parameters():
                loss += torch.norm(p.data)
            self.metrics_dict["expected-loss"] = torch.norm(loss)

        
        return self.metrics_dict