from abc import ABC, abstractmethod
from enum import auto
from code import DictEnum
from collections import defaultdict
import torch
# from code.models import load_model
import code.models
# from code.datasets import load_dataset
import code.datasets
import torch
import numpy as np
import os
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy


class Loss(DictEnum):
    MSELoss = auto()
    CrossEntropyLoss = auto()


def load_loss(config):
    if config.loss == Loss.MSELoss:
        return torch.nn.MSELoss(reduction='mean')
    if config.loss == Loss.CrossEntropyLoss:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError()


class _ProblemBase(ABC):
    def __init__(self, config, rank) -> None:
        self.master_node = 0
        self.rank = rank
        self.size = config.npeers

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

        dataset = getattr(code.datasets, config.dataset['name'])(config, rank)
        # self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.device = torch.device('cuda:4')

        self.accuracy = None
        if hasattr(dataset, 'classes'):
            # self.accuracy = Accuracy(task="multiclass", num_classes=len(dataset.classes), top_k=1).to(self.device)
            self.accuracy = Accuracy(task="multiclass", num_classes=dataset.n_classes, top_k=1).to(self.device)

        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=config.batchsize, num_workers=0)
        self.iter = iter(self.loader)
        self.model = getattr(code.models, config.model['name'])(*dataset.model_args())
        # self.model.share_memory()
        self.model.to(self.device)
        self.model.train()

        if rank == 0:
            md_test_dataset = getattr(code.datasets, config.dataset['name'])(config, rank, train=False)
            self.md_test_loader = DataLoader(md_test_dataset, batch_size=config.batchsize, num_workers=0)
            self.md_test_iter = iter(self.md_test_loader)
            md_test_loader_full = DataLoader(md_test_dataset, batch_size=config.nsamples, num_workers=0)
            self.md_test_full = next(iter(md_test_loader_full))
            self.md_test_full[0] = self.md_test_full[0].to(self.device)
            self.md_test_full[1] = self.md_test_full[1].to(self.device)

            self.loss_star = 0  # md_test_dataset.loss_star(self.md_test_full, self.criterion)

        if rank == 0:
            test_dataset = getattr(code.datasets, config.dataset['name'])(config, rank, train=False)
            if len(test_dataset):
                self.test_loader = DataLoader(test_dataset, batch_size=config.batchsize, num_workers=0)
                self.test_loader_full = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0)
                self.test_full = next(iter(self.test_loader_full))
                self.test_full[0] = self.test_full[0].to(self.device)
                self.test_full[1] = self.test_full[1].to(self.device)
            else:
                self.test_full = None

    def batch_to_device(self):
        self.batch[0] = self.batch[0].to(self.device)
        self.batch[1] = self.batch[1].to(self.device)

    @torch.no_grad()
    def sample(self, full=False):
        if full:
            self.batch = self.full_batch
        else:
            try:
                self.batch = next(self.iter)
            except StopIteration:
                self.iter = iter(self.loader)
                self.batch = next(self.iter)
            self.batch_to_device()

    @torch.no_grad()
    def sample_test(self, full=False):
        if full:
            self.batch = self.md_test_full
        else:
            try:
                self.batch = next(self.md_test_iter)
            except StopIteration:
                self.md_test_iter = iter(self.md_test_loader)
                self.batch = next(self.md_test_iter)
            self.batch_to_device()

    @torch.no_grad()
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

    @torch.no_grad()
    def metrics(self) -> float:
        self.model.eval()
        correct = 0
        total_loss = 0
        for data, labels in self.loader:
            data, labels = data.to(self.device), labels.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, labels)
            total_loss += loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
        self.metrics_dict["train-loss"] = total_loss / len(self.loader)
        self.metrics_dict["train-accuracy"] = 100. * correct / len(self.loader.dataset)

        # if self.accuracy is not None:
        #     inputs = self.batch[0]
        #     outputs = self.model(inputs)
        #     # outputs.to(self.device)
        #     self.metrics_dict["train-accuracy"] = self.accuracy(outputs, self.batch[1])

        if self.test_full is not None:
            self.batch = self.test_full
            self.batch_to_device()
            self.metrics_dict["test-loss"] = self.loss().item() - self.loss_star

            if self.accuracy is not None:
                inputs = self.batch[0]
                outputs = self.model(inputs)
                # outputs.to(self.device)
                self.metrics_dict["test-accuracy"] = self.accuracy(outputs, self.batch[1])
 
        else:
            loss = 0.
            for p in self.model.parameters():
                loss += torch.norm(p.data)
            self.metrics_dict["expected-loss"] = torch.norm(loss)
        self.model.train()
        return self.metrics_dict

#         self.model.eval()
#         correct = 0
#         total_loss = 0
#         for data, labels in self.loader:
#             data, labels = data.to(self.device), labels.to(self.device)
#             output = self.model(data)
#             loss = self.criterion(output, labels)
#             total_loss += loss.item()

#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(labels.view_as(pred)).sum().item()

#         self.metrics_dict["train-accuracy"] = 100. * correct / len(self.loader.dataset)
#         self.metrics_dict["train-loss"] = total_loss / len(self.loader)

#         self.model.eval()
#         correct = 0
#         total_loss = 0
#         for data, labels in self.test_loader:
#             data, labels = data.to(self.device), labels.to(self.device)
#             output = self.model(data)
#             loss = self.criterion(output, labels)
#             total_loss += loss.item()

#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(labels.view_as(pred)).sum().item()

#         self.metrics_dict["test-accuracy"] = 100. * correct / len(self.test_loader.dataset)
#         self.metrics_dict["test-loss"] = total_loss / len(self.test_loader)

#         self.model.train()
        return self.metrics_dict