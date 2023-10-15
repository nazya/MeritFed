import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from .base import _ProblemBase
from .utils import create_matrix, create_bias


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class MeanModel(torch.nn.Module):
    def __init__(self, dim):
        super(MeanModel, self).__init__()
        self.mu = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.mu
    
class Normal(Dataset):
    def __init__(self, dim, n_samples, rank, transform=None):
        self.n_samples = n_samples
        mean = torch.ones(dim) * (rank%2)
        std = torch.eye(dim)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, std)
        self.dset = dist.sample((n_samples,))
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # print(idx)
        idx = torch.IntTensor(idx)
        return torch.index_select(self.dset, 0, idx)

class Quadratic(_ProblemBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)

        dim = 10
        self.dim = dim
        self.model = MeanModel(dim)
        
        train_dataset = Normal(dim, config.n_samples, rank)
        
        sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(train_dataset),
                batch_size=config.batch_size,
                drop_last=False)

        

        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True)

        self.train_loader = DataLoader(train_dataset,
            sampler=sampler)
        
        self.full_loader = DataLoader(dataset=train_dataset,
                                      batch_size=len(train_dataset))

        self.batch = next(iter(self.train_loader))

    def sample(self, full=False):
        if full:
            loader = self.full_loader
        else:
            loader = self.train_loader
        self.batch = next(iter(loader))

    def loss(self):
        # self.batch = next(iter(self.train_loader))
        inputs = self.batch[0].view(-1, self.dim)
        outputs = self.model(inputs)

        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, self.batch[1])
        return loss.data

    def grad(self):
        self.model.zero_grad()

        inputs = self.batch[0].view(-1, self.dim)
        outputs = self.model(inputs)

        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, self.batch[1])

        loss.backward()
        grads = []
        for p in self.model.parameters():
            # print('rank: ', self.rank, torch.norm(param))
            grads.append(p.grad.detach())
        return grads

    def metrics(self) -> float:
        self.sample(full=True)
        self.metrics_dict["loss"] = self.loss().item()
        print(self.metrics_dict["loss"])
        return self.metrics_dict



class Quadratic_old(_ProblemBase):
    def __init__(self, rank, config):
        super().__init__(rank, config)

        if rank is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(rank)

        self.matrix = create_matrix(config.dim, config.n_samples, 1, 10)
        self.bias = create_bias(config.dim, config.n_samples, False)
        self.params = torch.ones(config.dim)
        self.true = torch.zeros_like(self.params)

        self.dim = sum(p.numel() for p in self.params)

    def grad(self, index) -> torch.Tensor:
        grad = 0.
        for i in index:
            grad += torch.matmul(self.matrix[i], self.params) + self.bias[i]
        return grad/len(index)

        grads = torch.matmul(self.matrix, self.params) + self.bias
        grads = torch.index_select(grads, 0, index).mean(dim=0)
        return grads

    def metrics(self) -> float:
        metrics = defaultdict(float)
        metrics["loss"] = 0.5*(self.params @ self.matrix @ self.params).mean() + (self.bias @ self.params).mean()
        print(metrics["f"])
        metrics["dist2opt"] = torch.linalg.norm(self.true - self.params)
        return metrics