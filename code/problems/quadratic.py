import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from .base import _ProblemBase
from .utils import create_matrix, create_bias




import numpy as np


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
class MeanModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MeanModel, self).__init__()
        self.mu = torch.nn.Parameter(10*torch.ones(output_dim))

    def forward(self, x):
        return self.mu

class MyDataset(Dataset):
    def __init__(self):
        N = 10 # number of data points
        m = .9
        c = 1
        np.random.seed(0)
        x = np.linspace(0,2*np.pi,N)
        y = m*x + c + np.random.normal(0,.3,x.shape)
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor([1,self.x[idx]], dtype=torch.float32), torch.tensor([self.y[idx]], dtype=torch.float32)

    
class Normal(Dataset):
    def __init__(self, dim, n_samples, rank, transform=None):
        
        self.n_samples = n_samples
        mean = torch.ones(dim) * (rank%2)
        std = torch.eye(dim)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, std)
        self.dset = dist.sample((n_samples,))
        print(self.mean())
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.zeros((1,1)), self.dset[idx]
        # print(idx)
        # idx = torch.IntTensor(idx)
        # return torch.zeros(len(idx)), torch.index_select(self.dset[1], 0, idx)
        
    def mean(self):
        return self.dset.mean(dim=0)
    

class Quadratic(_ProblemBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        

        dim = 10
        train_dataset = Normal(dim, config.n_samples, rank)
        # train_dataset = MyDataset()
        # sampler = torch.utils.data.sampler.BatchSampler(
        #         torch.utils.data.sampler.RandomSampler(train_dataset),
        #         batch_size=config.batch_size,
        #         drop_last=False)
        # self.train_loader = DataLoader(train_dataset,
        #     sampler=sampler)
        
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True)
        self.full_loader = DataLoader(dataset=train_dataset,
                                      batch_size=len(train_dataset))
        
        
        self.batch = next(iter(self.full_loader))
        self.loss_star = self.criterion(train_dataset.mean(), self.batch[1]).data
        
        
        self.batch = next(iter(self.train_loader))
        
        
        input_dim = self.batch[0].shape[1]
        self.input_dim = input_dim
        output_dim = self.batch[1].shape[1]
        self.model = MeanModel(input_dim, output_dim)
        # self.model = LinearRegressionModel(input_dim, output_dim)
        

    def sample(self, full=False):
        if full:
            loader = self.full_loader
        else:
            loader = self.train_loader
        self.batch = next(iter(loader))

    def loss(self):
        
        # self.batch = next(iter(self.train_loader))
        inputs = self.batch[0]#.view(-1, self.input_dim)
        outputs = self.model(inputs)
        # print(self.batch[0].size(), outputs.size(), self.batch[1].size())

        # criterion = torch.nn.MSELoss(reduction='mean')
        # criterion = torch.nn.MSELoss()
        loss = self.criterion(outputs, self.batch[1])
        return loss.data

    def grad(self):
        self.model.zero_grad()

        # inputs = self.batch[0].view(-1, self.input_dim)
        inputs = self.batch[0]#.view(-1, self.input_dim)
        outputs = self.model(inputs)
        # print(self.batch[0].size(), outputs.size(), self.batch[1].size())

        # criterion = torch.nn.MSELoss(reduction='mean')
        loss = self.criterion(outputs, self.batch[1])
        
        # l = 0
        # for i in self.batch[1]:
        #     l += self.criterion(outputs, i)
        # l /= len(self.batch[1])
        # print('loss', l.item(), loss.item())
        # l.backward()
        loss.backward()
        grads = []
        for p in self.model.parameters():
            # print('rank: ', self.rank, torch.norm(param))
            grads.append(p.grad.detach())
        return grads

    def metrics(self) -> float:
        self.sample(full=True)
        self.metrics_dict["loss"] = self.loss().item() - self.loss_star
        print(self.metrics_dict["loss"].item() - self.loss_star)
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