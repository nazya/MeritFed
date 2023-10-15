import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from .base import _ProblemBase


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out


class Logistic(_ProblemBase):
    def __init__(self, config, rank):
        super().__init__(config, rank)

        train_dataset = dsets.MNIST(root='/tmp',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    # target_transform=transforms.ToTensor(),
                                    download=True)

        self.input_dim = 28*28
        output_dim = 10
        self.model = LogisticRegressionModel(self.input_dim, output_dim)

#         per_worker = int(math.ceil(len(train_dataset) / float(config.n_peers)))
#         if per_worker < config.n_samples:
#             # print("n_samples set to ", per_worker)
#             self.per_worker = per_worker
#         else:
#             self.per_worker = config.n_samples

#         beg = rank * per_worker
#         end = min(beg + per_worker, len(train_dataset)-1)
#         train_dataset.targets = train_dataset.targets[beg:end]
#         train_dataset.data = train_dataset.data[beg:end]

#         train_dataset.targets = train_dataset.targets.clone()
#         mask = train_dataset.targets == rank % output_dim

#         ratio = config.h_ratio
#         # cond = int(mask.sum()/len(mask) > ratio)
#         # n_switch = round((1-2*cond)*(len(mask)*ratio - int(mask.sum())))
#         # idx = torch.where(mask == cond)[0]
#         # idx_switch = np.random.choice(idx, size=n_switch, replace=False)
#         # mask[idx_switch] = not cond
#         n_switch = round(ratio*(len(mask) - int(mask.sum())))
#         idx = torch.where(mask == False)[0]
#         idx_switch = np.random.choice(idx, size=n_switch, replace=False)
#         mask[idx_switch] = True

#         train_dataset.targets = train_dataset.targets[mask]
#         train_dataset.data = train_dataset.data[mask]
#         print('rank %i: n_samples %i' % (rank, len(train_dataset.targets)))

        # mask = train_dataset.targets == rank % output_dim 
    
        mask = train_dataset.targets != 0
        indices = torch.nonzero(mask).squeeze()
        per_worker = int(math.ceil(len(indices) / float(config.n_peers)))
            
        beg = rank * per_worker
        end = min(beg + per_worker, len(indices)-1)
        indices = indices[beg:end]
        
        
        if rank % output_dim == 0:
            mask = train_dataset.targets == 0 
            g_indices = torch.nonzero(mask).squeeze()
            n_g = int(config.n_peers / output_dim + config.n_peers % output_dim)
            print(n_g)
            
            per_worker = int(math.ceil(len(g_indices) / float(n_g)))
            for i in range(n_g):
                if i == int(rank / output_dim):
                    beg = i * per_worker
                    end = min(beg + per_worker, len(g_indices)-1)
                    g_indices = g_indices[beg:end]
                    print('skdjf', len(g_indices))
                    ratio = config.h_ratio
                    n = math.ceil(config.n_samples*(1-ratio))
                    indices[:n] = g_indices[:n]


        indices = indices[:config.n_samples]
        if len(indices) < config.n_samples:
            print("n_samples set to ", len(indices))

        mask = torch.zeros_like(mask).scatter_(0, indices, 1)
        train_dataset.targets = train_dataset.targets[mask]
        train_dataset.data = train_dataset.data[mask]
        print('rank %i: n_samples %i' % (rank, len(train_dataset.targets)))

            
        
        
        
        self.train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=config.batch_size,
                                       shuffle=True)

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
        inputs = self.batch[0].view(-1, self.input_dim)
        outputs = self.model(inputs)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, self.batch[1])
        return loss.data

    def grad(self):
        self.model.zero_grad()

        inputs = self.batch[0].view(-1, self.input_dim)
        outputs = self.model(inputs)

        criterion = torch.nn.CrossEntropyLoss()
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
        # print(self.metrics_dict["loss"])
        return self.metrics_dict




#         self.data = torch.randn(config.n_samples, config.dim).to(torch.double)
#         self.true = (torch.randn(config.dim)).to(torch.double)
#         self.labels = torch.rand(config.n_samples).to(torch.double) < 1 / (1+torch.exp(self.data@self.true))
#         self.labels = self.labels.double()
#         self.labels = 2*self.labels - 1

#         self.params = Variable(torch.ones(config.dim).to(torch.double), requires_grad=True)

#     def loss(self, index):
#         loss = 0.
#         for i in index:
#             pred = torch.sigmoid(self.data[i]@self.params)
#             # print('la', pred, self.labels[i])
#             loss += -self.labels[i]*torch.log(pred)
#             loss += -(1-self.labels[i])*torch.log(1-pred)
#         ret = loss/len(index)
#         return ret

#     def grad(self, index: int) -> torch.Tensor:
#         loss = self.loss(index)
#         ret = autograd.grad(loss, self.params, retain_graph=True)[0]
#         grad = 0.
#         for i in index:
#             pred = torch.sigmoid(self.data[i]@self.params)
#             grad += (pred-self.labels[i])*self.data[i]
#         # print('norm', torch.norm(grad/len(index) - ret))
#         return ret

#     def metrics(self) -> float:
#         metrics = defaultdict(float)
#         metrics["loss"] = self.loss(self.full()).item()
#         print(metrics["loss"])
#         return metrics




