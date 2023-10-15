import torch
from enum import auto
from code import DictEnum


class Model(DictEnum):
    Linear = auto()
    Mean = auto()


def load_model(config, args):
    if config.model == Model.Linear:
        return Linear(*args)
    if config.model == Model.Mean:
        return Mean(*args)
    else:
        raise ValueError()


class Linear(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        # print(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        # out = self.linear(x.view(-1, self.input_dim))
        out = self.linear(x)
        return out


class Mean(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.ones(dim) / (dim**0.5))

    def forward(self, x):
        # print(self.mu.expand((len(x), *self.mu.shape)).shape)
        return self.mu.expand((len(x), *self.mu.shape))