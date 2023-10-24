import torch
from enum import auto
from code import DictEnum

import torch.nn as nn
import torch.nn.functional as F


class Model(DictEnum):
    Linear = auto()
    Mean = auto()
    Net = auto()


def load_model(config, args):
    if config.model == Model.Linear:
        return Linear(*args)
    elif config.model == Model.Mean:
        return Mean(*args)
    elif config.model == Model.Net:
        return Net(*args)
    else:
        raise ValueError()


class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # x = x.view(20, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        return x
        # output = F.log_softmax(x, dim=1)
        # output = F.softmax(x, dim=1)
        # return output


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