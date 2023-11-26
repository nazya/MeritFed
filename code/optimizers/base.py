from enum import auto
from abc import ABC, abstractmethod
from collections import defaultdict

from code import DictEnum
# from code.problems import load_problem

from code.problem import Problem


class Optimizer(DictEnum):
    SGD = auto()
    MeritFed = auto()


class _OptimizerBase(ABC):
    def __init__(self, config, rank):
        self.master_node = 0
        self.rank = rank
        self.i = 0

        self.lr = config.lr
        self.npeers = config.npeers
        self.batchsize = config.batchsize
        self.problem = Problem(config, rank)

        if rank == self.master_node:
            self.metrics_dict = defaultdict(float)

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def metrics(self):
        pass