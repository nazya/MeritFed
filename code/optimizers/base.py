from enum import auto
from abc import ABC, abstractmethod

from code import DictEnum
from code.problems import load_problem


class Optimizer(DictEnum):
    SGD = auto()
    SGDMD = auto()


class _OptimizerBase(ABC):
    def __init__(self, config, rank):
        self.rank = rank
        self.i = 0

        self.lr = config.lr
        self.n_peers = config.n_peers
        self.batch_size = config.batch_size
        self.problem = load_problem(config, rank)

    @abstractmethod
    def step(self):
        pass