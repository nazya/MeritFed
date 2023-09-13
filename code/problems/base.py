from abc import ABC, abstractmethod
from enum import auto
from code import DictEnum
from collections import defaultdict
import torch
# from .robust_regression import RobustLinRegConfig, RobustLinReg, RobustLogReg


class Problem(DictEnum):
    Quadratic = auto()
    Logistic = auto()
    # ROBUST_LOGISTIC_REG = "robust_logistic_regression"


class _ProblemBase(ABC):
    def __init__(self, config, rank) -> None:
        self.master_node = 0
        self.rank = rank
        self.size = config.n_peers

        if rank is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(rank)

        if rank == self.master_node:
            self.metrics_dict = defaultdict(float)

    @abstractmethod
    def metrics(self):
        pass