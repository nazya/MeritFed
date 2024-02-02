from enum import auto
from abc import ABC, abstractmethod
from collections import defaultdict

from codes import DictEnum


class Optimizer(DictEnum):
    SGD = auto()
    MeritFed = auto()
    TAWT = auto()
    FedAdp = auto()