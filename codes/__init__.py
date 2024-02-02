from .utils import DictEnum#, haha_seed

from enum import auto
# from codes import DictEnum


class Loss(DictEnum):
    MSELoss = auto()
    CrossEntropyLoss = auto()