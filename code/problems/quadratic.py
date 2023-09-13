import torch
from collections import defaultdict
from .base import _ProblemBase
from .utils import create_matrix, create_bias


class Quadratic(_ProblemBase):
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