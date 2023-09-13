import math
import torch


def random_vector(dim: int) -> torch.Tensor:
    x = torch.zeros(dim).normal_() / math.sqrt(dim)
    x.requires_grad_()
    return x


def create_matrix(dim, n_samples, mu, ell):
    A = torch.randn(n_samples, dim, dim,
                    requires_grad=False)
    e, V = torch.linalg.eigh(A@A.transpose(1, 2))
    e -= torch.min(e, 1)[0][:, None]
    e /= torch.max(e, 1)[0][:, None]
    e *= ell - mu
    e += mu
    A = V @ torch.diag_embed(e) @ V.transpose(1, 2)
    return A.clone()


def create_bias(dim, n_samples, with_bias):
    bias = torch.zeros(n_samples, dim)
    if with_bias:
        bias = 10*bias.normal_() / math.sqrt(dim)
    # s, _ = torch.linalg.eigh(A[0])
    # print('Matrix generated', s.min(), s.max())
    return bias.clone()
