from collections import defaultdict
from torch.utils.data import Subset
import random as random
import itertools


def split_test(dset, cfg, mdnsamples):
    if hasattr(dset, 'split_test'):
        return dset.split_test(mdnsamples)

    # mdnsamples = cfg.mdnsamples
    nclasses = cfg.nclasses
    val_inds, test_inds = list(), list()
    for i in range(nclasses):
        indices = [j for j, x in enumerate(dset.targets) if x == i]
        random.shuffle(indices)

        val_inds += indices[:mdnsamples]
        test_inds += indices[mdnsamples:]

#     for i in range(nclasses, len(dset.classes)):
#         indices = [j for j, x in enumerate(dset.targets) if x == i]
#         random.shuffle(indices)

#         val_inds += indices[:mdnsamples]

    return Subset(dset, val_inds), Subset(dset, test_inds)


# def split_train(dset, cfg):
#     if hasattr(dset, 'split_train'):
#         return dset.split_train()

#     npeers = cfg.npeers
#     hratio = cfg.hratio
#     # nclasses = 4
#     d = defaultdict(list)
#     m = len(dset)
#     for i, c in enumerate(dset.classes):
#         indices = [j for j, x in enumerate(dset.targets) if x == i]
#         random.shuffle(indices)
#         d[i] = indices
#         m = min(m, len(indices))

#     for i, _ in enumerate(dset.targets):
#         pass
#         # dset.targets[i] %= nclasses

#     target_rank_below, near_target_rank_below = 1, 11

#     trueweights = [1 / target_rank_below if i < target_rank_below else 0 for i in range(npeers)]

#     indices_split = [list() for _ in range(npeers)]

#     # m //= npeers
#     m //= near_target_rank_below - target_rank_below
#     if hratio is None:
#         hratio = 0.5  # no matter
#     rest = None
#     for rank, _ in enumerate(indices_split):
#         if rank < target_rank_below:
#             for i in range(nclasses):
#                 if len(d[i]) < m:
#                     raise RuntimeError("Wrong indexing")
#                 indices_split[rank] += d[i][:m]
#                 d[i] = d[i][m:]
#         elif target_rank_below <= rank and rank < near_target_rank_below:
#             for i in range(nclasses):
#                 n = int(m * hratio)

#                 if len(d[i+nclasses]) < m-n:
#                     raise RuntimeError("Wrong indexing")
#                 indices_split[rank] += d[i+nclasses][:m-n]
#                 d[i+nclasses] = d[i+nclasses][m-n:]

#                 if len(d[i]) < n:
#                     raise RuntimeError("Wrong indexing")
#                 indices_split[rank] += d[i][:n]
#                 d[i] = d[i][n:]

#         else:
#             if rest is None:
#                 rest = list()
#                 for i in range(nclasses, len(dset.classes)):
#                     rest += d[i]

#             if len(rest) < nclasses*m:
#                 raise RuntimeError("Wrong indexing")
#             indices_split[rank] += rest[:nclasses*m]
#             rest = rest[nclasses*m:]

#     # for i, ind in enumerate(indices_split):
#     #     print(f"{len(ind)=}")

#     return [Subset(dset, inds) for inds in indices_split], trueweights, (None, 10)
#     return [Subset(dset, inds) for inds in indices_split], trueweights, (None, nclasses)


# release
def split_train(dset, cfg):
    if hasattr(dset, 'split_train'):
        return dset.split_train()

    nclasses = cfg.nclasses
    if nclasses > 3:
        raise RuntimeError("Too many classes for that split")

    npeers = cfg.npeers
    hratio = cfg.hratio

    d = defaultdict(list)
    m = len(dset)
    for i, c in enumerate(dset.classes):
        indices = [j for j, x in enumerate(dset.targets) if x == i]
        random.shuffle(indices)
        d[i] = indices
        m = min(m, len(indices))

    # for i, _ in enumerate(dset.targets):
    #     dset.targets[i] %= nclasses

    target_rank_below, near_target_rank_below = 1, 11

    trueweights = [1 / target_rank_below if i < target_rank_below else 0 for i in range(npeers)]

    indices_split = [list() for _ in range(npeers)]

    m //= npeers
    # print(f"{m=}")
    if hratio is None:
        hratio = 0.5  # no matter
    for rank, _ in enumerate(indices_split):
        if rank < target_rank_below:
            for i in range(nclasses):
                if len(d[i]) < m:
                    raise RuntimeError("Wrong indexing")
                indices_split[rank] += d[i][:m]
                d[i] = d[i][m:]
        elif target_rank_below <= rank and rank < near_target_rank_below:
            for i in range(nclasses):
                n = int(m * hratio)
                if len(d[i+nclasses]) < m-n:
                    raise RuntimeError("Wrong indexing")
                indices_split[rank] += d[i+nclasses][:m-n]
                d[i+nclasses] = d[i+nclasses][m-n:]

                indices_split[rank] += d[i][:n]
                d[i] = d[i][n:]
        else:
            for i in range(nclasses):
                if len(d[i+2*nclasses]) < m:
                    print(f"{len(d[i+2*nclasses])=}")
                    raise RuntimeError("Wrong indexing")
                indices_split[rank] += d[i+2*nclasses][:m]
                d[i+2*nclasses] = d[i+2*nclasses][m:]

    return [Subset(dset, inds) for inds in indices_split], trueweights, (None, 10)




def split_train40(dset, cfg):
    if hasattr(dset, 'split_train'):
        return dset.split_train()

    nclasses = cfg.nclasses
    if nclasses > 3:
        raise RuntimeError("Too many classes for that split")

    if cfg.npeers != 40:
        raise RuntimeError("Only for 40 peers")
        
    npeers = cfg.npeers // 2
    hratio = cfg.hratio

    d = defaultdict(list)
    m = len(dset)
    for i, c in enumerate(dset.classes):
        indices = [j for j, x in enumerate(dset.targets) if x == i]
        random.shuffle(indices)
        d[i] = indices
        m = min(m, len(indices))

    # for i, _ in enumerate(dset.targets):
    #     dset.targets[i] %= nclasses

    target_rank_below, near_target_rank_below = 1, 11

    trueweights = [1 / target_rank_below if i < 2*target_rank_below else 0 for i in range(cfg.npeers)]

    indices_split = [list() for _ in range(npeers)]

    m //= npeers
    # print(f"{m=}")
    if hratio is None:
        hratio = 0.5  # no matter
    for rank, _ in enumerate(indices_split):
        if rank < target_rank_below:
            for i in range(nclasses):
                if len(d[i]) < m:
                    raise RuntimeError("Wrong indexing")
                indices_split[rank] += d[i][:m]
                d[i] = d[i][m:]
        elif target_rank_below <= rank and rank < near_target_rank_below:
            for i in range(nclasses):
                n = int(m * hratio)
                if len(d[i+nclasses]) < m-n:
                    raise RuntimeError("Wrong indexing")
                indices_split[rank] += d[i+nclasses][:m-n]
                d[i+nclasses] = d[i+nclasses][m-n:]

                indices_split[rank] += d[i][:n]
                d[i] = d[i][n:]
        else:
            for i in range(nclasses):
                if len(d[i+2*nclasses]) < m:
                    print(f"{len(d[i+2*nclasses])=}")
                    raise RuntimeError("Wrong indexing")
                indices_split[rank] += d[i+2*nclasses][:m]
                d[i+2*nclasses] = d[i+2*nclasses][m:]

    indices_split = list(itertools.chain.from_iterable([[i,i] for i in indices_split]))
    return [Subset(dset, inds) for inds in indices_split], trueweights, (None, 10)
