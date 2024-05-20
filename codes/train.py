import os
import attrs
import torch
import argparse

# import numpy as np
from numpy import linspace
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
# import mlflow
# from mlflow import MlflowClient
# from mlflow.entities import ViewType

from codes.utils import fix_seed
from codes.logger import MLFlowLogger
from codes.data_utils import split_train, split_test

import codes.models as models
import codes.datasets as datasets
import codes.optimizers.single_thread as optimizers

from sklearn.metrics import accuracy_score
import numpy as np


def expected_loss(model):
    loss = 0.
    for p in model.parameters():
        loss += torch.sum(p*p, dim=0)
    return loss


def metrics(model, loader, prefix, criterion, device, classes=False):
    correct, total_loss = 0, 0

    model.eval()
    for batch in loader:
        batch = [data.to(device) for data in batch]
        output = model(*batch[:-1])
        if hasattr(output, "logits"):
            output = output.logits
        loss = criterion(output, batch[-1])
        total_loss += loss.item()
        if classes:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(batch[-1].view_as(pred)).sum().item()

    # for data, labels in loader:
    #     # print(f"{abels=}")
    #     data, labels = data.to(device), labels.to(device)
    #     output = model(data)
    #     loss = criterion(output, labels)
    #     total_loss += loss.item()
    #     if classes:
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct += pred.eq(labels.view_as(pred)).sum().item()

    m = defaultdict(float)
    m[prefix+'-loss'] = total_loss / len(loader)
    if classes:
        # print(f"{len(loader.dataset)=}")
        m[prefix+'-accuracy'] = 100. * correct / len(loader.dataset)
    return m


def train(cfg):
    if cfg.valenabled_ == False and cfg.mdfull_ == True:
        print(f"Incompatible config {cfg.valenabled_=} and {cfg.mdfull_=}")
        return
    config = attrs.asdict(cfg)
    logger = MLFlowLogger()
    if logger.check_exist(config):
        return
    logger.enabled = eval(os.environ['MLFLOW_VERBOSE'])
    logger.init(config)

    fix_seed(cfg.seed)
    device = torch.device(os.environ["TORCH_DEVICE"])

    train_data = getattr(datasets, cfg.dataset['name'])(cfg, train=True)
    if cfg.npeers == 40:
        train_split, weights, shapes = split_train(train_data, cfg)
    else:
        train_split, weights, shapes = split_train(train_data, cfg)

    config.update({'weights': weights})
    cfg = argparse.Namespace(**config)

    dl_kwargs = {'batch_size': cfg.batchsize, 'shuffle': True,  # 'pin_memory': True,
                 'num_workers': 0}
    train_loaders = [DataLoader(i, **dl_kwargs) for i in train_split]

    model = getattr(models, cfg.model['name'])(*shapes).to(device)
    criterion = getattr(torch.nn, cfg.loss['name'])().to(device)
    Optimizer = getattr(optimizers, cfg.optimizer['name'])
    optimizer = Optimizer(model.parameters(), cfg, device)

    test_data = getattr(datasets, cfg.dataset['name'])(cfg, train=False)
    classes = True if hasattr(test_data, 'classes') else False

    val_loader = None
    nsamples = cfg.valnsamples_
    if cfg.valenabled_ is False:
        _, test_data = split_test(test_data, cfg, nsamples)
        val_data = train_split[0]
    else:
        val_data, test_data = split_test(test_data, cfg, nsamples)
    # print(f"{len(val_data)=}")
    # print(f"{len(train_split[0])=}")
    # print(f"{len(test_data)=}")
    # dl_kwargs.update({'batch_size': len(val_data) if cfg.mdfull_ else cfg.mdbatchsize_})
    dl_kwargs.update({'batch_size': len(val_data) if cfg.mdfull_ else cfg.mdbatchsize_})
    val_loader = DataLoader(val_data, **dl_kwargs)
    if test_data is not None:
        # dl_kwargs.update({'batch_size': 1000})  # any number fitting memmory
        dl_kwargs.update({'batch_size': 1})  # any number fitting memmory
        test_loader = DataLoader(test_data, **dl_kwargs)

    if logger.enabled:
        m = defaultdict(float)
        m.update(metrics(model, train_loaders[0], 'train', criterion, device, classes))

        if test_data is None:
            m["expected-loss"] = expected_loss(model)
        else:
            m.update(metrics(model, test_loader, 'test', criterion, device, classes))

        m.update(optimizer.metrics())
        logger.log_metrics(m, 0)

    if hasattr(optimizer, 'init_losses'):
        optimizer.init_losses(model, train_loaders, criterion)
    nticks = min(cfg.nepochs, 50)
    # log_ticks = np.linspace(0, cfg.nepochs, nticks, endpoint=True).round().astype(int)
    log_ticks = linspace(0, cfg.nepochs, nticks, endpoint=True).round().astype(int)
    for e in range(1, cfg.nepochs+1):
        model.train()
        running_loss, correct, samples_per_epoch = 0, 0, 0
        train_loader_iter = [iter(loader) for loader in train_loaders]
        iter_steps = len(train_loaders[0])
        for _ in range(iter_steps):
            for w_id,  iteraror in enumerate(train_loader_iter):
                batch = next(iteraror)
                batch = [data.to(device) for data in batch]

                # print(f"")
                # print(f"{len(batch)=}")
                output = model(*batch[:-1])
                if hasattr(output, "logits"):
                    output = output.logits
                loss = criterion(output, batch[-1])
#                 print(f"{w_id, loss=}")

#                 print(f"")
                # running_loss += loss.item()
                if w_id == 0:
                    running_loss += loss.item() * len(batch[-1])
                # data, labels = next(iteraror)
                # data, labels = data.to(device), labels.to(device)
                # output = model(data)
                # loss = criterion(output, labels)
                # loss.backward()
                # running_loss += loss.item()
                # if w_id == 0:
                #     running_loss += loss.item() * len(labels)
                    # running_loss += loss.item()
                    # samples_per_epoch += len(labels)
                    # print(f"{len(labels)=}")
                    if logger.enabled and classes:
                        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                        correct += pred.eq(batch[-1].view_as(pred)).sum().item()
                if hasattr(optimizer, 'before_backward'):
                    optimizer.before_backward(w_id, model)
                loss.backward()
                # if cfg.optimizer['name'] == ''
                # optimizer.step(w_id, model, train_loaders[w_id], criterion, device)
                optimizer.step(w_id, model, val_loader, criterion, device)
                optimizer.zero_grad()
        # train_loss = running_loss/(iter_steps*cfg.npeers)

#     for e in range(1, cfg.nepochs+1):
#         model.train()
#         running_loss, correct, samples_per_epoch = 0, 0, 0
#         train_iters = [iter(loader) for loader in train_loaders]
#         while True:
#             try:
#                 for w_id,  iteraror in enumerate(train_iters):
#                     data, labels = next(iteraror)
#                     data, labels = data.to(device), labels.to(device)
#                     optimizer.zero_grad()
#                     output = model(data)
#                     loss = criterion(output, labels)
#                     loss.backward()
#                     if w_id == 0:
#                         running_loss += loss.item() * len(labels)
#                         samples_per_epoch += len(labels)
#                         if logger.enabled and classes:
#                             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#                             correct += pred.eq(labels.view_as(pred)).sum().item()

#                     optimizer.step(w_id, model, val_loader, criterion, device)

#             except StopIteration:
#                 break
        if logger.enabled and e in log_ticks:
            m.update(metrics(model, train_loaders[0], 'train', criterion, device, classes))

            # m["train-loss"] = running_loss / len(train_loaders[0].dataset)
            # m["train-loss"] = running_loss / samples_per_epoch

            if classes:
                running_accuracy = 100. * correct / len(train_loaders[0].dataset)
                # running_accuracy = 100. * correct / samples_per_epoch
                m["train-accuracy"] = running_accuracy

            if test_data is None:
                m["expected-loss"] = expected_loss(model)
            else:
                m.update(metrics(model, test_loader, 'test', criterion, device, classes))

            m.update(optimizer.metrics())
            logger.log_metrics(m, optimizer.i)

    logger.terminate()


#     for e in range(epochs):
#         model.train()
#         running_loss = 0
#         train_loader_iter = [iter(loader) for loader in train_loaders]
#         iter_steps = len(train_loaders[0])
#         for _ in range(iter_steps):
#             for w_id,  iteraror in enumerate(train_loader_iter):
#                 data, labels = next(iteraror)
#                 data, labels = data.to(device), labels.to(device)
#                 output = model(data)
#                 loss = criterion(output, labels)
#                 loss.backward()
#                 running_loss += loss.item()
#                 optimizer.step_local_global(w_id)
#                 optimizer.zero_grad()
#         train_loss = running_loss/(iter_steps*cfg.npeers)