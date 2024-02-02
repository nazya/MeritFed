import os
import attrs
import torch
import random
import argparse

# import numpy as np
from numpy import linspace
from torch.utils.data import DataLoader
from collections import defaultdict
# import mlflow
# from mlflow import MlflowClient
# from mlflow.entities import ViewType

from codes.utils import fix_seed
from codes.logger import MLFlowLogger
from codes.data_utils import split_train, split_test

import codes.models as models
import codes.datasets as datasets
import codes.optimizers.distributed as optimizers


import json
import socket
from contextlib import closing
import torch.distributed as dist
import torch.multiprocessing as mp


def expected_loss(model):
    loss = 0.
    for p in model.parameters():
        loss += torch.sum(p*p, dim=0)
    return loss


def metrics(model, loader, prefix, criterion, device, classes=False):
    correct, total_loss = 0, 0

    model.eval()
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()
        if classes:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    m = defaultdict(float)
    m[prefix+'-loss'] = total_loss / len(loader)
    if classes:
        m[prefix+'-accuracy'] = 100. * correct / len(loader.dataset)
    return m


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def train(cfg):
    config = attrs.asdict(cfg)
    logger = MLFlowLogger()
    if logger.check_exist(config):
        return
    logger.enabled = eval(os.environ['MLFLOW_VERBOSE'])
    logger.init(config)

    fix_seed(cfg.seed)
    device = torch.device(os.environ["TORCH_DEVICE"])

    train_data = getattr(datasets, cfg.dataset['name'])(cfg, train=True)
    train_split, weights, shapes = split_train(train_data, cfg)
    config.update({'weights': weights})

    model = getattr(models, cfg.model['name'])(*shapes).to(device)
    model.share_memory()

    shared_grads = list()
    for i in range(cfg.npeers):
        p_grads = list()
        for p in model.parameters():
            p_grads.append(torch.zeros_like(p, device=device).share_memory_())
        shared_grads.append(p_grads)
    # model = None

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = find_free_port()
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(cfg.npeers):
        p = mp.Process(target=_train,
                       args=(rank, json.dumps(config), train_split[rank],
                             model, device, shared_grads, logger))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    for p in processes:
        p.kill()

    # for i in range(cfg.npeers):
    #     for j, p in enumerate(model.parameters()):
    #         del shared_grads[i][-1]
    #     # del shared_grads[i]
    # del shared_grads
    # del model

    # dist.destroy_process_group()
    # mp.spawn(_train,
    #          args=(master_addr, master_port,cfg,),
    #          nprocs=nprocs)


def _train(rank, config, train_data, model, device, shared_grads, logger):
    config = json.loads(config)
    cfg = argparse.Namespace(**config)
    if cfg.valenabled == False and cfg.mdfull_ == True:
        print(f"Incompatible config {cfg.valenabled=} and {cfg.mdfull_=}")
        return
    fix_seed(cfg.seed + rank, False)
    dist.init_process_group('gloo', rank=rank, world_size=cfg.npeers)

    criterion = getattr(torch.nn, cfg.loss['name'])().to(device)
    Optimizer = getattr(optimizers, cfg.optimizer['name'])
    optimizer = Optimizer(model.parameters(), rank, cfg, device)

    dl_kwargs = {'batch_size': cfg.batchsize, 'shuffle': True,
                 'num_workers': 0}  # 'pin_memory': True,

    train_loader = DataLoader(train_data, **dl_kwargs)
    val_loader = None
    if not rank:
        test_data = getattr(datasets, cfg.dataset['name'])(cfg, train=False)
        classes = True if hasattr(test_data, 'classes') else False

        nsamples = cfg.valnsamples
        if cfg.valenabled is False:
            _, test_data = split_test(test_data, cfg, nsamples)
            val_data = train_data
        else:
            val_data, test_data = split_test(test_data, cfg, nsamples)
            
        dl_kwargs.update({'batch_size': len(val_data) if cfg.mdfull_ else cfg.mdbatchsize_})
        val_loader = DataLoader(val_data, **dl_kwargs)

        if test_data is not None:
            dl_kwargs.update({'batch_size': 1000})
            test_loader = DataLoader(test_data, **dl_kwargs)

    logger.enabled = logger.enabled and not rank
    if logger.enabled:
        m = defaultdict(float)
        m.update(metrics(model, train_loader, 'train', criterion, device, classes))

        if test_data is None:
            m["expected-loss"] = expected_loss(model)
        else:
            m.update(metrics(model, test_loader, 'test', criterion, device, classes))

        m.update(optimizer.metrics())
        logger.log_metrics(m, 0)

    nticks = min(cfg.nepochs, 50)
    log_ticks = linspace(0, cfg.nepochs, nticks, endpoint=True).round().astype(int)
    for e in range(1, cfg.nepochs+1):
        model.train()
        running_loss, correct = 0, 0
        train_iter = iter(train_loader)
        while True:
            try:
                data, labels = next(train_iter)
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                running_loss += loss.item() * len(labels)
                if logger.enabled and classes:
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(labels.view_as(pred)).sum().item()

                optimizer.step(shared_grads, model, val_loader, criterion, device)
            except StopIteration:
                break

        if logger.enabled and e in log_ticks:
            m["train-loss"] = running_loss / len(train_loader.dataset)

            if classes:
                running_accuracy = 100. * correct / len(train_loader.dataset)
                m["train-accuracy"] = running_accuracy

            if test_data is None:
                m["expected-loss"] = expected_loss(model)
            else:
                m.update(metrics(model, test_loader, 'test', criterion, device, classes))

            m.update(optimizer.metrics())
            logger.log_metrics(m, optimizer.i)

    logger.terminate()
    dist.destroy_process_group()


# def _train(rank, cfg, train_data):
#     # mp.set_sharing_strategy("file_system")
#     cfg = json.loads(cfg)
#     dist.init_process_group('gloo', rank=rank, world_size=cfg['npeers'])

#     verbose = os.environ['MLFLOW_VERBOSE']
#     if verbose and rank == 0:
#         tracking_uri = os.path.expanduser('~/mlruns/')
#         experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
#         client = MlflowClient(tracking_uri=tracking_uri)
#         e = client.get_experiment_by_name(experiment_name)
#         e_id = client.create_experiment(experiment_name)\
#             if e is None else e.experiment_id
#         r = client.create_run(experiment_id=e_id,
#                               run_name=os.environ['MLFLOW_RUN_NAME'])
#         r_id = r.info.run_id
#         client.log_dict(r_id, cfg, 'cfg.json')
#         for key, value in cfg.items():
#             if isinstance(value, dict):
#                 value = value['name']
#             client.log_param(r_id, key, value)
#         # client.log_param(r_id, 'Title', os.environ['MLFLOW_RUN_TITLE'])

#     cfg = namedtuple('cfg', cfg.keys())(**cfg)

#     optimizer = getattr(codes.optimizers.distributed, cfg.optimizer['name'])
#     optimizer = optimizer(cfg, rank)

#     if verbose and rank == 0:
#         metrics = optimizer.metrics()
#         for key in metrics.keys():
#             client.log_metric(r_id, key,
#                               metrics[key],
#                               # timestamp=0.,
#                               step=optimizer.i)

#         metrics = optimizer.problem.metrics()
#         for key in metrics.keys():
#             client.log_metric(r_id, key,
#                               metrics[key],
#                               # timestamp=0.,
#                               step=optimizer.i)
#     sys.stdout.write('Iterations left: ' +
#                              str(cfg.niters - 0 - 1) + 9*' ' + '\r')
#     sys.stdout.flush()


#     log_ticks = np.linspace(0, cfg.niters-1, 50, endpoint=True).round().astype(int)
#     for i in range(1, cfg.niters+1):
#         optimizer.step()

#         if verbose and rank == 0 and i in log_ticks:
#             metrics = optimizer.metrics()
#             for key in metrics.keys():
#                 client.log_metric(r_id, key,
#                                   metrics[key],
#                                   # timestamp=0.,
#                                   step=optimizer.i)

#             metrics = optimizer.problem.metrics()
#             for key in metrics.keys():
#                 client.log_metric(r_id, key,
#                                   metrics[key],
#                                   # timestamp=0.,
#                                   step=optimizer.i)
#             sys.stdout.write('Iterations left: ' +
#                              str(cfg.niters - i - 1) + 9*' ' + '\r')
#             sys.stdout.flush()

#     if verbose and rank == 0:
#         client.set_terminated(r_id)
#     dist.destroy_process_group()
#     # print(f"{rank=} destroy complete")