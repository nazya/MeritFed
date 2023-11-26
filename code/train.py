import os
import sys
import json
import numpy as np
from collections import namedtuple
# import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
import torch.distributed as dist
import torch.multiprocessing as mp
import code.optimizers
import socket
from contextlib import closing

from attrs import asdict

import argparse

import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
from collections import defaultdict

import code.optimizers.single_thread


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def loss_and_accuracy(model, loader, criterion, device):
    correct = 0
    total_loss = 0

    model.eval()
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(loader.dataset)
    total_loss = total_loss / len(loader)

    return total_loss, accuracy


def train(args):
    # args = args.__dict__
    args = asdict(args)

    check_exist = eval(os.environ['MLFLOW_CHECK_EXIST'])
    if check_exist:
        tracking_uri = os.path.expanduser('~/mlruns/')
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        client = MlflowClient(tracking_uri=tracking_uri)
        e = client.get_experiment_by_name(experiment_name)
        if e is not None:
            filter_string = list()
            for key in args.keys():
                if isinstance(args[key], dict):
                    value = args[key]['name']
                else:
                    value = args[key]
                filter_string.append(f'params.{key}="{value}"')
            filter_string.append('attributes.status="FINISHED"')

            # print(f"{filter_string=}")
            filter_string = ' and '.join(filter_string)
            runs = client.search_runs(experiment_ids=[e.experiment_id],
                                      filter_string=filter_string,
                                      run_view_type=ViewType.ACTIVE_ONLY)
            if len(runs):
                return

    verbose = eval(os.environ['MLFLOW_VERBOSE'])
    if verbose:
        tracking_uri = os.path.expanduser('~/mlruns/')
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        client = MlflowClient(tracking_uri=tracking_uri)
        e = client.get_experiment_by_name(experiment_name)
        e_id = client.create_experiment(experiment_name)\
            if e is None else e.experiment_id
        r = client.create_run(experiment_id=e_id,
                              run_name=os.environ['MLFLOW_RUN_NAME'])
        r_id = r.info.run_id
        client.log_dict(r_id, args, 'config.json')
        
    args = argparse.Namespace(**args)
        
    set_seed(args.seed)

    device = torch.device(os.environ["TORCH_DEVICE"])

    model = getattr(code.models, args.model['name'])().to(device)
    criterion = getattr(torch.nn, args.loss['name'])().to(device)
    train_data, test_data = getattr(code.datasets, args.dataset['name'])()

    dataloader_kwargs = {'batch_size': args.batchsize,
                         'shuffle': True}
    use_cuda = os.environ["TORCH_DEVICE"] != 'cpu'
    if use_cuda:
        dataloader_kwargs.update({'num_workers': 0,
                                  'pin_memory': True})
    # dataset = train_dataset
    # train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    optimizer = getattr(code.optimizers.single_thread, args.optimizer['name'])
    # optimizer = globals()[args.optimizer['name']]
    optimizer = optimizer(model.parameters(), lr=args.lr, n_workers=args.npeers, momentum=0.5)

    n = len(train_data)

    indices = np.arange(n)
    np.random.shuffle(indices)

    indices = np.arange(n)
    n_val = int(np.floor(0.1 * n))
    val_data = Subset(train_data, indices=indices[:n_val])
    indices = indices[n_val:]
    n = len(indices)
    a = int(np.floor(n / args.npeers))
    top_ind = a * args.npeers
    seq = range(a, top_ind, a)
    split = np.split(indices[:top_ind], seq)

    train_loaders = list()
    for ind in split:
        # train_loaders[b] = DataLoader(Subset(train_data, ind), **dataloader_kwargs)
        train_loaders.append(torch.utils.data.DataLoader(Subset(train_data, ind), **dataloader_kwargs))
        # train_loaders.append(torch.utils.data.DataLoader(train_data, **dataloader_kwargs))
    test_loader = DataLoader(test_data, **dataloader_kwargs)
    val_loader = DataLoader(val_data, **dataloader_kwargs)

    test_acc = 0
    test_loss, test_acc = -1, -1
    nticks = min(args.nepochs, 50)
    log_ticks = np.linspace(0, args.nepochs, nticks, endpoint=True).round().astype(int)

    metrics = defaultdict(float)
    metrics["test-loss"], metrics["test-accuracy"] = loss_and_accuracy(model, test_loader, criterion, device)
    if verbose:
        for key in metrics.keys():
            client.log_metric(r_id, key,
                              metrics[key],
                              # timestamp=0.,
                              # step=optimizer.i)
                              step=0)
    print("Epoch: {}/{}.. Train Loss: {}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
          .format(0, args.nepochs, '-', metrics["test-loss"], metrics["test-accuracy"]), end='\r')
    
    # metrics["train-loss"], _
    for e in range(1,args.nepochs+1):
        model.train()
        running_loss = 0
        iters_per_epoch = 0
        train_loader_iter = [iter(loader) for loader in train_loaders]
        while True:
            try:
                for w_id,  iteraror in enumerate(train_loader_iter):
                    data, labels = next(iteraror)
                    data, labels = data.to(device), labels.to(device)
                    output = model(data)
                    loss = criterion(output, labels)
                    loss.backward()
                    running_loss += loss.item()
                    optimizer.step_local_global(w_id)
                    optimizer.zero_grad()
                iters_per_epoch += 1
            except StopIteration:
                break
                
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
#         train_loss = running_loss/(iter_steps*args.npeers)

        metrics["train-loss"] = running_loss/(args.npeers*iters_per_epoch)
        # metrics["train-accuracy"] = 
        
        metrics["test-loss"], metrics["test-accuracy"] = loss_and_accuracy(model, test_loader, criterion, device)
        if verbose and e in log_ticks:
            for key in metrics.keys():
                client.log_metric(r_id, key,
                                  metrics[key],
                                  # timestamp=0.,
                                  # step=optimizer.i)
                                  step=e)

        print("Epoch: {}/{}.. Train Loss: {:.5f}, Test Loss: {:.5f}, Test accuracy: {:.2f} "
              .format(e, args.nepochs, metrics["train-loss"], metrics["test-loss"], metrics["test-accuracy"]), end='\r')

    if verbose:
        client.set_terminated(r_id)




