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
import socket
from contextlib import closing

import code.optimizers.distributed
from code.data_utils import split
from attrs import asdict
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import defaultdict
from torch.utils.data import DataLoader, Subset


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def _train(rank, cfg, train_data):
    # mp.set_sharing_strategy("file_system") 
    cfg = json.loads(cfg)
    dist.init_process_group('gloo', rank=rank, world_size=cfg['npeers'])
    
    verbose = os.environ['MLFLOW_VERBOSE']
    if verbose and rank == 0:
        tracking_uri = os.path.expanduser('~/mlruns/')
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        client = MlflowClient(tracking_uri=tracking_uri)
        e = client.get_experiment_by_name(experiment_name)
        e_id = client.create_experiment(experiment_name)\
            if e is None else e.experiment_id
        r = client.create_run(experiment_id=e_id,
                              run_name=os.environ['MLFLOW_RUN_NAME'])
        r_id = r.info.run_id
        client.log_dict(r_id, cfg, 'cfg.json')
        for key, value in cfg.items():
            if isinstance(value, dict):
                value = value['name']
            client.log_param(r_id, key, value)
        # client.log_param(r_id, 'Title', os.environ['MLFLOW_RUN_TITLE'])

    cfg = namedtuple('cfg', cfg.keys())(**cfg)
    
    
    optimizer = getattr(code.optimizers.distributed, cfg.optimizer['name'])
    optimizer = optimizer(cfg, rank)
    
    if verbose and rank == 0:
        metrics = optimizer.metrics()
        for key in metrics.keys():
            client.log_metric(r_id, key,
                              metrics[key],
                              # timestamp=0.,
                              step=optimizer.i)

        metrics = optimizer.problem.metrics()
        for key in metrics.keys():
            client.log_metric(r_id, key,
                              metrics[key],
                              # timestamp=0.,
                              step=optimizer.i)
    sys.stdout.write('Iterations left: ' +
                             str(cfg.niters - 0 - 1) + 9*' ' + '\r')
    sys.stdout.flush()
    

    log_ticks = np.linspace(0, cfg.niters-1, 50, endpoint=True).round().astype(int)
    for i in range(1, cfg.niters+1):
        optimizer.step()

        
        if verbose and rank == 0 and i in log_ticks:
            metrics = optimizer.metrics()
            for key in metrics.keys():
                client.log_metric(r_id, key,
                                  metrics[key],
                                  # timestamp=0.,
                                  step=optimizer.i)

            metrics = optimizer.problem.metrics()
            for key in metrics.keys():
                client.log_metric(r_id, key,
                                  metrics[key],
                                  # timestamp=0.,
                                  step=optimizer.i)
            sys.stdout.write('Iterations left: ' +
                             str(cfg.niters - i - 1) + 9*' ' + '\r')
            sys.stdout.flush()

    if verbose and rank == 0:
        client.set_terminated(r_id)
    dist.destroy_process_group()
    # print(f"{rank=} destroy complete")


def train(cfg):
    nprocs = cfg.npeers
    cfg = asdict(cfg)
    
    check_exist = eval(os.environ['MLFLOW_CHECK_EXIST'])
    if check_exist:
        tracking_uri = os.path.expanduser('~/mlruns/')
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        client = MlflowClient(tracking_uri=tracking_uri)
        e = client.get_experiment_by_name(experiment_name)
        if e is not None:
            filter_string = list()
            for key in cfg.keys():
                if isinstance(cfg[key], dict):
                    value = cfg[key]['name']
                else:
                    value = cfg[key]
                filter_string.append(f'params.{key}="{value}"')
            filter_string.append('attributes.status="FINISHED"')

            # print(f"{filter_string=}")
            filter_string = ' and '.join(filter_string)
            runs = client.search_runs(experiment_ids=[e.experiment_id],
                                      filter_string=filter_string,
                                      run_view_type=ViewType.ACTIVE_ONLY)
            if len(runs):
                return
    
    

    root = '/tmp'
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    train_split = split(train_data, cfg['npeers'], cfg['hratio'])
    
    
    
    cfg = json.dumps(cfg)

    master_addr = '127.0.0.1'
    master_port = find_free_port()
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    processes = []
    for rank in range(nprocs):
        p = mp.Process(target=_train, args=(rank,cfg, train_split[rank],))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
        
        
    # mp.spawn(_train,
    #          args=(master_addr, master_port,cfg,),
    #          nprocs=nprocs)