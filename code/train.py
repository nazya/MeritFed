import os
import sys
import json
import numpy as np
from collections import namedtuple
# import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from code.optimizers import load_distributed_optimizer
import socket
from contextlib import closing



def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def _train(rank: int, addr: int, port: str, config):
    # mp.set_sharing_strategy("file_system")
    config = json.loads(config)
    setup(rank, addr, port, config['n_peers'])

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
        client.log_dict(r_id, config, 'config.json')
        for key in config.keys():
            if isinstance(config[key], dict):
                value = config[key]['name']
            else:
                value = config[key]
            client.log_param(r_id, key, value)

    config = namedtuple('Config', config.keys())(**config)
    optimizer = load_distributed_optimizer(config, rank)
    log_ticks = np.linspace(0, config.n_iters-1, 50, endpoint=True).round().astype(int)
    for i in range(config.n_iters):
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
                             str(config.n_iters - i - 1) + 9*' ' + '\r')
            sys.stdout.flush()

        optimizer.step()
    if verbose and rank == 0:
        client.set_terminated(r_id)
    dist.destroy_process_group()
    # print(f"{rank=} destroy complete")


def setup(rank, master_addr, master_port, world_size, backend='gloo'):
    # print(f'setting up {rank=} {world_size=} {backend=}')
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    # print(f"{master_addr=} {master_port=}")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # print(f"{rank=} init complete")

    # dist.destroy_process_group()
    # print(f"{rank=} destroy complete")


def train(config):
    torch.device('cuda:5')
    nprocs = config.n_peers
    config = config.__dict__

    check_exist = os.environ['MLFLOW_CHECK_EXIST']
    if check_exist:
        tracking_uri = os.path.expanduser('~/mlruns/')
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        client = MlflowClient(tracking_uri=tracking_uri)
        e = client.get_experiment_by_name(experiment_name)
        if e is not None:
            filter_string = list()
            for key in config.keys():
                if isinstance(config[key], dict):
                    value = config[key]['name']
                else:
                    value = config[key]
                filter_string.append(f'params.{key}="{value}"')
            filter_string.append('attributes.status="FINISHED"')

            # print(f"{filter_string=}")
            filter_string = ' and '.join(filter_string)
            runs = client.search_runs(experiment_ids=[e.experiment_id],
                                      filter_string=filter_string,
                                      run_view_type=ViewType.ACTIVE_ONLY)
            if len(runs):
                return

    config = json.dumps(config)

    master_addr = '127.0.0.1'
    master_port = find_free_port()
    mp.spawn(_train,
             args=(master_addr, master_port, config,),
             nprocs=nprocs)