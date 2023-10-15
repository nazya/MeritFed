import os
import sys
import json
import random
import numpy as np
from collections import namedtuple
# import mlflow
from mlflow import MlflowClient
import torch.distributed as dist
import torch.multiprocessing as mp
from code.optimizers.base import Optimizer
from code.optimizers import load_distributed_optimizer
# from .train import TrainConfig



class PortNotAvailableError(Exception):
    pass


def _train(rank: int, port: str, config):
    mp.set_sharing_strategy("file_system") 
    config = json.loads(config)
    setup(rank, config['n_peers'], port)

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
        # client.log_param(r_id, 'Title', os.environ['MLFLOW_RUN_TITLE'])

    config = namedtuple('Config', config.keys())(**config)
    optimizer = load_distributed_optimizer(config, rank)
    log_ticks = np.linspace(0, config.n_iters-1, 100, endpoint=True).round().astype(int)
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
            # sys.stdout.write(str(optimizer.game.dist())+'\r')
            # print('----------------------------------------------------------iter end', i)
            sys.stdout.write('Iterations left: ' +
                             str(config.n_iters - i - 1) + 9*' ' + '\r')
            sys.stdout.flush()

        optimizer.step()
    if verbose and rank == 0:
        client.set_terminated(r_id)
    dist.destroy_process_group()


def setup(rank: int, size: int, port: str, backend: str = 'gloo'):
    os.environ['MASTER_ADDR'] = '127.0.1.1'
    os.environ['MASTER_PORT'] = port
    try:
        dist.init_process_group(backend, rank=rank, world_size=size)
    except:
        raise PortNotAvailableError


def train(config):
    nprocs = config.n_peers
    # config = config.__dict__
    config = json.dumps(config.__dict__)
    # data = data.__dict__
    # Tries to allocate a port until a port is available
    while True:
        port = str(random.randrange(1030, 49151))
        print("Trying port %s" % port)
        try:
            mp.spawn(_train,
                     args=(port, config),
                     # nprocs=config['n_peers'],
                     # nprocs=config.n_peers,
                     nprocs=nprocs,
                     join=True)
            break
        except PortNotAvailableError:
            print("Port %s not available" % port)
        else:
            raise


# if __name__ == "__main__":
#     train()