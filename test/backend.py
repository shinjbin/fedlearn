import os
import copy
import random
import time


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process


"""논-블로킹(non-blocking) 점-대-점 간 통신"""


def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def send(parameters, dst, group=None, tag=0):
    for parameter in parameters:
        dist.send(parameter, dst, group, tag)


def recv(parameters, src, group=None, tag=0):
    for parameter in parameters:
        dist.recv(parameter, src, group, tag)


def k_selection(parameters_list, k):
    return random.choices(list=parameters_list, k=k)