import os
import time
import subprocess

import numpy as np
import torch
import torch.distributed as dist

from data import create_dataset, custom_collate_fn

try:
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex")

class Engine(object):
    def __init__(self, opt):
        self.devices = None
        self.distributed = False

        self.opt = opt

        # multi node mode on slurm cluster
        if 'SLURM_JOB_NUM_NODES' in os.environ and int(os.environ['SLURM_JOB_NUM_NODES']) > 1:
            # number of nodes / node ID
            node_id = int(os.environ['SLURM_NODEID'])
            n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])

            # local rank on the current node / global rank
            local_rank = self.opt.local_rank
            global_rank = int(os.environ['SLURM_NODEID']) * int(os.environ['WORLD_SIZE']) + local_rank

            # number of processes / GPUs per node
            world_size = int(os.environ['SLURM_NTASKS']) * int(os.environ['WORLD_SIZE'])

            # define master address and master port
            hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
            master_addr = hostnames.split()[0].decode('utf-8')

            # set environment variables for 'env://'
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = str(29500)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['RANK'] = str(global_rank)
        else:
            node_id = 0
            n_nodes = 1
            global_rank = "na"

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            self.local_rank = self.opt.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            self.devices = [i for i in range(self.world_size)]
            self.is_main = node_id == 0 and self.opt.local_rank == 0
        else:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices =  [i for i in range(len(gpus.split(',')))]
            self.local_rank = 0
            self.world_size = 1
            self.is_main = True

        # wait a little while so that all processes do not print at once
        time.sleep(global_rank * 0.1 if global_rank != "na" else self.local_rank * 0.1)
        print(f"Initializing node {node_id + 1} / {n_nodes}, rank {self.opt.local_rank} (local) {global_rank} (global) / {self.world_size}")

    def data_parallel(self, model):
        if self.distributed:
            model = DistributedDataParallel(model, delay_allreduce=True)
        return model

    def create_dataset(self, opt, phase="train", fold=None, from_vid=False, load_vid=False):
        opt = opt["base"] if type(opt) is dict else opt
        verbose = self.is_main
        return create_dataset(opt, phase=phase, fold=fold, from_vid=from_vid, load_vid=load_vid, verbose=verbose)

    def create_dataloader(self, dataset, batch_size=None, num_workers=1, is_train=True):
        datasampler = None
        is_shuffle = is_train or self.opt.shuffle_valid
        drop_last = True
        pin_memory = True
        if self.distributed:
            datasampler = torch.utils.data.distributed.DistributedSampler(dataset)
            batch_size = batch_size // self.world_size
            is_shuffle = False

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 drop_last=drop_last,
                                                 shuffle=is_shuffle,
                                                 pin_memory=pin_memory,
                                                 sampler=datasampler,
                                                 worker_init_fn=lambda _: np.random.seed(),
                                                 collate_fn=custom_collate_fn)

        return dataloader, datasampler, batch_size

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return tensor

    def all_gather_tensor(self, tensor):
        if self.distributed:
            tensor_list = [torch.ones_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(tensor_list, tensor)
            return tensor_list
        else:
            return [tensor]

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            print("An exception occurred during Engine initialization, "
                  "give up running process")
            return False

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)
    return tensor

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor