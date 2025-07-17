import logging

import torch
from torch.distributed import all_reduce, ReduceOp


def print_multi_gpu(str, local_rank):
    if local_rank == 0:
        print(str)


def log_multi_gpu(str, local_rank):
    if local_rank == 0:
        logging.info(str)


def reduce_results_dict(results):
    new_results = dict()
    for k, v in results.items():
        metric = torch.Tensor([v]).cuda()
        all_reduce(metric, op=ReduceOp.SUM)
        avg_metric = metric.item() / torch.cuda.device_count()
        new_results[k] = avg_metric
    return new_results
