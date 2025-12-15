from __future__ import print_function

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from train_utils import prepare_optimizer
from focusMIL_trainer import FocusTrainer
from multi_gpu_utils import print_multi_gpu, log_multi_gpu
from trainer_simple import SimpleTrainer
from multi_gpu_utils import print_multi_gpu, log_multi_gpu, reduce_results_dict
# from torchinfo import summary

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')

from train_utils import prepare_dataloader, pick_model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

np.seterr(divide='ignore', invalid='ignore')

os.environ["WANDB__SERVICE_WAIT"] = "300"

torch.backends.cudnn.benchmark = True


@hydra.main(config_path="config", config_name="config_MIL", version_base='1.1')
def main(cfg: DictConfig):
    if cfg.training.multi_gpu:
        init_process_group(backend="nccl", timeout=timedelta(seconds=3600))
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0

    n_gpus = torch.cuda.device_count()
    log_multi_gpu("Using {} GPUs".format(n_gpus), local_rank)

    log_multi_gpu(OmegaConf.to_yaml(cfg), local_rank)
    log_multi_gpu(f"Running {cfg.experiment}, Work in {os.getcwd()}", local_rank)

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print_multi_gpu('\nGPU is ON!', local_rank)

    log_multi_gpu('Load Train and Test Set', local_rank)

    train_loader, test_loader = prepare_dataloader(cfg)
    log_multi_gpu('Init Model', local_rank)
    model = pick_model(cfg)

    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        sd = torch.load(os.path.join(cfg.checkpoint, 'best_model.pth'), map_location='cuda:0')
        new_sd = {key.replace("module.", ""): value for key, value in sd.items()}
        missing, unexpected = model.load_state_dict(state_dict=new_sd)
        if local_rank == 0:
            if missing:
                print("Missing keys:")
                for k in missing:
                    print(f"  - {k}")
            if unexpected:
                print("Unexpected keys:")
                for k in unexpected:
                    print(f"  - {k}")

    if cfg.training.multi_gpu == True:

        torch.cuda.set_device(local_rank)
        model.cuda()
        model = DDP(  # <- We need to wrap the model with DDP
            model,
            device_ids=[local_rank],  # <- and specify the device_ids/output_device
            find_unused_parameters=True  # was True before
        )
    else:
        if torch.cuda.is_available():
            model.cuda()

    trainer = SimpleTrainer(cfg)
    #loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    #optimizer = prepare_optimizer(cfg, model)
    #trainer = FocusTrainer(optimizer=optimizer,loss_function=loss_function,cfg=cfg)
    epoch_results = dict()
    train_results, train_df = trainer.run_one_epoch(model, train_loader, 1, train=False, local_rank=local_rank)
    test_results, test_df = trainer.run_one_epoch(model, test_loader, 1, train=False, local_rank=local_rank)

    train_results = {k + '_train': v for k, v in train_results.items()}
    test_results = {k + '_test': v for k, v in test_results.items()}

    epoch_results.update(train_results)
    epoch_results.update(test_results)

    if cfg.training.multi_gpu == True:
        epoch_results = reduce_results_dict(epoch_results)
        if local_rank == 0:
            print(epoch_results)

    if cfg.training.multi_gpu == True:
        torch.distributed.barrier()
        # Debug: Print local DF shapes before gather
        print(
            f"Rank {dist.get_rank()}: Sending train_df shape {train_df.shape if hasattr(train_df, 'shape') else 'None/Invalid'}")
        print(
            f"Rank {dist.get_rank()}: Sending test_df shape {test_df.shape if hasattr(test_df, 'shape') else 'None/Invalid'}")

        # ---- Gather TRAIN ----
        if local_rank == 0:
            gathered_train_dfs = [None for _ in range(n_gpus)]
            dist.gather_object(
                train_df,
                object_gather_list=gathered_train_dfs,
                dst=0
            )
        else:
            torch.distributed.gather_object(train_df, dst=0)

        # ---- Gather TEST ----
        if local_rank == 0:
            gathered_test_dfs = [None for _ in range(n_gpus)]
            dist.gather_object(
                test_df,
                object_gather_list=gathered_test_dfs,
                dst=0
            )
        else:
            dist.gather_object(test_df, dst=0)

        # ---- Concatenate on rank 0 ----
        if local_rank == 0:
            train_df = pd.concat(gathered_train_dfs, ignore_index=True)
            test_df = pd.concat(gathered_test_dfs, ignore_index=True)

    if local_rank == 0:
        print("path: ", os.getcwd())
        train_df.to_csv("train_results.csv", index=False)
        test_df.to_csv("test_results.csv", index=False)
        print("results saved to csv files")

    if cfg.check:
        log_multi_gpu("Model check completed", local_rank)
        return


if __name__ == "__main__":
    main()
