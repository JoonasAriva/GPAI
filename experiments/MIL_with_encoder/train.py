from __future__ import print_function

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from multi_gpu_utils import print_multi_gpu, log_multi_gpu, reduce_results_dict

# from torchinfo import summary

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')

from train_utils2 import prepare_dataloader, pick_model, pick_trainer, prepare_optimizer

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

    # steps_in_epoch = 480  # with pärnu and kirc and kits its 1250, if just TUH, its 480
    steps_in_epoch = 577*2 # with extra data
    steps_in_epoch = (steps_in_epoch // n_gpus) // cfg.data.batch_size
    if cfg.data.dataloader == 'kidney_pasted':
        steps_in_epoch = 478 // n_gpus

    if cfg.data.dataloader == "kidney_real":
        proj_name = "MIL_encoder_24"
    elif cfg.data.dataloader == "synthetic" or "kidney_synth":
        proj_name = "MIL_encoder_synth24"
    if 'depth' in cfg.experiment:
        if cfg.data.dataloader == "abdomen_atlas":
            proj_name = "abdomen_atlas_depth"
            steps_in_epoch = 4666
        else:
            proj_name = "depth"
    if cfg.experiment == 'swin':
        proj_name = "swin_compass"
    if cfg.data.dataloader == 'kidney_pasted':
        proj_name = "MIL_encoder_24"

    log_multi_gpu('Init Model', local_rank)
    model = pick_model(cfg)

    if cfg.model.pretrained:
        # for 2d slice attention model
        # sd = torch.load(
        #     '/users/arivajoo/results/depth/train/resnetdepth/kidney_real/2025-07-14/11-17-49/checkpoints/best_model.pth',
        #     map_location='cuda:0')
        if cfg.model.model_type == '2D':
            sd = torch.load(
                '/users/arivajoo/results/patches2D_depth/train/depth_patches2D/kidney_real/2025-11-14/11-18-30/checkpoints/best_model.pth',
                map_location='cuda:0')
        elif cfg.model.model_type == '3D':
            sd = torch.load(
                '/users/arivajoo/results/patches3D_depth/train/depth_patches3D/kidney_real/2025-11-13/15-32-42/checkpoints/best_model.pth',
                map_location='cuda:0')
        new_sd = {key.replace("module.", ""): value for key, value in sd.items()}
        missing, unexpected = model.load_state_dict(state_dict=new_sd, strict=False)

        if missing:
            print("Missing keys:")
            for k in missing:
                print(f"  - {k}")
        if unexpected:
            print("Unexpected keys:")
            for k in unexpected:
                print(f"  - {k}")

        logging.info('Loaded depth pretrained model')

    # if you need to continue training
    resume_run = False
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        sd = torch.load(os.path.join(cfg.checkpoint, 'last_model.pth'), map_location='cuda:0')
        new_sd = {key.replace("module.", ""): value for key, value in sd.items()}
        model.load_state_dict(state_dict=new_sd)
        resume_run = True

    if cfg.training.multi_gpu == True:

        torch.cuda.set_device(local_rank)
        model.cuda()
        model = DDP(  # <- We need to wrap the model with DDP
            model,
            device_ids=[local_rank],  # <- and specify the device_ids/output_device
            find_unused_parameters=False  # was True before
        )
    else:
        if torch.cuda.is_available():
            model.cuda()

    if cfg.experiment == 'compass_twostage_adv':
        optimizer, optimizer_adv = prepare_optimizer(cfg, model)

    else:
        optimizer = prepare_optimizer(cfg, model)

    number_of_epochs = cfg.training.epochs

    total_steps = number_of_epochs * steps_in_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = min(
            (step - warmup_steps) / max(1, total_steps - warmup_steps),
            1.0
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    if "checkpoint" in cfg.keys():
        optimizer.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'current_optimizer.pth')))
        scheduler.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'current_scheduler.pth')))
    trainer = pick_trainer(cfg, optimizer, scheduler, steps_in_epoch,
                           adv_optimizer=optimizer_adv if cfg.experiment == "compass_twostage_adv" else None,
                           warmup_steps=warmup_steps)

    if not cfg.check and local_rank == 0:
        experiment = wandb.init(project=proj_name, anonymous='must',settings=wandb.Settings(init_timeout=120))
        if not resume_run:
            experiment.config.update(
                dict(epochs=cfg.training.epochs,
                     learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                     weight_decay=cfg.training.weight_decay, config=cfg))
            logging.info('Start Training')
        else:
            logging.info('Resume Training')

    best_loss = 10000
    best_f1 = 0
    best_epoch = 0
    not_improved_epochs = 0

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_results = dict()
        train_results = trainer.run_one_epoch(model, train_loader, epoch, train=True, local_rank=local_rank)
        epoch_results["learning_rate"] = scheduler.get_last_lr()[0]

        test_results = trainer.run_one_epoch(model, test_loader, epoch, train=False, local_rank=local_rank)

        train_results = {k + '_train': v for k, v in train_results.items()}
        test_results = {k + '_test': v for k, v in test_results.items()}

        epoch_results.update(train_results)
        epoch_results.update(test_results)

        if cfg.training.multi_gpu == True:
            epoch_results = reduce_results_dict(epoch_results)

        test_f1 = epoch_results["f1_score_test"]
        # test_loss = epoch_results["loss_test"]

        # epoch_results = reduce_results_dict(epoch_results)
        # print_multi_gpu(epoch_results, local_rank)
        if cfg.check:
            log_multi_gpu("Model check completed", local_rank)
            return

        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

        if test_f1 > best_f1:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            log_multi_gpu(f"Best new model at epoch {epoch} (lowest test f1 score: {test_f1})!", local_rank)

            # best_loss = test_loss
            best_f1 = test_f1
            best_epoch = epoch
            not_improved_epochs = 0

        else:
            if not_improved_epochs > 30:
                log_multi_gpu("Model has not improved for the last 30 epochs, stopping training", local_rank)
                break
            not_improved_epochs += 1

        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # all ranks sync here

        if local_rank == 0:
            logging.info("Logging results to wandb")
            experiment.log(epoch_results)

            torch.save(model.module.state_dict(), str(dir_checkpoint / 'last_model.pth'))
            torch.save(optimizer.state_dict(), str(dir_checkpoint / 'current_optimizer.pth'))
            torch.save(scheduler.state_dict(), str(dir_checkpoint / 'current_scheduler.pth'))
            logging.info("Checkpoint saved")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # wait until save finished

    if local_rank == 0:
        torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
        log_multi_gpu(f'Last checkpoint! Checkpoint {epoch} saved!', local_rank)
        log_multi_gpu(f"Training completed, best_metric (f1-test): {best_f1:.4f} at epoch: {best_epoch}", local_rank)


if __name__ == "__main__":
    main()
