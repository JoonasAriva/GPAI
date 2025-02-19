from __future__ import print_function

import datetime
import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.optim as optim
import wandb
from omegaconf import OmegaConf, DictConfig
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# from torchinfo import summary

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')

from train_utils import prepare_dataloader, pick_model, pick_trainer, prepare_optimizer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

np.seterr(divide='ignore', invalid='ignore')

os.environ["WANDB__SERVICE_WAIT"] = "300"

def print_slurm_env():
    slurm_env = "\n".join(
        [
            "=" * 80,
            f"SLURM Process: {os.environ.get('SLURM_PROCID', 'N/A')=}",
            "=" * 80,
            f"{os.environ.get('SLURM_NTASKS', 'N/A')=}",
            f"{os.environ.get('SLURM_LOCALID', 'N/A')=}",
            f"{os.environ.get('RANK', 'N/A')=}",
            f"{os.environ.get('LOCAL_RANK', 'N/A')=}",
            f"{os.environ.get('WORLD_SIZE', 'N/A')=}",
            f"{os.environ.get('MASTER_ADDR', 'N/A')=}",
            f"{os.environ.get('MASTER_PORT', 'N/A')=}",
            f"{os.environ.get('ROCR_VISIBLE_DEVICES', 'N/A')=}",
            f"{os.environ.get('SLURM_JOB_GPUS', 'N/A')=}",
            f"{os.sched_getaffinity(0)=}",
            f"{os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING', 'N/A')=}",
            "-" * 80 + "\n",
        ]
    )
    print(slurm_env, flush=True)

@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.experiment}, Work in {os.getcwd()}")

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print('\nGPU is ON!')

    if cfg.training.multi_gpu:
        #torch.multiprocessing.set_start_method("spawn")
        #local_rank = int(os.environ['LOCAL_RANK'])
        init_process_group(backend="nccl")
    print_slurm_env()
    print('Load Train and Test Set')

    train_loader, test_loader = prepare_dataloader(cfg)
    steps_with_parnu = 1250
    steps_in_epoch = steps_with_parnu // cfg.data.batch_size  # before pärnu it was 500
    if cfg.data.dataloader == "kidney_real":
        proj_name = "MIL_encoder_24"
    elif cfg.data.dataloader == "synthetic" or "kidney_synth":
        proj_name = "MIL_encoder_synth24"
    if cfg.experiment == 'depth':
        if cfg.data.dataloader == "abdomen_atlas":
            proj_name = "abdomen_atlas_depth"
            steps_in_epoch = 4666
        else:
            proj_name = "depth"
    if cfg.experiment == 'swin':
        proj_name = "swin_compass"

    logging.info('Init Model')
    model = pick_model(cfg)

    if cfg.model.pretrained:
        sd = torch.load(
            '/users/arivajoo/results/swin/train/swincompassV1/kidney_real/2025-01-29/15-23-40/checkpoints/best_model.pth')
        model.load_state_dict(state_dict=sd, strict=False)
        logging.info('Loaded compass pretrained model')

    # if you need to continue training
    resume_run = False
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(
            torch.load(os.path.join(cfg.checkpoint, 'current_model.pth')))
        resume_run = True
    #if torch.cuda.is_available():
    #   model.cuda()

    if cfg.training.multi_gpu == True:

        n_gpus = torch.cuda.device_count()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        model.cuda()
        print("Using {} GPUs".format(n_gpus))
        print("Local rank: {}".format(local_rank))
        model = DDP(  # <- We need to wrap the model with DDP
            model,
            device_ids=[local_rank],  # <- and specify the device_ids/output_device
        )

    optimizer = prepare_optimizer(cfg, model)

    number_of_epochs = cfg.training.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=int(
        number_of_epochs * steps_in_epoch / cfg.training.weight_update_freq),
                                              pct_start=0.2, max_lr=[cfg.training.learning_rate,
                                                                     cfg.training.learning_rate * 10] if cfg.experiment == "compass_twostage" else cfg.training.learning_rate)
    if "checkpoint" in cfg.keys():
        optimizer.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'current_optimizer.pth')))
        scheduler.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'current_scheduler.pth')))
    # summary(model,input_size=(300,3,512,512))

    trainer = pick_trainer(cfg, optimizer, scheduler, steps_in_epoch)

    if not cfg.check:
        experiment = wandb.init(project=proj_name, anonymous='must')
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
        train_results = trainer.run_one_epoch(model, train_loader, epoch, train=True)
        epoch_results["learning_rate"] = scheduler.get_last_lr()[0]

        test_results = trainer.run_one_epoch(model, test_loader, epoch, train=False)

        train_results = {k + '_train': v for k, v in train_results.items()}
        test_results = {k + '_test': v for k, v in test_results.items()}

        test_loss = test_results["loss_test"]

        epoch_results.update(train_results)
        epoch_results.update(test_results)

        if cfg.check:
            logging.info("Model check completed")
            return

        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

        if test_loss < best_loss:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch} (lowest test loss: {test_loss})!")

            best_loss = test_loss
            best_epoch = epoch
            not_improved_epochs = 0

        else:
            if not_improved_epochs > 20:
                logging.info("Model has not improved for the last 20 epochs, stopping training")
                # break
            not_improved_epochs += 1

        experiment.log(epoch_results)
        torch.save(model.state_dict(), str(dir_checkpoint / 'current_model.pth'))
        torch.save(optimizer.state_dict(), str(dir_checkpoint / 'current_optimizer.pth'))
        torch.save(scheduler.state_dict(), str(dir_checkpoint / 'current_scheduler.pth'))

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric (f1-test): {best_f1:.4f} at epoch: {best_epoch}")


if __name__ == "__main__":
    main()
