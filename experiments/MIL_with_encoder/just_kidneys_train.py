from __future__ import print_function

import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import wandb
from omegaconf import OmegaConf, DictConfig

from losses import AttentionLoss

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
from trainer import Trainer
from data.dataloaders import CT_DataloaderPatches
from models import ResNet18Attention, ResNet18AttentionV2, ResNetAttentionV3

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

# read in statistics about the scans for validation metrics
test_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_test_ROIS.csv")
train_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS.csv")
train_ROIS_extra = pd.read_csv(
    "/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS_from_test.csv")
train_ROIS = pd.concat([train_ROIS, train_ROIS_extra])


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.project}, Work in {os.getcwd()}")

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')

    train_dataset = CT_DataloaderPatches(dataset_type="train", as_rgb=True)
    test_dataset = CT_DataloaderPatches(dataset_type="test", as_rgb=True)

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    logging.info('Init Model')

    if cfg.model.name == 'resnet18':
        model = ResNet18Attention(neighbour_range=cfg.model.neighbour_range, num_attention_heads=cfg.model.num_heads)
        # Let's freeze the backbone
        # model.backbone.requires_grad_(False)
    elif cfg.model.name == 'resnet18V2':
        model = ResNet18AttentionV2(neighbour_range=cfg.model.neighbour_range,
                                    num_attention_heads=cfg.model.num_heads, instnorm=cfg.model.inst_norm)

    elif cfg.model.name == 'resnet18V3':
        model = ResNetAttentionV3(neighbour_range=cfg.model.neighbour_range,
                                  num_attention_heads=cfg.model.num_heads, instnorm=cfg.model.inst_norm, resnet_type="18")

        # if you need to continue training
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(
            torch.load(os.path.join('/gpfs/space/home/joonas97/GPAI/experiments/MIL_with_encoder/', cfg.checkpoint)))
    if torch.cuda.is_available():
        model.cuda()

    # summary(model, (3, 512, 512))
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()
    attention_loss = AttentionLoss().cuda()
    trainer = Trainer(optimizer=optimizer, loss_function=loss_function, attention_loss=attention_loss, check=cfg.check,
                      nth_slice=cfg.data.take_every_nth_slice, crop_size=cfg.data.crop_size)

    if not cfg.check:
        experiment = wandb.init(project='MIL_encoder_just_kidney24', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay))
    logging.info('Start Training')

    best_test_error = 1  # should be 1
    best_train_error = 1
    best_test_loss = 10
    best_epoch = 0
    not_improved_epochs = 0

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_results = dict()
        train_results = trainer.run_one_epoch(model, train_loader, epoch, train=True)
        test_results = trainer.run_one_epoch(model, test_loader, epoch, train=False)

        train_results = {k + '_train': v for k, v in train_results.items()}
        test_results = {k + '_test': v for k, v in test_results.items()}

        test_error = test_results["error_test"]
        train_error = train_results["error_train"]
        test_loss = test_results["loss_test"]
        epoch_results.update(train_results)
        epoch_results.update(test_results)

        if cfg.check:
            logging.info("Model check completed")
            return

        if test_error < best_test_error:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch} (smallest test error)!")

            best_test_error = test_error
            best_epoch = epoch
            not_improved_epochs = 0
        elif test_loss < best_test_loss:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            best_model_path = str(dir_checkpoint / 'best_loss_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best new model at epoch {epoch} (smallest test loss)!")

            best_test_loss = test_loss
            not_improved_epochs = 0
        elif train_error < best_train_error:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            best_model_path = str(dir_checkpoint / 'best_train_error_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best new train error at epoch {epoch}!")

            best_train_error = train_error
            not_improved_epochs = 0

        else:
            if not_improved_epochs > 20:
                logging.info("Model has not improved for the last 20 epochs, stopping training")
                break
            not_improved_epochs += 1
        experiment.log(epoch_results)

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric: {best_test_error:.4f} at epoch: {best_epoch}")
    return best_model_path


if __name__ == "__main__":
    main()
