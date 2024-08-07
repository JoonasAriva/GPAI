from __future__ import print_function

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

# from torchinfo import summary

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')

from trainer import Trainer
from utils import prepare_dataloader, pick_model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

np.seterr(divide='ignore', invalid='ignore')

os.environ["WANDB__SERVICE_WAIT"] = "300"


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

    train_loader, test_loader = prepare_dataloader(cfg)
    if cfg.data.dataloader == "kidney_real":
        steps_in_epoch = 500
        proj_name = "MIL_encoder_24"
    elif cfg.data.dataloader == "synthetic":
        steps_in_epoch = 500
        proj_name = "MIL_encoder_synth24"

    logging.info('Init Model')
    model = pick_model(cfg)

    # if you need to continue training
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(
            torch.load(os.path.join('/gpfs/space/home/joonas97/GPAI/experiments/MIL_with_encoder/', cfg.checkpoint)))
    if torch.cuda.is_available():
        model.cuda()

    # summary(model,input_size=(300,3,512,512))
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)

    number_of_epochs = 40
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=number_of_epochs * steps_in_epoch,
                                              pct_start=0.2, max_lr=cfg.training.learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()

    trainer = Trainer(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                      steps_in_epoch=steps_in_epoch)

    if not cfg.check:
        experiment = wandb.init(project=proj_name, anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay, config=cfg))
    logging.info('Start Training')

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

        test_f1 = test_results["f1_score_test"]

        epoch_results.update(train_results)
        epoch_results.update(test_results)

        if cfg.check:
            logging.info("Model check completed")
            return

        if test_f1 > best_f1:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch} (highest f1 test score)!")

            best_f1 = test_f1
            best_epoch = epoch
            not_improved_epochs = 0

        else:
            if not_improved_epochs > 20:
                logging.info("Model has not improved for the last 20 epochs, stopping training")
                break
            not_improved_epochs += 1
        experiment.log(epoch_results)
        torch.save(model.state_dict(), str(dir_checkpoint / 'current_model.pth'))

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric (f1-test): {best_f1:.4f} at epoch: {best_epoch}")


if __name__ == "__main__":
    main()
