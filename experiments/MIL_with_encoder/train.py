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

from trainer_two_stage import TrainerTwoStage
from trainer import Trainer
from trainer_multi_two_stage import TrainerTwoStageMulti
from trainer_two_stage_two_heads import TrainerTwoStageTwoHeads
from depth_trainer import TrainerDepth, DepthLossV2
from utils import prepare_dataloader, pick_model

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

np.seterr(divide='ignore', invalid='ignore')

os.environ["WANDB__SERVICE_WAIT"] = "300"


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.experiment}, Work in {os.getcwd()}")

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')

    train_loader, test_loader = prepare_dataloader(cfg)
    steps_in_epoch = 500
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


    logging.info('Init Model')
    model = pick_model(cfg)
    if cfg.model.pretrained_depth:
        sd = torch.load(
            '/users/arivajoo/results/depth/train/resnetdepth/kidney_real/2024-10-31/17-01-11/checkpoints/best_model.pth')
        model.load_state_dict(state_dict=sd, strict=False)
        logging.info('Loaded depth pretrained weights')

    # if you need to continue training
    resume_run = False
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(
            torch.load(os.path.join(cfg.checkpoint, 'current_model.pth')))
        resume_run = True
    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)

    number_of_epochs = cfg.training.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=int(
        number_of_epochs * steps_in_epoch / cfg.training.weight_update_freq),
                                              pct_start=0.2, max_lr=cfg.training.learning_rate)
    if "checkpoint" in cfg.keys():
        optimizer.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'current_optimizer.pth')))
        scheduler.load_state_dict(torch.load(os.path.join(cfg.checkpoint, 'current_scheduler.pth')))
    # summary(model,input_size=(300,3,512,512))

    if "twostage" in cfg.model.name:
        loss_function = torch.nn.CrossEntropyLoss().cuda()
        if "multi" in cfg.model.name:
            trainer = TrainerTwoStageMulti(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                                           cfg=cfg,
                                           steps_in_epoch=steps_in_epoch)
        elif "two_heads" in cfg.model.name:
            loss_function = torch.nn.BCEWithLogitsLoss().cuda()
            trainer = TrainerTwoStageTwoHeads(optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                                              steps_in_epoch=steps_in_epoch)
        else:
            trainer = TrainerTwoStage(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                                      steps_in_epoch=steps_in_epoch)
    elif cfg.experiment == "depth":
        loss_function = DepthLossV2(step=0.01).cuda()
        trainer = TrainerDepth(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                               steps_in_epoch=steps_in_epoch)
    else:
        loss_function = torch.nn.BCEWithLogitsLoss().cuda()
        trainer = Trainer(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                          steps_in_epoch=steps_in_epoch)

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
    best_test_attention = 0
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_results = dict()
        train_results = trainer.run_one_epoch(model, train_loader, epoch, train=True)
        epoch_results["learning_rate"] = scheduler.get_last_lr()[0]

        test_results = trainer.run_one_epoch(model, test_loader, epoch, train=False)

        train_results = {k + '_train': v for k, v in train_results.items()}
        test_results = {k + '_test': v for k, v in test_results.items()}

        # test_f1 = test_results["f1_score_test"]
        test_loss = test_results["loss_test"]
        # test_attention = test_results["attention_map_cases_full_kidney_test"]

        epoch_results.update(train_results)
        epoch_results.update(test_results)

        if cfg.check:
            logging.info("Model check completed")
            return

        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        # if test_attention > best_test_attention:
        #     best_test_attention = test_attention
        #     logging.info(f"Best new attention at epoch {epoch} (highest mAp on cases on full kidney region)!")
        #
        #     torch.save(model.state_dict(), str(dir_checkpoint / 'best_attention.pth'))
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
