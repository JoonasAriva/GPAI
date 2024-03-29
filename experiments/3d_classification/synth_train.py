from __future__ import print_function

import gc
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import wandb
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from monai.networks.nets import resnet18
sys.path.append('/gpfs/space/home/joonas97/GPAI/')
from data.synth_dataloaders import SynthDataloader
from train_utils import create_pretrained_medical_resnet



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

# read in statistics about the scans for validation metrics
test_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_test_ROIS.csv")
train_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS.csv")
train_ROIS_extra = pd.read_csv(
    "/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS_from_test.csv")
train_ROIS = pd.concat([train_ROIS, train_ROIS_extra])


class Trainer:
    def __init__(self, optimizer, loss_function, crop_size: int = 120, nth_slice: int = 4, check: bool = False):

        self.check = check
        self.device = torch.device("cuda")
        self.optimizer = optimizer
        self.loss_function = loss_function
        # for attention accuracy
        self.crop_size = crop_size
        self.nth_slice = nth_slice

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True):

        results = dict()

        epoch_loss = 0.
        epoch_error = 0.
        step = 0
        # attention related accuracies
        all_attention_acc = 0.
        tumor_only_attention_acc = 0.

        tepoch = tqdm(data_loader, unit="batch", ascii=True)
        model.train()

        if not train:
            model.disable_dropout()

        scaler = torch.cuda.amp.GradScaler()
        self.optimizer.zero_grad(set_to_none=True)

        for data, bag_label in tepoch:

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            calculate_attention_accuracy = False

            data = torch.unsqueeze(data[:,0,:,:,:],1)
            print(data.shape)


            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            bag_label = bag_label.to(self.device, non_blocking=True)

            # calculate loss and metrics
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():
                Y_prob= model.forward(data)
                loss = self.loss_function(Y_prob, bag_label.float())

            if train:
                if (step) % 1 == 0 or (step) == len(data_loader):
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

            epoch_loss += loss.item()

            Y_hat = torch.nn.functional.sigmoid(Y_prob)
            Y_hat = torch.ge(Y_hat, 0.5)

            error = 1. - Y_hat.eq(bag_label.float()).cpu().float().mean().data.item()
            epoch_error += error


            if step >= 5 and self.check:
                break

        # calculate loss and error for epoch
        epoch_loss /= len(data_loader)
        epoch_error /= len(data_loader)

        print(
            '{}: loss: {:.4f}, enc error: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss, epoch_error))


        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["error"] = epoch_error


        return results


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

    train_dataset = SynthDataloader(length=30)
    test_dataset = SynthDataloader(length=10)

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    logging.info('Init Model')

    if cfg.model.name == 'resnet18':
        logging.info("Using 3d model")

        model, inside = create_pretrained_medical_resnet(
            pretrained_path='/gpfs/space/home/joonas97/GPAI/pretrained_models/medicalnet/resnet_18_23dataset.pth',
            model_constructor=resnet18, shortcut_type='A')

        # if you need to continue training
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(
            torch.load(os.path.join('/gpfs/space/home/joonas97/GPAI/experiments/3d_classification/', cfg.checkpoint)))
    if torch.cuda.is_available():
        model.cuda()

    # summary(model, (3, 512, 512))
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()

    trainer = Trainer(optimizer=optimizer, loss_function=loss_function, check=cfg.check,
                      nth_slice=cfg.data.take_every_nth_slice, crop_size=cfg.data.crop_size)

    if not cfg.check:
        experiment = wandb.init(project='3d_classification_synth24', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay))
    logging.info('Start Training')

    best_test_error = 1  # should be 1
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

        else:
            if not_improved_epochs > 20:
                logging.info("Model has not improved for the last 20 epochs, stopping training")
                break
            not_improved_epochs += 1
        experiment.log(epoch_results)

    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric: {best_test_error:.4f} at epoch: {best_epoch}")
    return best_model_path, train_loader, test_loader

if __name__ == "__main__":
    main()
