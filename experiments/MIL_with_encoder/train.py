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
from monai.transforms import *
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
from data.dataloaders import CT_dataloader
from models import ResNet18Attention
from utils.train_utils import find_case_id, attention_accuracy, center_crop_dataframe

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

    def run_one_epoch(self, model, data_loader, epoch: int, TUH_length: int, train: bool = True):

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

        for data, bag_label, file_path in tepoch:

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            if 'tuh_kidney' in file_path[0]:
                calculate_attention_accuracy = True
                case_id = find_case_id(file_path, start_string='(?:cases|controls)/', end_string='.nii.gz')

                if train:
                    ROIS = train_ROIS
                else:
                    ROIS = test_ROIS

                rois = ROIS.loc[ROIS["file_name"] == case_id + ".nii.gz"].copy()[::self.nth_slice]
                rois = center_crop_dataframe(rois, self.crop_size)

            else:
                calculate_attention_accuracy = False

            data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            bag_label = bag_label.to(self.device, non_blocking=True)

            # calculate loss and metrics
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():
                Y_prob, Y_hat, A = model.forward(data)
                loss = self.loss_function(Y_prob, bag_label.float())

            if train:
                if (step) % 1 == 0 or (step) == len(data_loader):
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

            epoch_loss += loss.item()
            error = model.calculate_classification_error(bag_label, Y_hat)
            epoch_error += error

            if calculate_attention_accuracy:
                attention = A.cpu().detach()[0]
                all_acc, tumor_only_acc, all_recall = attention_accuracy(attention=np.array(attention), df=rois)
                all_attention_acc += all_acc
                tumor_only_attention_acc += tumor_only_acc

            if step >= 5 and self.check:
                break

        # calculate loss and error for epoch
        epoch_loss /= len(data_loader)
        epoch_error /= len(data_loader)
        all_attention_acc /= TUH_length
        tumor_only_attention_acc /= TUH_length

        print(
            '{}: loss: {:.4f}, enc error: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss, epoch_error))
        print('attention accuracy, kidney {:.4f} and  tumor {:.4f}'.format(all_attention_acc, tumor_only_attention_acc))

        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["attention_accuracy"] = all_attention_acc
        results["tumor_attention_accuracy"] = tumor_only_attention_acc

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

    transforms_train = Compose(
        [
            RandRotate(range_x=1, prob=1),
            RandGaussianNoise(prob=0.5, mean=0, std=0.2),
            RandAffine(prob=0.5, scale_range=(-0.1, 0.1), translate_range=(-50, 50),
                       padding_mode="border")
        ])
    dataloader_params = {'datasets': cfg.data.datasets,
                         'only_every_nth_slice': cfg.data.take_every_nth_slice, 'as_rgb': True,
                         'plane': 'axial', 'center_crop': cfg.data.crop_size, 'downsample': False}
    train_dataset = CT_dataloader(dataset_type="train", augmentations=transforms_train, **dataloader_params)
    test_dataset = CT_dataloader(dataset_type="test", **dataloader_params)

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    TUH_length_train = train_dataset.TUH_length
    TUH_length_test = test_dataset.TUH_length

    logging.info('Init Model')

    if cfg.model.name == 'resnet18':
        model = ResNet18Attention(neighbour_range=cfg.model.neighbour_range, num_attention_heads=cfg.model.num_heads)
        # Let's freeze the backbone
        # model.backbone.requires_grad_(False)

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

    trainer = Trainer(optimizer=optimizer, loss_function=loss_function, check=cfg.check, nth_slice=cfg.data.take_every_nth_slice)

    if not cfg.check:
        experiment = wandb.init(project='MIL_encoder_24', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay))
    logging.info('Start Training')

    best_test_error = 1  # should be 1
    best_epoch = 0
    best_attention = 0
    not_improved_epochs = 0

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_results = dict()
        train_results = trainer.run_one_epoch(model, train_loader, epoch, TUH_length=TUH_length_train, train=True)
        test_results = trainer.run_one_epoch(model, test_loader, epoch, TUH_length=TUH_length_test, train=False)

        train_results = {k + '_train': v for k, v in train_results.items()}
        test_results = {k + '_test': v for k, v in test_results.items()}

        test_error = test_results["error_test"]
        test_attention = test_results["attention_accuracy_test"]

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

        elif test_attention > best_attention:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_attention_model.pth'))
            logging.info(f"Best new attention at epoch {epoch}!")

            best_attention = test_attention
            best_epoch = epoch
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


if __name__ == "__main__":
    main()
