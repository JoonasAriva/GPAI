import gc
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from misc_utils import get_percentage_of_scans_from_dataframe


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


class RegularizedAttentionLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super().__init__()
        self.cls_loss = torch.nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction_scores, bag_label, attention_logits):
        cls_loss = self.cls_loss(prediction_scores, bag_label)
        # attention loss
        l1_term = self.alpha * torch.norm(attention_logits, p=1)
        # Variance regularization to prevent all logits from being the same
        variance_term = self.beta * torch.var(attention_logits, dim=1).mean()
        return cls_loss, l1_term, variance_term


class TrainerReg:
    def __init__(self, optimizer, loss_function, cfg, steps_in_epoch: int = 0, scheduler=None):

        self.check = cfg.check
        self.device = torch.device("cuda")
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.crop_size = cfg.data.crop_size
        self.nth_slice = cfg.data.take_every_nth_slice
        self.steps_in_epoch = steps_in_epoch
        self.scheduler = scheduler
        self.roll_slices = cfg.data.roll_slices
        self.important_slices_only = cfg.data.important_slices_only
        self.update_freq = cfg.training.weight_update_freq
        self.slice_level_supervision = cfg.data.slice_level_supervision
        if cfg.data.no_lungs:
            path = '/scratch/project_465001111/ct_data/kidney/slice_statistics.csv'
            self.statistics = pd.read_csv(path)
            if self.slice_level_supervision > 0:
                self.scan_names = get_percentage_of_scans_from_dataframe(self.statistics, self.slice_level_supervision)
        else:
            path = "/users/arivajoo/GPAI/slice_statistics/"
            # path = "/gpfs/space/projects/BetterMedicine/joonas/kidney/slice_statistics/"
            self.statistics = pd.concat([pd.read_csv(path + "slice_info_kits_kirc_train.csv"),
                                         pd.read_csv(path + "slice_info_tuh_train.csv"),
                                         pd.read_csv(path + "slice_info_tuh_test_for_train.csv"),
                                         pd.read_csv(path + "slice_info_kits_kirc_test.csv"),
                                         pd.read_csv(path + "slice_info_tuh_test_for_test.csv")
                                         ])

    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True):

        results = dict()
        epoch_loss = 0.
        class_loss = 0.
        l1_loss = 0.
        variance_loss = 0.
        epoch_error = 0.
        step = 0
        nr_of_batches = len(data_loader)

        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=self.steps_in_epoch if self.steps_in_epoch > 0 and train else len(data_loader))

        if train:
            model.train()
        else:
            model.eval()
            # model.disable_dropout()

        scaler = torch.cuda.amp.GradScaler()
        self.optimizer.zero_grad(set_to_none=True)
        data_times = []
        forward_times = []
        backprop_times = []
        outputs = []
        targets = []

        attention_scores = dict()
        attention_scores["all_scans"] = [[], []]  # first is for all label acc, second is tumor specific
        attention_scores["cases"] = [[], []]
        attention_scores["controls"] = [[], []]
        time_data = time.time()
        for data, bag_label, meta in tepoch:

            if self.check:
                print("data shape: ", data.shape, flush=True)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))  # if training a swin model, disable this line

            # data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            data = data.cuda(non_blocking=True).to(dtype=torch.float16)
            bag_label = bag_label.cuda(non_blocking=True)
            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                out = model.forward(data, scan_end=meta[3])

                Y_hat = out['predictions']
                Y_prob = out['scores']
                attention_logits = out['attention_weights']
                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                cls_loss, l1_loss, var_loss = self.loss_function(Y_prob, bag_label.float(), attention_logits)

                total_loss = cls_loss + l1_loss - var_loss

                total_loss /= self.update_freq

                outputs.append(Y_hat.cpu())
                targets.append(bag_label.cpu())

            if train:
                time_backprop = time.time()
                scaler.scale(total_loss).backward()
                backprop_time = time.time() - time_backprop
                backprop_times.append(backprop_time)
                if (step) % self.update_freq == 0 or (step) == len(data_loader):

                    scaler.step(self.optimizer)
                    scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            # torch.cuda.empty_cache()
            # torch.cuda.reset_max_memory_allocated()
            # torch.cuda.reset_peak_memory_stats()

            epoch_loss += total_loss.item() * self.update_freq
            class_loss += cls_loss.item()
            l1_loss += l1_loss.item()
            variance_loss += var_loss.item()

            error = calculate_classification_error(bag_label, Y_hat)

            epoch_error += error

            if step >= 6 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        # calculate loss and error for epoch

        epoch_loss /= nr_of_batches
        epoch_error /= nr_of_batches
        class_loss /= nr_of_batches
        l1_loss /= nr_of_batches
        variance_loss /= nr_of_batches

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, outputs, average='macro')

        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3))

        print(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss, epoch_error, f1))
        print(
            f"Main classification loss: {round(class_loss, 3)}")

        # print(f"Attention mAP for kidney region all scans: {round(np.mean(attention_scores['all_scans'][0]), 3)}")

        results["epoch"] = epoch
        results["class_loss"] = class_loss
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["f1_score"] = f1
        results["l1_loss"] = l1_loss
        results["variance_loss"] = variance_loss

        return results
