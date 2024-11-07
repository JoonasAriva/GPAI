import gc
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from utils import get_percentage_of_scans_from_dataframe
import torch.nn as nn


class DepthLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(DepthLoss, self).__init__()
        self.alpha = alpha

    def forward(self, depth_scores):
        depth_matrix = depth_scores.reshape(-1, 1) - depth_scores
        depth_order_loss = (torch.triu(depth_matrix) / (len(depth_scores) ** 2)).sum()  # division is for normalization
        regularization = depth_scores.abs().sum() / len(depth_scores)
        loss = depth_order_loss + self.alpha * regularization
        return loss


class DepthLossV2(nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def create_step_matrix(self, step_value, distance_matrix):
        matrix_size = distance_matrix.shape[0]
        steps = torch.zeros(matrix_size, matrix_size)
        steps = torch.diagonal_scatter(steps, torch.zeros(matrix_size), 0)
        for i in range(matrix_size):
            steps = torch.diagonal_scatter(steps, (1 + i) * step_value * torch.ones(matrix_size - i - 1), 1 + i)
            steps = torch.diagonal_scatter(steps, (1 + i) * step_value * torch.ones(matrix_size - i - 1), -1 - i)
        return steps

    def forward(self, predictions, z_spacing, nth_slice):
        # nth slice - slice sample frequency
        # if we only take every 3rd slice for example
        acceptable_step = self.step * z_spacing * nth_slice
        predictions = predictions[:, 0]
        distance_matrix = predictions.reshape(-1, 1) - predictions
        steps = self.create_step_matrix(acceptable_step, distance_matrix).type(torch.HalfTensor).cuda()
        # acceptable steps calculated only on slice pairings where the order is correctly predicted
        idxs = distance_matrix >= 0
        distance_matrix[idxs] = distance_matrix[idxs] - 0.2 * steps[idxs]

        idxs = distance_matrix >= 0  # this is updated now
        # remove the remaining part of allowed step
        distance_matrix[idxs] = torch.maximum(distance_matrix[idxs] - 0.8 * steps[idxs],
                                              torch.zeros_like(distance_matrix[idxs]))
        # so: loss is coming from:
        # a) pairings which are correctly ordered but the distance is too big
        # b) pairings where order is incorrect (notice abs()), negative ordering gives negative values in distance matrix
        # c) pairings which are correctly ordered but the distance is too small e.g: (legs : 0.1, neck: 0.15)
        # TODO: masking for similar values
        loss = torch.tril(distance_matrix).abs().sum() / (len(predictions) ** 2) # division is for normalization
        return loss


class TrainerDepth:
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
        loss_times = []
        outputs = []
        targets = []

        attention_scores = dict()
        attention_scores["all_scans"] = [[], []]  # first is for all label acc, second is tumor specific
        attention_scores["cases"] = [[], []]
        attention_scores["controls"] = [[], []]
        time_data = time.time()
        for data, bag_label, meta in tepoch:

            if self.check:
                print("data shape: ", data.shape)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            bag_label = bag_label.to(self.device, non_blocking=True)

            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                position_scores = model.forward(data)

                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                time_loss = time.time()
                loss = self.loss_function(position_scores, z_spacing=meta[2].item(), nth_slice=meta[1].item())
                loss_time = time.time() - time_loss
                loss_times.append(loss_time)
                total_loss = loss / self.update_freq

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

            epoch_loss += total_loss.item() * self.update_freq

            if step >= 6 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        # calculate loss and error for epoch

        epoch_loss /= nr_of_batches

        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3), "loss speed: ", round(np.mean(loss_times), 3), )

        print(
            '{}: loss: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss))

        results["epoch"] = epoch
        results["loss"] = epoch_loss

        return results
