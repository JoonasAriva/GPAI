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
from misc_utils import get_percentage_of_scans_from_dataframe
import torch.nn as nn


class DepthLoss3D(nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def calc_manhattan_distances_in_3d(self, matrix):
        return matrix.reshape(-1, 1, 3) - matrix.float()

    def create_index_tensor(self, h, w, d):
        return torch.stack(list(torch.unravel_index(torch.arange(0, h * w * d), (h, w, d))), dim=1)

    def forward(self, predictions, spacings, shape):
        # TODO: add spacings into step matrix
        h, w, d = shape
        # meta information about scan spacing/slice thickness is not in right shape and order for this algorithm
        new_spacing = torch.unsqueeze(torch.permute(torch.stack(spacings), (1, 0)), 0)
        # go from h,w,d order to d,h,w order
        # indexes = [2, 0, 1]
        # new_spacing = new_spacing[:, :, indexes]
        index_tensor = self.create_index_tensor(h, w, d)
        # multiply indexing tensor with reordered 3d spacings
        # self.step is just for scaling
        index_tensor = index_tensor * new_spacing * self.step
        # calculate the manhattan distances for predictions and index matrix
        distance_matrix = self.calc_manhattan_distances_in_3d(predictions)
        steps = self.calc_manhattan_distances_in_3d(index_tensor).cuda()

        # acceptable steps calculated only on slice pairings where the order is correctly predicted
        # We define the "correct" order to be positive
        idxs = distance_matrix >= 0
        distance_matrix[idxs] = distance_matrix[idxs] - 0.2 * steps[idxs]
        idxs = distance_matrix >= 0  # this is updated now
        # remove the remaining part of allowed step
        distance_matrix[idxs] = torch.maximum(distance_matrix[idxs] - 0.8 * steps[idxs],
                                              torch.zeros_like(distance_matrix[idxs]))

        # Each distance is calcualted twice: dist(x,y) and dist(y,x), where dist(x,y) = - dist(y,x)
        # We only want to take one distance per pair (generate loss once per pair)
        to_consider_idxs = steps >= 0
        distance_matrix[~to_consider_idxs] = 0

        # As some patches share a height level, previous lines will not catch cases where the index distance is 0
        # example: patches share z-coordinate
        # therefore we halve the loss from these patch pairs, since they are taken twice into loss calculation
        same_level_idxs = steps == 0
        distance_matrix[same_level_idxs] = 0.5 * distance_matrix[same_level_idxs]

        # so: loss is coming from:
        # a) pairings which are correctly ordered but the distance is too big
        # b) pairings where order is incorrect (notice abs()), negative ordering gives negative values in distance matrix
        # c) pairings which are correctly ordered but the distance is too small e.g: (legs : 0.1, neck: 0.15)

        losses = [distance_matrix[:, :, i].abs().sum(dim=(0, 1)) / (len(distance_matrix) ** 2) for i in range(3)]

        return losses  # order:  y_loss, x_loss, z_loss


class Trainer3DDepth:
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
        depth_loss = 0.
        d_loss = 0.
        h_loss = 0.
        w_loss = 0.
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
        for data, bag_label, meta, grid in tepoch:

            if self.check:
                print("data shape: ", data.shape)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            # data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            # bag_label = bag_label.to(self.device, non_blocking=True)

            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                # position_scores, shape = model.forward(data)
                position_scores = model.forward(data)  # d,h,w
                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                time_loss = time.time()

                hloss, wloss, dloss = self.loss_function(position_scores, spacings=meta[2], shape=grid)

                loss_time = time.time() - time_loss
                loss_times.append(loss_time)
                depth_loss = (dloss + hloss + wloss) / self.update_freq

            if train:
                time_backprop = time.time()
                scaler.scale(depth_loss).backward()
                backprop_time = time.time() - time_backprop
                backprop_times.append(backprop_time)
                if (step) % self.update_freq == 0 or (step) == len(data_loader):

                    scaler.step(self.optimizer)
                    scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

            epoch_loss += depth_loss.item() * self.update_freq
            # depth_loss += d_loss * self.update_freq
            d_loss += dloss * self.update_freq
            w_loss += wloss * self.update_freq
            h_loss += hloss * self.update_freq

            if step >= 6 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        # calculate loss and error for epoch

        epoch_loss /= nr_of_batches
        # depth_loss /= nr_of_batches
        d_loss /= nr_of_batches
        h_loss /= nr_of_batches
        w_loss /= nr_of_batches

        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3), "loss speed: ", round(np.mean(loss_times), 3), )

        print(
            '{}: loss: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss))

        results["epoch"] = epoch
        results["loss"] = epoch_loss
        # results["depth_loss"] = depth_loss
        results["d_loss"] = d_loss
        results["h_loss"] = h_loss
        results["w_loss"] = w_loss

        return results
