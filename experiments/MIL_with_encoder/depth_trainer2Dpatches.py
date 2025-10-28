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
from multi_gpu_utils import print_multi_gpu
from attention_loss import AttentionLossPatches2D

ATTENTION_loss = AttentionLossPatches2D(step=1).cuda()


class DepthLoss2DPatches(nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def calc_manhattan_distances_in_3d(self, matrix):
        return matrix.reshape(-1, 1, 3) - matrix.float()

    def create_index_tensor(self, h, w, d):
        index_tensor = torch.stack(list(torch.unravel_index(torch.arange(0, h * w * d), (h, w, d))), dim=1)
        return index_tensor

    def forward(self, predictions, spacings, shape, filtered_indices, nth_slice):
        # TODO: add spacings into step matrix

        h, w, d = shape
        # meta information about scan spacing/slice thickness is not in right shape and order for this algorithm
        new_spacing = torch.unsqueeze(torch.permute(torch.stack(spacings), (1, 0)), 0)
        # print("new spacing shape: ", new_spacing.shape)
        new_spacing[:, :, 2] = new_spacing[:, :, 2] * nth_slice  # take into account slice sampling

        index_tensor = self.create_index_tensor(h.item(), w.item(), d.item())

        # multiply indexing tensor with reordered 3d spacings
        # self.step is just for scaling
        index_tensor = index_tensor * new_spacing * self.step
        # calculate the manhattan distances for predictions and index matrix
        distance_matrix = self.calc_manhattan_distances_in_3d(predictions)
        steps = self.calc_manhattan_distances_in_3d(index_tensor).float().cuda()

        # acceptable steps calculated only on slice pairings where the order is correctly predicted
        # We define the "correct" order to be positive
        idxs = distance_matrix >= 0

        # try:
        distance_matrix[idxs] = distance_matrix[idxs] - 0.2 * steps[idxs]
        # except:
        #     print("pred shape: ", predictions.shape)
        #     print("spacings: ", new_spacing)
        #     print("distance_matrix: ", distance_matrix.shape)
        #     print("idxs: ", idxs.shape)
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

        # do not generate loss from distances to patches, which have been filtered out
        # filtered indices - indices we want to keep, real patches
        distance_matrix[~filtered_indices, :, :] = 0
        distance_matrix[:, ~filtered_indices, :] = 0

        # so: loss is coming from:
        # a) pairings which are correctly ordered but the distance is too big
        # b) pairings where order is incorrect (notice abs()), negative ordering gives negative values in distance matrix
        # c) pairings which are correctly ordered but the distance is too small e.g: (legs : 0.1, neck: 0.15)

        losses = [distance_matrix[:, :, i].abs().sum(dim=(0, 1)) / (len(distance_matrix) ** 2) for i in range(3)]

        return losses  # order:  y_loss, x_loss, z_loss


class DepthLossSamplePatches(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_manhattan_distances_in_3d(self, matrix):
        return matrix.reshape(-1, 1, 3) - matrix.float()

    def forward(self, predictions, real_coordinates):
        voxel_spacing = torch.tensor([2.0, 0.84, 0.84])
        scaled_coordinates = real_coordinates * voxel_spacing
        steps = self.calc_manhattan_distances_in_3d(scaled_coordinates).float().cuda()

        # multiply indexing tensor with reordered 3d spacings

        # calculate the manhattan distances for predictions and index matrix
        pred_distance_matrix = self.calc_manhattan_distances_in_3d(predictions)
        for i in range(3):
            steps[:, :, i] = torch.tril(steps[:, :, i])
            pred_distance_matrix[:, :, i] = torch.tril(pred_distance_matrix[:, :, i])

        pred_distance_norm = torch.norm(pred_distance_matrix, dim=2)
        steps_norm = torch.norm(steps, dim=2)

        # THE DECOUPLED PART
        pred_distance_matrix = pred_distance_matrix - steps

        losses = [pred_distance_matrix[:, :, i].abs().sum(dim=(0, 1)) / (len(pred_distance_matrix) ** 2) for i in
                  range(3)]

        # THE COUPLED PART
        pred_distance_norm = pred_distance_norm - steps_norm

        norm_loss = pred_distance_norm.abs().sum() / (len(pred_distance_matrix) ** 2)

        return losses, norm_loss  # order:  y_loss, x_loss, z_loss # really should be z,y,x


class Trainer2DPatchDepth:
    def __init__(self, optimizer, loss_function, cfg, steps_in_epoch: int = 0, scheduler=None,
                 attention_exp: bool = False):

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
        self.scaler = torch.cuda.amp.GradScaler()
        self.model_type = cfg.model.model_type
        self.attention_exp = attention_exp
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

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True, local_rank: int = 0):

        print_multi_gpu(f"START EPOCH {epoch}: allocated MB:{torch.cuda.memory_allocated() / 1024 ** 2}",
                        local_rank)

        results = dict()
        epoch_loss = 0.
        d_loss = 0.
        h_loss = 0.
        w_loss = 0.
        dist_loss = 0.
        att_loss = 0.
        step = 0
        nr_of_batches = len(data_loader)

        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=self.steps_in_epoch if self.steps_in_epoch > 0 and train else len(data_loader))

        if train:
            model.train()
        else:
            model.eval()

        self.optimizer.zero_grad(set_to_none=True)
        data_times = []
        forward_times = []
        backprop_times = []
        loss_times = []

        time_data = time.time()

        # patches, y, (case_id, new_spacings, path, nth_slice), filtered_indices, num_patches, patch_centers
        for data, bag_label, meta, filtered_indices, num_patches, patch_centers in tepoch:

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            if self.model_type == '3D':
                data = torch.permute(data, (1, 0, 2, 3, 4))
            else:
                data = torch.squeeze(data)

            if self.check:
                print("data shape: ", data.shape, flush=True)

                print("patches_cpu dtype:", data.dtype)
                print("patches_cpu shape:", data.shape)
                print("patches_cpu is_contiguous:", data.is_contiguous())
                print("patches_cpu requires_grad:", data.requires_grad)

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)

            time_forward = time.time()
            grad_ctx = torch.no_grad() if not train else nullcontext()
            with torch.cuda.amp.autocast(), grad_ctx:
                features, att_head_weights = model.forward(data)


                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                time_loss = time.time()

                (dloss, hloss, wloss), norm_loss = self.loss_function(features, patch_centers)

                loss_time = time.time() - time_loss
                loss_times.append(loss_time)
                depth_loss = (dloss + hloss + wloss + norm_loss) / self.update_freq

                if self.attention_exp:
                    aloss = 0
                    for i in range(att_head_weights.shape[1]):
                        aloss = ATTENTION_loss()

            if train:
                time_backprop = time.time()
                self.scaler.scale(depth_loss).backward()
                backprop_time = time.time() - time_backprop
                backprop_times.append(backprop_time)
                if (step) % self.update_freq == 0 or (step) == len(data_loader):

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            epoch_loss += depth_loss.item()

            d_loss += dloss.item()
            w_loss += wloss.item()
            h_loss += hloss.item()
            dist_loss += norm_loss.item()

            if step >= 6 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        epoch_loss /= nr_of_batches
        d_loss /= nr_of_batches
        h_loss /= nr_of_batches
        w_loss /= nr_of_batches
        dist_loss /= nr_of_batches

        print_multi_gpu(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}",
            local_rank)

        print_multi_gpu(
            '{}: loss: {:.4f} '.format(
                "Train" if train else "Validation", epoch_loss), local_rank)

        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["d_loss"] = d_loss
        results["h_loss"] = h_loss
        results["w_loss"] = w_loss
        results["dist_loss"] = dist_loss

        print_multi_gpu(f"END EPOCH {epoch}: allocated MB:{torch.cuda.memory_allocated() / 1024 ** 2}",
                        local_rank)
        print_multi_gpu(f"END EPOCH {epoch}: reserved  MB:{torch.cuda.memory_reserved() / 1024 ** 2}",
                        local_rank)

        # del data, features, hloss, wloss, dloss, depth_loss, dist_loss
        torch.cuda.empty_cache()
        gc.collect()

        return results
