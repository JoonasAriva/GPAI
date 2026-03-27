import gc
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
import torch.nn as nn

from misc_utils import get_percentage_of_scans_from_dataframe
from multi_gpu_utils import print_multi_gpu


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

    def create_step_matrix(self, step_value, matrix_size):
        # matrix_size = distance_matrix.shape[0]
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
        print("distance_matrix: ", distance_matrix)
        print("distance_matrix shape: ", distance_matrix.shape)
        steps = self.create_step_matrix(acceptable_step, len(predictions)).type(torch.HalfTensor).cuda()
        print("steps: ", steps)
        # acceptable steps calculated only on slice pairings where the order is correctly predicted
        idxs = distance_matrix >= 0
        distance_matrix[idxs] = distance_matrix[idxs] - 0.2 * steps[idxs]

        idxs = distance_matrix >= 0  # this is updated now
        # remove the remaining part of allowed step
        distance_matrix[idxs] = torch.maximum(distance_matrix[idxs] - 0.8 * steps[idxs],
                                              torch.zeros_like(distance_matrix[idxs]))
        print("changed dist_matr", distance_matrix)
        # so: loss is coming from:
        # a) pairings which are correctly ordered but the distance is too big
        # b) pairings where order is incorrect (notice abs()), negative ordering gives negative values in distance matrix
        # c) pairings which are correctly ordered but the distance is too small e.g: (legs : 0.1, neck: 0.15)
        # TODO: masking for similar values
        loss = torch.tril(distance_matrix).abs().sum() / (len(predictions) ** 2)  # division is for normalization
        print("loss: ", loss)
        return loss


class DepthLossV3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, z_spacing, nth_slice):
        # nth slice - slice sample frequency
        # if we only take every 3rd slice for example
        predictions = predictions[:, 0].float()
        pred_distance_matrix = predictions.reshape(-1, 1) - predictions

        true_distance = torch.arange(0, len(predictions))
        true_distance = true_distance * nth_slice * z_spacing

        true_distance_matrix = (true_distance.reshape(-1, 1) - true_distance).float().cuda()

        error = pred_distance_matrix - true_distance_matrix

        loss = torch.tril(error).abs().sum() / (len(predictions) ** 2)  # division is for normalization
        return loss


class DepthLossV4(nn.Module):
    def __init__(self, tolerance: float = 0.0):
        """
        tolerance: fractional slack around the true distance.
                   0.0 = exact match required (pure V3 behaviour)
                   0.2 = allow 20% deviation before penalizing
        """
        super().__init__()
        self.tolerance = tolerance

    def forward(self, predictions, z_spacing, nth_slice):
        # predictions: (N, 1) or (N,)
        predictions = predictions.reshape(-1).float()
        N = len(predictions)
        device = predictions.device

        # Predicted pairwise distances: shape (N, N)
        # pred_dist[i,j] = pred[i] - pred[j]
        pred_dist = predictions.unsqueeze(1) - predictions.unsqueeze(0)

        # Ground truth pairwise distances
        indices = torch.arange(N).float()
        true_dist = ((indices.unsqueeze(1) - indices.unsqueeze(0)) * nth_slice * z_spacing * 0.01).to(device)

        # Signed error
        error = pred_dist - true_dist  # (N, N)

        if self.tolerance > 0.0:
            # Shrink error by tolerance band, then clamp negatives to 0
            # This means small deviations within tolerance are free
            slack = self.tolerance * true_dist.abs()
            error = torch.clamp(error.abs() - slack.to(device), min=0.0)

        # Lower triangle only (avoid double-counting the antisymmetric matrix)
        loss = torch.tril(error).abs().sum() / (N * (N - 1) / 2)  # normalize by actual pair count

        return loss


class CompassLoss(nn.Module):
    def __init__(self, step):
        super().__init__()

        self.cls_loss = torch.nn.BCEWithLogitsLoss()
        self.depth_loss = DepthLossV2(step=step)

    def forward(self, depth_predictions, z_spacing, nth_slice, tumor_prob, bag_label):
        d_loss = self.depth_loss(depth_predictions, z_spacing, nth_slice)
        cls_loss = self.cls_loss(tumor_prob, bag_label)

        return d_loss, cls_loss


class TrainerDepth:
    def __init__(self, optimizer, loss_function, cfg, steps_in_epoch: int = 0, scheduler=None):

        self.check = cfg.check
        # self.device = torch.device("cuda")
        #
        # self.device = torch.device(int(os.environ['LOCAL_RANK']))
        # print("trainer device:", self.device)
        # self.device = device
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
        self.scaler = torch.amp.GradScaler('cuda')
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

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True, local_rank=0):

        results = defaultdict(int)
        epoch_loss = 0.
        depth_loss = 0.
        class_loss = 0.
        step = 0
        nr_of_batches = len(data_loader)

        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=self.steps_in_epoch if self.steps_in_epoch > 0 and train else len(data_loader))
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        if train:
            model.train()
        else:
            model.eval()

            device = torch.device(f'cuda:{local_rank}')  # each GPU checks its own memory
            free, total = torch.cuda.mem_get_info(device)
            mem_used_MB = (total - free) / 1024 ** 2
            print(f"GPU {local_rank} memory usage: {mem_used_MB}")

        self.optimizer.zero_grad(set_to_none=True)
        data_times = []
        forward_times = []
        backprop_times = []
        loss_times = []

        time_data = time.time()
        # x, y, slice_class, path
        gc.collect()

        for data, bag_label, slice_class, nth_slice, spacing, scan_end in tepoch:

            if self.check:
                free, total = torch.cuda.mem_get_info(torch.device(f'cuda:{local_rank}'))
                print(
                    f"GPU {local_rank} batch {step} free: {free / 1024 ** 2:.0f}MB, scan end: {scan_end} and data shape: {data.shape}")

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1

            data = torch.squeeze(data)
            # data = torch.permute(data, (0, 4, 1, 2, 3))

            # data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            data = data.cuda(non_blocking=True).to(dtype=torch.float16)

            time_forward = time.time()

            #dev = torch.device('cuda:0')
            #torch.cuda.reset_peak_memory_stats(dev)
            #alloc = torch.cuda.memory_allocated(dev) / 1e9
            #res = torch.cuda.memory_reserved(dev) / 1e9
            #print(f"Batch {step} | Before forward - Alloc: {alloc:.2f}GB | Reserved: {res:.2f}GB | scan_end={scan_end}")

            try:
                with torch.amp.autocast(device_type='cuda'):
                    position_scores = model.forward(data, scan_end=scan_end)

                    forward_time = time.time() - time_forward
                    forward_times.append(forward_time)

                    time_loss = time.time()

                    d_loss = self.loss_function(position_scores, z_spacing=spacing[2],  # second 2 is for z spacing
                                                nth_slice=nth_slice)

                loss_time = time.time() - time_loss
                loss_times.append(loss_time)
                total_loss = d_loss

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    print(f"\n=== OOM DETECTED on GPU {local_rank} at batch {step} ===")
                    print(f"scan_end = {scan_end}")
                    print(f"Data shape: {data.shape if 'data' in locals() else 'N/A'}")
                    print(torch.cuda.memory_summary(device, abbreviated=False))
                    print(f"Full error: {e}")
                    # Optional: dump snapshot (see below)
                    torch.cuda.memory._dump_snapshot(f"oom_snapshot_rank{local_rank}_batch{step}.pickle")
                    raise  # re-raise so SLURM still kills cleanly, or handle gracefully
                else:
                    raise

            if train:
                time_backprop = time.time()
                self.scaler.scale(total_loss).backward()
                backprop_time = time.time() - time_backprop
                backprop_times.append(backprop_time)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            results["loss"] += total_loss.item()
            results["depth_loss"] += d_loss.item()

            if step >= 8 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break
            del data, position_scores, d_loss, total_loss

            time_data = time.time()

        # calculate loss and error for epoch

        for key, value in results.items():
            results[key] = value / nr_of_batches
        results["epoch"] = epoch  # epoch number

        print_multi_gpu(
            f"data speed: {round(np.mean(data_times), 3)} forward speed:{round(np.mean(forward_times), 3)} backprop speed:  {round(np.mean(backprop_times), 3)} loss speed: {round(np.mean(loss_times), 3)}",
            local_rank)

        print_multi_gpu(
            '{}: loss: {:.4f}'.format(
                "Train" if train else "Validation", results["depth_loss"]), local_rank)

        return results
