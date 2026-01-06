import gc
import sys
import time
from collections import defaultdict
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
from attention_loss import AttentionLossPatches2D
from multi_gpu_utils import print_multi_gpu
from depth_trainer2Dpatches import DepthLossSamplePatches

DEPTH_loss = DepthLossSamplePatches().cuda()
DANN_loss = nn.BCEWithLogitsLoss().cuda()
# ATTENTION_loss = AttentionLoss(step=1).cuda()
ATTENTION_loss = AttentionLossPatches2D(step=1).cuda()


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


# for diff loss gating
def smooth_gate(x, threshold, sharpness):
    # sigmoid gate that turns on smoothly near the threshold
    return torch.sigmoid(sharpness * (x - threshold))


class FocusTrainer:
    def __init__(self, optimizer, loss_function, cfg, steps_in_epoch: int = 0, scheduler=None, warmup_steps: int = 0):

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
        self.slice_level_supervision = cfg.data.slice_level_supervision
        self.classification = cfg.model.classification
        self.num_heads = cfg.model.num_heads
        self.cysts = cfg.model.cysts
        print("num heads: ", self.num_heads)
        if cfg.model.dann:
            print("Using DANN")
            self.dann = True
        else:
            self.dann = False
        if "depth" in cfg.model.name or cfg.model.depth:
            print("Continuing depth training")
            self.depth = True
        else:
            self.depth = False

        if cfg.experiment == "Attention":
            self.attention = True
        else:
            self.attention = False

        self.global_steps = 0
        self.warmup_steps = warmup_steps

        if cfg.data.no_lungs:
            path = '/scratch/project_465001111/ct_data/kidney/slice_statistics.csv'
            self.statistics = pd.read_csv(path)
            if self.slice_level_supervision > 0:
                self.scan_names = get_percentage_of_scans_from_dataframe(self.statistics, self.slice_level_supervision)
        else:
            path = "/users/arivajoo/GPAI/slice_statistics/"
            self.statistics = pd.concat([pd.read_csv(path + "slice_info_kits_kirc_train.csv"),
                                         pd.read_csv(path + "slice_info_tuh_train.csv"),
                                         pd.read_csv(path + "slice_info_tuh_test_for_train.csv"),
                                         pd.read_csv(path + "slice_info_kits_kirc_test.csv"),
                                         pd.read_csv(path + "slice_info_tuh_test_for_test.csv")
                                         ])

        path = "/users/arivajoo/GPAI/experiments/calculate_tumor_sizes/"
        self.tumor_sizes = pd.concat([pd.read_csv(path + "train_set.csv"), pd.read_csv(path + "test_set.csv")])

    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True, local_rank=0):

        results = defaultdict(int)  # all metrics start at zero
        step = 0
        nr_of_batches = len(data_loader)
        self.train = train
        disable_tqdm = False if local_rank == 0 else True
        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=self.steps_in_epoch if self.steps_in_epoch > 0 and train else len(data_loader),
                      disable=disable_tqdm)

        if train:
            model.train()
        else:
            model.eval()

        scaler = torch.cuda.amp.GradScaler()
        self.optimizer.zero_grad(set_to_none=True)
        data_times = []
        forward_times = []
        backprop_times = []
        outputs = []
        targets = []
        patch_outputs = []
        patch_targets = []

        time_data = time.time()

        for data, bag_idx, bag_label, patch_class, patch_centers, path in tepoch:
            # was just: data, bag_label, meta before 2d patch sampling
            #bag_idx = torch.Tensor(np.array([1]))
            if self.check:
                print("data shape: ", data.shape, flush=True)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1

            data = torch.squeeze(data, dim=0)
            # next line disabled for 2d patch model
            # data = torch.permute(data, (1, 0, 2, 3, 4))
            # data = torch.squeeze(data)

            data = data.cuda(non_blocking=True).to(dtype=torch.float16)
            bag_label = bag_label.cuda(non_blocking=True)
            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                out = model.forward(data, bag_idx, training=train)
                bag_label = bag_label.view(-1, 1).float()  # [B]->[B,1]
                # Main classification loss
                cls_loss = self.loss_function(out["scores"], bag_label)
                KL_loss = out["KL_loss"]
                results["KL_loss"] += KL_loss.item()
                total_loss = (
                        1 * KL_loss
                        + 1 * cls_loss
                )

                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                Y_hat = out["predictions"]

                total_loss += cls_loss
                results["class_loss"] += cls_loss.item()

                outputs.append(Y_hat.detach().cpu())
                targets.append(bag_label.detach().cpu())
                error = calculate_classification_error(bag_label, Y_hat)
                results["error"] += error

                individual_predictions = out["instance_scores"]
                logit_class = individual_predictions > 0.05
                patch_outputs.append(logit_class.cpu())
                patch_targets.append(patch_class)


            if train:
                time_backprop = time.time()
                scaler.scale(total_loss).backward()
                backprop_time = time.time() - time_backprop
                backprop_times.append(backprop_time)

                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()


            results["loss"] += total_loss.item()

            if step >= 6 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        gc.collect()
        torch.cuda.empty_cache()

        # calculate epoch averages
        for key, value in results.items():
            results[key] = value / nr_of_batches
        results["epoch"] = epoch  # epoch number

        f1 = 0
        if self.classification:

            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)
            patch_outputs = np.concatenate(patch_outputs)
            patch_targets = np.concatenate(patch_targets)

            f1 = f1_score(targets, outputs, average='macro')
            patch_f1 = f1_score(patch_targets, patch_outputs, average='macro')

            results["patch_f1"] = patch_f1
            results["f1_score"] = f1


        print_multi_gpu(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}",
            local_rank)

        print_multi_gpu(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f} '.format(
                "Train" if train else "Validation", results["loss"], results["error"], f1), local_rank)
        print_multi_gpu(
            f"Main classification loss: {round(results['class_loss'], 3)}", local_rank)

        return results
