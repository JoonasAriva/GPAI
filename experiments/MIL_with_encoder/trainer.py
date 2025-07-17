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
from data.data_utils import get_source_label, map_source_label
from depth_trainer import DepthLossV2
from attention_loss import AttentionLoss
from multi_gpu_utils import print_multi_gpu

DEPTH_loss = DepthLossV2(step=0.05).cuda()
DANN_loss = nn.CrossEntropyLoss().cuda()
ATTENTION_loss = AttentionLoss(step=1).cuda()


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


class Trainer:
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

        if cfg.model.name == "DANN":
            self.dann = True
        else:
            self.dann = False
        if "depth" in cfg.model.name:
            self.depth = True
        else:
            self.depth = False

        if cfg.experiment == "Attention":
            self.attention = True
        else:
            self.attention = False

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

        results = dict()
        epoch_loss = 0.
        class_loss = 0.
        epoch_error = 0.
        domain_loss = 0.
        attn_loss = 0.
        depth_loss = 0.
        step = 0
        nr_of_batches = len(data_loader)

        disable_tqdm = False if local_rank == 0 else True
        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=self.steps_in_epoch if self.steps_in_epoch > 0 and train else len(data_loader),
                      disable=disable_tqdm)

        if train:
            model.train()
        else:
            model.eval()
        # model.train()

        scaler = torch.cuda.amp.GradScaler()
        self.optimizer.zero_grad(set_to_none=True)
        data_times = []
        forward_times = []
        backprop_times = []
        outputs = []
        targets = []
        domain_predictions = []
        target_sources = []

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

            data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))  # if training a swin model, disable this line

            # data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            data = data.cuda(non_blocking=True).to(dtype=torch.float16)
            bag_label = bag_label.cuda(non_blocking=True)
            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                path = meta[4][0]
                source_label = get_source_label(path)
                out = model.forward(data, label=bag_label, scan_end=meta[3], source_label=source_label)
                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                Y_hat = out["predictions"]
                Y_prob = out["scores"]

                cls_loss = self.loss_function(Y_prob, bag_label.float())
                total_loss = cls_loss

                if self.dann:
                    domain_pred = out['domain_pred']
                    path = meta[4][0]
                    source_label = get_source_label(path)
                    int_label = map_source_label(source_label).type(torch.LongTensor)

                    dann_loss = DANN_loss(domain_pred, int_label.cuda())
                    total_loss = cls_loss + 0.5 * dann_loss
                    domain_loss += dann_loss.item()
                    domain_predictions.append(torch.argmax(domain_pred).item())
                    target_sources.append(int_label)

                if self.depth:
                    depth_scores = out['depth_scores']
                    d_loss = DEPTH_loss(depth_scores, z_spacing=meta[2][2],  # second 2 is for z spacing
                                        nth_slice=meta[1])
                    depth_loss += d_loss.item()
                    total_loss = total_loss + d_loss
                if self.attention:
                    attention_scores = out['attention_weights'].flatten()
                    attn_loss = ATTENTION_loss(attention_scores, z_spacing=meta[2][2],
                                               nth_slice=meta[1]).item()
                    total_loss = total_loss + attn_loss

                total_loss /= self.update_freq

                outputs.append(Y_hat.detach().cpu())
                targets.append(bag_label.detach().cpu())

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

            error = calculate_classification_error(bag_label, Y_hat)

            epoch_error += error

            if step >= 6 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        gc.collect()
        torch.cuda.empty_cache()
        # calculate loss and error for epoch

        epoch_loss /= nr_of_batches
        epoch_error /= nr_of_batches
        class_loss /= nr_of_batches
        domain_loss /= nr_of_batches
        depth_loss /= nr_of_batches
        attn_loss /= nr_of_batches

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, outputs, average='macro')

        if self.dann:
            target_sources = np.concatenate(target_sources)
            domain_predictions = np.array(domain_predictions)
            domain_f1 = f1_score(target_sources, domain_predictions, average='macro')
            results["domain_f1_score"] = domain_f1

        print_multi_gpu(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}",
            local_rank)

        print_multi_gpu(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f} '.format(
                "Train" if train else "Validation", epoch_loss, epoch_error, f1), local_rank)
        print_multi_gpu(
            f"Main classification loss: {round(class_loss, 3)}", local_rank)

        if self.attention:
            print_multi_gpu(f"Attention loss: {round(attn_loss, 3)}", local_rank)

        # print(f"Attention mAP for kidney region all scans: {round(np.mean(attention_scores['all_scans'][0]), 3)}")

        results["epoch"] = epoch
        results["class_loss"] = class_loss
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["f1_score"] = f1
        results["domain_loss"] = domain_loss
        results["depth_loss"] = depth_loss
        results["attention_loss"] = attn_loss

        return results
