import gc
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from utils import get_percentage_of_scans_from_dataframe
import torch
from sklearn.metrics import f1_score
from depth_trainer import DepthLossV2


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


def calculate_scale_step(start_scale, end_scale, steps_per_epoch, epochs):
    span = end_scale - start_scale
    return span / (steps_per_epoch * epochs)


class TwoStageCompassLoss(nn.Module):
    def __init__(self, step, fixed_compass=False):
        super().__init__()
        self.fixed_compass = fixed_compass
        self.cls_loss = torch.nn.BCEWithLogitsLoss()
        pos_weight = torch.tensor([2])
        self.rel_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.depth_loss = DepthLossV2(step=step)

    def forward(self, depth_predictions, z_spacing, nth_slice, tumor_prob, bag_label, rel_prob=None, rel_labels=None):
        if not self.fixed_compass:
            d_loss = self.depth_loss(depth_predictions, z_spacing, nth_slice)
        else:
            d_loss = torch.tensor([0]).cuda()
        cls_loss = self.cls_loss(tumor_prob, bag_label)
        if rel_labels is not None:
            rel_loss = self.rel_loss(rel_prob, rel_labels)

            return d_loss, cls_loss, rel_loss
        else:
            return d_loss, cls_loss, torch.Tensor([0]).cuda()


class TrainerCompassTwoStage:
    def __init__(self, optimizer, loss_function, cfg, steps_in_epoch: int = 0, scheduler=None,
                 progressive_sigmoid_scaling=False):

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
        self.relevancy_head = cfg.model.relevancy_head
        self.scaling = progressive_sigmoid_scaling

        if self.scaling:
            self.scale_step = calculate_scale_step(start_scale=20, end_scale=100, steps_per_epoch=self.steps_in_epoch,
                                                   epochs=30)
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
        if cfg.model.fixed_compass:
            self.fixed_compass = True
            compass_path = "/users/arivajoo/GPAI/experiments/MIL_with_encoder/"
            self.compass_scores = pd.concat([pd.read_csv(compass_path + "fixed_depth_scores_test.csv"),
                                             pd.read_csv(compass_path + "fixed_depth_scores_train.csv")])
        else:
            self.fixed_compass = False

    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True):

        results = dict()
        epoch_loss = 0.
        epoch_error = 0.
        depth_loss = 0.
        class_loss = 0.
        relevancy_loss = 0.
        span_loss = 0.

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

            if self.fixed_compass:
                case_id = meta[0][0]
                depth_scores = torch.Tensor(
                    self.compass_scores.loc[self.compass_scores["case_id"] == case_id, "weights"].to_numpy()).cuda()
                depth_scores = torch.unsqueeze(depth_scores, 1)
            else:
                depth_scores = None
            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                position_scores, tumor_score, Y_hat, relevancy_scores, rel_labels = model.forward(data,
                                                                                                  depth_scores=depth_scores,
                                                                                                  scan_end=meta[3])

                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                time_loss = time.time()
                d_loss, cls_loss, rel_loss = self.loss_function(position_scores, z_spacing=meta[2].item(),
                                                                nth_slice=meta[1].item(),
                                                                tumor_prob=tumor_score, bag_label=bag_label.float(),
                                                                rel_prob=relevancy_scores if self.relevancy_head else None,
                                                                rel_labels=rel_labels.float() if self.relevancy_head else None)
                range = model.depth_range[1] - model.depth_range[0]
                range_loss = torch.max(torch.Tensor([0]).cuda(), -1 * range).sum() \
                             + torch.max(torch.Tensor([0.5]).cuda() - range, torch.Tensor([0]).cuda()).sum() \
                             + torch.max(range - torch.Tensor([1]).cuda(), torch.Tensor([0]).cuda()).sum()
                # 1) generate loss if range gets flipped
                # 2) generate loss if range get smaller than 0.5
                # 3) generate loss if range gets bigger than 1
            max_pos = torch.max(position_scores).item()
            min_pos = torch.min(position_scores).item()

            range_loss2 = torch.max(model.depth_range[1] - max_pos, torch.Tensor([0]).cuda()).sum() + torch.min(
                model.depth_range[0] - min_pos, torch.Tensor([0]).cuda()).abs().sum()

            # 1) generate loss if max position score is smaller than upper range parameter
            # 2) vice versa, generate loss if min pos score is larger than lower range parameter
            # in summary, range paramters should be inside maximum and minimum scores
            range_loss += range_loss2
            loss_time = time.time() - time_loss
            loss_times.append(loss_time)
            total_loss = (d_loss + cls_loss + rel_loss + range_loss) / self.update_freq

            error = calculate_classification_error(bag_label, Y_hat)
            epoch_error += error
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

                if self.scaling and model.scale < 100:  # for progressively steepening the sigmoid curve (100 is end value)
                    model.scale += self.scale_step

            epoch_loss += total_loss.item() * self.update_freq
            depth_loss += d_loss.item() * self.update_freq
            class_loss += cls_loss.item() * self.update_freq
            relevancy_loss += rel_loss.item() * self.update_freq
            span_loss += range_loss.item() * self.update_freq

            if step >= 6 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        # calculate loss and error for epoch

        epoch_loss /= nr_of_batches
        depth_loss /= nr_of_batches
        class_loss /= nr_of_batches
        epoch_error /= nr_of_batches
        relevancy_loss /= nr_of_batches
        span_loss /= nr_of_batches

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, outputs, average='macro')

        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3), "loss speed: ", round(np.mean(loss_times), 3), )

        print(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss, epoch_error, f1))
        print(
            f"Main classification loss: {round(class_loss, 3)}")
        print(
            f"Span loss: {span_loss}, Range: {round(model.depth_range[0].item(), 3)}--{round(model.depth_range[1].item(), 3)}")

        if self.scaling:
            results["sigmoid_scale"] = model.scale
        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["depth_loss"] = depth_loss
        results["class_loss"] = class_loss
        results["error"] = epoch_error
        results["relevancy_loss"] = relevancy_loss
        results["span_loss"] = span_loss
        results["f1_score"] = f1
        results["range1"] = model.depth_range[0].item()
        results["range2"] = model.depth_range[1].item()

        return results
