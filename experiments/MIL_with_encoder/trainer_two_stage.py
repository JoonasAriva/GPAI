import gc
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch.nn as nn

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from utils import evaluate_attention, prepare_statistics_dataframe


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).float().mean().data.item()

    return error


class TwoStageLoss(nn.Module):
    def __init__(self):
        super(TwoStageLoss, self).__init__()

    def forward(self, roi_scores):
        return torch.abs(1 + roi_scores.min()) + torch.abs(1 - roi_scores.max())


class TrainerTwoStage:
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
        if cfg.model.name == "twostagenet_simple":
            self.simple = True
        else:
            self.simple = False

        if cfg.data.dataloader == "synthetic":
            self.synthetic = True
        else:
            self.synthetic = False

        # path = "/users/arivajoo/GPAI/slice_statistics/"
        path = "/gpfs/space/projects/BetterMedicine/joonas/kidney/slice_statistics/"
        self.train_statistics = pd.concat([pd.read_csv(path + "slice_info_kits_kirc_train.csv"),
                                           pd.read_csv(path + "slice_info_tuh_train.csv"),
                                           pd.read_csv(path + "slice_info_tuh_test_for_train.csv")])

        self.test_statistics = pd.concat([pd.read_csv(path + "slice_info_kits_kirc_test.csv"),
                                          pd.read_csv(path + "slice_info_tuh_test_for_test.csv")])

        self.two_stage_loss = TwoStageLoss().cuda()

    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True):

        results = dict()
        epoch_loss = 0.
        class_loss = 0.
        polar_loss = 0
        non_relevant_loss = 0.
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

            (case_id, nth_slice) = meta
            nth_slice = nth_slice.item()
            if self.check:
                print("data shape: ", data.shape)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            bag_label = bag_label[0].to(self.device, non_blocking=True)

            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.autograd.set_detect_anomaly(True), torch.no_grad() if not train else nullcontext():

                if self.simple:
                    df = prepare_statistics_dataframe(self.train_statistics if train else self.test_statistics,
                                                      case_id[0],
                                                      self.crop_size, nth_slice, self.roll_slices)

                    important_probs, non_important_probs, rois = model.forward(data, df)

                else:
                    important_probs, non_important_probs, rois = model.forward(data)

                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                loss_cls = self.loss_function(important_probs, bag_label.long())
                loss_polar = self.two_stage_loss(rois)
                total_loss = loss_cls + loss_polar
                if not non_important_probs.isnan().any():
                    loss_roi = self.loss_function(non_important_probs, torch.Tensor([2]).long().cuda())
                    total_loss += loss_roi
                else:
                    loss_roi = 0
                predictions = torch.argmax(important_probs[:2],
                                           dim=-1).cpu()  # we do not care for non-relevant class for f1 calculation
                outputs.append(predictions)
                targets.append(bag_label.cpu())

                if not self.synthetic:
                    # calculate attention accuracy
                    ap_all, ap_tumor = evaluate_attention(rois.cpu()[0],
                                                          self.train_statistics if train else self.test_statistics,
                                                          case_id[0],
                                                          self.crop_size, nth_slice, bag_label=bag_label,
                                                          roll_slices=self.roll_slices)
                    attention_scores["all_scans"][0].append(ap_all)

                    if bag_label:
                        attention_scores["cases"][0].append(ap_all)
                        attention_scores["cases"][1].append(ap_tumor)
                    else:
                        attention_scores["controls"][0].append(ap_all)

            if train:
                if (step) % 1 == 0 or (step) == len(data_loader):
                    time_backprop = time.time()
                    scaler.scale(total_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    backprop_time = time.time() - time_backprop
                    backprop_times.append(backprop_time)

            epoch_loss += total_loss.item()
            class_loss += loss_cls.item()
            polar_loss += loss_polar.item()
            if not non_important_probs.isnan().any():
                non_relevant_loss += loss_roi.item()

            error = calculate_classification_error(bag_label.cpu(), predictions)

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
        non_relevant_loss /= nr_of_batches
        polar_loss /= nr_of_batches

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, outputs, average='macro')

        if not self.synthetic:
            results["attention_map_all_scans_full_kidney"] = np.mean(attention_scores["all_scans"][0])
            results["attention_map_cases_full_kidney"] = np.mean(attention_scores["cases"][0])
            results["attention_map_cases_tumor"] = np.mean(attention_scores["cases"][1])
            results["attention_map_controls_full_kidney"] = np.mean(attention_scores["controls"][0])

        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3))

        print(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss, epoch_error, f1))
        print(
            f"Main classification loss: {round(class_loss, 3)}, non_rel loss: {round(non_relevant_loss, 3)}, polar loss: {round(polar_loss, 3)}")

        if not self.synthetic:
            print(f"Attention mAP for kidney region all scans: {round(np.mean(attention_scores['all_scans'][0]), 3)}")

        results["classification_loss"] = class_loss
        results["non_relevant_loss"] = non_relevant_loss
        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["f1_score"] = f1
        results["polar_loss"] = polar_loss

        return results
