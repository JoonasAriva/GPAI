import gc
import sys
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from utils import evaluate_attention


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

        path = "/users/arivajoo/GPAI/slice_statistics/"
        self.train_statistics = pd.concat([pd.read_csv(path + "slice_info_kits_kirc_train.csv"),
                                           pd.read_csv(path + "slice_info_tuh_train.csv"),
                                           pd.read_csv(path + "slice_info_tuh_test_for_train.csv")])

        self.test_statistics = pd.concat([pd.read_csv(path + "slice_info_kits_kirc_test.csv"),
                                          pd.read_csv(path + "slice_info_tuh_test_for_test.csv")])

    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True):

        results = dict()
        epoch_loss = 0.
        class_loss = 0.
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
        for data, bag_label, case_id in tepoch:

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

                Y_prob, Y_hat, attention = model.forward(data)

                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                loss = self.loss_function(Y_prob, bag_label.float())

                total_loss = loss

                outputs.append(Y_hat.cpu())
                targets.append(bag_label.cpu())

                # calculate attention accuracy
                ap_all, ap_tumor = evaluate_attention(attention.cpu()[0],
                                                      self.train_statistics if train else self.test_statistics,
                                                      case_id[0],
                                                      self.crop_size, self.nth_slice, bag_label=bag_label)
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
            class_loss += loss.item()

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

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, outputs, average='macro')

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
            f"Main classification loss: {round(class_loss, 3)}")

        print(f"Attention mAP for kidney region all scans: {round(np.mean(attention_scores['all_scans'][0]), 3)}")

        results["classification_loss"] = class_loss
        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["f1_score"] = f1

        return results
