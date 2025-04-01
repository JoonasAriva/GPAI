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

DANN_loss = nn.CrossEntropyLoss().cuda()


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
        epoch_error = 0.
        domain_loss = 0.
        step = 0
        nr_of_batches = len(data_loader)

        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=self.steps_in_epoch if self.steps_in_epoch > 0 and train else len(data_loader))

        # if train:
        #     model.train()
        # else:
        #     model.eval()
        model.train()

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
            gc.collect()

            data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))  # if training a swin model, disable this line

            # data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            data = data.cuda(non_blocking=True).to(dtype=torch.float16)
            bag_label = bag_label.cuda(non_blocking=True)
            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():

                out = model.forward(data, scan_end=meta[3])
                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                Y_hat = out["predictions"]
                Y_prob = out["scores"]

                loss = self.loss_function(Y_prob, bag_label.float())

                if self.dann:
                    domain_pred = out['domain_pred']
                    path = meta[4][0]
                    source_label = get_source_label(path)
                    int_label = map_source_label(source_label).type(torch.LongTensor)
                    if int_label.item() == 0:  # remove TUH data from domain head training
                        dann_loss = torch.Tensor([0]).cuda()
                    else:
                        dann_loss = DANN_loss(domain_pred, int_label.cuda())
                    total_loss = loss + 0.5 * dann_loss
                    domain_loss += dann_loss.item()
                else:
                    total_loss = loss

                total_loss /= self.update_freq

                domain_predictions.append(torch.argmax(domain_pred).item())
                target_sources.append(int_label)

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
        domain_loss /= nr_of_batches

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, outputs, average='macro')
        target_sources = np.concatenate(target_sources)
        domain_predictions = np.array(domain_predictions)
        # print("domain preds:", domain_predictions, domain_predictions.shape)
        # print("trgt sources:", target_sources, target_sources.shape)
        domain_predictions = domain_predictions[target_sources != 0]
        target_sources = target_sources[target_sources != 0]

        domain_f1 = f1_score(target_sources, domain_predictions, average='macro')

        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3))

        print(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f} domain f1 score: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss, epoch_error, f1, domain_f1))
        print(
            f"Main classification loss: {round(class_loss, 3)}")

        # print(f"Attention mAP for kidney region all scans: {round(np.mean(attention_scores['all_scans'][0]), 3)}")

        results["epoch"] = epoch
        results["class_loss"] = class_loss
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["f1_score"] = f1
        results["domain_loss"] = domain_loss
        results["domain_f1_score"] = domain_f1

        return results
