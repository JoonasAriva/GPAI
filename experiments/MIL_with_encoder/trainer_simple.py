import gc
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from multi_gpu_utils import print_multi_gpu


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


class SimpleTrainer:
    def __init__(self, cfg, optimizer, loss_function, steps_in_epoch: int = 0, scheduler=None):

        self.check = cfg.check
        self.device = torch.device("cuda")
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.steps_in_epoch = steps_in_epoch
        self.scheduler = scheduler

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

        scaler = torch.amp.GradScaler('cuda')
        self.optimizer.zero_grad(set_to_none=True)
        data_times = []
        forward_times = []
        backprop_times = []
        outputs = []
        targets = []
        slice_outputs = []
        slice_targets = []

        time_data = time.time()

        for data, bag_idx, bag_label, slice_class, path in tepoch:

            if self.check:
                print("data shape: ", data.shape, flush=True)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1

            data = torch.squeeze(data, dim=0)
            data = data.cuda(non_blocking=True).to(dtype=torch.float16)
            bag_label = bag_label.cuda(non_blocking=True)
            time_forward = time.time()
            with torch.amp.autocast(device_type='cuda'), torch.no_grad() if not train else nullcontext():

                out = model.forward(data, bag_idx, training=train)
                bag_label = bag_label.view(-1, 1).float()  # [B]->[B,1]
                # Main classification loss
                cls_loss = self.loss_function(out["scores"], bag_label)

                KL_loss = out["KL_loss"]
                results["KL_loss"] += KL_loss.item()
                total_loss = (
                        0.001 * KL_loss
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
                slice_outputs.append(logit_class.cpu())
                slice_targets.append(slice_class)

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

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        # patch_outputs = np.concatenate(slice_outputs)
        # patch_targets = np.concatenate(slice_targets)

        f1 = f1_score(targets, outputs, average='macro')
        # patch_f1 = f1_score(patch_targets, patch_outputs, average='macro')

        # results["slice_f1"] = patch_f1
        results["f1_score"] = f1

        print_multi_gpu(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}",
            local_rank)

        print_multi_gpu(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f} '.format(
                "Train" if train else "Validation", results["loss"], results["error"], f1), local_rank)
        print_multi_gpu(
            f"Main classification loss: {round(results['class_loss'], 3)}",
            local_rank)

        return results
