import gc
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from data.data_utils import get_source_label
from multi_gpu_utils import print_multi_gpu


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


class SimpleTrainer:
    def __init__(self, cfg):

        self.check = cfg.check
        self.device = torch.device("cuda")

        self.crop_size = cfg.data.crop_size
        self.nth_slice = cfg.data.take_every_nth_slice
        self.roll_slices = cfg.data.roll_slices
        self.important_slices_only = cfg.data.important_slices_only
        self.slice_level_supervision = cfg.data.slice_level_supervision
        self.classification = cfg.model.classification
        self.num_heads = cfg.model.num_heads
        print("num heads: ", self.num_heads)

    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True, local_rank=0):

        results = defaultdict(int)  # all metrics start at zero
        step = 0
        nr_of_batches = len(data_loader)

        disable_tqdm = False if local_rank == 0 else True
        tepoch = tqdm(data_loader, unit="batch", ascii=True,
                      total=len(data_loader),
                      disable=disable_tqdm)

        model.eval()

        data_times = []
        forward_times = []
        backprop_times = []
        outputs = []
        targets = []
        patch_outputs = []
        patch_targets = []

        attention_scores = dict()
        attention_scores["all_scans"] = [[], []]  # first is for all label acc, second is tumor specific
        attention_scores["cases"] = [[], []]
        attention_scores["controls"] = [[], []]
        time_data = time.time()

        rows = []
        for data, bag_idx, bag_label, patch_class, patch_centers, path in tepoch:
            # was just: data, bag_label, meta before 2d patch sampling

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
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():
                total_loss = 0
                # path = meta[2][0]  # was 4 before (4-->2)

                source_label = [get_source_label(p) for p in path]
                out = model.forward(data, label=bag_label, training=train, bag_idx= bag_idx)
            forward_time = time.time() - time_forward
            forward_times.append(forward_time)

            Y_hat = out["predictions"]
            # Y_prob = out["scores"]

            outputs.append(Y_hat.detach().cpu())
            targets.append(bag_label.detach().cpu())
            error = calculate_classification_error(bag_label, Y_hat)
            results["error"] += error
            # results["loss"] += total_loss.item()

            individual_predictions = out["instance_scores"]
            logit_class = individual_predictions > 0.05
            patch_outputs.append(logit_class.cpu())
            patch_targets.append(patch_class)


            for i in range(len(bag_label)):
                rows.append({"bag_label": bag_label[i].cpu().item(), "predicted_label": Y_hat[i].cpu().item(),
                             "path": path[i].split("/")[-1],
                             "source_label": source_label[i]})

            if step >= 6 and self.check:
                break

            time_data = time.time()

        gc.collect()
        torch.cuda.empty_cache()

        # calculate epoch averages
        for key, value in results.items():
            results[key] = value / nr_of_batches
        results["epoch"] = epoch  # epoch number

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        patch_outputs = np.concatenate(patch_outputs)
        patch_targets = np.concatenate(patch_targets)
        f1 = f1_score(targets, outputs, average='macro')
        patch_f1 = f1_score(patch_targets, patch_outputs, average='macro')
        #patch_accuracy = ((patch_targets == patch_outputs).sum()) / sum(patch_targets)
        results["patch_f1"] = patch_f1
        results["f1_score"] = f1

        print_multi_gpu(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}",
            local_rank)
        df = pd.DataFrame(rows)
        return results, df
