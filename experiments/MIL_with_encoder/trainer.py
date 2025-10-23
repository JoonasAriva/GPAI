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
from attention_loss import AttentionLossPatches2D
from multi_gpu_utils import print_multi_gpu
from depth_trainer2Dpatches import DepthLossSamplePatches

DEPTH_loss = DepthLossSamplePatches().cuda()
DANN_loss = nn.CrossEntropyLoss().cuda()
# ATTENTION_loss = AttentionLoss(step=1).cuda()
ATTENTION_loss = AttentionLossPatches2D(step=1).cuda()


def calculate_classification_error(Y, Y_hat):
    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

    return error


# for diff loss gating
def smooth_gate(x, threshold, sharpness=50.0):
    # sigmoid gate that turns on smoothly near the threshold
    return torch.sigmoid(sharpness * (x - threshold))


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
        self.classification = cfg.model.classification
        self.num_heads = cfg.model.num_heads

        if cfg.model.name == "DANN":
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
        f1 = 0.
        domain_loss = 0.
        attn_loss = 0.
        depth_loss = 0.
        difference_loss = 0.
        ent_loss = 0.
        step = 0
        d_loss = 0.
        h_loss = 0.
        w_loss = 0.
        dist_loss = 0.
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
        for data, bag_label, meta, filtered_indices, num_patches, patch_centers in tepoch:
            # was just: data, bag_label, meta before 2d patch sampling

            if self.check:
                print("data shape: ", data.shape, flush=True)

            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1

            # next line disabled for 2d patch model
            # data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))  # if training a swin model, disable this line
            data = torch.squeeze(data)

            # data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            data = data.cuda(non_blocking=True).to(dtype=torch.float16)
            bag_label = bag_label.cuda(non_blocking=True)
            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():
                total_loss = 0
                path = meta[2][0]  # was 4 before (4-->2)
                source_label = get_source_label(path)
                out = model.forward(data, label=bag_label, scan_end=num_patches, source_label=source_label)
                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                if self.classification:
                    Y_hat = out["predictions"]
                    Y_prob = out["scores"]

                    cls_loss = self.loss_function(Y_prob, bag_label.float())
                    total_loss += cls_loss
                    class_loss += cls_loss.item()

                    outputs.append(Y_hat.detach().cpu())
                    targets.append(bag_label.detach().cpu())
                    error = calculate_classification_error(bag_label, Y_hat)
                    epoch_error += error

                if self.dann:
                    domain_pred = out['domain_pred']
                    path = meta[4][0]
                    source_label = get_source_label(path)
                    int_label = map_source_label(source_label).type(torch.LongTensor)

                    dann_loss = DANN_loss(domain_pred, int_label.cuda())
                    total_loss += 0.5 * dann_loss
                    domain_loss += dann_loss.item()
                    domain_predictions.append(torch.argmax(domain_pred).item())
                    target_sources.append(int_label)

                if self.depth:
                    depth_scores = out['depth_scores']

                    (dloss, hloss, wloss), norm_loss = DEPTH_loss(depth_scores, patch_centers)
                    depth_loss = (dloss + hloss + wloss + norm_loss)
                    total_loss += depth_loss
                if self.attention:
                    # attention_scores = out['attention_weights'].flatten()
                    # attn_loss = ATTENTION_loss(attention_scores, z_spacing=meta[2][2],
                    #                           nth_slice=meta[1]).item()
                    a_loss = 0
                    attention_scores = out['all_attention']

                    for i in range(self.num_heads):
                        head_attention = attention_scores[:, i].flatten()

                        a_loss += ATTENTION_loss(head_attention, patch_centers)
                        # entropy part
                        # N = torch.log(torch.tensor(len(head_attention), dtype=torch.float16)).cuda()
                        # entropy = 0 * -0.005 * (head_attention * torch.log(head_attention + 1e-12)).sum(dim=0) / N
                        # ent_loss += 0.5 * entropy

                    total_loss += 1000 * a_loss  # + entropy)
                    attn_loss += a_loss
                    if self.num_heads > 1:
                        diff_loss = torch.sum(attention_scores[:, 0] * attention_scores[:, 1])
                        gated_diff_loss = smooth_gate(diff_loss, 0.0003, 50)
                        total_loss += gated_diff_loss
                        difference_loss += gated_diff_loss

                total_loss /= self.update_freq

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

            epoch_loss += total_loss.item() * self.update_freq

            if self.depth:
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

        gc.collect()
        torch.cuda.empty_cache()
        # calculate loss and error for epoch

        epoch_loss /= nr_of_batches
        epoch_error /= nr_of_batches
        class_loss /= nr_of_batches
        domain_loss /= nr_of_batches
        depth_loss /= nr_of_batches
        attn_loss /= nr_of_batches
        ent_loss /= nr_of_batches
        difference_loss /= nr_of_batches

        if self.dann:
            target_sources = np.concatenate(target_sources)
            domain_predictions = np.array(domain_predictions)
            domain_f1 = f1_score(target_sources, domain_predictions, average='macro')
            results["domain_f1_score"] = domain_f1
        if self.classification:
            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)
            f1 = f1_score(targets, outputs, average='macro')

        print_multi_gpu(
            f"data speed: {round(np.mean(data_times), 3)}, forward speed ,{round(np.mean(forward_times), 3)},backprop speed: , {round(np.mean(backprop_times), 3)}",
            local_rank)

        print_multi_gpu(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f} '.format(
                "Train" if train else "Validation", epoch_loss, epoch_error, f1), local_rank)
        print_multi_gpu(
            f"Main classification loss: {round(class_loss, 3)}", local_rank)

        if self.attention:
            print_multi_gpu(f"Attention loss: {round(attn_loss.item(), 6)}", local_rank)

        # print(f"Attention mAP for kidney region all scans: {round(np.mean(attention_scores['all_scans'][0]), 3)}")

        results["epoch"] = epoch
        results["class_loss"] = class_loss
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["f1_score"] = f1
        results["domain_loss"] = domain_loss
        results["depth_loss"] = depth_loss
        results["attention_loss"] = attn_loss
        results["difference_loss"] = difference_loss
        results["entropy_loss"] = ent_loss
        results["d_loss"] = d_loss
        results["h_loss"] = h_loss
        results["w_loss"] = w_loss
        results["dist_loss"] = dist_loss

        return results
