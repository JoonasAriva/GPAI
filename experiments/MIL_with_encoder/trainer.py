import gc
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')
from utils.train_utils import find_case_id, attention_accuracy, center_crop_dataframe
import pandas as pd

# read in statistics about the scans for validation metrics
test_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_test_ROIS.csv")
train_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS.csv")
train_ROIS_extra = pd.read_csv(
    "/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS_from_test.csv")
train_ROIS = pd.concat([train_ROIS, train_ROIS_extra])


class Trainer:
    def __init__(self, optimizer, loss_function, attention_loss, crop_size: int = 120, nth_slice: int = 4,
                 check: bool = False, calculate_attention_accuracy=False, steps_in_epoch: int = 0):

        self.check = check
        self.device = torch.device("cuda")
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.attention_loss = attention_loss
        self.crop_size = crop_size
        self.nth_slice = nth_slice
        self.calculate_attention_accuracy = calculate_attention_accuracy
        self.steps_in_epoch = steps_in_epoch

    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def run_one_epoch(self, model, data_loader, epoch: int, train: bool = True):

        results = dict()

        epoch_loss = 0.
        class_loss = 0.
        att_loss = 0.
        epoch_error = 0.
        mean_loss = 0.
        max_loss = 0.
        min_loss = 0.
        step = 0
        # attention related accuracies
        all_attention_acc = 0.
        tumor_only_attention_acc = 0.
        attention_count = 0

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
        time_data = time.time()
        data_times = []
        forward_times = []
        backprop_times = []

        outputs = []
        targets = []

        for data, bag_label, file_path in tepoch:

            if self.check:
                print("data shape: ", data.shape)
            data_time = time.time() - time_data
            data_times.append(data_time)

            tepoch.set_description(f"Epoch {epoch}")
            step += 1
            gc.collect()

            calculate_attention_accuracy = False

            data = torch.permute(torch.squeeze(data), (3, 0, 1, 2))

            data = data.to(self.device, dtype=torch.float16, non_blocking=True)
            bag_label = bag_label.to(self.device, non_blocking=True)

            # calculate loss and metrics

            time_forward = time.time()
            with torch.cuda.amp.autocast(), torch.no_grad() if not train else nullcontext():
                Y_prob, Y_hat, A = model.forward(data, return_unnorm_attention=True)

                forward_time = time.time() - time_forward
                forward_times.append(forward_time)

                loss = self.loss_function(Y_prob, bag_label.float())
                attention_loss, terms = self.attention_loss(A)
                total_loss = loss + attention_loss

                outputs.append(Y_hat.cpu())
                targets.append(bag_label.cpu())

            if self.calculate_attention_accuracy and "tuh_kidney" in file_path[0]:
                attention_count += 1
                case_id = find_case_id(file_path, start_string='(?:cases|controls)/', end_string='.nii.gz')

                if train:
                    ROIS = train_ROIS
                else:
                    ROIS = test_ROIS

                rois = ROIS.loc[ROIS["file_name"] == case_id + ".nii.gz"].copy()[::self.nth_slice]
                rois = center_crop_dataframe(rois, self.crop_size)
                attention = A.cpu().detach()[0]

                all_acc, tumor_only_acc, all_recall = attention_accuracy(attention=np.array(attention), df=rois)
                if self.check:
                    print("attention metrics: ", all_acc, tumor_only_acc, all_recall)
                all_attention_acc += all_acc
                tumor_only_attention_acc += tumor_only_acc

            if train:
                if (step) % 1 == 0 or (step) == len(data_loader):
                    time_backprop = time.time()
                    scaler.scale(total_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    backprop_time = time.time() - time_backprop
                    backprop_times.append(backprop_time)
            mean_loss, max_loss, min_loss = terms

            mean_loss += mean_loss.item()
            max_loss += max_loss.item()
            min_loss += min_loss.item()

            epoch_loss += total_loss.item()
            class_loss += loss.item()
            att_loss += attention_loss.item()

            error = self.calculate_classification_error(bag_label, Y_hat)

            epoch_error += error

            if step >= 5 and self.check:
                break

            if train and self.steps_in_epoch > 0:
                if step >= self.steps_in_epoch:
                    nr_of_batches = self.steps_in_epoch
                    break

            time_data = time.time()

        # calculate loss and error for epoch
        mean_loss /= nr_of_batches
        max_loss /= nr_of_batches
        min_loss /= nr_of_batches
        epoch_loss /= nr_of_batches
        epoch_error /= nr_of_batches
        class_loss /= nr_of_batches
        att_loss /= nr_of_batches

        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)

        f1 = f1_score(targets, outputs, average='macro')

        # print("debug mean and max (item): ", mean_loss.item(), max_loss.item())
        # print("debug class loss: ", class_loss)
        print("data speed: ", round(np.mean(data_times), 3), "forward speed ", round(np.mean(forward_times), 3),
              "backprop speed: ", round(np.mean(backprop_times), 3))

        print(
            '{}: loss: {:.4f}, enc error: {:.4f}, f1 macro score: {:.4f}'.format(
                "Train" if train else "Validation", epoch_loss, epoch_error, f1))
        print(
            f"Main classification loss: {round(class_loss, 3)}, attention loss: {round(att_loss, 3)} = (not scaled)(inc: {round(mean_loss.item(), 3)}, exc: {round(max_loss.item(), 3)})")

        results["attention_loss"] = att_loss
        results["classification_loss"] = class_loss
        results["epoch"] = epoch
        results["loss"] = epoch_loss
        results["error"] = epoch_error
        results["f1_score"] = f1

        results["mean_loss"] = mean_loss.item()
        results["max_loss"] = max_loss.item()
        results["min_loss"] = min_loss.item()

        if self.calculate_attention_accuracy:
            all_attention_acc /= attention_count + 0.001
            tumor_only_attention_acc /= attention_count + 0.001
            results["attention_accuracy"] = all_attention_acc
            results["tumor_attention_accuracy"] = tumor_only_attention_acc
            print("Attention accuracy: ", round(all_attention_acc, 3))

        # cm = wandb.plot.confusion_matrix(probs=None, y_true=targets.reshape(-1), preds=outputs.reshape(-1),class_names=['controls','cases'])
        # results["cm"] = cm

        return results
