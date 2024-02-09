# fix datasets
# train
# find best treshold
# evaluate all scans
# ouput metrics
import logging
import os
import sys
from collections import defaultdict
from os import makedirs
from pathlib import Path

import hydra
import matplotlib.animation as animation
import raster_geometry as rg
import torch
import torch.utils.data as data_utils
from matplotlib.colors import *
from omegaconf import DictConfig

import synth_train
from data.synth_dataloaders import SynthDataloader
from data.visualize_utils import get_animation_with_masks
from models import ResNet18Attention
from modules.ScoreCAM.cam.scorecam import ScoreCAM_for_attention

# for discrete colormap
discrete_cmap = ListedColormap(['#53c972', '#e3c634', '#2a5fbd'])
discrete_cmap.set_under(color='white', alpha=0)
boundaries = [0.1, 1.1, 2.1, 3.1]
norm = BoundaryNorm(boundaries, discrete_cmap.N)


def calculate_IOU(x, gt):
    gt = np.array(gt, dtype=bool)
    x = np.array(x, dtype=bool)

    overlap = x * gt  # Logical AND
    union = x + gt  # Logical OR

    IOU = overlap.sum() / float(union.sum())

    return IOU


def reconstruct_synth_gt(data, circle_coordinates):
    add_circle, z, y, x, radius = circle_coordinates

    gt = torch.zeros_like(data[0, :, :, :])

    size = gt.shape[1]
    height = gt.shape[2]

    sphere_mask = rg.sphere((size, size, height), radius.item(), (y.item(), x.item(), z.item()))
    ones = torch.ones_like(gt)

    gt[sphere_mask] = ones[sphere_mask]
    return gt


def test_tresholds(att_map, gt, i):
    tresholds = np.linspace(0.5, 0.95, 10)
    ious = []
    best_IOU = 0
    for tresh in tresholds:
        treshold_map = torch.ones_like(att_map)
        treshold_map[att_map <= tresh] = 0

        IOU = calculate_IOU(treshold_map, gt)

        ious.append(IOU)
        if IOU > best_IOU:
            best_IOU = IOU
            #ani = get_animation_with_masks(gt, treshold_map, use_zoom=False)
            #writergif = animation.PillowWriter(fps=15)
            #ani.save(f'{Path(dir_vis)}/animation_{i}_{tresh}_{round(IOU, 4)}.gif', writer=writergif)

    iou_dict = dict(zip(tresholds, ious))
    return iou_dict


def compute_best_treshold(ious_per_tresh):
    best_iou = 0
    best_tresh = 0
    for k, v in ious_per_tresh.items():
        iou = np.mean(v)
        print("tresh: ", k, "mean iou: ", iou)
        if iou > best_iou:
            best_iou = iou
            best_tresh = k
    print("Best treshold is ", best_tresh, "with an (mean) IOU score of ", best_iou)
    return best_tresh


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
dir_checkpoint = Path('./checkpoints/')
dir_vis = Path('./visuals')


#@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main():
    train_model = False

    #print(f"Running {cfg.project}, Work in {os.getcwd()}")

    np.random.seed(1)
    torch.manual_seed(1)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        print('\nGPU is ON!')
    if train_model:
        model_path = synth_train.main()
    else:
        model_path ='./results/synth_train/resnet18/2024-01-23/11-07-56/checkpoints/best_loss_model.pth'
    #'/gpfs/space/home/joonas97/GPAI/experiments/MIL_with_encoder/results/synth_train/2024-01-12/15-44-50/checkpoints/best_loss_model.pth' #first default good model
    train_dataset = SynthDataloader(length=60, return_meta=True)
    test_dataset = SynthDataloader(length=20, return_meta=True)
    val_dataset = SynthDataloader(length=10, return_meta=True)

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    model = ResNet18Attention(neighbour_range=0, num_attention_heads=1)


    model.load_state_dict(torch.load(model_path))

    cam_obj = ScoreCAM_for_attention(dict(type='custom_model', arch=model, layer_name="backbone.7"))

    ious_per_tresh = defaultdict(lambda: [])

    step = 0
    model.eval()
    makedirs(Path(dir_vis), exist_ok=True)
    for data, label, coordinates in val_loader:

        if label == 0:
            continue
        step += 1

        # reconstruct the ground truth mask
        circle_coordinates = coordinates[0]

        gt = reconstruct_synth_gt(torch.squeeze(data, 0), circle_coordinates)

        data = torch.permute(torch.squeeze(data, 0), (3, 0, 1, 2))  # (h, c, x, y)
        gt = torch.permute(gt, (2, 0, 1))  # (h, c, x, y)
        data, bag_label = data.cuda(), label.cuda()

        with torch.cuda.amp.autocast():

            att_map, predicted_class, att = cam_obj(data)
            att_map = att_map.cpu()

            logging.info(f"predicted: {predicted_class} bag label: {bag_label}")
            iou_dict = test_tresholds(att_map, gt, step)
        print("data shape:", data.shape)
        print("att map shaoe:", att_map.shape)
        ani = get_animation_with_masks(data[:, 0, :, :].cpu(), att_map, use_zoom=False)
        writergif = animation.PillowWriter(fps=12)
        ani.save(f'{Path(dir_vis)}/animation_{step}.gif', writer=writergif)
        for k, v in iou_dict.items():
            ious_per_tresh[k].append(v)

        if step > 4:
            best_tresh = compute_best_treshold(ious_per_tresh)
            break
    best_tresh = 0.85
    test_counter = 0
    ious = 0
    for data, label, coordinates in test_loader:

        if label == 0:
            continue
        test_counter += 1

        # reconstruct the ground truth mask
        circle_coordinates = coordinates[0]
        gt = reconstruct_synth_gt(torch.squeeze(data, 0), circle_coordinates)

        data = torch.permute(torch.squeeze(data, 0), (3, 0, 1, 2))  # (h, c, x, y)
        gt = torch.permute(gt, (2, 0, 1))

        data, label = data.cuda(), label.cuda()

        with torch.cuda.amp.autocast():

            att_map, predicted_class, att = cam_obj(data)
            att_map = att_map.cpu()

            treshold_map = torch.ones_like(att_map)
            treshold_map[att_map <= best_tresh] = 0
            IOU = calculate_IOU(treshold_map, gt)
            ious += IOU
    mean_iou = ious / test_counter
    logging.info(f"Got avg IOU of {mean_iou} with treshold {best_tresh}")


if __name__ == "__main__":
    main()
