import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
os.chdir('/users/arivajoo/GPAI/experiments/MIL_with_encoder/')
sys.path.append('/users/arivajoo/GPAI')
print(os.getcwd())
from experiments.MIL_with_encoder.model_zoo import ResNetDepth

sys.path.append('/users/arivajoo/GPAI')
sys.path.append('/gpfs/space/home/joonas97/GPAI/')

model = ResNetDepth(instnorm=True)
sd = torch.load(
    '/users/arivajoo/results/depth/train/resnetdepth/kidney_real/2025-07-14/11-17-49/checkpoints/best_model.pth',
    map_location='cuda:0')
new_sd = {key.replace("module.", ""): value for key, value in sd.items()}
model.load_state_dict(state_dict=new_sd)
model.cuda()
print("mode loaded!")

from data.kidney_dataloader import KidneyDataloader
import torch.utils.data as data_utils

dataloader_params = {
    'only_every_nth_slice': 3, 'as_rgb': True,
    'plane': 'axial', 'center_crop': 302, 'downsample': False,
    'roll_slices': True, 'no_lungs': False}
train_dataset = KidneyDataloader(type="train",
                                 augmentations=None,
                                 **dataloader_params)
test_dataset = KidneyDataloader(type="test", **dataloader_params)

loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)
train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=False, **loader_kwargs)


def calculate_compass_scores(model, loader):
    main_df = pd.DataFrame()
    model.eval()
    for data, _, meta in tqdm(loader):
        torch.cuda.empty_cache()

        df = pd.DataFrame()
        data = torch.permute(torch.squeeze(data, 0), (3, 0, 1, 2))  # (h, c, x, y)
        data = data.to(torch.device("cuda"), dtype=torch.float16, non_blocking=True)

        with torch.cuda.amp.autocast(), torch.no_grad():
            weights = model(data, scan_end=meta[3])
            weights = weights.cpu()[:, 0]
            df["weights"] = pd.Series(weights)
            df["case_id"] = meta[0][0]
            df["index"] = df.index
        main_df = pd.concat([main_df, df])
    return main_df


print("start calculating compass scores")
train_scores = calculate_compass_scores(model, train_loader)
train_scores.to_csv("train_compass_scores.csv", index=False)
print("train scores calculated!")
test_scores = calculate_compass_scores(model, test_loader)
test_scores.to_csv("test_compass_scores.csv", index=False)
print("test scores calculated!")
