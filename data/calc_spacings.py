import os
import sys

import nibabel as nib
import pandas as pd
from tqdm import tqdm

os.chdir('/users/arivajoo/GPAI/experiments/MIL_with_encoder/')
print(os.getcwd())

sys.path.append('/users/arivajoo/GPAI')

from data.kidney_dataloader import KidneyDataloader

dataloader_params = {
    'only_every_nth_slice': 3, 'as_rgb': True,
    'plane': 'axial', 'center_crop': 302, 'downsample': False,
    'roll_slices': True, 'no_lungs': False}
test_dataset = KidneyDataloader(type="test",
                                augmentations=None,
                                **dataloader_params)

train_dataset = KidneyDataloader(type="train",
                                 augmentations=None,
                                 **dataloader_params)


def get_source_label(path):
    if "tuh_test" in path or "tuh_train" in path:
        if "control" in path:
            data_class = "TUH DATA CONTROL"
        else:
            data_class = "TUH DATA CASE"

    elif "parnu" in path:
        data_class = "PÄRNU"
    elif "kits" in path:
        data_class = "KITS"
    elif "TCGA" in path:
        data_class = "KIRC"
    return data_class


def compute_spacings(dataset):
    scan_paths = dataset.img_paths
    main_df = pd.DataFrame()
    for i, path in enumerate(tqdm(scan_paths)):
        label = get_source_label(path)

        scan = nib.load(path)
        # scan = set_orientation(scan, path, "axial")
        spacings = [*scan.header.get_zooms()]
        row = pd.DataFrame([spacings], columns=["h_spacing", "w_spacing", "d_spacing"])
        row[["shape_h", "shape_w", "shape_d"]] = scan.shape
        row["class"] = label
        row["tumor"] = dataset.labels[i]
        row["filename"] = path.split("/")[-1]
        main_df = pd.concat([main_df, row], axis=0)

        # hists[label] += hist
    return main_df
    # return hists, bin_edges


print("Starting computation of train set spacings")
train_df = compute_spacings(train_dataset)
train_df.to_csv("train_spacings.csv", index=False)

print("Starting computation of test set spacings")
test_df = compute_spacings(test_dataset)
test_df.to_csv("test_spacings.csv", index=False)
print("Finished!")
