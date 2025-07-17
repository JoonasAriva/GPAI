import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import set_orientation

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


def compute_aggregated_hu_histograms(scan_paths, bins=400):
    hists = {}

    # hist_total = np.zeros(bins)
    bin_edges = np.linspace(-150, 250, bins + 1)
    main_df = pd.DataFrame()
    for i, path in enumerate(tqdm(scan_paths)):

        label = get_source_label(path)

        if label not in hists:
            hists[label] = np.zeros(bins)

        scan = nib.load(path).get_fdata()
        scan = set_orientation(scan, path, "axial")

        scan = np.clip(scan, -150, 250)
        hist, _ = np.histogram(scan.flatten(), bins=bin_edges)

        row = pd.DataFrame([hist], columns=[str(i) for i in range(bins)])
        row["class"] = label
        row["tumor"] = test_dataset.labels[i]
        # data = pd.DataFrame(arr, columns=[str(i) for i in range(3)])
        main_df = pd.concat([main_df, row], axis=0)

        # hists[label] += hist
    return main_df
    # return hists, bin_edges


print("Starting computation of aggregated HU histograms")
paths = test_dataset.img_paths
main_df = compute_aggregated_hu_histograms(paths)
print("Finished computation of aggregated HU histograms")
main_df.to_csv("histograms_test.csv", index=False)
# with open('train_data_dict.pkl', 'wb') as f:
#    pickle.dump(hists, f)
print("Finished saving HU histograms into dataframe")
