import os
import sys

import nibabel as nib
import numpy as np
from tqdm import tqdm

os.chdir('/users/arivajoo/GPAI/experiments/MIL_with_encoder/')
print(os.getcwd())
import torch

sys.path.append('/users/arivajoo/GPAI')

from data.kidney_dataloader import KidneyDataloader
from data_utils import set_orientation_nib
from patch3d_dataloader import threshold_f, resize_array, remove_table_3d

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

from monai.transforms import *

air_cropper = CropForeground(select_fn=threshold_f, margin=0, allow_smaller=False)


def compute_bbox(dataset):
    scan_paths = dataset.img_paths

    for i, path in enumerate(tqdm(scan_paths)):
        match = path.split('/')[-1]
        case_id = match.replace("_0000.nii", ".nii")

        scan = nib.load(path)
        scan = set_orientation_nib(scan)

        spacings = scan.header.get_zooms()  # [2]-for only z, currently changed to take all spacings | h,w,d (order ?)
        x = scan.get_fdata()
        new_spacings = (0.84, 0.84, 2)
        x = resize_array(x, spacings, new_spacings)
        x = np.clip(x, -1024, None)
        x, mask = remove_table_3d(torch.Tensor(x), offline=True)

        box_start, box_end = air_cropper.compute_bounding_box(x)

        np.savez_compressed(
            f"/scratch/project_465001979/ct_data/kidney/preprocess_files/{case_id}_preproc.npz",
            mask=mask.astype(np.uint8),
            bbox=np.array([box_start, box_end])
        )

print("Starting computation of air bbox")
compute_bbox(train_dataset)
print("Starting computation of test air bbox")
compute_bbox(test_dataset)
print("Finished!")
