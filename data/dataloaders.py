import random
from typing import List

import nibabel as nib
import numpy as np
import torch
from monai.transforms import *
import sys
sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
from data_utils import get_dataset_paths, set_orientation, downsample_scan, normalize_scan


class CT_dataloader(torch.utils.data.Dataset):
    def __init__(self, datasets: List[str], dataset_type: str, only_every_nth_slice: int = 1, downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 120,
                 percentage: float = 1):
        super(CT_dataloader, self).__init__()
        self.as_rgb = as_rgb
        self.augmentations = augmentations
        self.nth_slice = only_every_nth_slice
        self.sample_shifting = sample_shifting
        self.plane = plane
        self.downsample = downsample

        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))  # 500

        self.resizer = Resize(spatial_size=512, size_mode="longest")
        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)
        print("USING ", 100 * percentage, "% of the dataset")

        control, tumor, TUH_length = get_dataset_paths(datasets=datasets, dataset_type=dataset_type,
                                                       percentage=percentage)

        control_labels = [[False]] * len(control)
        tumor_labels = [[True]] * len(tumor)

        self.TUH_length = TUH_length
        self.img_paths = control + tumor
        self.labels = control_labels + tumor_labels

        print("Data length: ", len(self.img_paths), "Label length: ", len(self.labels))
        print(
            f"control: {len(control)}, tumor: {len(tumor)}")

        self.classes = ["control", "tumor"]

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        path = self.img_paths[index]
        get_segmentation = False
        x = nib.load(path).get_fdata()

        x = set_orientation(x, path, self.plane)

        if self.downsample:
            x = downsample_scan(x)

        x = x[:, :, ::self.nth_slice] # sample slices

        norm_x = normalize_scan(x)
        norm_x = self.center_cropper(norm_x)

        x = torch.squeeze(norm_x)

        y = torch.tensor(self.labels[index])

        if self.augmentations is not None:

            for i in range(x.shape[2]):
                x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))
            # data = self.augmentations(image=x) # for albumentations
            # x = data["image"]

        if self.as_rgb:  # if we need 3 channel input
            x = torch.stack([x, x, x], dim=0)
            x = torch.squeeze(x)

        x = x.as_tensor()

        return x, y, self.img_paths[index]
