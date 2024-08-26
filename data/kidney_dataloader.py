import sys

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from monai.transforms import *
import re
sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
sys.path.append('/users/arivajoo/GPAI/data')
from data_utils import get_kidney_datasets, set_orientation, downsample_scan, normalize_scan, remove_table_3d


class KidneyDataloader(torch.utils.data.Dataset):
    def __init__(self, only_every_nth_slice: int = 1, type: str = "train", downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 120, roll_slices=False, model_type="2D"):
        super(KidneyDataloader, self).__init__()
        self.roll_slices = roll_slices
        self.as_rgb = as_rgb
        self.augmentations = augmentations
        self.nth_slice = only_every_nth_slice
        self.sample_shifting = sample_shifting
        self.plane = plane
        self.downsample = downsample
        self.model_type = model_type

        if roll_slices:
            center_crop = center_crop
        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))  # 500

        self.resizer = Resize(spatial_size=512, size_mode="longest")
        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)

        control, tumor = get_kidney_datasets(type)

        control_labels = [[False]] * len(control)
        self.controls = len(control)

        tumor_labels = [[True]] * len(tumor)
        self.cases = len(tumor)

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

        # find scan id from path
        print("path: ", path)
        match = path.split('/')[-1]
        print("match: ", match)
        case_id = match.replace("_0000.nii",".nii")
        print("case_id: ", case_id)

        x = nib.load(path).get_fdata()

        x = set_orientation(x, path, self.plane)

        if self.downsample:
            x = downsample_scan(x)

        x = x[:, :, ::self.nth_slice]  # sample slices
        x = np.expand_dims(x, 0)  # needed for cropper
        x = self.center_cropper(x).as_tensor()
        x = np.squeeze(x)
        w, h, d = x.shape

        y = torch.Tensor(self.labels[index])

        x = np.clip(x, -1024,None)
        #x = remove_table_3d(x)

        if self.as_rgb:  # if we need 3 channel input
            if self.roll_slices:

                roll_forward = np.roll(x, 1, axis=(2))
                # mirror the first slice
                #roll_forward[:, :, 0] = roll_forward[:, :, 1]
                roll_back = np.roll(x, -1, axis=(2))
                # mirror the first slice
                #roll_back[:, :, -1] = roll_back[:, :, -2]
                x = np.stack((roll_back, x, roll_forward), axis=0)
                x = x[:,:,:, 1:-1]
            else:
                x = np.concatenate((x, x, x), axis=0)

        if self.augmentations is not None:

            if self.roll_slices:
                x = self.augmentations(x)

            else:
                for i in range(x.shape[2]):
                    x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))


        x = normalize_scan(x, single_channel=not self.as_rgb, model_type=self.model_type)

        if w < 512 or h < 512:
            if w >= h:
                x = tio.Resize((512, int(512 * h / w), d))(x)
            else:
                x = tio.Resize((int(512 * w / h), 512, d))(x)
        # x = x.as_tensor()

        return x, y, case_id
