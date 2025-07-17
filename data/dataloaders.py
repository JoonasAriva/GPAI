import sys
from typing import List

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from monai.transforms import *

sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
sys.path.append('/users/arivajoo/GPAI')
from data_utils import get_dataset_paths, set_orientation, downsample_scan, normalize_scan_new, normalize_scan, compass_filter
from scipy import ndimage as ndi
import time


class CT_dataloader(torch.utils.data.Dataset):
    def __init__(self, datasets: List[str], dataset_type: str, only_every_nth_slice: int = 1, downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 120,
                 percentage: float = 1, roll_slices=True):
        super(CT_dataloader, self).__init__()
        self.roll_slices = roll_slices
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
        print(f"control: {len(control)}, tumor: {len(tumor)}")

        self.classes = ["control", "tumor"]

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):
        time1 = time.time()
        path = self.img_paths[index]
        get_segmentation = False
        x = nib.load(path).get_fdata()

        time2 = time.time()

        x = set_orientation(x, path, self.plane)

        if self.downsample:
            x = downsample_scan(x)

        x = x[:, :, ::self.nth_slice]  # sample slices
        x = np.expand_dims(x, 0)  # needed for cropper

        x = self.center_cropper(x).as_tensor()
        w, h, d = np.squeeze(x).shape

        y = torch.Tensor(self.labels[index])

        time3 = time.time()
        if self.as_rgb:  # if we need 3 channel input
            if self.roll_slices:

                roll_forward = np.roll(x, 1, axis=(3))
                # mirror the first slice
                roll_forward[:, :, :, 0] = roll_forward[:, :, :, 1]
                roll_back = np.roll(x, -1, axis=(3))
                # mirror the first slice
                roll_back[:, :, :, -1] = roll_back[:, :, :, -2]
                x = np.stack((roll_back, x, roll_forward), axis=0)
            else:
                x = np.concatenate((x, x, x), axis=0)
            x = np.squeeze(x)
        time5 = time.time()

        if self.augmentations is not None:

            if self.roll_slices:
                x = self.augmentations(x)

            else:
                for i in range(x.shape[2]):
                    x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))

        time6 = time.time()
        # print("b4 norm: ", x.shape)
        x = normalize_scan_new(x)

        if w < 512 or h < 512:
            if w >= h:
                x = tio.Resize((512, int(512 * h / w), d))(x)
            else:
                x = tio.Resize((int(512 * w / h), 512, d))(x)
        # x = x.as_tensor()

        # print("loading: ", round(time2 - time1, 3))
        # print("rolling+stacking: ", round(time5 - time3, 3))
        # print("augmenting: ", round(time6 - time5, 3))
        return x, y, self.img_paths[index]


class CT_DataloaderPatches(torch.utils.data.Dataset):
    def __init__(self, dataset_type: str, only_every_nth_slice: int = 2, downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 200, roll_slices: bool = True,
                 percentage: float = 1):
        super(CT_DataloaderPatches, self).__init__()
        self.as_rgb = as_rgb
        self.roll_slices = roll_slices
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

        control, tumor, TUH_length, all_labels = get_dataset_paths(datasets=["TUH_kidney"], dataset_type=dataset_type,
                                                                   percentage=percentage, return_predictions=True)
        self.all_labels = all_labels
        control_labels = [[False]] * len(control)
        tumor_labels = [[True]] * len(tumor)

        self.TUH_length = TUH_length
        self.img_paths = control + tumor
        self.labels = control_labels + tumor_labels

        print("Data length: ", len(self.img_paths), "Label length: ", len(self.labels))
        print(f"control: {len(control)}, tumor: {len(tumor)}")

        self.classes = ["control", "tumor"]

        self.patch_coordinates = []
        for i in range(len(self.labels)):
            single_coords = []
            for j in range(2):
                x, y, z = np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
                single_coords.append([x, y, z])
            self.patch_coordinates.append(single_coords)

    def get_segmentation(self, path):
        file_id = path.split("/")[-1]
        file_id = file_id.strip('.nii.gz')
        file_id = file_id.strip('_0000')

        match = list(filter(lambda x: file_id in x, self.all_labels))[0]
        segmentation = nib.load(match).get_fdata()
        return segmentation

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        path = self.img_paths[index]
        x = nib.load(path).get_fdata()
        segmentation = self.get_segmentation(path)
        x = np.stack([x, segmentation], axis=-1)

        x = set_orientation(x, path, self.plane)

        if self.downsample:
            x = downsample_scan(x)

        x = x[:, :, ::self.nth_slice]  # sample slices
        ####
        x = np.transpose(np.squeeze(x), (3, 0, 1, 2))
        x = self.center_cropper(x)
        x = np.transpose(np.squeeze(x), (1, 2, 3, 0))  # and then back again
        ####
        scan, segmentation = x[:, :, :, 0], x[:, :, :, 1]

        upper_mask = segmentation < 3
        lower_mask = segmentation > 0
        full_mask = upper_mask * lower_mask

        diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        dilated = ndi.binary_dilation(full_mask, diamond, iterations=30)

        size = scan.shape[1]
        height = scan.shape[2]
        # print(scan.shape)
        filtered_vol = np.ones((size, size, height)) * -1000  # .view(-1,-1,1)
        # filtered_vol = np.random.randn(size, size, height) * -400

        filtered_vol[dilated] = scan[dilated]

        # for i in range(2):
        #     sphere_coords = self.patch_coordinates[index]
        #     x, y, z = sphere_coords[i]
        #     sphere_mask = rg.sphere((size, size, height), 70, (x, y, z))
        #     filtered_vol[sphere_mask] = scan[sphere_mask]

        x = filtered_vol
        # norm_x = normalize_scan(filtered_vol)

        # x = torch.squeeze(norm_x)

        y = torch.tensor(self.labels[index])

        if self.augmentations is not None:

            for i in range(x.shape[2]):
                x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))
            # data = self.augmentations(image=x) # for albumentations
            # x = data["image"]
        x = np.expand_dims(x, 0)
        if self.as_rgb:  # if we need 3 channel input
            if self.roll_slices:

                roll_forward = np.roll(x, 1, axis=(3))
                # mirror the first slice
                roll_forward[:, :, :, 0] = roll_forward[:, :, :, 1]
                roll_back = np.roll(x, -1, axis=(3))
                # mirror the first slice
                roll_back[:, :, :, -1] = roll_back[:, :, :, -2]
                x = np.stack((roll_back, x, roll_forward), axis=0)
            else:
                x = np.concatenate((x, x, x), axis=0)
            x = np.squeeze(x)

        if self.augmentations is not None:

            if self.roll_slices:
                x = self.augmentations(x)

            else:
                for i in range(x.shape[2]):
                    x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))

        x = normalize_scan(x, single_channel=False)

        return x, y, self.img_paths[index]


class CT_DataloaderLimited(torch.utils.data.Dataset):
    def __init__(self, dataset_type: str, only_every_nth_slice: int = 2, downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 200,
                 percentage: float = 1, roll_slices=True):
        super(CT_DataloaderLimited, self).__init__()
        self.roll_slices = roll_slices
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

        control, tumor, TUH_length, all_labels = get_dataset_paths(datasets=["TUH_kidney"], dataset_type=dataset_type,
                                                                   percentage=percentage, return_predictions=True)
        self.all_labels = all_labels
        control_labels = [[False]] * len(control)
        tumor_labels = [[True]] * len(tumor)

        self.TUH_length = TUH_length
        self.img_paths = control + tumor
        self.labels = control_labels + tumor_labels

        print("Data length: ", len(self.img_paths), "Label length: ", len(self.labels))
        print(f"control: {len(control)}, tumor: {len(tumor)}")

        self.classes = ["control", "tumor"]

    def get_segmentation(self, path):
        file_id = path.split("/")[-1]
        file_id = file_id.strip('.nii.gz')
        file_id = file_id.strip('_0000')

        match = list(filter(lambda x: file_id in x, self.all_labels))[0]
        segmentation = nib.load(match).get_fdata()
        return segmentation

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        path = self.img_paths[index]
        x = nib.load(path).get_fdata()
        segmentation = self.get_segmentation(path)
        x = np.stack([x, segmentation], axis=-1)

        x = set_orientation(x, path, self.plane)

        if self.downsample:
            x = downsample_scan(x)

        x = x[:, :, ::self.nth_slice]  # sample slices

        scan, segmentation = x[:, :, :, 0], x[:, :, :, 1]

        upper_mask = segmentation < 3
        lower_mask = segmentation > 0
        full_mask = upper_mask * lower_mask
        center = full_mask.sum(axis=(0, 1)).argsort()[-1]
        x = scan[:, :, center - 20:center + 20]
        x = np.expand_dims(x, 0)

        y = torch.tensor(self.labels[index])

        if self.as_rgb:  # if we need 3 channel input
            if self.roll_slices:

                roll_forward = np.roll(x, 1, axis=(3))
                # mirror the first slice
                roll_forward[:, :, :, 0] = roll_forward[:, :, :, 1]
                roll_back = np.roll(x, -1, axis=(3))
                # mirror the first slice
                roll_back[:, :, :, -1] = roll_back[:, :, :, -2]
                x = np.stack((roll_back, x, roll_forward), axis=0)
            else:
                x = np.concatenate((x, x, x), axis=0)
            x = np.squeeze(x)

        if self.augmentations is not None:

            if self.roll_slices:
                x = self.augmentations(x)

            else:
                for i in range(x.shape[2]):
                    x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))

        x = normalize_scan(x, single_channel=False)

        return x, y, self.img_paths[index]
