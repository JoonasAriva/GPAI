import sys
from typing import List

import nibabel as nib
import numpy as np
import torch
from monai.transforms import *

sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
from data_utils import get_dataset_paths, set_orientation, downsample_scan, normalize_scan
from scipy import ndimage as ndi
import raster_geometry as rg


class CT_dataloader(torch.utils.data.Dataset):
    def __init__(self, datasets: List[str], dataset_type: str, only_every_nth_slice: int = 1, downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 120,
                 percentage: float = 1, roll_slices = True):
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

        x = x[:, :, ::self.nth_slice]  # sample slices

        norm_x = normalize_scan(x)
        norm_x = self.center_cropper(norm_x)

        x = torch.squeeze(norm_x)

        y = torch.tensor(self.labels[index])

        if self.as_rgb:  # if we need 3 channel input

            if self.roll_slices:
                roll_forward = torch.roll(x, 1, dims=(2))
                # mirror the first slice
                roll_forward[:, :, 0] = roll_forward[:, :, 1]

                roll_back = torch.roll(x, -1, dims=(2))
                # mirror the first slice
                roll_back[:, :, -1] = roll_back[:, :, -2]
                x = torch.stack([roll_back, x, roll_forward], dim=0)
            else:
                x = torch.stack([x, x, x], dim=0)
            x = torch.squeeze(x)

        if self.augmentations is not None:

            for i in range(x.shape[3]):
                x[:,:, :, i] = self.augmentations(np.expand_dims(x[:,:, :, i], 0))
            # data = self.augmentations(image=x) # for albumentations
            # x = data["image"]



        x = x.as_tensor()

        return x, y, self.img_paths[index]


class CT_DataloaderPatches(torch.utils.data.Dataset):
    def __init__(self, dataset_type: str, only_every_nth_slice: int = 2, downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 200,
                 percentage: float = 1):
        super(CT_DataloaderPatches, self).__init__()
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
        scan = normalize_scan(scan)
        scan = torch.squeeze(scan)

        upper_mask = segmentation < 3
        lower_mask = segmentation > 0
        full_mask = upper_mask * lower_mask

        diamond = ndi.generate_binary_structure(rank=3, connectivity=1)
        dilated = ndi.binary_dilation(full_mask, diamond, iterations=30)

        size = scan.shape[1]
        height = scan.shape[2]
        #print(scan.shape)
        filtered_vol = torch.ones((size, size, height)) * scan.min(dim = 0).values.min(dim=0).values#.view(-1,-1,1)
        #filtered_vol = np.random.randn(size, size, height) * -400

        filtered_vol[dilated] = scan[dilated]

        for i in range(2):
            sphere_coords = self.patch_coordinates[index]
            x, y, z = sphere_coords[i]
            sphere_mask = rg.sphere((size, size, height), 70, (x, y, z))
            filtered_vol[sphere_mask] = scan[sphere_mask]

        x = filtered_vol
        #norm_x = normalize_scan(filtered_vol)

        #x = torch.squeeze(norm_x)

        y = torch.tensor(self.labels[index])

        if self.augmentations is not None:

            for i in range(x.shape[2]):
                x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))
            # data = self.augmentations(image=x) # for albumentations
            # x = data["image"]

        if self.as_rgb:  # if we need 3 channel input

            if self.roll_slices:
                roll_forward = torch.roll(x, 1, dims=(2))
                # mirror the first slice
                roll_forward[:, :, 0] = roll_forward[:, :, 1]

                roll_back = torch.roll(x, -1, dims=(2))
                # mirror the first slice
                roll_back[:, :, -1] = roll_back[:, :, -2]
                x = torch.stack([roll_back, x, roll_forward], dim=0)
            else:
                x = torch.stack([x, x, x], dim=0)
            x = torch.squeeze(x)

        # x = x.as_tensor()

        return x, y, self.img_paths[index]
