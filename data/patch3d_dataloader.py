import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import *

sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
sys.path.append('/users/arivajoo/GPAI/data')
from data_utils import get_kidney_datasets, set_orientation_nib, \
    get_pasted_small_dateset, normalize_scan_new, CompassFilter, remove_table_3d


def threshold_f(x):
    # threshold at 1
    return x > -800


def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape  # [2:]  # (D, H, W)
    array = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(array.copy(), dtype=torch.float32), dim=0), dim=0)
    # print("before scaling", array.shape)
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False)
    # print("after scaling", resized_array.shape)
    resized_array = torch.squeeze(resized_array).numpy()
    # print("after squueezing", resized_array.shape)
    return resized_array


class KidneyDataloader3D(torch.utils.data.Dataset):
    def __init__(self, type: str = "train", nth_slice: int = 3,
                 augmentations: callable = None, plane: str = 'axial',
                 center_crop: int = 120, no_lungs: bool = False,
                 pasted_experiment: bool = False,
                 TUH_only: bool = False,
                 compass_filtering: bool = False, rgb: bool = True, roll_slices: bool = True,
                 patch_size: tuple = (1, 200, 200), **kwargs):
        super().__init__()
        self.augmentations = augmentations
        self.nth_slice = nth_slice
        self.plane = plane
        self.as_rgb = rgb
        self.roll_slices = roll_slices
        self.type = type
        self.patch_size = patch_size
        self.compass_filtering = compass_filtering
        if compass_filtering:
            self.COMPASS = CompassFilter(
                dataframe_path_train='/users/arivajoo/GPAI/experiments/MIL_with_encoder/train_compass_scores.csv',
                dataframe_path_test='/users/arivajoo/GPAI/experiments/MIL_with_encoder/test_compass_scores.csv')

        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)

        if pasted_experiment and type == 'train':
            control, tumor = get_pasted_small_dateset()
        else:
            control, tumor = get_kidney_datasets(type, no_lungs=no_lungs, TUH_only=TUH_only)

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
        self.patch_size = patch_size
        self.stride = patch_size
        self.grid = (4,3)
        self.grid_split = GridSplit(grid=self.grid, size=None)
        self.grid_patch = GridPatch(
            patch_size=self.patch_size,
            stride=self.stride
        )
        self.foreground_threshold = 0.3
        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))
        self.air_cropper = CropForeground(select_fn=threshold_f, margin=0, allow_smaller=False)
        self.exact_path = None

    def pick_sample_frequency(self, nr_of_original_slices: int, nth_slice: int):

        if nth_slice == 1:
            return nth_slice

        if nr_of_original_slices / nth_slice < 50:
            return self.pick_sample_frequency(nr_of_original_slices, nth_slice - 1)
        else:
            return nth_slice

    def extract_patches(self, x):
        """
        Extract (1, n, n) patches from each slice of x (shape: C, D, H, W).
        Returns: Tensor of shape (total_patches, 1, n, n).
        """
        patches = []
        for i in range(x.shape[1]):  # Iterate over depth (D)
            slice_data = x[:, i, :, :]  # Shape: (C, H, W)
            slice_patches = self.grid_split(slice_data)  # List of (C, n, n) patches
            patches.extend(slice_patches)
        return np.stack(patches, axis=0)  # Shape: (total_patches, C, n, n)

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        if self.exact_path is None:
            path = self.img_paths[index]
        else:
            path = self.exact_path
        # find scan id from path

        match = path.split('/')[-1]

        case_id = match.replace("_0000.nii", ".nii")
        y = torch.Tensor(self.labels[index])
        x = nib.load(path)

        x = set_orientation_nib(x)
        spacings = x.header.get_zooms()  # [2]-for only z, currently changed to take all spacings | h,w,d (order ?)
        x = x.get_fdata()

        nr_of_original_slices = x.shape[-1]

        new_spacings = (0.84, 0.84, 2)
        x = resize_array(x, spacings, new_spacings)
        # print("after adjusting spacings: ", x.shape)
        nth_slice = self.pick_sample_frequency(nr_of_original_slices, self.nth_slice)

        x = x[:, :, ::nth_slice]  # sample slices
        x = np.expand_dims(x, 0)  # needed for cropper
        x = self.center_cropper(x).as_tensor()
        x = np.squeeze(x)
        # if self.compass_filtering:
        #     original_len = x.shape[-1]
        #     low_index, high_index = self.COMPASS.compass_filter_indexes(case_id,
        #                                                                 train=True if self.type == 'train' else False)
        #     x = x[:, :, low_index:high_index]
        #     after_len = x.shape[-1]
        #     filter_effect = original_len - after_len
        # else:
        #     filter_effect = 0
        x = np.clip(x, -1024, None)
        x = remove_table_3d(x)
        shape_after_table = x.shape
        x = self.air_cropper(x)
        # print("after cropping table and air: ", x.shape)

        if self.as_rgb:  # if we need 3 channel input
            if self.roll_slices:

                roll_forward = np.roll(x, 1, axis=(2))
                # mirror the first slice
                # roll_forward[:, :, 0] = roll_forward[:, :, 1]
                roll_back = np.roll(x, -1, axis=(2))
                # mirror the first slice
                # roll_back[:, :, -1] = roll_back[:, :, -2]
                x = np.stack((roll_back, x, roll_forward), axis=0)
                x = x[:, :, :, 1:-1]
            else:
                x = np.concatenate((x, x, x), axis=0)

        x = normalize_scan_new(x)

        x = np.transpose(x, (0, 3, 1, 2))
        d = x.shape[1]
        patches = torch.as_tensor(self.extract_patches(x))

        # print("all patches: ", patches.shape)
        # print("before filtering: ", patches.shape)
        self.patch_size = (1,patches.shape[2],patches.shape[3])
        num_patches = patches.shape[0]
        indices = torch.arange(num_patches)
        keep_mask = (patches > 0.05).float().mean(dim=(1, 2, 3)) > self.foreground_threshold
        patches = patches[keep_mask]
        filtered_indices = indices[keep_mask]
        # print("after filtering: ", patches.shape)
        # print("after filtering: ", patches.shape)
        # Stack patches into (N, C, D, H, W)

        if self.augmentations is not None:
            x = self.augmentations(x)

        # 5. Compute grid dims
        # print("x shape: ",x.shape)
        # grid_D = (x.shape[1] - self.patch_size[0]) // self.stride[0] + 1
        # grid_H = max((x.shape[2] - self.patch_size[1]) // self.stride[1] + 1, 1)
        # grid_W = max((x.shape[3] - self.patch_size[2]) // self.stride[2] + 1, 1)
        # print(grid_D, grid_H, grid_W)
        grid_H = self.grid[0]
        grid_W = self.grid[1]
        grid_D = d
        return patches, y, (case_id, new_spacings, path, nth_slice), (
            grid_H, grid_W, grid_D), filtered_indices, num_patches
