import sys

import nibabel as nib
import numpy as np
import torch
from monai.transforms import GridPatch

sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
sys.path.append('/users/arivajoo/GPAI/data')
from data_utils import get_kidney_datasets, set_orientation_nib, \
    get_pasted_small_dateset, normalize_scan_new, CompassFilter


class KidneyDataloader(torch.utils.data.Dataset):
    def __init__(self, type: str = "train",
                 augmentations: callable = None, plane: str = 'axial',
                 center_crop: int = 120, patch_size=(64, 64, 64), no_lungs: bool = False,
                 pasted_experiment: bool = False,
                 TUH_only: bool = False,
                 compass_filtering: bool = False):
        super().__init__()
        self.augmentations = augmentations

        self.plane = plane

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
        self.patch_size = (100,100,100)
        self.stride = (100,100,100)
        self.grid_patch = GridPatch(
            patch_size=self.patch_size,
            stride=self.stride
        )
        self.foreground_threshold = 0.6

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        path = self.img_paths[index]

        # find scan id from path

        match = path.split('/')[-1]

        case_id = match.replace("_0000.nii", ".nii")
        y = torch.Tensor(self.labels[index])
        x = nib.load(path)

        x = set_orientation_nib(x)
        spacings = x.header.get_zooms()  # [2]-for only z, currently changed to take all spacings | h,w,d (order ?)
        x = x.get_fdata()

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

        if self.augmentations is not None:
            x = self.augmentations(np.expand_dims(x, 0))

        x = normalize_scan_new(x)
        print("orginal shape: ", x.shape)

        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        patches = self.grid_patch(x)

        print("all patches: ", patches.shape)

        if self.foreground_threshold > 0:
            keep_mask = (patches > 0.05).float().mean(dim=(1, 2, 3, 4)) > self.foreground_threshold
            patches = patches[keep_mask]
        print("after filtering: ", patches.shape)
        # Stack patches into (N, C, D, H, W)

        # 5. Compute grid dims
        grid_D = (x.shape[1] - self.patch_size[0]) // self.stride[0] + 1
        grid_H = (x.shape[2] - self.patch_size[1]) // self.stride[1] + 1
        grid_W = (x.shape[3] - self.patch_size[2]) // self.stride[2] + 1

        return patches, y, (case_id, spacings, path), (grid_D, grid_H, grid_W)
