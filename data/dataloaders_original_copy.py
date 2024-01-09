
import random
from typing import List

import nibabel as nib
import numpy as np

import torch
import torch.nn.functional as F
from monai.transforms import *

from data_utils import add_random_sphere, get_dataset_paths

class CT_dataloader(torch.utils.data.Dataset):
    def __init__(self, datasets: List[str], dataset_type: str, only_every_nth_slice: int = 1,
                 interpolation: bool = False, downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 120, artificial_spheres: bool = False, highlight_experiment: bool = False,
                 dimming_experiment: bool = False,
                 percentage: float = 1, output_segmentations: bool = False,
                 output_segmentations_percentage: float = 0.1):
        super(CT_dataloader, self).__init__()
        self.as_rgb = as_rgb
        self.augmentations = augmentations
        self.nth_slice = only_every_nth_slice
        self.interpolation = interpolation
        self.sample_shifting = sample_shifting
        self.plane = plane
        self.artifical_spheres = artificial_spheres
        self.downsample = downsample
        self.highlight_experiment = highlight_experiment
        self.dimming_experiment = dimming_experiment
        self.output_segmentations = output_segmentations

        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))  # 500

        self.resizer = Resize(spatial_size=512, size_mode="longest")
        self.padder = DivisiblePad((32, 32))
        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)
        print("ARTIFICIAL SPHERES EXPERIMENT: ", artificial_spheres)
        print("DIMMING EXPERIMENT: ", dimming_experiment)
        print("HIGHLIGHT EXPERIMENT: ", highlight_experiment)
        print("USING ", 100 * percentage, "% of the dataset")


        if highlight_experiment or dimming_experiment or output_segmentations:
            control, tumor, TUH_length, all_labels = get_dataset_paths(datasets=datasets, dataset_type=dataset_type,
                                                                       percentage=percentage, return_predictions=True)
            self.all_labels = all_labels
        else:
            control, tumor, TUH_length = get_dataset_paths(datasets=datasets, dataset_type=dataset_type,
                                                           percentage=percentage)

        control_labels = [[False]] * len(control)
        tumor_labels = [[True]] * len(tumor)

        self.TUH_length = TUH_length
        self.img_paths = control + tumor
        self.labels = control_labels + tumor_labels

        if self.output_segmentations:
            control_seg_output = np.random.choice([0, 1], len(control), p=[1 - output_segmentations_percentage,
                                                                           output_segmentations_percentage])
            tumor_seg_output = np.random.choice([0, 1], len(tumor), p=[1 - output_segmentations_percentage,
                                                                       output_segmentations_percentage])
            self.seg_output = np.concatenate((control_seg_output, tumor_seg_output))

        print("Data length: ", len(self.img_paths), "Label length: ", len(self.labels))
        print(
            f"control: {len(control)}, tumor: {len(tumor)}")

        self.classes = ["control", "tumor"]

    def get_segmentation(self, path):
        file_id = path.split("/")[-1]
        file_id = file_id.replace('_0000.nii', '.nii')
        file_id = file_id.strip('.nii.gz')

        match = list(filter(lambda x: file_id in x, self.all_labels))[0]
        segmentation = nib.load(match).get_fdata()
        return segmentation

    def set_orientation(self, x, path):

        # PLANES: set it into default plane (axial)
        # if transformations are needed we start from this position
        if not "kits23" in path:
            x = np.flip(x, axis=1)
            if len(x.shape) == 3:
                x = np.transpose(x, (1, 0, 2))
            elif len(x.shape) == 4:
                x = np.transpose(x, (1, 0, 2, 3))
        else:  # kits is in another orientation
            x = np.transpose(x, (1, 2, 0))
        # this should give the most common axial representation: (patient on their back)
        if self.plane == "axial":
            pass
            # originally already in axial format
        elif self.plane == "coronal":
            x = np.transpose(x, (2, 1, 0))
        elif self.plane == "sagital":
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError('plane is not correctly specified')

        return x

    def sample_slices(self, x, path):

        if not self.sample_shifting:

            if x.shape[2] < 80 and "tuh_kidney" not in path:
                # take more slices if the scan is small, currently cannot do it for tuh study as attention accuracy is measured only on certain slices
                x = x[:, :, ::max(int(self.nth_slice - 2), 1)]
            else:
                x = x[:, :, ::self.nth_slice]
        else:  # do not shift samples == take always the same slices # currently do not use with segmentations
            shift = random.randint(0, self.nth_slice - 1)
            x = x[:, :, shift::self.nth_slice]
        return x

    def downsample_scan(self, x, scale_factor=0.5):

        x = torch.from_numpy(x.copy())
        x = F.interpolate(torch.unsqueeze(torch.unsqueeze(x, 0), 0), scale_factor=scale_factor,
                          mode='trilinear', align_corners=False)
        x = np.array(torch.squeeze(x))
        return x

    def normalize_scan(self, x, segmentation: bool = False):

        if segmentation:
            seg = x[:, :, :, 1]
            x = x[:, :, :, 0]
        if not self.dimming_experiment:
            clipped_x = np.clip(x, np.percentile(x, q=0.05), np.percentile(x, q=99.5))
        else:
            clipped_x = np.clip(x, np.percentile(x, q=55), np.percentile(x, q=100))

        norm_x = (clipped_x - np.mean(clipped_x, axis=(0, 1))) / (
                np.std(clipped_x, axis=(0, 1)) + 1)  # mean 0, std 1 norm
        if segmentation:
            norm_x = np.stack([norm_x, seg], axis=-1)
        norm_x = torch.unsqueeze(torch.from_numpy(norm_x), 0)
        return norm_x

    def custom_highlight_function(self, x, segmentation):

        if self.highlight_experiment:
            x[np.where((segmentation > 0) & (segmentation < 3))] *= 3
        if self.dimming_experiment:
            x = np.clip(x, np.percentile(x, q=0), np.percentile(x, q=99.5))
            x[np.where(segmentation == 0)] *= 0.2

        return x

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        path = self.img_paths[index]
        get_segmentation = False
        x = nib.load(path).get_fdata()

        if self.highlight_experiment or self.dimming_experiment:
            segmentation = self.get_segmentation(path)
            x = self.custom_highlight_function(x, segmentation)

        if self.output_segmentations:
            if self.seg_output[index] == 1:
                segmentation = self.get_segmentation(path)
                x = np.stack([x, segmentation], axis=-1)

                get_segmentation = True

        x = self.set_orientation(x, path)

        if self.downsample:
            x = self.downsample_scan(x)

        x = self.sample_slices(x, path)
        norm_x = self.normalize_scan(x, segmentation=get_segmentation)

        if get_segmentation:  # we need to move segmentation to first dim for channel first requirement

            norm_x = torch.permute(torch.squeeze(norm_x), (3, 0, 1, 2))
            norm_x = self.center_cropper(norm_x)
            norm_x = torch.permute(norm_x, (1, 2, 3, 0))  # and then back again
        else:
            norm_x = self.center_cropper(norm_x)

        x = torch.squeeze(norm_x)
        # x = np.array(norm_x)

        if self.artifical_spheres:  # add spheres to 50% of the scans
            label = np.random.randint(0, 2)
            if label == 1:
                x = add_random_sphere(x)  # create sphere
                y = torch.tensor([True])
            else:
                y = torch.tensor([False])  # do not create sphere
        else:
            y = torch.tensor(self.labels[index])

        # x = torch.permute(x, (2, 0, 1))
        # x = self.resizer(x) # needed to resize independently from augementations, monai resizer needed channel first tensor
        # x = self.padder(x)
        # x = torch.permute(x, (1, 2, 0))

        if self.augmentations is not None:
            if get_segmentation:
                x = torch.permute(x, (3, 0, 1, 2))
                for i in range(x.shape[3]):
                    x[:, :, :, i] = self.augmentations(x[:, :, :, i])
                x = torch.permute(x, (1, 2, 3, 0))
                x[:, :, :, 1] = torch.round(x[:, :, :, 1])
            else:
                for i in range(x.shape[2]):
                    x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))

            # data = self.augmentations(image=x) # for albumentations
            # x = data["image"]

        if get_segmentation:
            segmentation = x[:, :, :, 1].as_tensor()
            x = torch.squeeze(x[:, :, :, 0])
        if self.as_rgb:  # if we need 3 channel input
            x = torch.stack([x, x, x], dim=0)
            x = torch.squeeze(x)

        x = x.as_tensor()

        return x, y, self.img_paths[index], segmentation if get_segmentation else torch.tensor(0)
