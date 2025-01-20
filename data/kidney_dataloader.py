import random
import sys
from glob import glob

import nibabel as nib
import numpy as np
import raster_geometry as rg
import torch
import torchio as tio
from monai.transforms import *
from sklearn.model_selection import train_test_split

sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
sys.path.append('/users/arivajoo/GPAI/data')
from data_utils import get_kidney_datasets, downsample_scan, normalize_scan, set_orientation_nib


def make_single_sphere_coords():
    synth_coords = dict()
    synth_coords["xyz"] = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), np.random.uniform(0.3,
                                                                                                      0.7)
    synth_coords["circle_radius"] = np.random.randint(30, 40)
    synth_coords["circ_in_circ_radius"] = synth_coords["circle_radius"] - np.random.randint(10, 20)
    synth_coords["circ_in_circ_xyz"] = synth_coords["xyz"][0] + np.random.uniform(-0.02, 0.02), \
                                       synth_coords["xyz"][1] + np.random.uniform(-0.02, 0.02), \
                                       synth_coords["xyz"][2] + np.random.uniform(-0.02, 0.02)
    return synth_coords


def make_sphere_coords(img_paths, labels, deterministic: bool = True):
    synth_data = []
    if deterministic:
        np.random.seed(555)
    for i in range(len(img_paths)):
        synth_coords = make_single_sphere_coords()
        synth_data.append(synth_coords)

    np.random.shuffle(labels)

    return synth_data


class KidneyDataloader(torch.utils.data.Dataset):
    def __init__(self, only_every_nth_slice: int = 1, type: str = "train", downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 120, roll_slices=False, model_type="2D", generate_spheres: bool = False,
                 patchify: bool = False, patch_size: int = 128, no_lungs: bool = False,
                 random_experiment: bool = False):
        super().__init__()
        self.roll_slices = roll_slices
        self.as_rgb = as_rgb
        self.augmentations = augmentations
        self.nth_slice = only_every_nth_slice
        self.sample_shifting = sample_shifting
        self.plane = plane
        self.downsample = downsample
        self.model_type = model_type
        self.generate_spheres = generate_spheres
        self.type = type
        self.patchify = patchify
        self.patch_size = patch_size
        self.no_lungs = no_lungs

        if roll_slices:
            center_crop = center_crop
        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))  # 500
        self.padder_z = SpatialPad(spatial_size=(-1, -1, center_crop - 2), method="end", constant_values=0)
        self.resizer = Resize(spatial_size=512, size_mode="longest")
        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)

        control, tumor = get_kidney_datasets(type, no_lungs=no_lungs)

        control_labels = [[False]] * len(control)
        self.controls = len(control)

        tumor_labels = [[True]] * len(tumor)
        self.cases = len(tumor)

        self.img_paths = control + tumor
        self.labels = control_labels + tumor_labels

        if random_experiment:
            random.shuffle(self.labels)
        print("Data length: ", len(self.img_paths), "Label length: ", len(self.labels))
        print(
            f"control: {len(control)}, tumor: {len(tumor)}")

        self.classes = ["control", "tumor"]

        if self.generate_spheres:
            self.synth_data = make_sphere_coords(self.img_paths, self.labels, deterministic=True)

    def pick_sample_frequency(self, nr_of_original_slices: int, nth_slice: int):

        if nth_slice == 1:
            return nth_slice

        if nr_of_original_slices / nth_slice < 50:
            return self.pick_sample_frequency(nr_of_original_slices, nth_slice - 1)
        else:
            return nth_slice

    def add_synth_tumor(self, scan, synth_coords, add_hole):
        radius = synth_coords["circle_radius"]
        x, y, z = synth_coords["xyz"]
        sphere_mask = rg.sphere(scan.shape, radius, (x, y, z))
        if add_hole:
            cx, cy, cz = synth_coords["circ_in_circ_xyz"]
            hole_radius = synth_coords["circ_in_circ_radius"]
            hole_mask = rg.sphere(scan.shape, hole_radius, (cx, cy, cz))

            sphere_mask = np.logical_xor(sphere_mask, hole_mask)

        gaussian_noise_circle = torch.FloatTensor(np.random.randn(*scan.shape) * 20 + 210)
        scan[sphere_mask] = gaussian_noise_circle[sphere_mask]
        return scan

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
        z_spacing = x.header.get_zooms()[2]
        x = x.get_fdata()

        if self.downsample:
            x = downsample_scan(x)

        if self.generate_spheres:
            if self.type == "test":
                synth_coords = self.synth_data[index]
                x = self.add_synth_tumor(x, synth_coords, add_hole=y)
            elif self.type == "train":
                synth_coords = make_single_sphere_coords()
                y = torch.Tensor([np.random.choice(a=[False, True])])
                x = self.add_synth_tumor(x, synth_coords, add_hole=y)

        nr_of_original_slices = x.shape[-1]

        nth_slice = self.pick_sample_frequency(nr_of_original_slices, self.nth_slice)

        x = x[:, :, ::nth_slice]  # sample slices
        x = np.expand_dims(x, 0)  # needed for cropper
        x = self.center_cropper(x).as_tensor()
        x = np.squeeze(x)
        w, h, d = x.shape

        x = np.clip(x, -1024, None)
        # x = remove_table_3d(x)

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

        if self.augmentations is not None:

            if self.roll_slices:
                x = self.augmentations(x)

            else:
                for i in range(x.shape[2]):
                    x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))

        x = normalize_scan(x, single_channel=not self.as_rgb, model_type=self.model_type, remove_bones=False)

        if w < 512 or h < 512:
            if w >= h:
                x = tio.Resize((512, int(512 * h / w), d))(x)
            else:
                x = tio.Resize((int(512 * w / h), 512, d))(x)
        # x = x.as_tensor()

        if self.patchify:
            x = torch.Tensor(x)

            if self.type == "train":
                shift_x, shift_y = np.random.choice([0, -64, 64]), np.random.choice([0, -64, 64])
                x = torch.roll(x, shifts=(shift_x, shift_y), dims=(1, 2))
            patches = x.unfold(1, 150, 128).unfold(2, 150, 128)

            last_fold1 = torch.permute(x[:, -150:, :, :].unfold(2, 150, 128), (0, 2, 3, 1, 4)).reshape(3, -1, 150, 150)

            # last_fold2 = x[:, :, -150:, :].unfold(1, 150, 128)

            # last_fold2 = torch.permute(last_fold2, (0, 1, 3, 4, 2))

            # last_fold2 = last_fold2.reshape(3, -1, 150, 150)

            last_fold2 = torch.permute(x[:, :, -150:, :].unfold(1, 150, 128), (0, 1, 3, 4, 2)).reshape(3, -1, 150, 150)

            last_fold3 = torch.permute(x[:, -150:, -150:, :], (0, 3, 1, 2)).reshape(3, -1, 150, 150)
            patches = patches.reshape(3, -1, 150, 150)

            patches_final = torch.cat((patches, last_fold1, last_fold2, last_fold3), dim=1)
            patches_final = torch.permute(patches_final, (0, 2, 3, 1))
            return patches_final, y, (case_id, nth_slice, x)
            # x = remove_empty_tiles(x)

        height_before_padding = x.shape[-1]
        x = self.padder_z(x).as_tensor()

        return x, y, (case_id, nth_slice, z_spacing, height_before_padding,path)


class AbdomenAtlasLoader(torch.utils.data.Dataset):
    def __init__(self, only_every_nth_slice: int = 1, type: str = "train", downsample: bool = False,
                 augmentations: callable = None, as_rgb: bool = False,
                 sample_shifting: bool = False, plane: str = 'axial',
                 center_crop: int = 120, roll_slices=False, model_type="2D", dynamic_sampling: bool = True):
        super().__init__()
        self.roll_slices = roll_slices
        self.as_rgb = as_rgb
        self.augmentations = augmentations
        self.nth_slice = only_every_nth_slice
        self.model_type = model_type
        self.type = type
        self.dynamic_sampling = dynamic_sampling

        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))  # 500
        self.padder = SpatialPad(spatial_size=(512, 512, -1), mode="minimum")
        self.padder_z = SpatialPad(spatial_size=(-1, -1, center_crop - 2), method="end", constant_values=0)

        self.resizer = Resize(spatial_size=512, size_mode="longest")
        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)

        img_paths = glob('/scratch/project_465001111/abdomen_atlas/*/ct.nii.gz')
        self.train_scans, self.test_scans = train_test_split(img_paths, test_size=0.1, random_state=42)

        if type == 'train':
            self.img_paths = self.train_scans
        else:
            self.img_paths = self.test_scans
        print("Data length: ", len(self.img_paths))

    def pick_sample_frequency(self, nr_of_original_slices: int, nth_slice: int):

        if nth_slice == 1:
            return nth_slice

        if nr_of_original_slices / nth_slice < 50:
            return self.pick_sample_frequency(nr_of_original_slices, nth_slice - 1)
        else:
            return nth_slice

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        path = self.img_paths[index]

        # find scan id from path

        case_id = path.split('/')[-2]

        x = nib.load(path)

        x = set_orientation_nib(x)
        z_spacing = x.header.get_zooms()[2]
        x = x.get_fdata()

        nr_of_original_slices = x.shape[-1]

        if self.dynamic_sampling:
            nth_slice = self.pick_sample_frequency(nr_of_original_slices, self.nth_slice)
        else:
            nth_slice = self.nth_slice
        x = x[:, :, ::nth_slice]  # sample slices

        x = np.expand_dims(x, 0)  # needed for cropper
        x = self.padder(self.center_cropper(x)).as_tensor()
        x = np.squeeze(x)

        x = np.clip(x, -1024, None)

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

        if self.augmentations is not None:

            if self.roll_slices:
                x = self.augmentations(x)

            else:
                for i in range(x.shape[2]):
                    x[:, :, i] = self.augmentations(np.expand_dims(x[:, :, i], 0))

        x = normalize_scan(x, single_channel=not self.as_rgb, model_type=self.model_type, remove_bones=False)
        height_before_padding = x.shape[-1]
        x = self.padder_z(x).as_tensor()

        bag_label = torch.Tensor([1])  # placeholder
        return x, bag_label, (case_id, nth_slice, z_spacing, height_before_padding)
