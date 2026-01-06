import sys
import time

import nibabel as nib
import numpy as np
import raster_geometry as rg
import torch.nn.functional as F
from monai.transforms import *

sys.path.append('/gpfs/space/home/joonas97/GPAI/data/')
sys.path.append('/users/arivajoo/GPAI/data')
from data_utils import get_kidney_datasets, set_orientation_nib, \
    get_pasted_small_dateset, CompassFilter, normalize_scan_per_patch

import torch


# def extract_patches_3d(x: torch.Tensor, patch_centers, patch_size):
#     """
#     Efficient 3D patch extraction without advanced indexing broadcast.
#
#     Args:
#         x: Tensor of shape (C, D, H, W)
#         patch_centers: (N, 3) numpy array or torch.Tensor with (d, h, w) centers
#         patch_size: tuple of (pd, ph, pw)
#
#     Returns:
#         patches: Tensor of shape (N, C, pd, ph, pw)
#     """
#     patch_centers = torch.as_tensor(patch_centers, dtype=torch.long)
#
#     C, D, H, W = x.shape
#     pd, ph, pw = patch_size
#     N = patch_centers.shape[0]
#
#     patches = []
#     for d, h, w in patch_centers:
#         d0, h0, w0 = d - pd // 2, h - ph // 2, w - pw // 2
#         d1, h1, w1 = d0 + pd, h0 + ph, w0 + pw
#         # boundary check
#         if d0 < 0 or h0 < 0 or w0 < 0 or d1 > D or h1 > H or w1 > W:
#             raise ValueError(f"Invalid center {(d, h, w)} for patch size {patch_size} in volume {x.shape}")
#
#         patch = x[:, d0:d1, h0:h1, w0:w1]
#         patches.append(patch)
#     patches = torch.stack(patches, dim=0)  # (N, C, pd, ph, pw)
#
#     return patches


def extract_patches_3d(x: torch.Tensor, patch_centers, patch_size):
    patch_centers = torch.as_tensor(patch_centers, dtype=torch.long)
    C, D, H, W = x.shape
    pd, ph, pw = patch_size
    N = patch_centers.shape[0]

    d0 = patch_centers[:, 0] - pd // 2
    h0 = patch_centers[:, 1] - ph // 2
    w0 = patch_centers[:, 2] - pw // 2

    if ((d0 < 0).any() or (h0 < 0).any() or (w0 < 0).any() or
            ((d0 + pd) > D).any() or ((h0 + ph) > H).any() or ((w0 + pw) > W).any()):
        raise ValueError("Invalid patch centers")

    patches = x.new_empty((N, C, pd, ph, pw))
    for i in range(N):
        patches[i] = x[:, d0[i]:d0[i] + pd, h0[i]:h0[i] + ph, w0[i]:w0[i] + pw]

    return patches


# def extract_patches_3d(x: torch.Tensor, patch_centers, patch_size):
#     """
#     Fully vectorized 3D patch extraction.
#
#     Args:
#         x: Tensor of shape (C, D, H, W)
#         patch_centers: (N, 3) numpy array or torch.Tensor with (d, h, w) centers
#         patch_size: tuple of (pd, ph, pw)
#
#     Returns:
#         patches: Tensor of shape (N, C, pd, ph, pw)
#     """
#
#     patch_centers = torch.as_tensor(patch_centers, dtype=torch.long)
#
#     C, D, H, W = x.shape
#     N = patch_centers.shape[0]
#     pd, ph, pw = patch_size
#
#     # compute start indices for each patch
#     d_start = patch_centers[:, 0] - pd // 2
#     h_start = patch_centers[:, 1] - ph // 2
#     w_start = patch_centers[:, 2] - pw // 2
#
#     assert (d_start >= 0).all() and (h_start >= 0).all() and (w_start >= 0).all(), "Patch starts < 0"
#     assert (d_start + pd <= D).all() and (h_start + ph <= H).all() and (w_start + pw <= W).all(), \
#         "Patch exceeds scan bounds"
#
#     # build index grids
#     d_idx = d_start[:, None, None, None] + torch.arange(pd).view(1, pd, 1, 1)
#     h_idx = h_start[:, None, None, None] + torch.arange(ph).view(1, 1, ph, 1)
#     w_idx = w_start[:, None, None, None] + torch.arange(pw).view(1, 1, 1, pw)
#
#     # expand to match patch shape
#     d_idx = d_idx.expand(N, pd, ph, pw)
#     h_idx = h_idx.expand(N, pd, ph, pw)
#     w_idx = w_idx.expand(N, pd, ph, pw)
#
#     # gather patches
#     patches = x[:, d_idx, h_idx, w_idx]  # x is (C,D,H,W)
#     # output shape: (C, N, pd, ph, pw) -> permute to (N, C, pd, ph, pw)
#     patches = patches.permute(1, 0, 2, 3, 4).contiguous()
#
#     return patches


def threshold_f(x):
    # threshold at 1
    return x > -800


def resize_array(array, current_spacing, target_spacing, mask: bool = False):
    original_shape = array.shape  # [2:]  # (D, H, W)
    array = torch.unsqueeze(torch.unsqueeze(array, dim=0), dim=0)
    # print("before scaling", array.shape)
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    if mask:
        mode = 'nearest'
    else:
        mode = 'trilinear'
    resized_array = F.interpolate(array, size=new_shape, mode=mode, align_corners=False if not mask else None)
    # print("after scaling", resized_array.shape)
    resized_array = torch.squeeze(resized_array)
    # print("after squueezing", resized_array.shape)
    return resized_array


def add_sphere(scan, synth_coords, rng, simplified=False):
    radius = rng.integers(25, 30)  # 15-20

    relative_coords = tuple(round(i / j, 3) for i, j in zip(synth_coords, scan.shape))
    sphere_mask = rg.sphere(scan.shape, radius, (relative_coords))

    gaussian_noise_circle = torch.FloatTensor(np.ones_like(scan) * 1000)
    scan[sphere_mask] = gaussian_noise_circle[sphere_mask]
    if simplified:
        gaussian_noise = torch.FloatTensor(np.random.rand(*scan.shape) * 20)
        scan[~sphere_mask] = gaussian_noise[~sphere_mask]
    return scan


class KidneyDataloader3D(torch.utils.data.Dataset):
    def __init__(self, type: str = "train", nth_slice: int = 3,
                 augmentations: callable = None, plane: str = 'axial',
                 center_crop: int = 120, no_lungs: bool = False,
                 pasted_experiment: bool = False,
                 TUH_only: bool = False, TUH_extra_data: bool = False,
                 compass_filtering: bool = False, model_type: str = '3D',
                 patch_size: tuple = (1, 150, 150), nr_of_patches: int = 100, sample: str = 'grid',
                 spheres: bool = False, process_masks: bool = False, **kwargs):
        super().__init__()
        self.augmentations = augmentations
        self.nth_slice = nth_slice
        self.plane = plane
        self.type = type
        self.kind = model_type
        self.patch_size = patch_size
        self.compass_filtering = compass_filtering
        self.sample = sample
        self.process_masks = process_masks
        self.spheres = spheres
        if self.spheres:
            self.process_masks = True

        if compass_filtering:
            self.COMPASS = CompassFilter(
                dataframe_path_train='/users/arivajoo/GPAI/experiments/MIL_with_encoder/train_compass_scores.csv',
                dataframe_path_test='/users/arivajoo/GPAI/experiments/MIL_with_encoder/test_compass_scores.csv')

        print("PLANE: ", plane)
        print("CROP SIZE: ", center_crop)

        if pasted_experiment and type == 'train':
            control, tumor = get_pasted_small_dateset()
        # elif self.spheres:
        #     control = get_TUH_control(type)
        #     tumor = []
        # else:
        control, tumor = get_kidney_datasets(type, no_lungs=no_lungs, TUH_only=TUH_only, TUH_extra_data=TUH_extra_data)

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
        self.grid = (4, 3)
        self.grid_split = GridSplit(grid=self.grid, size=None)
        self.grid_patch = GridPatch(
            patch_size=self.patch_size,
            stride=self.stride
        )
        self.foreground_threshold = 0.25
        self.center_cropper = CenterSpatialCrop(roi_size=(512, 512, center_crop))
        self.air_cropper = CropForeground(select_fn=threshold_f, margin=0, allow_smaller=False)
        self.exact_path = None
        self.nr_of_patches = nr_of_patches

        self.mode = "coarse"
        self.cached_coords = {}

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

    def set_refinement_targets(self, top_centers, volume):
        """Cache high-attention centers for refinement sampling."""
        self.top_centers = top_centers
        self.mode = "refined"
        self.x = volume

    def find_depth_sample_rate(self, D, max_patches, axial_patches_nr, sample_rate):

        how_many_patches = (D // sample_rate) * axial_patches_nr

        if how_many_patches > max_patches:
            return self.find_depth_sample_rate(D, max_patches, axial_patches_nr, sample_rate + 1)
        elif how_many_patches <= max_patches:
            return sample_rate

    def find_axial_sample_rate(self, dimension, patch_size, edge_buffer, sample_rate):

        axial_patches_nr = ((dimension - 2 * edge_buffer) // int(patch_size * sample_rate))

        if axial_patches_nr < 2:

            return self.find_axial_sample_rate(dimension, patch_size, edge_buffer, sample_rate - 0.1)
        else:
            return sample_rate

    def sample_patch_coordinates(self, x, sphere_options, tumor_options=None):

        D, H, W = x.shape[-3:]

        edge_buffer = 70
        if self.kind == "2D":
            depth_buffer = 1
            patch_size = (1, 128, 128)
        else:
            if x.shape[1] > 65:
                depth_buffer = 33
                patch_size = (64, 128, 128)
            else:
                patch_size = (x.shape[1] - 2, 128, 128)
                depth_buffer = (x.shape[1] - 1) // 2

        if sphere_options is not None and self.sample != 'grid':
            sphere_options = np.array(sphere_options)

            idxs = np.random.randint(len(sphere_options), size=self.nr_of_patches * 8, dtype=int)

            coordinates = sphere_options[idxs]

            offsets = np.random.uniform(np.array([0, -64, -64]),
                                        np.array([0, 64, 64]),
                                        size=np.array([self.nr_of_patches * 8, 3])).astype(int)

            coordinates = coordinates + offsets
            possible_coords = []

            for coord in coordinates:
                d, h, w = coord
                if d > D - depth_buffer or h > H - edge_buffer or w > W - edge_buffer:
                    continue
                if d < depth_buffer or h < edge_buffer or w < edge_buffer:
                    continue
                possible_coords.append([coord])

            coordinates = np.concatenate(possible_coords)
            if tumor_options is not None:
                if len(tumor_options) > 0:
                    tumor_idx = np.random.randint(len(tumor_options), size=1, dtype=int)
                    tumor_coord = tumor_options[tumor_idx]
                    coordinates = np.concatenate((tumor_coord, coordinates))

            coordinates = coordinates[:self.nr_of_patches]

            return coordinates, patch_size

        if self.mode == "coarse":
            if self.sample == "uniform":
                patch_centers = np.random.uniform(np.array([depth_buffer, edge_buffer, edge_buffer]),
                                                  np.array([D - depth_buffer, H - edge_buffer, W - edge_buffer]),
                                                  size=np.array([self.nr_of_patches, 3]))
                patch_centers = np.round(patch_centers).astype(int)


            elif self.sample == "grid":
                patch_centers = []
                # how many patches?
                H_sample_rate = self.find_axial_sample_rate(H, patch_size[1], edge_buffer, sample_rate=1)
                W_sample_rate = self.find_axial_sample_rate(W, patch_size[2], edge_buffer, sample_rate=1)
                # print("axial sample_rate: ", H_sample_rate, W_sample_rate)
                axial_patches_nr = ((((H - 2 * edge_buffer) // int(patch_size[1] * H_sample_rate)) + 1) *
                                    (((W - 2 * edge_buffer) // int(patch_size[2] * W_sample_rate)) + 1))
                # print("axial patches nr: ", axial_patches_nr)
                sample_rate = self.find_depth_sample_rate(D, max_patches=120, axial_patches_nr=axial_patches_nr,
                                                          sample_rate=3)
                # print("depth sample_rate:", sample_rate)
                # print("patch size: ", patch_size)
                # print("d:", len(np.arange(depth_buffer, D - depth_buffer, int(patch_size[0] * sample_rate))))
                # print("h: ", len(np.arange(edge_buffer, H - edge_buffer, int(patch_size[1] * H_sample_rate))))
                # print("w: ", len(np.arange(edge_buffer, W - edge_buffer, int(patch_size[2] * W_sample_rate))))
                for d in range(depth_buffer, D - depth_buffer, int(patch_size[0] * sample_rate)):
                    for h in range(edge_buffer, H - edge_buffer, int(patch_size[1] * H_sample_rate)):
                        for w in range(edge_buffer, W - edge_buffer, int(patch_size[2] * W_sample_rate)):
                            patch_centers.append([[d, h, w]])
                patch_centers = np.concatenate(patch_centers)
                # print("patch_centers:", patch_centers.shape)

        # if self.mode == "refined":
        #     all_neighbors = []
        #     radius = 30
        #     n_samples = 50
        #
        #     for c in self.top_centers:
        #         # Sample uniform offsets around the center
        #         z, y, x = c
        #         min_bounds = np.array(
        #             [-min(radius, max(z - depth_buffer, 0)), -min(radius, max(y - edge_buffer, 0)),
        #              -min(radius, max(x - edge_buffer, 0))])
        #         max_bounds = np.array(
        #             [min(radius, max(D - depth_buffer - z, 0)), min(radius, max((H - edge_buffer - y, 0))),
        #              min(radius, max(W - edge_buffer - x, 0))])
        #
        #         offsets = np.random.uniform(min_bounds, max_bounds,
        #                                     size=np.array([n_samples, 3]))
        #
        #         neighbors = torch.squeeze(c) + offsets
        #         all_neighbors.append(neighbors)
        #         patch_centers = torch.cat(all_neighbors, dim=0)
        return patch_centers, patch_size

    def __len__(self):
        # a DataSet must know its size
        return len(self.img_paths)

    def __getitem__(self, index):

        t0 = time.time()
        path = self.img_paths[index]

        match = path.split('/')[-1]

        case_id = match.replace("_0000.nii", ".nii")
        y = torch.Tensor(self.labels[index])
        x = nib.load(path)
        t1 = time.time()
        x = set_orientation_nib(x)
        spacings = x.header.get_zooms()  # [2]-for only z, currently changed to take all spacings | h,w,d (order ?)
        x = torch.from_numpy(x.get_fdata().copy())
        new_spacings = (0.84, 0.84, 2)
        x = resize_array(x, spacings, new_spacings)
        if self.process_masks:
            # seed = hash(path) % (2 ** 32)
            # rng = np.random.default_rng(seed)
            seg_path = path.replace("_0000.nii.gz", ".nii.gz", ).replace("images", "labels")
            t2 = time.time()
            kidney_mask = set_orientation_nib(nib.load(seg_path))
            kidney_mask = np.asanyarray(kidney_mask.dataobj, dtype=np.uint8)
            kidney_mask = torch.from_numpy(kidney_mask.copy())
            t3 = time.time()
            kidney_mask = resize_array(kidney_mask, spacings, new_spacings, mask=True)

        t4 = time.time()
        x = torch.clamp(x, min=-1024)
        preproc_path = f"/scratch/project_465001979/ct_data/kidney/preprocess_files/{case_id}_preproc.npz"
        pre_proc = np.load(preproc_path)
        mask = torch.as_tensor(pre_proc["mask"], dtype=torch.bool)
        x[~mask] = -1024
        # x = remove_table_3d(x)
        t41 = time.time()
        # box_start_, box_end_ = self.air_cropper.compute_bounding_box(x)
        x = self.air_cropper.crop_pad(x, pre_proc["bbox"][0], pre_proc["bbox"][1]).as_tensor()

        if self.process_masks:
            kidney_mask = self.air_cropper.crop_pad(kidney_mask, pre_proc["bbox"][0], pre_proc["bbox"][1])
            kidney_mask = kidney_mask.permute(2, 0, 1)
            if self.kind == "2D":
                kidney_mask = kidney_mask[1:-1, :, :]
        t42 = time.time()
        if self.kind == "2D":
            # x shape: (H, W, D)
            H, W, D = x.shape
            C_out = 3
            D_out = D - 2  # after cropping depth

            # preallocate output tensor
            x_stack = torch.empty((C_out, H, W, D_out), dtype=x.dtype, device=x.device)

            # fill channels efficiently
            x_stack[0] = x[:, :, :-2]  # roll_back
            x_stack[1] = x[:, :, 1:-1]  # center
            x_stack[2] = x[:, :, 2:]  # roll_forward

            x = x_stack


        else:  # 3D patches
            x = x.unsqueeze(0)

        # transpose (C, D, H, W)
        x = x.permute(0, 3, 1, 2)

        x = normalize_scan_per_patch(x)
        t5 = time.time()
        if self.spheres:
            # keep only the kidney slices
            foreground = (kidney_mask.as_tensor() > 0).to(torch.bool)  # already boolean
            depth_foreground_mask = foreground.sum(dim=(1, 2)) > 0
            x = x[:, depth_foreground_mask, :, :]
            kidney_mask = kidney_mask[depth_foreground_mask, :, :]

            # calculate options for uniform sampling on the reduced volumes
            foreground = (kidney_mask.as_tensor() > 0).to(torch.bool)  # already boolean
            tumor_foreground = (kidney_mask.as_tensor() == 2).to(torch.bool)  # already boolean
            options = torch.nonzero(foreground, as_tuple=False)  # shape (M,3)
            tumor_options = torch.nonzero(tumor_foreground, as_tuple=False)

        patch_centers, patch_size = self.sample_patch_coordinates(x, sphere_options=options,
                                                                  tumor_options=tumor_options)
        kidney_mask = kidney_mask.unsqueeze(0)

        combined = torch.cat([x, kidney_mask], dim=0)

        combined_patches = extract_patches_3d(combined, patch_centers, patch_size)
        patches = combined_patches[:, :-1]
        mask_patches = combined_patches[:, -1:]

        t6 = time.time()
        if self.kind == "2D":
            patches = torch.squeeze(patches)
            threshold_dims = (1, 2, 3)
        else:
            threshold_dims = (1, 2, 3, 4)
        num_patches = patches.shape[0]
        indices = torch.arange(num_patches)

        keep_mask = (patches > 0.05).float().mean(dim=threshold_dims) > self.foreground_threshold
        patches = patches[keep_mask]
        mask_patches = torch.squeeze(mask_patches[keep_mask][:, 0])
        patch_class = ((2.5 > mask_patches) & (mask_patches > 1.5)).sum(axis=(1, 2))
        patch_class = patch_class > 0
        # try:
        patch_centers = patch_centers[keep_mask]
        # except:
        #    print("patch centers", patch_centers.shape)

        # take max 120 patches (GPU memory limitations)

        # patches = patches[:120]
        # patch_class = patch_class[:120]
        if self.augmentations is not None:
            x = self.augmentations(x)

        patch_centers = torch.Tensor(patch_centers)
        t7 = time.time()
        # print(
        #     f"load scan: {t1 - t0:.2f}, "
        #     f"load segmentation: {t3 - t2:.2f}, "
        #     f"process: {t5 - t4:.2f}"
        #     f"extract patches from volumes: {t6 - t5:.2f}, "
        #     f"apply augmentations: {t7 - t6:.2f}, "
        #     f"p1: {t41 - t4:.3f}, "
        #     f"p2: {t42 - t41:.3f}, "
        #     f"p3: {t5 - t42:.3f}, "
        # )
        return patches, y, patch_class, patch_centers, x, path
        # return patches, y, (
        # case_id, new_spacings, path, x, patch_class, num_patches), patch_centers

### FOR CIRCLES EXP
# if rng.random() < 0.5:
#
#     y = torch.Tensor([True])
#
#     kidney_mask[kidney_mask > 0] = 1
#     possible_locs = np.where(kidney_mask == 1)
#     options = list(zip(*possible_locs))
#     random_idx = rng.integers(len(options))
#     coords = options[random_idx]
#     x = add_sphere(x, coords, rng,simplified=False)
# else:
#     y = torch.Tensor([False])
# gaussian_noise = torch.FloatTensor(np.random.rand(*x.shape) * 20)
# x = np.array(gaussian_noise)
