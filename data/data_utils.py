import glob
import os
from typing import List

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import raster_geometry as rg
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage import morphology

matplotlib.rcParams['animation.embed_limit'] = 2 ** 128


def add_random_sphere(image):
    size = image.shape[1]
    height = image.shape[2]

    z, y, x = np.random.uniform(0.1, 0.9), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)

    radius = np.random.randint(20, 40)

    sphere_mask = rg.sphere((size, size, height), radius, (y, x, z))
    gaussian_noise = torch.DoubleTensor(np.random.randn(size, size, height) * 0.2 + 2.5)

    image[sphere_mask] = gaussian_noise[sphere_mask]

    return image


def get_dataset_paths(datasets: List[str], dataset_type: str, percentage: float = 1, return_predictions: bool = False):
    ''' currently possible dataset types: TUH_kidney, totalsegmentor
    dataset type is either train or test'''

    TUH_data_path = '/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/'
    total_segmentor_data_path = '/gpfs/space/projects/BetterMedicine/joonas/totalsegmentor/'
    kits23_data_path = '/gpfs/space/projects/BetterMedicine/joonas/kits23/'

    all_controls = []
    all_tumors = []
    all_labels = []
    TUH_study_length = 1
    print("-------------------------")
    print("DATASET TYPE (TRAIN/TEST):", dataset_type)
    print("using the combination of ", len(datasets), "datasets")

    if "TUH_kidney" in datasets:
        data_path = os.path.join(TUH_data_path, dataset_type)

        control_path = data_path + '/controls/*nii.gz'
        tumor_path = data_path + '/cases/*nii.gz'
        print("PATHS TUH:")
        print(control_path)
        print(tumor_path)
        control = glob.glob(control_path)
        tumor = glob.glob(tumor_path)
        if dataset_type == "train":
            extra_control_path = data_path + '/from_test_set_cases/*nii.gz'
            extra_tumor_path = data_path + '/from_test_set_controls/*nii.gz'

            extra_controls = glob.glob(extra_control_path)
            extra_tumors = glob.glob(extra_tumor_path)

            control.extend(extra_controls)
            tumor.extend(extra_tumors)

        all_controls.extend(control)
        all_tumors.extend(tumor)

        TUH_study_length = len(control) + len(tumor)
        if return_predictions:
            if dataset_type == "train":
                all_labels.extend(glob.glob(data_path + '/labels/cases/*.nii.gz'))
                all_labels.extend(glob.glob(data_path + '/labels/cases_from_test/*.nii.gz'))
                all_labels.extend(glob.glob(data_path + '/labels/controls/*.nii.gz'))
                all_labels.extend(glob.glob(data_path + '/labels/controls_from_test/*.nii.gz'))
            else:
                all_labels.extend(glob.glob(data_path + '/case_labels/*.nii.gz'))
                all_labels.extend(glob.glob(data_path + '/control_labels/*.nii.gz'))

    if "totalsegmentor" in datasets:
        data_path = os.path.join(total_segmentor_data_path + 'model_ready_dataset', dataset_type)

        control_path = data_path + '/control/*nii.gz'
        tumor_path = data_path + '/tumor/*nii.gz'
        cyst_path = data_path + '/cyst/*nii.gz'

        print("PATHS totalsegmentor:")
        print(control_path)
        print(tumor_path)
        print(cyst_path)

        control = glob.glob(control_path)
        tumor = glob.glob(tumor_path)
        cyst = glob.glob(cyst_path)

        all_controls.extend(control)
        all_controls.extend(cyst)  # also use cysts as controls, might alter in the future
        all_tumors.extend(tumor)

        if return_predictions:
            all_labels.extend(glob.glob(total_segmentor_data_path + 'abdomen_preds/*.nii.gz'))

    if "kits23" in datasets:
        data_path = os.path.join(kits23_data_path + 'model_ready_dataset', dataset_type)
        data_path = data_path + "/*nii.gz"
        print("PATH kits23:")
        print(data_path)
        # kits only has tumor cases
        tumor = glob.glob(data_path)
        # take only fraction of kits to keep dataset class balance
        tumor = tumor[:int(len(tumor) * 0.43)]
        all_tumors.extend(tumor)

        if return_predictions:
            print(kits23_data_path + 'dataset/*/segmentation.nii.gz')
            all_labels.extend(glob.glob(kits23_data_path + 'dataset/*/segmentation.nii.gz'))

    if percentage < 1:
        all_controls = all_controls[:int(len(all_controls) * percentage)]
        all_tumors = all_tumors[:int(len(all_tumors) * percentage)]

    if return_predictions:
        print("segmentation paths length: ", len(all_labels))
        return all_controls, all_tumors, TUH_study_length, all_labels
    else:
        return all_controls, all_tumors, TUH_study_length


def set_orientation(x, path, plane):
    # PLANES: set it into default plane (axial)
    # if transformations are needed we start from this position

    if "kits" in path:  # kits is in another orientation
        x = np.transpose(x, (1, 2, 0))
    if "parnu" in path:
        print("parnu in path")
        x = np.flip(x, axis=1)

        if len(x.shape) == 3:
            x = np.transpose(x, (1, 0, 2))
        elif len(x.shape) == 4:
            x = np.transpose(x, (1, 0, 2, 3))
    else:
        x = np.flip(x, axis=1)
        if len(x.shape) == 3:
            x = np.transpose(x, (1, 0, 2))
        elif len(x.shape) == 4:
            x = np.transpose(x, (1, 0, 2, 3))
    # this should give the most common axial representation: (patient on their back)
    if plane == "axial":
        pass
        # originally already in axial format
    elif plane == "coronal":
        x = np.transpose(x, (2, 1, 0))
    elif plane == "sagital":
        x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError('plane is not correctly specified')

    return x


def set_orientation_nib(x):
    orig_ornt = nib.io_orientation(x.affine)
    targ_ornt = nib.orientations.axcodes2ornt("RAS")
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    x = x.as_reoriented(transform)
    return x


def downsample_scan(x, scale_factor=0.5):
    x = torch.from_numpy(x.copy())
    x = F.interpolate(torch.unsqueeze(torch.unsqueeze(x, 0), 0), scale_factor=scale_factor,
                      mode='trilinear', align_corners=False)
    x = np.array(torch.squeeze(x))
    return x


def normalize_scan(x, single_channel=False, model_type="2D", remove_bones=False):
    if single_channel:
        xy_dims = (0, 1)
    else:
        xy_dims = (1, 2)

    if remove_bones:
        x[np.where(x > 300)] = -150
    clipped_x = np.clip(x, -150, 250)  # soft tissue window

    if model_type == "2D":
        norm_x = (clipped_x - np.expand_dims(np.mean(clipped_x, axis=xy_dims), xy_dims)) / (
                np.expand_dims(np.std(clipped_x, axis=xy_dims), xy_dims) + 1)  # mean 0, std 1 norm
        norm_x = np.squeeze(norm_x)
        return norm_x
    else:

        norm_x = (clipped_x - np.mean(clipped_x)) / (np.std(clipped_x) + 1)  # mean 0, std 1 norm
        # norm_x = np.squeeze(norm_x)
        return norm_x


def normalize_scan_new(x):
    clipped_x = np.clip(x, -150, 250)  # soft tissue window
    norm_x = (clipped_x - np.min(clipped_x)) / (np.max(clipped_x) - np.min(clipped_x))

    return norm_x


def normalize_scan_per_patch(x):
    clipped_x = np.clip(x, -150, 250)  # soft tissue window
    mins = np.min(clipped_x, axis=(1, 2, 3), keepdims=True)
    maxs = np.max(clipped_x, axis=(1, 2, 3), keepdims=True)
    norm_x = (clipped_x - mins) / (maxs - mins)

    return norm_x


def remove_empty_tiles(data):
    # 3, 128, 128, 1520
    # find minimum values per patch
    min_per_row = torch.min(data, dim=1)[0]  # Shape becomes [100, 512]
    min_per_channel = torch.min(min_per_row, dim=1)[0]
    min_values = torch.min(min_per_channel, dim=0)[0]

    mask = data < min_values + 0.05  # filter out pixels with small values
    small_vals = torch.sum(mask, dim=(0, 1, 2))
    all_vals = data.shape[:3].numel()
    relative_percentage_of_small_vals = small_vals / all_vals
    filter_mask = relative_percentage_of_small_vals > 0.4

    filtered_data = data[:, :, :, ~filter_mask]
    return filtered_data


# def custom_highlight_function(x, segmentation):
#
#     if self.highlight_experiment:
#         x[np.where((segmentation > 0) & (segmentation < 3))] *= 3
#     if self.dimming_experiment:
#         x = np.clip(x, np.percentile(x, q=0), np.percentile(x, q=99.5))
#         x[np.where(segmentation == 0)] *= 0.2
#
#     return x


def get_kidney_datasets(type: str, no_lungs: bool = False, TUH_only: bool = False):
    # type = train, test
    base_path = '/scratch/project_465001979/ct_data/'  # lumi

    tuh_train_data_path = base_path + 'kidney/tuh_train/'
    tuh_test_data_path = base_path + 'kidney/tuh_test/'
    tuh_extra_data_path = base_path + 'kidney/tuh_extra/'
    other_datasets_path = base_path + 'kidney/data'

    all_controls = []
    all_tumors = []

    for data_path in [tuh_train_data_path, tuh_test_data_path, tuh_extra_data_path]:
        control_path = data_path + 'controls/images/' + type + '/*nii.gz'
        tumor_path = data_path + 'cases/images/' + type + '/*nii.gz'

        control = glob.glob(control_path)
        tumor = glob.glob(tumor_path)

        all_controls.extend(control)
        all_tumors.extend(tumor)
    if TUH_only:
        return all_controls, all_tumors

    else:
        # # kits dataset + kirc dataset
        tumor = glob.glob(other_datasets_path + '/imagesTr/' + type + '/*nii.gz')
        all_tumors.extend(tumor)

        # parnu_data_path = base_path + 'kidney/parnu/'
        # control_path = parnu_data_path + 'controls/images/' + type + '/*nii.gz'
        # tumor_path = parnu_data_path + 'cases/images/' + type + '/*nii.gz'
        # control = glob.glob(control_path)
        # tumor = glob.glob(tumor_path)
        # all_controls.extend(control)
        # all_tumors.extend(tumor)

        return all_controls, all_tumors


def get_TUH_control(type: str):
    base_path = '/scratch/project_465001979/ct_data/'  # lumi

    tuh_train_data_path = base_path + 'kidney/tuh_train/'
    tuh_test_data_path = base_path + 'kidney/tuh_test/'

    all_controls = []

    for data_path in [tuh_train_data_path, tuh_test_data_path]:
        control_path = data_path + 'controls/images/' + type + '/*nii.gz'

        control = glob.glob(control_path)

        all_controls.extend(control)

    return all_controls


def get_source_label(path):
    if "tuh_test" in path or "tuh_train" in path:
        data_class = "TUH DATA"
    elif "parnu" in path:
        data_class = "PÄRNU"
    elif "kits" in path:
        data_class = "KITS"
    elif "TCGA" in path:
        data_class = "KIRC"
    return data_class


def map_source_label(source_label):
    classes = ["TUH DATA", "PÄRNU", "KITS", "KIRC"]
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    int_label = torch.Tensor([label_to_index[source_label]])
    return int_label


def get_pasted_dateset():
    base_path = '/project/project_465001979/ct_data/'
    tuh_train_data_path = base_path + 'kidney/tuh_train/'
    tuh_test_data_path = base_path + 'kidney/tuh_test/'
    pasted_tumors_path = '/scratch/project_465001979/ct_data/pasted_tumors_small/*.nii.gz'

    all_controls = []

    for data_path in [tuh_train_data_path, tuh_test_data_path]:
        control_path = data_path + 'controls/images/train/*nii.gz'

        control = glob.glob(control_path)
        all_controls.extend(control)

    all_tumors = glob.glob(pasted_tumors_path)

    return all_controls, all_tumors


def get_pasted_small_dateset():
    base_path = '/project/project_465001979/ct_data/'
    tuh_train_data_path = base_path + 'kidney/tuh_train/'
    tuh_test_data_path = base_path + 'kidney/tuh_test/'
    pasted_tumors_path = '/scratch/project_465001979/ct_data/pasted_tumors_small/*.nii.gz'
    pasted_tumors_path = '/scratch/project_465001979/ct_data/pasted_tumors_small/tuh_control_2.25.196698621768415934049516370456628718614*'
    all_controls = []

    for data_path in [tuh_train_data_path, tuh_test_data_path]:
        control_path = data_path + 'controls/images/train/*nii.gz'

        control = glob.glob(control_path)
        all_controls.extend(control)

    all_tumors = glob.glob(pasted_tumors_path)

    filter_list = []
    for path in all_tumors:

        id = path.replace("_0000.nii.gz", "").split("_")[-2]
        if id not in filter_list:
            filter_list.append(id)

    filtered_controls = [s for s in all_controls if any(sub in s for sub in filter_list)]

    return filtered_controls, all_tumors


import time


def remove_table_3d(img):
    start = time.time()
    thresh = -800  # in HU units, it should filter out air :)
    binary = img >= thresh

    # for working with 3d components, add binary:true layer to both end to close off any cavities
    true_layer = torch.unsqueeze(torch.ones_like(img[:, :, 0], dtype=torch.bool), dim=2)
    binary = torch.concat((true_layer, binary, true_layer), dim=2)

    # Fill the largest region completely (the body of the patient)
    filled_region = ndi.binary_fill_holes(binary)

    keep_mask = filled_region[:, :, 1:-1]

    n = 3
    structure = torch.concat((torch.zeros(3, 3, 1), torch.ones(3, 3, 1), torch.zeros(3, 3, 1)), dim=2)

    eroded_binary = ndi.binary_erosion(keep_mask, iterations=n, structure=structure)
    dilated_mask = ndi.binary_dilation(eroded_binary, iterations=n, structure=structure)

    # remove small objects not connected to the main body. 40000 pixels per slice seems to do the work for 512x512 resolution
    # changed to 20 000 due to resizing/ spacing normalization
    min_size = 40000 * img.shape[2] * (img.shape[0] / 512)
    keep_mask = morphology.remove_small_objects(dilated_mask, min_size=min_size)
    # keep_mask = np.expand_dims(keep_mask, 0)

    maskedimg = img.detach().clone()
    # specific fill value does not matter that much if you apply ct windowing afterwards
    maskedimg[~keep_mask] = -1024  # put regular min negative value back
    end = time.time()
    # print("time: ",end-start)
    return maskedimg


class CompassFilter:
    def __init__(self, dataframe_path_train: str, dataframe_path_test: str):
        super().__init__()

        self.train = pd.read_csv(dataframe_path_train)
        self.test = pd.read_csv(dataframe_path_test)

        gb = self.train.groupby(['case_id'])
        grouped_train = gb.agg({'weights': [np.min, np.max]})

        self.cutoff_min = grouped_train["weights"]["amin"].quantile(q=0.8)
        self.cutoff_max = grouped_train["weights"]["amax"].quantile(q=0.2)
        print("Compass cutoffs: ", self.cutoff_min, self.cutoff_max)

    def compass_filter_indexes(self, case_id: str, train: bool = True):
        if train:
            scan = self.train.loc[self.train["case_id"] == case_id]
        else:
            scan = self.test.loc[self.test["case_id"] == case_id]

        filtered_scan = scan.loc[(scan["weights"] > self.cutoff_min) & (scan["weights"] < self.cutoff_max)]
        index_last = filtered_scan.iloc[-1]["index"]
        index_first = filtered_scan.iloc[0]["index"]
        # better_filtered_scan = scan.loc[(scan["index"] > index_first) & (scan["index"] < index_last)]
        return index_first, index_last
