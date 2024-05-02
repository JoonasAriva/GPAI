import glob
import os
from typing import List

import matplotlib
import numpy as np
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
    if not "kits" in path:
        x = np.flip(x, axis=1)
        if len(x.shape) == 3:
            x = np.transpose(x, (1, 0, 2))
        elif len(x.shape) == 4:
            x = np.transpose(x, (1, 0, 2, 3))
    else:  # kits is in another orientation
        x = np.transpose(x, (1, 2, 0))
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


def downsample_scan(x, scale_factor=0.5):
    x = torch.from_numpy(x.copy())
    x = F.interpolate(torch.unsqueeze(torch.unsqueeze(x, 0), 0), scale_factor=scale_factor,
                      mode='trilinear', align_corners=False)
    x = np.array(torch.squeeze(x))
    return x


def normalize_scan(x, single_channel=False, model_type= "2D"):
    if single_channel:
        xy_dims = (0, 1)
    else:
        xy_dims = (1, 2)


    clipped_x = np.clip(x, -150, 250)  # soft tissue window

    if model_type == "2D":
        norm_x = (clipped_x - np.expand_dims(np.mean(clipped_x, axis=xy_dims), xy_dims)) / (
                np.expand_dims(np.std(clipped_x, axis=xy_dims), xy_dims) + 1)  # mean 0, std 1 norm
        norm_x = np.squeeze(norm_x)
        return norm_x
    else:
        norm_x = (clipped_x - np.mean(clipped_x)) / (np.std(clipped_x) + 1)  # mean 0, std 1 norm
        #norm_x = np.squeeze(norm_x)
        return norm_x


# def custom_highlight_function(x, segmentation):
#
#     if self.highlight_experiment:
#         x[np.where((segmentation > 0) & (segmentation < 3))] *= 3
#     if self.dimming_experiment:
#         x = np.clip(x, np.percentile(x, q=0), np.percentile(x, q=99.5))
#         x[np.where(segmentation == 0)] *= 0.2
#
#     return x


def get_kidney_datasets(type: str):
    # type = train, test
    base_path = '/gpfs/space/projects/BetterMedicine/joonas/'  # hpc
    #base_path = '/project/project_465001111/ct_data/'  # lumi
    tuh_train_data_path = base_path + 'kidney/tuh_train/'
    tuh_test_data_path = base_path + 'kidney/tuh_test/'
    # ts_data_path = '/gpfs/space/projects/BetterMedicine/joonas/kidney/total_segmentor'
    other_datasets_path = base_path + 'kidney/data'

    all_controls = []
    all_tumors = []
 #
    # tuh
    for data_path in [tuh_train_data_path, tuh_test_data_path]:
        control_path = data_path + 'controls/images/' + type + '/*nii.gz'
        tumor_path = data_path + 'cases/images/' + type + '/*nii.gz'


        control = glob.glob(control_path)
        tumor = glob.glob(tumor_path)


        all_controls.extend(control)
        all_tumors.extend(tumor)

    # total segmentor
    # control_path = ts_data_path + '/ts_controls/images/*nii.gz'
    # tumor_path = ts_data_path + '/ts_tumors/images/*nii.gz'
    # control = glob.glob(control_path)
    # tumor = glob.glob(tumor_path)
    # all_controls.extend(control)
    # all_tumors.extend(tumor)

    # # kits + kirc
    tumor = glob.glob(other_datasets_path + '/imagesTr/' + type + '/*nii.gz')
    all_tumors.extend(tumor)

    return all_controls, all_tumors


import time


def remove_table_3d(img):
    start = time.time()
    thresh = -200  # in HU units, it should filter out air :)
    binary = img > thresh

    # for working with 3d components, add binary:true layer to both end to close off any cavities
    true_layer = torch.unsqueeze(torch.ones_like(img[:, :, 0], dtype=torch.bool), dim=2)
    binary = torch.concat((true_layer, binary, true_layer), dim=2)

    # Fill the largest region completely (the body of the patient)
    filled_region = ndi.binary_fill_holes(binary)

    keep_mask = filled_region[:, :, 1:-1]

    # remoev small objects not connected to the main body. 40000 pixels per slice seems to do the work for 512x512 resolution
    min_size = 40000 * img.shape[2]
    keep_mask = morphology.remove_small_objects(keep_mask, min_size=min_size)
    keep_mask = np.expand_dims(keep_mask, 0)

    maskedimg = img.detach().clone()
    # specific fill value does not matter that much if you apply ct windowing afterwards
    maskedimg[~keep_mask] = thresh
    end = time.time()
    # print("time: ",end-start)
    return maskedimg
