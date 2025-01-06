from __future__ import print_function

import random
import re

import numpy as np
import torch
import torch.utils.data as data_utils
import torchio as tio
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score

from data.kidney_dataloader import KidneyDataloader, AbdomenAtlasLoader
from data.synth_dataloaders import SynthDataloader
from model_zoo import ResNetAttentionV3, ResNetSelfAttention, ResNetTransformerPosEnc, ResNetTransformerPosEmbed, \
    ResNetTransformer, ResNetGrouping, SelfSelectionNet, TwoStageNet, TwoStageNetSimple, TwoStageNetMaskedAttention, \
    MultiHeadTwoStageNet, TwoStageNetTwoHeads, TransMIL, TwoStageNetTwoHeadsV2, ResNetDepth, TransDepth, CompassModel, \
    CompassModelV2, TwoStageCompass, TwoStageCompassV2, TwoStageCompassV3


def prepare_dataloader(cfg: DictConfig):
    if "kidney" in cfg.data.dataloader:
        transforms = tio.Compose(
            [tio.RandomFlip(axes=(0, 1)),
             tio.RandomAffine(scales=(1, 1.2), degrees=(0, 0, 10), translation=(50, 50, 0))])
        dataloader_params = {
            'only_every_nth_slice': cfg.data.take_every_nth_slice, 'as_rgb': True,
            'plane': 'axial', 'center_crop': cfg.data.crop_size, 'downsample': False,
            'roll_slices': cfg.data.roll_slices,
            'generate_spheres': True if cfg.data.dataloader == 'kidney_synth' else False, 'patchify': cfg.data.patchify,
            'no_lungs': cfg.data.no_lungs}
        train_dataset = KidneyDataloader(type="train",
                                         augmentations=None if not cfg.data.data_augmentations else transforms,
                                         **dataloader_params)
        test_dataset = KidneyDataloader(type="test", **dataloader_params)

        loader_kwargs = {'num_workers': 7, 'pin_memory': True} if torch.cuda.is_available() else {}

        # create sampler for training set
        class_sample_count = [train_dataset.controls, train_dataset.cases]
        weights = 1 / torch.Tensor(class_sample_count)
        samples_weight = np.array([weights[int(t[0])] for t in train_dataset.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                 replacement=False)

        train_loader = data_utils.DataLoader(train_dataset, batch_size=1, sampler=sampler, **loader_kwargs)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    elif cfg.data.dataloader == "synthetic":
        train_dataset = SynthDataloader(length=250, premade=True,
                                        train=True)
        test_dataset = SynthDataloader(length=100, premade=True, train=False)

        loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    elif cfg.data.dataloader == "abdomen_atlas":

        transforms = tio.Compose(
            [tio.RandomFlip(axes=(0, 1)),
             tio.RandomAffine(scales=(1, 1.2), degrees=(0, 0, 10), translation=(50, 50, 0))])

        dataloader_params = {
            'only_every_nth_slice': cfg.data.take_every_nth_slice, 'as_rgb': True,
            'plane': 'axial', 'center_crop': cfg.data.crop_size,
            'roll_slices': cfg.data.roll_slices, 'patchify': cfg.data.patchify}
        train_dataset = AbdomenAtlasLoader(type="train",
                                           augmentations=None if not cfg.data.data_augmentations else transforms,
                                           **dataloader_params)

        test_dataset = AbdomenAtlasLoader(type="test",
                                          augmentations=None,
                                          **dataloader_params)
        loader_kwargs = {'num_workers': 7, 'pin_memory': True} if torch.cuda.is_available() else {}

        train_loader = data_utils.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    else:
        print("Wrong type specified")
    return train_loader, test_loader


def pick_model(cfg: DictConfig):
    if cfg.model.name == 'resnet18V3':
        model = ResNetAttentionV3(neighbour_range=cfg.model.neighbour_range,
                                  num_attention_heads=cfg.model.num_heads, instnorm=True, resnet_type="18")
    elif cfg.model.name == 'resnet34V3':
        model = ResNetAttentionV3(neighbour_range=cfg.model.neighbour_range,
                                  num_attention_heads=cfg.model.num_heads, instnorm=True, resnet_type="34")
    elif cfg.model.name == 'resnetselfattention':
        model = ResNetSelfAttention(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'posembed':
        model = ResNetTransformerPosEmbed(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'posenc':
        model = ResNetTransformerPosEnc(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'transformer':
        model = ResNetTransformer(nr_of_blocks=cfg.model.nr_of_blocks)
    elif cfg.model.name == 'resnetgroup':
        model = ResNetGrouping(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'selfselectionnet':
        model = SelfSelectionNet(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'twostagenet':
        model = TwoStageNet(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'twostagenet_simple':
        model = TwoStageNetSimple(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'twostagenet_masked':
        model = TwoStageNetMaskedAttention(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'twostagenet_multi':
        model = MultiHeadTwoStageNet(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'twostagenet_two_heads':
        model = TwoStageNetTwoHeads(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'twostagenet_two_headsV2':
        model = TwoStageNetTwoHeadsV2(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'transmil':
        model = TransMIL()
    elif cfg.model.name == 'resnetdepth':
        model = ResNetDepth(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'transdepth':
        model = TransDepth(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'compass_model':
        model = CompassModel(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'compass_modelV2':
        model = CompassModelV2(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'compass_modelV3':
        model = TwoStageCompass(instnorm=cfg.model.inst_norm)
    elif cfg.model.name == 'twostagecompassV2':
        model = TwoStageCompassV2(instnorm=cfg.model.inst_norm, fixed_compass=cfg.model.fixed_compass)
    elif cfg.model.name == 'twostagecompassV3':
        model = TwoStageCompassV3(instnorm=cfg.model.inst_norm, fixed_compass=cfg.model.fixed_compass)
    return model


# def attention_accuracy(attention, df):
#     df.loc[(df["kidney"] > 0) | (df["tumor"] > 0) | (df["cyst"] > 0), "important_all"] = 1
#     df["important_all"] = df["important_all"].fillna(0)
#
#     df.loc[df["tumor"] > 0, "important_tumor"] = 1
#     df["important_tumor"] = df["important_tumor"].fillna(0)
#
#     # print("attention min, max and nan values: ",np.min(attention),np.max(attention),np.isnan(attention).any())
#     treshold = threshold_otsu(attention)
#     # otsu threshold
#     # binarize attention
#     attention[attention > treshold] = 1
#     attention[attention != 1] = 0
#
#     matching_all = np.sum((attention == True) & (np.array(df["important_all"] == True)))
#     matching_tumor = np.sum((attention == True) & (np.array(df["important_tumor"] == True)))
#     all_relevant_attention = np.sum(attention)
#     all_relevant_slices = np.sum(df["important_all"])
#
#     accuracy_all = matching_all / all_relevant_attention
#     recall_all = matching_all / all_relevant_slices
#     accuracy_tumor = matching_tumor / all_relevant_attention
#
#     return round(accuracy_all, 2), round(accuracy_tumor, 2), round(recall_all, 2)


def find_case_id(path, start_string, end_string):
    match = re.search(start_string + '(.*)' + end_string, path[0]).group(1)
    match = match.strip("_0000")
    return match


def center_crop_dataframe(df, crop_size):
    if len(df) > crop_size:
        # simulate center cropper
        midpoint = int(len(df) / 2)
        df = df[int(midpoint - crop_size / 2):int(midpoint + crop_size / 2)].copy()

        df.reset_index(inplace=True)
    return df


def prepare_statistics_dataframe(df, case_id, crop_size, nth_slice, roll_slices):
    scan_df = df.loc[df["file_name"] == case_id]

    scan_df = scan_df.copy()[::nth_slice]
    # print("len after nth slice sampling: ", len(scan_df))
    cropped_scan_df = center_crop_dataframe(scan_df, crop_size)
    if roll_slices:
        cropped_scan_df = cropped_scan_df[1:-1].copy()
    return cropped_scan_df


def evaluate_attention(attention, df, case_id, crop_size, nth_slice, bag_label, roll_slices):
    cropped_scan_df = prepare_statistics_dataframe(df, case_id, crop_size, nth_slice, roll_slices)
    ap_all, ap_tumor = calculate_ap(attention, cropped_scan_df, bag_label)

    return ap_all, ap_tumor


def calculate_ap(attention, df, bag_label):
    df.loc[(df["kidney"] > 0) | (df["tumor"] > 0) | (df["cyst"] > 0), "important_all"] = 1
    df["important_all"] = df["important_all"].fillna(0)

    df.loc[df["tumor"] > 0, "important_tumor"] = 1
    df["important_tumor"] = df["important_tumor"].fillna(0)

    attention = attention.detach().cpu().numpy()

    ap_all = average_precision_score(df["important_all"], attention)

    if bag_label:
        ap_tumor = average_precision_score(df["important_tumor"], attention)
    else:
        ap_tumor = 0
    return round(ap_all, 2), round(ap_tumor, 2)


def get_percentage_of_scans_from_dataframe(df, percentage):
    grouped_stats = df.groupby("file_name").sum()
    no_tumor = grouped_stats[grouped_stats["tumor"] == 0].reset_index().file_name.unique()
    tumor = grouped_stats[grouped_stats["tumor"] > 0].reset_index().file_name.unique()
    random.shuffle(tumor)
    random.shuffle(no_tumor)
    tumor = tumor[:int(percentage * len(tumor))]
    no_tumor = no_tumor[:int(percentage * len(no_tumor))]

    return np.concatenate([tumor, no_tumor])
