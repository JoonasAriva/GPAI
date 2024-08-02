from __future__ import print_function

import numpy as np
import torch

import torch.utils.data as data_utils
import torchio as tio

from omegaconf import OmegaConf, DictConfig
from data.kidney_dataloader import KidneyDataloader
from data.synth_dataloaders import SynthDataloader

from model_zoo import ResNetAttentionV3, ResNetSelfAttention, ResNetTransformerPosEnc, ResNetTransformerPosEmbed, \
    ResNetTransformer, ResNetGrouping


def prepare_dataloader(cfg: DictConfig):
    if cfg.data.dataloader == "kidney_real":
        transforms = tio.Compose(
            [tio.RandomFlip(axes=(0, 1)),
             tio.RandomAffine(scales=(1, 1.2), degrees=(0, 0, 10), translation=(50, 50, 0))])
        dataloader_params = {
            'only_every_nth_slice': cfg.data.take_every_nth_slice, 'as_rgb': True,
            'plane': 'axial', 'center_crop': cfg.data.crop_size, 'downsample': False,
            'roll_slices': cfg.data.roll_slices}
        train_dataset = KidneyDataloader(type="train",
                                         augmentations=None if not cfg.data.data_augmentations else transforms,
                                         **dataloader_params)
        test_dataset = KidneyDataloader(type="test", **dataloader_params)

        loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

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
        train_dataset = SynthDataloader(length=1000, premade=True,
                                        train=True)
        test_dataset = SynthDataloader(length=200, premade=True, train=False)

        loader_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
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
        model = ResNetTransformerPosEmbed(instnorm = cfg.model.inst_norm)
    elif cfg.model.name == 'posenc':
        model = ResNetTransformerPosEnc(instnorm = cfg.model.inst_norm)
    elif cfg.model.name == 'transformer':
        model = ResNetTransformer(nr_of_blocks=cfg.model.nr_of_blocks)
    elif cfg.model.name == 'resnetgroup':
        model = ResNetGrouping(instnorm = cfg.model.inst_norm)

    return model
