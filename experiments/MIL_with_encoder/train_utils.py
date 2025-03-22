from __future__ import print_function

import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torchio as tio
from omegaconf import DictConfig
from torch.utils.data.distributed import DistributedSampler

from compass_trainer import TrainerCompass
from compass_two_stage_adversary import TrainerCompassTwoStageAdv
from compass_two_stage_trainer import TwoStageCompassLoss, TrainerCompassTwoStage
from data.DistriputedDataCustomSampler import DistributedSamplerWrapper
from data.kidney_dataloader import KidneyDataloader, AbdomenAtlasLoader
from data.synth_dataloaders import SynthDataloader
from depth_trainer import TrainerDepth, DepthLossV2, CompassLoss
from depth_trainer3D import Trainer3DDepth, DepthLoss3D
from model_zoo import ResNetAttentionV3, ResNetSelfAttention, ResNetTransformerPosEnc, ResNetTransformerPosEmbed, \
    ResNetTransformer, ResNetGrouping, SelfSelectionNet, TwoStageNet, TwoStageNetSimple, TwoStageNetMaskedAttention, \
    MultiHeadTwoStageNet, TwoStageNetTwoHeads, TransMIL, TwoStageNetTwoHeadsV2, ResNetDepth, TransDepth, CompassModel, \
    CompassModelV2, TwoStageCompass, TwoStageCompassV2, TwoStageCompassV3, TwoStageCompassV4, TwoStageCompassV5, \
    TwoStageCompassV6
from swin_models import SWINCompass, SWINClassifier
from trainer import Trainer
from trainer_reg import TrainerReg, RegularizedAttentionLoss


def prepare_dataloader(cfg: DictConfig):
    if "kidney" in cfg.data.dataloader:
        if "pasted" in cfg.data.dataloader:
            pasted_experiment = True
        else:
            pasted_experiment = False
        transforms = tio.Compose(
            [tio.RandomFlip(axes=(0, 1)),
             tio.RandomAffine(scales=(1, 1.2), degrees=(0, 0, 10), translation=(50, 50, 0))])
        dataloader_params = {
            'only_every_nth_slice': cfg.data.take_every_nth_slice, 'as_rgb': cfg.data.as_rgb,
            'plane': 'axial', 'center_crop': cfg.data.crop_size, 'downsample': False,
            'roll_slices': cfg.data.roll_slices, 'model_type': cfg.model.model_type,
            'generate_spheres': True if cfg.data.dataloader == 'kidney_synth' else False, 'patchify': cfg.data.patchify,
            'no_lungs': cfg.data.no_lungs, "pasted_experiment": pasted_experiment}
        train_dataset = KidneyDataloader(type="train",
                                         augmentations=None if not cfg.data.data_augmentations else transforms,
                                         **dataloader_params, random_experiment=cfg.data.random_experiment)
        test_dataset = KidneyDataloader(type="test", **dataloader_params)

        loader_kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}

        # create sampler for training set
        class_sample_count = [train_dataset.controls, train_dataset.cases]
        weights = 1 / torch.Tensor(class_sample_count)
        samples_weight = np.array([weights[int(t[0])] for t in train_dataset.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                 replacement=False)

        if cfg.training.multi_gpu == True:
            sampler = DistributedSamplerWrapper(sampler=sampler, num_replicas=int(torch.cuda.device_count()),
                                                rank=int(os.environ["LOCAL_RANK"]), shuffle=True)
            # sampler_train = DistributedSampler(train_dataset, num_replicas=2, rank=int(os.environ["LOCAL_RANK"]),
            #                                   shuffle=True)
            sampler_test = DistributedSampler(test_dataset, num_replicas=int(torch.cuda.device_count()),
                                              rank=int(os.environ["LOCAL_RANK"]),
                                              shuffle=True)
        train_loader = data_utils.DataLoader(train_dataset, batch_size=cfg.data.batch_size, sampler=sampler,
                                             **loader_kwargs)
        test_loader = data_utils.DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False,
                                            sampler=sampler_test if cfg.training.multi_gpu == True else None,
                                            **loader_kwargs)

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
    elif cfg.model.name == 'twostagecompassV4':
        model = TwoStageCompassV4(instnorm=cfg.model.inst_norm, fixed_compass=cfg.model.fixed_compass)
    elif cfg.model.name == 'twostagecompassV5':
        model = TwoStageCompassV5(instnorm=cfg.model.inst_norm, fixed_compass=cfg.model.fixed_compass,
                                  ghostnorm=cfg.model.ghostnorm, range_0=-4,
                                  range_1=4)
    elif cfg.model.name == 'twostagecompassV6':
        model = TwoStageCompassV6(instnorm=cfg.model.inst_norm, fixed_compass=cfg.model.fixed_compass)
    elif cfg.model.name == 'swincompassV1':
        model = SWINCompass()
    elif cfg.model.name == 'swinV1':
        model = SWINClassifier()
    return model


def pick_trainer(cfg, optimizer, scheduler, steps_in_epoch, adv_optimizer=None):
    if cfg.experiment == "depth":
        loss_function = DepthLossV2(step=0.05).cuda()  # was 0.01 # then 0.1, but it might be too sparse
        trainer = TrainerDepth(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function,
                               cfg=cfg,
                               steps_in_epoch=steps_in_epoch)
    elif cfg.experiment == "compass":
        loss_function = CompassLoss(step=0.01).cuda()
        trainer = TrainerCompass(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                                 steps_in_epoch=steps_in_epoch)
    elif cfg.experiment == "compass_twostage":
        loss_function = TwoStageCompassLoss(step=0.01, fixed_compass=cfg.model.fixed_compass).cuda()
        trainer = TrainerCompassTwoStage(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                                         steps_in_epoch=steps_in_epoch,
                                         progressive_sigmoid_scaling=cfg.model.progressive_sigmoid_scaling)
    elif cfg.experiment == "swin":
        loss_function = DepthLoss3D(step=0.5).cuda()
        trainer = Trainer3DDepth(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                                 steps_in_epoch=steps_in_epoch)
    elif cfg.experiment == "attention_reg":
        loss_function = RegularizedAttentionLoss().cuda()
        trainer = TrainerReg(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                             steps_in_epoch=steps_in_epoch)
    elif cfg.experiment == "compass_twostage_adv":
        loss_function = TwoStageCompassLoss(step=0.01, fixed_compass=cfg.model.fixed_compass).cuda()
        trainer = TrainerCompassTwoStageAdv(optimizer_main=optimizer, optimizer_adv=adv_optimizer, scheduler=scheduler,
                                            loss_function=loss_function,
                                            cfg=cfg,
                                            steps_in_epoch=steps_in_epoch,
                                            progressive_sigmoid_scaling=cfg.model.progressive_sigmoid_scaling)
    else:
        loss_function = torch.nn.BCEWithLogitsLoss().cuda()
        trainer = Trainer(optimizer=optimizer, scheduler=scheduler, loss_function=loss_function, cfg=cfg,
                          steps_in_epoch=steps_in_epoch)
    return trainer


def prepare_optimizer(cfg, model):
    if 'compass_twostage' in cfg.experiment:

        boundary_name = ['depth_range']
        adversary_name = ['module.adversary_classifier.weight', 'module.ladversary_classifier.bias']

        boundary_params = [param for name, param in model.named_parameters() if name in boundary_name]

        base_params = [param for name, param in model.named_parameters() if
                       name not in boundary_name and name not in adversary_name]
        adversary_params = [param for name, param in model.named_parameters() if name in adversary_name]

        params = [
            {"params": [*base_params], "lr": cfg.training.learning_rate, 'weight_decay': cfg.training.weight_decay},
            # First group
            {"params": boundary_params, "lr": cfg.training.learning_rate * 1000, 'weight_decay': 0}
            # Second group, no decay for boundary parameters
        ]

        optimizer = optim.Adam(params, lr=cfg.training.learning_rate, betas=(0.9, 0.999))

        if len(adversary_params) > 0:
            optimizer_adv = optim.Adam(adversary_params, lr=cfg.training.learning_rate, betas=(0.9, 0.999))
            return optimizer, optimizer_adv
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                               weight_decay=cfg.training.weight_decay)

    return optimizer
