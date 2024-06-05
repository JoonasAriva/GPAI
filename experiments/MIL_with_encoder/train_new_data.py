from __future__ import print_function

import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torchio as tio
import wandb
from omegaconf import OmegaConf, DictConfig
#from torchinfo import summary

sys.path.append('/gpfs/space/home/joonas97/GPAI/')
sys.path.append('/users/arivajoo/GPAI')

# from models import ResNet18AttentionV2, ResNetAttentionV3
from current_model import ResNetAttentionV3, ResNetSelfAttention2, ResNetTransformerPosEnc, ResNetTransformerPosEmbed, ResNetTransformer
from losses import AttentionLossV2
from trainer import Trainer
from data.kidney_dataloader import KidneyDataloader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# Training settings

dir_checkpoint = Path('./checkpoints/')

# read in statistics about the scans for validation metrics
# test_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_test_ROIS.csv")
# train_ROIS = pd.read_csv("/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS.csv")
# train_ROIS_extra = pd.read_csv(
#     "/gpfs/space/projects/BetterMedicine/joonas/tuh_kidney_study/axial_train_ROIS_from_test.csv")
# train_ROIS = pd.concat([train_ROIS, train_ROIS_extra])

np.seterr(divide='ignore', invalid='ignore')

os.environ["WANDB__SERVICE_WAIT"] = "300"

@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.project}, Work in {os.getcwd()}")

    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.training.seed)
        print('\nGPU is ON!')

    print('Load Train and Test Set')


    transforms = tio.Compose(
        [tio.RandomFlip(axes=(0, 1)), tio.RandomAffine(scales=(1, 1.2), degrees=(0, 0, 10), translation=(50, 50, 0))])
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
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=False)

    train_loader = data_utils.DataLoader(train_dataset, batch_size=1, sampler=sampler, **loader_kwargs)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    logging.info('Init Model')

    # if cfg.model.name == 'resnet18V2':
    #     model = ResNet18AttentionV2(neighbour_range=cfg.model.neighbour_range, num_attention_heads=cfg.model.num_heads)

    if cfg.model.name == 'resnet18V3':
        model = ResNetAttentionV3(neighbour_range=cfg.model.neighbour_range,
                                  num_attention_heads=cfg.model.num_heads, instnorm=True, resnet_type="18")
    elif cfg.model.name == 'resnet34V3':
        model = ResNetAttentionV3(neighbour_range=cfg.model.neighbour_range,
                                  num_attention_heads=cfg.model.num_heads, instnorm=True, resnet_type="34")
    elif cfg.model.name == 'resnetselfattention':
        model = ResNetSelfAttention2()
    elif cfg.model.name == 'posembed':
        model = ResNetTransformerPosEmbed()
    elif cfg.model.name == 'posenc':
        model = ResNetTransformerPosEnc()
    elif cfg.model.name == 'transformer':
        model = ResNetTransformer(nr_of_blocks=cfg.model.nr_of_blocks)

    # if you need to continue training
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(
            torch.load(os.path.join('/gpfs/space/home/joonas97/GPAI/experiments/MIL_with_encoder/', cfg.checkpoint)))
    if torch.cuda.is_available():
        model.cuda()

    #summary(model,input_size=(300,3,512,512))
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.999),
                           weight_decay=cfg.training.weight_decay)
    steps_in_epoch = 500
    number_of_epochs = 40
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, total_steps=number_of_epochs * steps_in_epoch,
                                                    pct_start=0.2, max_lr=cfg.training.learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss().cuda()
    attention_loss = AttentionLossV2(gamma=0.8).cuda()
    trainer = Trainer(optimizer=optimizer,scheduler=scheduler,loss_function=loss_function, attention_loss=attention_loss, check=cfg.check,
                      nth_slice=cfg.data.take_every_nth_slice, crop_size=cfg.data.crop_size, steps_in_epoch=steps_in_epoch,
                      calculate_attention_accuracy=False)

    if not cfg.check:
        experiment = wandb.init(project='MIL_encoder_24', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs,
                 learning_rate=cfg.training.learning_rate, model_name=cfg.model.name,
                 weight_decay=cfg.training.weight_decay))
    logging.info('Start Training')

    best_f1 = 0
    best_epoch = 0
    best_attention = 0
    not_improved_epochs = 0

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_results = dict()
        train_results = trainer.run_one_epoch(model, train_loader, epoch, train=True)
        epoch_results["learning_rate"] = scheduler.get_last_lr()[0]

        test_results = trainer.run_one_epoch(model, test_loader, epoch, train=False)

        train_results = {k + '_train': v for k, v in train_results.items()}
        test_results = {k + '_test': v for k, v in test_results.items()}

        test_f1 = test_results["f1_score_test"]
        if trainer.calculate_attention_accuracy:
            test_attention = test_results["attention_accuracy_test"]
        else:
            test_attention = 0

        epoch_results.update(train_results)
        epoch_results.update(test_results)

        if cfg.check:
           logging.info("Model check completed")
           return

        if test_f1 > best_f1:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f"Best new model at epoch {epoch} (highest f1 test score)!")

            best_f1 = test_f1
            best_epoch = epoch
            not_improved_epochs = 0

        elif test_attention > best_attention:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_attention_model.pth'))
            logging.info(f"Best new attention at epoch {epoch}!")

            best_attention = test_attention
            best_epoch = epoch
            not_improved_epochs = 0
        else:
            if not_improved_epochs > 20:
                logging.info("Model has not improved for the last 20 epochs, stopping training")
                break
            not_improved_epochs += 1
        experiment.log(epoch_results)
        torch.save(model.state_dict(), str(dir_checkpoint / 'current_model.pth'))



    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    logging.info(f"Training completed, best_metric (f1-test): {best_f1:.4f} at epoch: {best_epoch}")


if __name__ == "__main__":
    main()
