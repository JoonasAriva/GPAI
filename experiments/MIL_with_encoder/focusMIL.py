import torch
import torch.distributions as dist
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet

from model_zoo import MyGroupNorm


###########################
# 1) Model Definition (Single-branch Version)
###########################
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=128, in_dim=512):
        super(VariationalEncoder, self).__init__()
        self.fc_initial = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 200),
        )
        self.mean = nn.Linear(200, latent_dim)
        self.logvar = nn.Linear(200, latent_dim)

    def forward(self, x):
        hidden = self.fc_initial(x)
        mu = self.mean(hidden)
        logvar = self.logvar(hidden)
        return mu, logvar


class AuxiliaryYFixed(nn.Module):
    def __init__(self, instance_latent_dim=128):
        super(AuxiliaryYFixed, self).__init__()
        self.fc_ins = nn.Linear(instance_latent_dim, 1)

    def forward(self, z_ins, bag_idx):
        """
        Single-branch scenario:
         - loc_ins: [N, 1] each instance score
         - M:       [B, 1] each bag score (max pooling)
        """
        loc_ins_logits = self.fc_ins(z_ins)  # [N,1]
        # print("logits:",loc_ins_logits)
        bags = bag_idx.unique()
        device = z_ins.device
        B = bags.shape[0]

        M = torch.zeros((B, 1), device=device)

        for i, bag_id in enumerate(bags):

            idxs_bag = (bag_idx == bag_id).nonzero(as_tuple=True)[0]

            if idxs_bag.numel() > 0:

                M[i, :] = loc_ins_logits[idxs_bag].max()

            else:
                M[i, :] = 0.0

        return M, loc_ins_logits


class FocusmilSingleBranch(nn.Module):
    """
    Simplified: only single-branch, without diff_loss or third_loss.
    """

    def __init__(self, instance_latent_dim=128):
        super(FocusmilSingleBranch, self).__init__()

        self.model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], norm_layer=MyGroupNorm)
        sd = resnet50(weights=ResNet50_Weights.DEFAULT).state_dict()

        self.model.load_state_dict(sd, strict=False)

        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.encoder = VariationalEncoder(latent_dim=instance_latent_dim, in_dim=self.num_features)
        self.aux_y = AuxiliaryYFixed(instance_latent_dim)

        # Main classification loss
        # You may use BCEWithLogitsLoss or BCELoss, depending on your need
        # self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, bag, bag_idx, training: bool = True, **kwargs):
        """
        Training forward:
          - Perform VAE encoding -> rsample() for each instance
          - Obtain instance-level scores (loc_ins) and bag-level score (M)
          - KL + main classification loss
        """

        # VAE encoder
        feats = self.model(bag)
        instance_mu, instance_logvar = self.encoder(feats)
        instance_std = (instance_logvar * 0.5).exp_()

        if training:
            qzx = dist.Normal(instance_mu, instance_std)
            z_ins = qzx.rsample()  # random sampling
        else:
            z_ins = instance_mu

        # Single-branch bag score
        bag_score, instance_scores = self.aux_y(z_ins, bag_idx)

        # KL
        KL_loss = 0.5 * (
                instance_mu.pow(2) + instance_std.pow(2)
                - 2 * torch.log(instance_std + 1e-8) - 1
        ).mean()
        Y_hat = torch.ge(torch.sigmoid(bag_score), 0.5).float()
        return {
            'scores': bag_score,
            'instance_scores': instance_scores,
            'KL_loss': KL_loss,
            'predictions': Y_hat
        }

    @torch.no_grad()
    def forward_no_sampling(self, bag, bag_idx):
        """
        Testing/inference forward:
         - No random sampling (directly use mu)
         - Obtain bag-level score M and instance-level score loc_ins
        """
        instance_mu, instance_logvar = self.encoder(bag)
        instance_std = (instance_logvar * 0.5).exp_()
        # no sampling
        z_ins = instance_mu
        M, loc_ins = self.aux_y(z_ins, bag_idx)

        KL_loss = 0.5 * (
                instance_mu.pow(2) + instance_std.pow(2)
                - 2 * torch.log(instance_std + 1e-8) - 1
        ).mean()

        return M, loc_ins, KL_loss

    @torch.no_grad()
    def predict_instance_score(self, bag, bag_idx):
        """
        Return instance-level scores [N]
        """
        M, loc_ins, KL_loss = self.forward_no_sampling(bag, bag_idx)
        return loc_ins.squeeze(-1)

    @torch.no_grad()
    def predict_bag_score(self, bag, bag_idx):
        """
        Return bag-level scores [B]
        """
        M, loc_ins, KL_loss = self.forward_no_sampling(bag, bag_idx)
        return M.squeeze(-1)
