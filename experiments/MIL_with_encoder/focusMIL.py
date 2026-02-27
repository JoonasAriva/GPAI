import torch
import torch.distributions as dist
import torch.nn as nn
from torchvision.models import resnet, resnet18, ResNet18_Weights

from model_zoo import MyGroupNorm


###########################
# 1) Model Definition (Single-branch Version)
###########################
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=128, in_dim=512):
        super(VariationalEncoder, self).__init__()
        self.fc_initial = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        hidden = self.fc_initial(x)
        mu = self.mean(hidden)
        logvar = self.logvar(hidden)
        return mu, logvar


class CLShead(nn.Module):
    def __init__(self, instance_latent_dim=128, cysts=False):
        super(CLShead, self).__init__()
        self.cysts = cysts
        if cysts == True:
            out_dim = 2
        else:
            out_dim = 1

        self.fc_ins = nn.Linear(instance_latent_dim, out_dim)

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
        C = torch.zeros((B, 1), device=device)

        for i, bag_id in enumerate(bags):

            idxs_bag = (bag_idx == bag_id).nonzero(as_tuple=True)[0]

            if idxs_bag.numel() > 0:

                M[i, :] = loc_ins_logits[idxs_bag, 0].max()

                if self.cysts == True:
                    C[i, :] = loc_ins_logits[idxs_bag, 1].max()

            else:
                M[i, :] = 0.0
                if self.cysts == True:
                    C[i, :] = 0.0

        if self.cysts == False:
            C = None

        return M, C, loc_ins_logits[:, 0]


class FocusMIL(nn.Module):
    """
    Simplified: only single-branch, without diff_loss or third_loss.
    """

    def __init__(self, instance_latent_dim=128, cysts=False):
        super(FocusMIL, self).__init__()

        self.backbone = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
        sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
        self.backbone.load_state_dict(sd, strict=False)
        self.cysts = cysts
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.encoder = VariationalEncoder(latent_dim=instance_latent_dim, in_dim=self.num_features)
        self.cls = CLShead(instance_latent_dim, cysts=cysts)
        # self.cls_cyst = CLShead(instance_latent_dim)
        if self.cysts == True:
            print("using cysts too!")

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
        feats = self.backbone(bag)
        instance_mu, instance_logvar = self.encoder(feats)
        instance_std = (instance_logvar * 0.5).exp_()

        if training:
            qzx = dist.Normal(instance_mu, instance_std)
            z_ins = qzx.rsample()  # random sampling
        else:
            z_ins = instance_mu

        # Single-branch bag score

        bag_score, cyst_bag_score, instance_scores = self.cls(z_ins, bag_idx)

        # cyst_bag_score, _ = self.cls_cyst(z_ins, bag_idx)

        # KL
        KL_loss = 0.5 * (
                instance_mu.pow(2) + instance_std.pow(2)
                - 2 * torch.log(instance_std + 1e-8) - 1
        ).mean()
        Y_hat = torch.ge(torch.sigmoid(bag_score), 0.5).float()
        if self.cysts:
            cyst_Y_hat = torch.ge(torch.sigmoid(cyst_bag_score), 0.5).float()
        else:
            cyst_Y_hat = None
        return {
            'scores': bag_score,
            'instance_scores': instance_scores,
            'KL_loss': KL_loss,
            'predictions': Y_hat,
            'cyst_scores': cyst_bag_score,
            'cyst_Y_hat': cyst_Y_hat,
        }
