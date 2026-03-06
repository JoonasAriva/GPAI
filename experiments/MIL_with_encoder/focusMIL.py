import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet, resnet18, ResNet18_Weights

from model_zoo import MyGroupNorm
from models3d import ResNetDepth2dPatches


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


class ClassificationHead(nn.Module):
    def __init__(self, instance_latent_dim=128, cysts=False):
        super(ClassificationHead, self).__init__()
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

    def __init__(self, instance_latent_dim=128):
        super(FocusMIL, self).__init__()

        self.backbone = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
        sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
        self.backbone.load_state_dict(sd, strict=False)
        self.num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.encoder = VariationalEncoder(latent_dim=instance_latent_dim, in_dim=self.num_features)
        self.cls = ClassificationHead(instance_latent_dim, cysts=False)

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


class FocusCompass(FocusMIL):

    def __init__(self):
        super().__init__()

        self.compass_head = nn.Linear(self.num_features, 3)
        self.projection_head = nn.Linear(self.num_features, 128)

    def forward(self, bag, bag_idx, training: bool = True, **kwargs):

        feats = self.backbone(bag)

        # FOCUSMIL PART

        # VAE encoder
        instance_mu, instance_logvar = self.encoder(feats)
        instance_std = (instance_logvar * 0.5).exp_()

        if training:
            qzx = dist.Normal(instance_mu, instance_std)
            z_ins = qzx.rsample()  # random sampling
        else:
            z_ins = instance_mu

        bag_score, cyst_bag_score, instance_scores = self.cls(z_ins, bag_idx)

        # KL
        KL_loss = 0.5 * (
                instance_mu.pow(2) + instance_std.pow(2)
                - 2 * torch.log(instance_std + 1e-8) - 1
        ).mean()
        Y_hat = torch.ge(torch.sigmoid(bag_score), 0.5).float()

        # PROJECTION PART
        proj_vectors = self.projection_head(feats)
        norm_vectors = F.normalize(proj_vectors, p=2, dim=1, eps=1e-12)

        # COMPASS PART

        compass_scores = self.compass_head(feats)

        return {
            'scores': bag_score,
            'instance_scores': instance_scores,
            'KL_loss': KL_loss,
            'predictions': Y_hat,
            'compass_scores': compass_scores,
            'proj_vectors': norm_vectors,
        }


class FocusCompassFixed(FocusMIL):

    def __init__(self):
        super().__init__()

        self.fixed_compass_model = ResNetDepth2dPatches()
        for param in self.fixed_compass_model.parameters():
            param.requires_grad = False

        pth = '/users/arivajoo/results/patches2D_depth/train/depth_patches2D/kidney_real/'
        resnet_pth = '2026-02-28/11-59-22/checkpoints/best_model.pth'  # resnet18
        sd = torch.load(pth + resnet_pth, map_location='cuda:0', weights_only=True)  # this resnet50
        new_sd = {key.replace("module.", ""): value for key, value in sd.items()}
        missing, unexpected = self.fixed_compass_model.load_state_dict(state_dict=new_sd, strict=False)
        if missing:
            print("Missing keys:")
            for k in missing:
                print(f"  - {k}")
        if unexpected:
            print("Unexpected keys:")
            for k in unexpected:
                print(f"  - {k}")
        print("loaded fixed compass model")

        self.projection_head = nn.Linear(self.num_features, 128)


    @torch.no_grad()
    def pred_compass(self, bag):
        return self.fixed_compass_model(bag)


    def forward(self, bag, bag_idx, training: bool = True, **kwargs):
        feats = self.backbone(bag)

        # FOCUSMIL PART

        # VAE encoder
        instance_mu, instance_logvar = self.encoder(feats)
        instance_std = (instance_logvar * 0.5).exp_()

        if training:
            qzx = dist.Normal(instance_mu, instance_std)
            z_ins = qzx.rsample()  # random sampling
        else:
            z_ins = instance_mu

        bag_score, cyst_bag_score, instance_scores = self.cls(z_ins, bag_idx)

        # KL
        KL_loss = 0.5 * (
                instance_mu.pow(2) + instance_std.pow(2)
                - 2 * torch.log(instance_std + 1e-8) - 1
        ).mean()
        Y_hat = torch.ge(torch.sigmoid(bag_score), 0.5).float()

        # PROJECTION PART
        proj_vectors = self.projection_head(feats)
        norm_vectors = F.normalize(proj_vectors, p=2, dim=1, eps=1e-12)

        # COMPASS PART

        compass_scores = self.fixed_compass_model(bag)

        return {
            'scores': bag_score,
            'instance_scores': instance_scores,
            'KL_loss': KL_loss,
            'predictions': Y_hat,
            'compass_scores': compass_scores,
            'proj_vectors': norm_vectors,
            'feature_vectors': feats,
        }


## LOSSES: FOCUSMIL (KL + CE), COMPASS, CONTRAST


"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
import torch
import torch.nn as nn


class ContastiveLoss(nn.Module):
    def __init__(self, compass_proximity_threshold=50, logit_confidence_threshold=1):
        super(ContastiveLoss, self).__init__()
        self.compass_proximity_threshold = compass_proximity_threshold  # in millimeters
        self.logit_confidence_threshold = logit_confidence_threshold
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        print("compass_proximity_threshold", compass_proximity_threshold)

    def calc_euclidian_distances_in_3d(self, matrix):
        # calculate manhattan distances between all coordinates
        # we go from N,3 --> N,N,3
        manhattan_dists = matrix.reshape(-1, 1, 3) - matrix.float()
        # go from N,N,3 --> N,N
        euclidian_dists = torch.norm(manhattan_dists, dim=2)
        return euclidian_dists

    def calc_positive_instance_pair_mask(self, compass_scores, bag_idxs, bag_labels, instance_scores):
        # To calculate positive pairs for contrastive loss we need to take account two variables
        # 1) position scores distances, 2) classes & instance scores

        # position scores
        euclidian_dists = self.calc_euclidian_distances_in_3d(compass_scores)
        #print("euclidian_dists", euclidian_dists)
        compass_positional_mask = euclidian_dists < self.compass_proximity_threshold
        #print("compass_positional_mask", compass_positional_mask)
        # classes & instance scores
        labels_per_instance = bag_labels[bag_idxs].cuda()
        # take all instances from negative bags and also negative instances from positive bags
        neg_instance_aware_mask = (labels_per_instance == 0) | (instance_scores < 0)
        neg_instance_aware_mask = neg_instance_aware_mask.unsqueeze(1) * neg_instance_aware_mask.unsqueeze(0)

        intra_patient_mask = bag_idxs.unsqueeze(1) == bag_idxs.unsqueeze(0)
        high_pos_mask = (instance_scores > self.logit_confidence_threshold) & (labels_per_instance == 1)
        high_pos_mask = high_pos_mask.unsqueeze(1) * high_pos_mask.unsqueeze(0)
        high_pos_inter_patient_mask = intra_patient_mask.cuda() * high_pos_mask

        # combine class and instance aware masks
        instance_mask = neg_instance_aware_mask | high_pos_inter_patient_mask
        # finally, combine with compass mask
        mask = instance_mask * compass_positional_mask
        mask = mask * (1 - torch.eye(mask.size(0), device=mask.device))  # remove diagonal

        return mask

    def forward(self, feature_vectors, bag_idxs, bag_labels, compass_scores, instance_scores):
        mask = self.calc_positive_instance_pair_mask(compass_scores, bag_idxs, bag_labels, instance_scores)
        #print("mask", mask)
        logits = self.cosine_sim(feature_vectors.unsqueeze(1), feature_vectors.unsqueeze(0))

        exp_logits = torch.exp(logits)
        exp_logits = exp_logits * (1 - torch.eye(exp_logits.size(0),
                                                 device=exp_logits.device))  # remove diagonal (self-contrast in not needed)
        denominator = torch.log(exp_logits.sum(1, keepdim=True)) + 1e-12  # calculate the denominator
        log_prob = logits - denominator  # log(a/b) = log(a) - log(b) and log(exp(a)) = a

        pos_count = mask.sum(1).clamp(min=1e-8)
        mean_log_prob_pos = (log_prob * mask).sum(1) / pos_count  # calculate log-likelihood over positive pairs
        loss = - mean_log_prob_pos.mean()

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, threshold=0.1, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, contrastive_method='simclr'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.threshold = threshold
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            if self.contrastive_method == 'gcl':
                mask = torch.eq(labels, labels.T).float().to(device)
            elif self.contrastive_method == 'pcl':
                mask = (torch.abs(
                    labels.T.repeat(batch_size, 1) - labels.repeat(1, batch_size)) < self.threshold).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

        # get compass scores
        # put three scan patches together, along with compass scores
        # N compass scores, N feature vectors
        # N x N compass distances, NxN cosine similarities
        # filter compass dist matrix to create a positive pair mask
        # also mask out self-contrast cases (diagonal)
