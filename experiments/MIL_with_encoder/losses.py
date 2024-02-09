import torch
import torch.nn as nn



class ExcludeLoss(nn.Module):
    def __init__(self):
        super(ExcludeLoss, self).__init__()

    def forward(self, attention):
        return torch.min(attention) ** 2


class IncludeLoss(nn.Module):
    def __init__(self):
        super(IncludeLoss, self).__init__()

    # TODO: if batch size is more than one, I have to change the losses
    def forward(self, attention, label):

        return (torch.max(attention) - label) ** 2


class AttentionLoss(nn.Module):

    def __init__(self, gamma=0.2):
        super(AttentionLoss, self).__init__()

        self.gamma = gamma
        self.include_loss = IncludeLoss()
        self.exclude_loss = ExcludeLoss()

    def forward(self, attention, label):
        inc_loss = self.include_loss(attention, label)
        exc_loss = self.exclude_loss(attention)
        return self.gamma * (inc_loss + exc_loss), (inc_loss, exc_loss)