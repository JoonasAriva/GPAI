import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights


class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=num_channels, num_channels=num_channels,
                                 eps=1e-5, affine=True)

    def forward(self, x):
        x = self.norm(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, L, D, K):
        super(AttentionHead, self).__init__()
        self.head = nn.Sequential(
            # nn.Dropout(0.25),
            nn.Linear(L, D),
            nn.Tanh(),  # was Relu
            # nn.Dropout(0.25),
            nn.Linear(D, K),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.head(x)


class ResNet18Attention(nn.Module):

    def __init__(self, neighbour_range=0, num_attention_heads=1):
        super(ResNet18Attention, self).__init__()
        self.neighbour_range = neighbour_range
        self.num_attention_heads = num_attention_heads
        print("Using neighbour attention with a range of ", self.neighbour_range)
        print("# of attention heads: ", self.num_attention_heads)
        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1

        # def init_weights(m):
        #     if isinstance(m, (nn.Linear, nn.Conv2d)):
        #         init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.attention_heads = nn.ModuleList([
            AttentionHead(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        # self.attention_heads.apply(init_weights)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1)
            ## nn.Sigmoid()
        )
        # self.classifier.apply(init_weights)
        self.sig = nn.Sigmoid()

        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

    def disable_dropout(self):
        # print("disabling dropout")
        for attention_head in self.attention_heads:
            attention_head.eval()

    def forward(self, x, scorecam_eval=False):

        H = self.backbone(x)

        H = self.adaptive_pooling(H)

        H = H.view(-1, 512 * 1 * 1)

        if self.neighbour_range != 0:
            combinedH = H.view(-1)
            combinedH = F.pad(combinedH, (self.L * self.neighbour_range, self.L * self.neighbour_range), "constant",
                              0)  # TODO: mirror padding?
            combinedH = combinedH.unfold(0, self.L * (self.neighbour_range * 2 + 1), self.L)

            H = 0.25 * combinedH[:, :self.L] + 0.5 * combinedH[:, self.L:2 * self.L] + 0.25 * combinedH[:,
                                                                                              2 * self.L:]
        # print("H:", torch.isnan(H).any())
        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)

        unnorm_A = torch.mean(attention_maps, dim=1)

        unnorm_A = unnorm_A.view(1, -1)

        A = unnorm_A / (torch.sum(unnorm_A) + 0.01)

        # A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if scorecam_eval:
            return Y_prob, Y_hat, unnorm_A

        else:
            return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

    def calculate_objective(self, Y, Y_prob):
        Y = Y.float()

        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (
                Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood


class AttentionHeadV2(nn.Module):
    def __init__(self, L, D, K):
        super(AttentionHeadV2, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),  # was Relu
            nn.Linear(D, K),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)

class AttentionHeadV3(nn.Module):
    def __init__(self, L, D, K):
        super(AttentionHeadV3, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),  # was Relu
            nn.Linear(D, K),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)

class ResNet18AttentionV2(nn.Module):

    def __init__(self, neighbour_range=0, num_attention_heads=1, instnorm=False):
        super(ResNet18AttentionV2, self).__init__()
        self.neighbour_range = neighbour_range
        self.num_attention_heads = num_attention_heads
        self.instnorm = instnorm
        print("Using neighbour attention with a range of ", self.neighbour_range)
        print("# of attention heads: ", self.num_attention_heads)
        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1

        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.attention_heads = nn.ModuleList([
            AttentionHeadV2(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        # self.attention_heads.apply(init_weights)

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))
        self.sig = nn.Sigmoid()

        if self.instnorm:
            # load the resnet with instance norm instead of batch norm
            model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
            sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            model.load_state_dict(sd, strict=False)
        else:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

    def disable_dropout(self):
        for attention_head in self.attention_heads:
            attention_head.eval()

    def forward(self, x, return_unnorm_attention=False):

        H = self.backbone(x)
        # print("h", H.shape)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        # print("after pooling", H.shape)
        # if self.neighbour_range != 0:
        #     combinedH = H.view(-1)
        #     combinedH = F.pad(combinedH, (self.L * self.neighbour_range, self.L * self.neighbour_range), "constant",
        #                       0)  # TODO: mirror padding?
        #     combinedH = combinedH.unfold(0, self.L * (self.neighbour_range * 2 + 1), self.L)
        #
        #     H = 0.25 * combinedH[:, :self.L] + 0.5 * combinedH[:, self.L:2 * self.L] + 0.25 * combinedH[:, 2 * self.L:]

        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)
        # print("attention", attention_maps.shape)
        unnorm_A = torch.mean(attention_maps, dim=1)
        # print("after mean", unnorm_A.shape)
        unnorm_A = unnorm_A.view(1, -1)
        # print("attention after view ", unnorm_A.shape)
        A = F.softmax(unnorm_A, dim=1)
        # print("after softmax", A.shape)
        # A = torch.abs(unnorm_A) / (torch.sum(torch.abs(unnorm_A)) + 0.01)

        M = torch.mm(A, H)
        # print("M", M.shape)
        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if return_unnorm_attention:
            return Y_prob, Y_hat, unnorm_A

        else:
            return Y_prob, Y_hat, unnorm_A, H, attention_maps

    # AUXILIARY METHODS
    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

class ResNetAttentionV3(nn.Module):

    def __init__(self, neighbour_range=0, num_attention_heads=1, instnorm=False, resnet_type = "18"):
        super(ResNetAttentionV3, self).__init__()
        self.neighbour_range = neighbour_range
        self.num_attention_heads = num_attention_heads
        self.instnorm = instnorm
        print("Using neighbour attention with a range of ", self.neighbour_range)
        print("# of attention heads: ", self.num_attention_heads)
        self.L = 512 * 1 * 1
        self.D = 128
        self.K = 1
        self.resnet_type = resnet_type
        self.adaptive_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))

        self.attention_heads = nn.ModuleList([
            AttentionHeadV3(self.L, self.D, self.K) for i in range(self.num_attention_heads)])

        # self.attention_heads.apply(init_weights)

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, 1))
        self.sig = nn.Sigmoid()

        if self.instnorm:
            # load the resnet with instance norm instead of batch norm

            if resnet_type == "18":
                model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], norm_layer=MyGroupNorm)
                sd = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
            elif resnet_type == "34":
                model = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], norm_layer=MyGroupNorm)
                sd = resnet34(weights=ResNet34_Weights.DEFAULT).state_dict()

            model.load_state_dict(sd, strict=False)
        else:
            if resnet_type == "18":
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
            if resnet_type == "34":
                model = resnet34(weights=ResNet34_Weights.DEFAULT)
        modules = list(model.children())[:-2]
        self.backbone = nn.Sequential(*modules)

    def disable_dropout(self):
        for attention_head in self.attention_heads:
            attention_head.eval()

    def forward(self, x, return_unnorm_attention=False):

        H = self.backbone(x)
        # print("h", H.shape)
        H = self.adaptive_pooling(H)
        H = H.view(-1, 512 * 1 * 1)
        # print("after pooling", H.shape)
        if self.neighbour_range != 0:
            combinedH = H.view(-1)
            combinedH = F.pad(combinedH, (self.L * self.neighbour_range, self.L * self.neighbour_range), "constant",
                              0)  # TODO: mirror padding?
            combinedH = combinedH.unfold(0, self.L * (self.neighbour_range * 2 + 1), self.L)

            H = 0.25 * combinedH[:, :self.L] + 0.5 * combinedH[:, self.L:2 * self.L] + 0.25 * combinedH[:, 2 * self.L:]

        attention_maps = [head(H) for head in self.attention_heads]
        attention_maps = torch.cat(attention_maps, dim=1)
        # print("attention", attention_maps.shape)
        unnorm_A = torch.mean(attention_maps, dim=1)
        # print("after mean", unnorm_A.shape)
        unnorm_A = unnorm_A.view(1, -1)
        # print("attention after view ", unnorm_A.shape)
        #A = F.softmax(unnorm_A, dim=1)
        # print("after softmax", A.shape)
        A = unnorm_A / (torch.sum(unnorm_A) + 0.01)

        M = torch.mm(A, H)
        # print("M", M.shape)
        Y_prob = self.classifier(M)
        Y_hat = self.sig(Y_prob)
        Y_hat = torch.ge(Y_hat, 0.5).float()

        if return_unnorm_attention:
            return Y_prob, Y_hat, unnorm_A

        else:
            return Y_prob, Y_hat, unnorm_A, H, attention_maps

    # AUXILIARY METHODS
    def calculate_classification_error(self, Y, Y_hat):
        Y = Y.float()
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
