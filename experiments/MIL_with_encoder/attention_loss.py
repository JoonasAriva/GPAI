import matplotlib.pyplot as plt
import torch

from depth_trainer import DepthLossV2


def create_step_matrix(step_value, matrix_size):
    steps = torch.zeros(matrix_size, matrix_size)
    steps = torch.diagonal_scatter(steps, torch.zeros(matrix_size), 0)
    for i in range(matrix_size):
        steps = torch.diagonal_scatter(steps, (1 + i) * step_value * torch.ones(matrix_size - i - 1), 1 + i)
        steps = torch.diagonal_scatter(steps, (1 + i) * step_value * torch.ones(matrix_size - i - 1), -1 - i)
    return steps


class AttentionLoss(DepthLossV2):
    def __init__(self, step):
        super().__init__(step)

    def forward(self, attention, z_spacing, nth_slice, verbose=False):
        acceptable_step_size = self.step * z_spacing * nth_slice
        step_matrix = create_step_matrix(acceptable_step_size, len(attention)).cuda(non_blocking=True)
        acceptable_distance = 100
        mercy_value = acceptable_distance * self.step
        mod_step_matrix = torch.maximum(step_matrix - mercy_value,
                                        torch.zeros_like(step_matrix).cuda(non_blocking=True))

        # attention = torch.sigmoid(attention)
        attention_matrix = attention * attention.reshape(-1, 1)
        loss_matrix = torch.tril(mod_step_matrix * attention_matrix)

        # constant = torch.Tensor([5]).cuda(non_blocking=True)
        # diag_loss = torch.maximum(constant - torch.diag(attention_matrix).sum(),
        #                          torch.Tensor([0]).cuda(non_blocking=True))

        if verbose:
            print("z_spacing:", z_spacing)
            print("nth_slice:", nth_slice)
            # print("diag_loss:", diag_loss, "constant:", constant)
            fig, axs = plt.subplots(1, 4, figsize=(24, 8))
            axs[0].imshow(step_matrix.cpu())
            axs[0].set_title("step matrix")

            axs[1].imshow(mod_step_matrix.cpu())
            axs[1].set_title("step matrix with mercy")

            axs[2].imshow(attention.cpu() * attention.reshape(-1, 1).cpu())
            axs[2].set_title("attention matrix")

            axs[3].imshow(loss_matrix.cpu())
            axs[3].set_title("loss_matrix")

        return loss_matrix.sum() / (len(attention) ** 2)  # + diag_loss

# class ClassAttentionLoss(nn.Module):
#     def __init__(self, step):
#         super().__init__()
#
#         self.attention_loss = AttentionLoss(step)
#         self.cls_loss = torch.nn.BCEWithLogitsLoss()
#
#     def forward(self, attention, z_spacing, nth_slice, prediction, bag_label):
#         att_loss = self.attention_loss(attention, z_spacing, nth_slice)
#         cls_loss = self.cls_loss(prediction, bag_label)
#         return att_loss, cls_loss
