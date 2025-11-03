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


class AttentionLossPatches2D(DepthLossV2):
    def __init__(self, step):
        super().__init__(step)

    def calc_manhattan_distances_in_3d(self, matrix):
        return matrix.reshape(-1, 1, 3) - matrix.float()

    def forward(self, attention, real_coordinates, verbose=False, z_only: bool = False, attention_matrix=None):
        voxel_spacing = torch.tensor([2.0, 0.84, 0.84])
        acceptable_distance = 100
        scaled_coordinates = real_coordinates * voxel_spacing
        real_distances = self.calc_manhattan_distances_in_3d(scaled_coordinates).float().cuda()
        if z_only:
            z_dist = torch.unsqueeze(real_distances[:, :, 0], 2)
            y_dist = real_distances[:, :, 1]
            # x_dist = torch.unsqueeze(real_distances[:, :, 2], 2)
            real_distances = z_dist
            acceptable_distance = 100

        dist_norm = torch.norm(real_distances, dim=2)
        mercy_value = acceptable_distance * self.step
        # mod_step_matrix = dist_norm - mercy_value
        mod_step_matrix = torch.maximum(dist_norm - mercy_value,
                                        torch.zeros_like(dist_norm).cuda(non_blocking=True))

        if z_only:
            loss_matrix_z = mod_step_matrix * attention_matrix

            loss_matrix_y = torch.maximum((-y_dist + 40), torch.zeros_like(y_dist).cuda(
                non_blocking=True)) * attention_matrix  # push heads apart in y direction
            loss_matrix = loss_matrix_z + loss_matrix_y
            # print("zx: ",loss_matrix_zx.sum() / (len(attention) ** 2))
            # print("y: ",loss_matrix_y.sum() / (len(attention) ** 2))
        else:
            attention_matrix = attention * attention.reshape(-1, 1)
            loss_matrix = torch.tril(mod_step_matrix * attention_matrix)

        if verbose:
            for i in range(3):
                sorted_idx = torch.argsort(scaled_coordinates[0, :, i], dim=0)
                sorted_coordinates = scaled_coordinates[:, sorted_idx]
                sorted_attention = attention[sorted_idx]

                real_distances = self.calc_manhattan_distances_in_3d(sorted_coordinates).float().cuda()

                dist_norm = torch.norm(real_distances, dim=2)

                acceptable_distance = 80
                mercy_value = acceptable_distance * self.step
                mod_step_matrix = torch.maximum(dist_norm - mercy_value,
                                                torch.zeros_like(dist_norm).cuda(non_blocking=True))

                loss_matrix = torch.tril(mod_step_matrix * attention_matrix)

                fig, axs = plt.subplots(1, 4, figsize=(24, 8))
                axs[0].imshow(dist_norm.cpu())
                axs[0].set_title("step matrix")

                axs[1].imshow(mod_step_matrix.cpu())
                axs[1].set_title("step matrix with mercy")

                axs[2].imshow(attention_matrix.cpu())
                axs[2].set_title("attention matrix")

                axs[3].imshow(loss_matrix.cpu())
                axs[3].set_title("loss_matrix")
                plt.show()

        return loss_matrix.sum() / (len(attention) ** 2)


class AttentionLossPatches2DReward(DepthLossV2):
    def __init__(self, step):
        super().__init__(step)

    def calc_manhattan_distances_in_3d(self, matrix):
        return matrix.reshape(-1, 1, 3) - matrix.float()

    def forward(self, attention, real_coordinates, verbose=False):
        voxel_spacing = torch.tensor([2.0, 0.84, 0.84])
        scaled_coordinates = real_coordinates * voxel_spacing

        real_distances = self.calc_manhattan_distances_in_3d(scaled_coordinates).float().cuda()

        dist_norm = torch.norm(real_distances, dim=2)

        acceptable_distance = 100
        mercy_value = acceptable_distance * self.step

        reward = torch.exp(-dist_norm / mercy_value)

        attention_matrix = attention * attention.reshape(-1, 1)
        loss_matrix = torch.tril(reward * attention_matrix)

        return -loss_matrix.sum() / (len(attention) ** 2)
