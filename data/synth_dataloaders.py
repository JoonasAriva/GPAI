import random

import nibabel as nib
import numpy as np
import raster_geometry as rg
import torch
import torch.nn.functional as F
from monai.transforms import *


class SynthDataloader2D(torch.utils.data.Dataset):
    def __init__(self, length: int):

        # init
        self.coordinates = []
        for i in range(length):
            add_circle = np.random.randint(0, 2)
            y, x = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
            radius = np.random.randint(25, 40)
            add_square = np.random.randint(0, 2)
            square_y, square_x = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
            square_side = np.random.randint(20, 35)
            self.coordinates.append([[add_circle, round(y, 3), round(x, 3), radius],
                                     [add_square, round(square_y, 3), round(square_x, 3), square_side]])

    def add_random_circle(self, image, y, x, radius):
        size = image.shape[1]

        circle_mask = rg.circle((size, size), radius, (y, x))

        gaussian_noise = torch.FloatTensor(np.random.randn(size, size) * 0.1 + 2)

        image[circle_mask] = gaussian_noise[circle_mask]

        return image

    def add_random_square(self, image, y, x, side):
        size = image.shape[1]

        square_mask = rg.square((size, size), side, (y, x))
        gaussian_noise = torch.FloatTensor(np.random.randn(size, size) * 0.1 + 2)

        image[square_mask] = gaussian_noise[square_mask]

        return image

    def __len__(self):
        # a DataSet must know its size
        return len(self.coordinates)

    def __getitem__(self, index):

        img = torch.rand((224, 224))
        circle_coordinates, square_coordinates = self.coordinates[index]
        add_circle, y, x, radius = circle_coordinates
        add_square, sy, sx, side = square_coordinates

        if add_circle == 0:
            # regular noise
            y = torch.tensor([False])
            if add_square == 1:
                img = self.add_random_square(img, sy, sx, side)
                img = self.add_random_square(img, sy/2,  sx/2, side)

        else:
            img = self.add_random_circle(img, y, x, radius)

            if add_square == 1:
                img = self.add_random_square(img, sy, sx, side)
                img = self.add_random_square(img, sy/2, sx/2, side)
            y = torch.tensor([True])

        clipped_img = np.clip(img, np.percentile(img, q=0.05), np.percentile(img, q=99.5))

        x = (clipped_img - torch.mean(clipped_img)) / (torch.std(clipped_img) + 1)  # mean 0, std 1 norm

        x = torch.stack([x, x, x], dim=0)
        x = torch.squeeze(x)

        return x, y


class SynthDataloader(torch.utils.data.Dataset):
    def __init__(self, length: int):

        # init
        self.coordinates = []
        for i in range(length):
            z, y, x = np.random.uniform(0.1, 0.9), np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)
            radius = np.random.randint(25, 50)
            self.coordinates.append([round(z, 3), round(y, 3), round(x, 3), radius])

    def add_random_sphere(self, image, z, y, x, radius):
        size = image.shape[1]
        height = image.shape[2]

        sphere_mask = rg.sphere((size, size, height), radius, (y, x, z))
        gaussian_noise = torch.FloatTensor(np.random.randn(size, size, height) * 0.1 + 2)

        image[sphere_mask] = gaussian_noise[sphere_mask]

        return image

    def __len__(self):
        # a DataSet must know its size
        return len(self.coordinates) * 2

    def __getitem__(self, index):

        img = torch.rand((224, 224, 100))
        if index >= len(self.coordinates):
            # regular noise
            y = torch.tensor([False])

        else:
            z, y, x, radius = self.coordinates[index]
            img = self.add_random_sphere(img, z, y, x, radius)
            y = torch.tensor([True])

        clipped_img = np.clip(img, np.percentile(img, q=0.05), np.percentile(img, q=99.5))

        x = (clipped_img - torch.mean(clipped_img, dim=(0, 1))) / (
                torch.std(clipped_img, dim=(0, 1)) + 1)  # mean 0, std 1 norm

        x = torch.stack([x, x, x], dim=0)
        x = torch.squeeze(x)

        return x, y