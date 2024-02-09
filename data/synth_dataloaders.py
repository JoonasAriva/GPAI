import glob

import numpy as np
import raster_geometry as rg
import torch


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
                img = self.add_random_square(img, sy / 2, sx / 2, side)

        else:
            img = self.add_random_circle(img, y, x, radius)

            if add_square == 1:
                img = self.add_random_square(img, sy, sx, side)
                img = self.add_random_square(img, sy / 2, sx / 2, side)
            y = torch.tensor([True])

        clipped_img = np.clip(img, np.percentile(img, q=0.05), np.percentile(img, q=99.5))

        x = (clipped_img - torch.mean(clipped_img)) / (torch.std(clipped_img) + 1)  # mean 0, std 1 norm

        x = torch.stack([x, x, x], dim=0)
        x = torch.squeeze(x)

        return x, y


class SynthDataloader(torch.utils.data.Dataset):
    def __init__(self, length: int, return_meta: bool = False, for_saving=False, premade=True, train=True):

        # init
        self.premade = premade
        self.train = train
        self.for_saving = for_saving
        self.return_meta = return_meta
        self.coordinates = []

        if self.premade:
            if self.train:
                path = "train/"
            else:
                path = "test/"

            data_pos = glob.glob('/gpfs/space/home/joonas97/GPAI/data/preloaded_synth_data/' + path + "pos/*.pt")[
                       :length]
            data_neg = glob.glob('/gpfs/space/home/joonas97/GPAI/data/preloaded_synth_data/' + path + "neg/*.pt")[
                       :length]

            neg_labels = [torch.tensor([0])] * len(data_neg)
            pos_labels = [torch.tensor([1])] * len(data_pos)
            self.labels = pos_labels + neg_labels
            self.paths = data_pos + data_neg

        for i in range(length):
            add_circle = np.random.randint(0, 2)
            z, y, x = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
            radius = np.random.randint(25, 35)
            add_square = np.random.randint(0, 2)

            square_coordinates = []
            square_side = np.random.randint(20, 40)
            for i in range(4):
                square_z, square_y, square_x = np.random.uniform(0.1, 0.9), np.random.uniform(0.1,
                                                                                              0.9), np.random.uniform(
                    0.1, 0.9)
                square_coordinates.append([square_x, square_y, square_z, square_side])
            cyl_z, cyl_y, cyl_x = 0.5, np.random.uniform(0.4, 0.6), np.random.uniform(
                0.3, 0.7)

            ellipsoid_coordinates = []
            for i in range(10):
                x_axis, y_axis, z_axis = np.random.randint(40, 50), np.random.randint(7, 15), np.random.randint(20, 50)
                elli_x, elli_y, elli_z = np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)
                coords = [x_axis, y_axis, z_axis, elli_x, elli_y, elli_z]
                ellipsoid_coordinates.append(coords)
            self.coordinates.append([[add_circle, round(z, 3), round(y, 3), round(x, 3), radius],
                                     [add_square, square_coordinates], [cyl_z, cyl_y, cyl_x, 10],
                                     ellipsoid_coordinates])

    def add_random_ellipsoid(self, image, z, y, x, axes):
        size = image.shape[1]
        height = image.shape[2]

        elli_mask = rg.ellipsoid((size, size, height), semiaxes=axes, position=(y, x, z))
        gaussian_noise = torch.FloatTensor(np.random.randn(size, size, height) * 0.2 + 1)

        image[elli_mask] = gaussian_noise[elli_mask]

        return image

    def add_random_cylinder(self, image, z, y, x, radius):
        size = image.shape[1]
        height = image.shape[2]

        cyl_mask = rg.cylinder((size, size, height), axis=2, radius=radius, height=150, position=(y, x, z))
        gaussian_noise = torch.FloatTensor(np.random.randn(size, size, height) * 0.2 + 1.5)

        image[cyl_mask] = gaussian_noise[cyl_mask]

        return image

    def add_random_sphere(self, image, z, y, x, radius):
        size = image.shape[1]
        height = image.shape[2]

        sphere_mask = rg.sphere((size, size, height), radius, (y, x, z))
        gaussian_noise = torch.FloatTensor(np.random.randn(size, size, height) * 0.4 + 1)

        image[sphere_mask] = gaussian_noise[sphere_mask]

        return image

    def add_random_cube(self, image, z, y, x, side):
        size = image.shape[1]
        height = image.shape[2]
        square_mask = rg.cube((size, size, height), side, (y, x, z))
        gaussian_noise = torch.FloatTensor(np.random.randn(size, size, height) * 0.3 + 1)

        image[square_mask] = gaussian_noise[square_mask]

        return image

    def __len__(self):
        # a DataSet must know its size
        if self.premade:
            return len(self.labels)
        else:
            return len(self.coordinates)

    def __getitem__(self, index):

        if self.premade:
            img = torch.load(self.paths[index])
            y = self.labels[index]
        else:
            img = torch.rand((512, 512, 200))  # 200 for MIL
            circle_coordinates, square_coordinates, cyl_coordinates, elli_coordinates = self.coordinates[index]
            add_circle, z, y, x, radius = circle_coordinates
            cz, cy, cx, cr = cyl_coordinates

            if add_circle == 0:
                # regular noise
                y = torch.tensor([0])
            else:
                img = self.add_random_sphere(img, z, y, x, radius)
                y = torch.tensor([1])
            for i in range(10):

                xax, yax, zax, elli_x, elli_y, elli_z = elli_coordinates[i]
                if i <= 5:
                    img = self.add_random_ellipsoid(img, elli_z, elli_y, elli_x, axes=(xax, yax, zax))
                else:
                    img = self.add_random_ellipsoid(img, elli_z, elli_y, elli_x, axes=(yax, xax, zax))
            img = self.add_random_cylinder(img, cz, cy, cx, radius=cr)

            if square_coordinates[0] == 1:
                for i in range(1):
                    sx, sy, sz, side = square_coordinates[1][i]
                    img = self.add_random_cube(img, sz, sy, sx, side)
                # img = self.add_random_cube(img, sz, sy / 2, sx / 2, side)

            if self.for_saving:
                return img, y

        clipped_img = np.clip(img, np.percentile(img, q=0.05), np.percentile(img, q=99.5))

        x = (clipped_img - torch.mean(clipped_img)) / (torch.std(clipped_img) + 1)  # mean 0, std 1 norm

        x = torch.stack([x, x, x], dim=0)
        x = torch.squeeze(x)

        # y_onehot = torch.nn.functional.one_hot(y, num_classes=2)

        if self.return_meta:
            return x, y, self.coordinates[index]
        else:
            return x, y, "no_path"
