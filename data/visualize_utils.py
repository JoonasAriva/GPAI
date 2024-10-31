from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import *

# for discrete colormap
discrete_cmap = ListedColormap(['#53c972', '#e3c634', '#2a5fbd'])
discrete_cmap.set_under(color='white', alpha=0)
boundaries = [0.1, 1.1, 2.1, 3.1]
norm = BoundaryNorm(boundaries, discrete_cmap.N)


# single CT scan visulistation (scan is reduced in size)
def get_animation(volume, use_zoom=True, title=None):
    if use_zoom:
        volume = zoom(volume, (0.3, 0.3, 0.3))
    fig = plt.figure()

    ims = []
    for image in range(0, volume.shape[0]):
        im = plt.imshow(volume[image, :, :],
                        animated=True, cmap=plt.cm.bone)

        plt.axis("off")
        ttl = plt.text(0.5, 1.2, title[image],horizontalalignment='center', verticalalignment='bottom')
        ims.append([im,ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)

    plt.close()
    return ani


# single scan visulaistation of labels (discrete colors)
def get_animation_discrete_color(volume, discrete_cmap, norm, use_zoom = True):
    if use_zoom:
        volume = zoom(volume, (0.3, 0.3, 0.3))
    fig = plt.figure()

    ims = []
    for image in range(0, volume.shape[0]):
        im = plt.imshow(volume[image, :, :],
                        animated=True, cmap=discrete_cmap, norm=norm)

        plt.axis("off")
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)

    plt.close()
    return ani


# scan + labels on top of it
def get_animation_with_discrete_masks(volume, mask, discrete_cmap, use_zoom=True):
    if use_zoom:
        volume = zoom(volume, (0.3, 0.3, 0.3))
        mask = zoom(mask, (0.3, 0.3, 0.3))
    fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    ims = []
    for image in range(0, volume.shape[0]):
        im = plt.imshow(volume[image, :, :], animated=True, cmap=plt.cm.bone)
        im2 = plt.imshow(mask[image, :, :], animated=True, cmap=discrete_cmap, norm=norm, alpha=0.4)

        plt.axis("off")
        ims.append([im, im2])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)

    plt.close()
    return ani

def get_animation_with_masks(volume, mask, use_zoom=True):
    if use_zoom:
        volume = zoom(volume, (0.3, 0.3, 0.3))
        mask = zoom(mask, (0.3, 0.3, 0.3))
    fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    ims = []
    for image in range(0, volume.shape[0]):
        im = plt.imshow(volume[image, :, :], animated=True, cmap=plt.cm.bone)
        im2 = plt.imshow(mask[image, :, :], animated=True, cmap=plt.cm.viridis, alpha=0.4)

        plt.axis("off")
        ims.append([im, im2])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)

    plt.close()
    return ani


def get_full_animation(volume, cams, cams_meta, mask=None, use_zoom=True, overlay_volume = False):
    '''Get animation of volume, all cam maps and add a ground truth mask to volume if it is included'''
    nr_of_rows = 1
    nr_of_cols = len(cams) + 1
    if len(cams) > 3:
        nr_of_rows = 2
        nr_of_cols = nr_of_cols // 2 + 1
    if nr_of_rows > 1:
        fig, (ax, ax2) = plt.subplots(nr_of_rows, nr_of_cols, figsize=(14, 7))
        fig.delaxes(ax2[2])
    else:
        fig, ax = plt.subplots(nr_of_rows, nr_of_cols, figsize=(10, 3))
    ims = []
    if use_zoom:
        volume = zoom(volume.astype(np.float32), (0.4, 0.4, 0.4))
        if  mask is not None:
            mask = zoom(mask.astype(np.float32), (0.4, 0.4, 0.4), mode='nearest')
    cams_reshaped = []
    for cam in cams:
        if use_zoom:
            cam = zoom(cam, (0.4, 0.4, 0.4))
        cams_reshaped.append(cam)

    for image in range(0, volume.shape[0]):

        slices = []
        im1 = ax[0].imshow(volume[image, :, :],
                           animated=True, cmap=plt.cm.bone)
        slices.append(im1)
        if mask is not None:
            im2 = ax[0].imshow(mask[image, :, :], animated=True, cmap=discrete_cmap, norm=norm, alpha=0.3)
            slices.append(im2)
        ax[0].set_title("Input image")
        for i, cam in enumerate(cams_reshaped):
            current_ax = (ax if i + 1 < nr_of_cols else ax2)
            current_index = (i + 1 if i + 1 < nr_of_cols else i + 1 - nr_of_cols)

            im = current_ax[current_index].imshow(cam[image, :, :],
                                                  animated=True)#, cmap=plt.cm.bone)
            current_ax[current_index].set_title(cams_meta[i])
            current_ax[current_index].axis('off')
            slices.append(im)

            if overlay_volume:
                im_over = current_ax[current_index].imshow(volume[image, :, :],
                                                      animated=True, cmap=plt.cm.bone, alpha = 0.3)
                slices.append(im_over)

        ims.append(slices)

    # plt.axis("off")
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)
    plt.close()
    return ani
