import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.fft import fft, fftshift
from skimage.transform import radon, iradon


def rotate_img(img, angle, if_show=False):
    center_y, center_x = np.array(img.shape) // 2
    R = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    img_rot = cv2.warpAffine(img, R, img.shape[::-1])

    if if_show:
        plt.imshow(img_rot, cmap="gray")
        plt.colorbar()
    return img_rot


def get_filter(size: int, if_show=False):
    s_time_direct = np.zeros(size)
    s_time_direct[0] = 0.25
    nn = np.concatenate([np.arange(1, size // 2 + 1, 2), np.arange(size // 2 - 1, 0, -2)])
    s_time_direct[1::2] = -1 / (np.pi * nn) ** 2
    if if_show:
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(s_time_direct, "--.")
        axes[1].plot(fftshift(np.real(fft(s_time_direct))))
        plt.show()

    return s_time_direct


def compare_recons(img, img_recons):
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    for axis, img_iter in zip(axes, [img, img_recons, img - img_recons]):
        handle = axis.imshow(img_iter, cmap="gray")
        plt.colorbar(handle, ax=axis, fraction=0.05)

    plt.tight_layout()
    plt.show()


def using_skimage(img: np.ndarray, if_show=True):
    theta_grid = np.linspace(0, 180, max(img.shape))
    sinogram = radon(img, theta=theta_grid)
    if if_show:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
        for axis, img_iter in zip(axes, [img, sinogram]):
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=0.05)
        axes[1].set_xlabel(r"$\theta$")
        axes[1].set_ylabel(r"$\rho$")

    img_recons = iradon(sinogram, theta=theta_grid, filter_name="ramp")
    compare_recons(img, img_recons)

    return sinogram, img_recons
