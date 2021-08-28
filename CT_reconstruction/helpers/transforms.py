import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.interpolate import interp1d
from .utils import rotate_img, get_filter


def radon_transform(img: np.ndarray, if_show=False):
    img_c = img.copy()
    H, W = img.shape
    radius = min(*img.shape)
    center_y, center_x = np.array(img.shape) // 2
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx -= center_x
    yy -= center_y
    dist = xx ** 2 + yy ** 2
    inds_outside = np.where(dist > radius ** 2)
    img_c[inds_outside] = 0

    radius = W
    theta_grid = np.linspace(0, 180, radius)
    sinogram = np.zeros((radius, radius))  # x: theta, y: rho (signed)

    for i, theta in enumerate(theta_grid):
        img_rot = rotate_img(img_c, theta)
        sinogram[:, i] = img_rot.sum(axis=0)  # vertically along y-axis

    if if_show:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
        for axis, img_iter in zip(axes, [img_c, sinogram]):
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=0.05)

        axes[1].set_xlabel(r"$\theta$")
        y_ticks = axes[1].get_yticks().astype(int)
        axes[1].set_yticklabels(y_ticks - W // 2)
        axes[1].set_ylabel(r"$\rho$")
        plt.tight_layout()
        plt.show()

    return sinogram


def iradon_transform(sinogram: np.ndarray, shape=None, use_filter=True):
    H, W = sinogram.shape
    assert H == W, "sinogram should be square"

    if shape is None:
        shape = sinogram.shape

    N = sinogram.shape[0]
    H, W = shape
    theta_grid = np.deg2rad(np.linspace(0, 180, N))
    img_recons = np.zeros(shape)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    center_y, center_x = np.array(shape) // 2
    xx -= center_x
    yy -= center_y

    for i, theta in enumerate(theta_grid):
        cosine, sine = np.cos(theta), np.sin(theta)
        rho = xx * cosine + yy * sine  # project each loc onto the line with which the film aligns
        sino_slice = sinogram[:, i]

        if use_filter:
            kernel = get_filter(N)
            sino_slice_freq = fft(sino_slice)
            kernel_freq = fft(kernel)
            sino_slice = np.real(ifft(kernel_freq * sino_slice_freq))

        interpolator = interp1d(np.arange(N) - N // 2, sino_slice, bounds_error=False, fill_value=0)
        img_recons += interpolator(rho)

    img_recons *= np.pi / N

    return img_recons
