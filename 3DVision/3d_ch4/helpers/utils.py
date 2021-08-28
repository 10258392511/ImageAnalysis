import numpy as np
import matplotlib.pyplot as plt
import cv2


def load_img_gray_float32(img_path: str, if_show=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

    if if_show:
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()

    return img


def save_img(img: np.ndarray, img_path: str, if_use_default_range=True, if_RGB2BGR=False):
    if img.dtype != np.uint8:
        if if_use_default_range:
            img = img * 255
        else:
            img = img / (np.max(img) - np.min(img)) * 255
        img = img.astype(np.uint8)

    if if_RGB2BGR:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(img_path, img)


def compute_gradient(img: np.ndarray, if_smooth=True, ksize=(5, 5), if_show=False):
    assert img.dtype == np.float32

    if if_smooth:
        img = cv2.GaussianBlur(img, ksize=ksize, sigmaX=0)

    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    if if_show:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8))
        for axis, deriv in zip(axes, [Ix, Iy]):
            handle = axis.imshow(deriv, cmap="gray")
            plt.colorbar(handle, ax=axis, fraction=0.04)

        plt.tight_layout()
        plt.show()

    return Ix, Iy
