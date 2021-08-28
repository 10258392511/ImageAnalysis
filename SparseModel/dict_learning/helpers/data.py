import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.util import view_as_windows


def load_synthetic_data(n=30, m=60, N=4000, k0=4, sigma=0.1):
    A = np.random.randn(n, m)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    X = np.zeros((m, N))  # (m, N)
    for i in range(N):
        inds = np.random.choice(m, k0, replace=False)
        X[inds, i] = np.random.randn(k0)
    Y = A @ X + np.random.randn(n, N) * sigma

    return A, X, Y


def load_img_data(img_path: str, split=0.6, window_shape=(8, 8), max_train=2000, max_test=1000, if_show=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img /= 255
    if if_show:
        plt.imshow(img, cmap="gray")
        plt.colorbar()
        plt.show()

    img_wins = view_as_windows(img, window_shape=window_shape)  # (n, m, 8, 8)
    img_data = img_wins.reshape((-1, np.prod(window_shape)))  # (N, 64)
    img_data = np.random.permutation(img_data)  # (N, 64)
    num_train = int(np.ceil(img_data.shape[0] * split))
    data_train, data_test = img_data[:num_train, :], img_data[num_train:, :]

    return data_train[:min(max_train, data_train.shape[0])].T, data_test[:min(max_test, data_test.shape[0])].T


def init_dict(shape_1d=(8, 11)):
    """
    Parameters
    ----------
    shape_1d: tuple
        Shape of 1D kernel to create the 2D kernel. e.g. If a 64-by-121 2D kernel is wanted, shape_1d should be (8, 11).


    Returns
    -------
    D: np.ndarray
    """
    n, m = shape_1d
    one_dim_kernel = np.arange(n).reshape((-1, 1)) @ np.arange(m).reshape((1, -1)) * np.pi / m
    one_dim_kernel = np.cos(one_dim_kernel)
    mean = one_dim_kernel.mean(axis=0)
    mean[0] = 0
    one_dim_kernel -= mean
    D = np.kron(one_dim_kernel, one_dim_kernel)
    D /= np.linalg.norm(D, axis=0)

    return D
