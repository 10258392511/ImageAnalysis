import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.utils import make_grid


def show_dict(D: np.ndarray):
    # D: (num_feats, N)
    D = D.T  # (N, num_feats)
    side = int(np.sqrt(D.shape[1]))
    D = D.reshape((-1, 1, side, side))
    D_tensor = torch.tensor(D)
    img_grid = make_grid(D_tensor, nrow=int(np.sqrt(D.shape[0])), normalize=True, padding=1)
    fig, axis = plt.subplots(figsize=(7.2, 7.2))
    handle = axis.imshow(img_grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.colorbar(handle, ax=axis, fraction=0.08)
    plt.show()


def dict_sim(D: np.ndarray, D_hat: np.ndarray, th=0.99):
    """
    Compare D_hat to D
    """
    inner_prod = np.abs(D_hat.T @ D)
    max_row = inner_prod.max(axis=1)
    count = (max_row >= th).sum()

    return count / D.shape[1]


def compute_error(D, X, Y: np.ndarray):
    Y_hat = D @ X  # (n, N)

    return np.linalg.norm(Y_hat - Y, axis=0).sum() / Y.shape[1]


def plot_curves(log_dict):
    """
    Parameters
    ----------
    log_dict: dict
        Keys: "recovered_atoms_percent", "train_error", "test_error"
    """
    # fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    axes = []
    log_rec_percent = log_dict["recovered_atoms_percent"]
    if log_rec_percent is not None:
        fig, axis = plt.subplots()
        axis.plot(log_rec_percent)
        axes.append(axis)

    fig, axis = plt.subplots()
    axis.plot(log_dict["train_error"], label="training")
    axis.plot(log_dict["test_error"], label="test")
    axis.legend()
    axes.append(axis)

    for axis in axes:
        axis.grid(True)

    plt.show()
