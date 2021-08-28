import numpy as np
import matplotlib.pyplot as plt
import cv2

from .utils import compute_gradient


def NMS(img: np.ndarray):
    """
    9-point NMS
    """
    H, W = img.shape
    center_x, center_y = np.meshgrid(np.arange(W), np.arange(H))
    top = np.clip(center_y - 1, 0, H - 1)
    bottom = np.clip(center_y + 1, 0, H - 1)
    left = np.clip(center_x - 1, 0, W - 1)
    right = np.clip(center_x + 1, 0, W - 1)

    row_iters = [top, center_y, bottom]
    col_iters = [left, center_x, right]
    mask = np.ones_like(img, dtype=bool)

    for i, row_iter in enumerate(row_iters):
        for j, col_iter in enumerate(col_iters):
            if i == 1 and j == 1:
                continue
            mask &= (img[center_y, center_x] > img[row_iter, col_iter])

    return mask


def compute_features(img: np.ndarray, win_size=11, min_eig_th=0.05, border=20, max_num_pts=1000,
                     min_dist_pix=10, if_show=False, if_return_st=False, **kwargs):
    """
    Returns
    -------
    x, y: np.ndarray
       Locations of features in xy-coordinate, of shape (N,)
    structure_tensor: np.ndarray
        [Ix, Iy, Ixx, Iyy, Ixy] if specifies "if_return_st = True", otherwise this is not returned
    """
    H, W = img.shape
    Ix, Iy = compute_gradient(img)
    Ixx = cv2.GaussianBlur(Ix ** 2, (win_size, win_size), sigmaX=0)
    Iyy = cv2.GaussianBlur(Iy ** 2, (win_size, win_size), sigmaX=0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (win_size, win_size), sigmaX=0)

    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2
    diff_half = np.sqrt((trace / 2) ** 2 - det)
    quality = np.minimum(np.abs(trace / 2 + diff_half), np.abs(trace / 2 - diff_half))  # lambda1 or lambda2

    mask_border = np.zeros_like(img, dtype=bool)
    mask_border[border : H - border, border : W - border] = True
    quality *= mask_border
    mask_thresh = (quality > min_eig_th * np.max(quality))
    quality *= mask_thresh
    mask_nms = NMS(quality)
    quality *= mask_nms

    quality_flatten = quality.ravel()
    inds = np.argsort(quality_flatten, axis=-1)[::-1]
    inds = inds[:min(len(inds), max_num_pts)]
    ind_rows, ind_cols = np.unravel_index(inds, img.shape)

    # filter by min dist
    num_features = len(ind_rows)
    pts = np.concatenate([ind_cols.reshape((1, -1)), ind_rows.reshape((1, -1))], axis=0)  # (2, N)
    pts_norm_sq = np.linalg.norm(pts, axis=0) ** 2  # (N,)
    D = np.tile(pts_norm_sq.reshape((-1, 1)), (1, num_features)) + \
        np.tile(pts_norm_sq.reshape((1, -1)), (num_features, 1)) - 2 * pts.T @ pts  # (N, N)
    D -= min_dist_pix ** 2
    D = np.tril(D, k=-1)
    D_sum = D.sum(axis=1)  # (N,)
    inds_sel = np.argwhere(D_sum >= 0)
    ind_rows, ind_cols = ind_rows[inds_sel], ind_cols[inds_sel]

    if if_show:
        img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_copy = (img_copy * 255).astype(np.uint8)
        for x, y in zip(ind_cols, ind_rows):
            img_copy = cv2.drawMarker(img_copy, (x, y), color=(0, 0, 255), markerSize=kwargs.get("markerSize", 10),
                                      thickness=kwargs.get("thickness", 2), line_type=cv2.LINE_AA)

        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.show()

    ind_cols = ind_cols.ravel()
    ind_rows = ind_rows.ravel()

    if if_return_st:
        return ind_cols, ind_rows, [Ix, Iy, Ixx, Iyy, Ixy]

    return ind_cols, ind_rows
