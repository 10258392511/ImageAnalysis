import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re
from io import StringIO


def read_and_resize(path: str, if_save=False, new_path=None, **kwargs):
    """
    Returns np.float32 image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255
    fx, fy = kwargs.get("fx", 1), kwargs.get("fy", 1)
    img = cv2.resize(img, dsize=None, fx=fx, fy=fy)

    if if_save:
        cv2.imwrite(new_path, (img * 255).astype(np.uint8))

    return img


def load_data(base_dir="./imgs_out", old_delimiter=", "):
    """
    Returns data matrix of shape (m, n, 2) where m is #frames and n #pts
    """
    all_paths = os.listdir(base_dir)
    pattern = re.compile(r"fr[0-9]+.*.txt")
    log_paths = sorted([os.path.join(base_dir, path) for path in all_paths if re.match(pattern, path)])
    arrays = []
    for path in log_paths:
        with open(path, "r") as rf:
            content = rf.read().replace(old_delimiter, " ")
        arr = np.loadtxt(StringIO(content))
        arrays.append(arr[np.newaxis, ...])

    return np.concatenate(arrays, axis=0)


def load_imgs(base_dir="./imgs_out"):
    """
        Returns image matrix of shape (m, H, W) where m is #frames, of dtype np.float32
        """
    all_paths = os.listdir(base_dir)
    pattern = re.compile(r"fr[0-9]_orig.png")
    img_paths = sorted([os.path.join(base_dir, path) for path in all_paths if re.match(pattern, path)])
    imgs = []
    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = (img / 255).astype(np.float32)
        imgs.append(img[np.newaxis, ...])

    return np.concatenate(imgs, axis=0)


def skew2vec(S: np.ndarray):
    assert np.allclose(S.T, -S), "Output is not skew-symmetric"
    assert S.shape == (3, 3), "Output is not 3-by-3"

    return np.array([-S[1, 2], S[0, 2], -S[0, 1]])


def vec2skew(s: np.ndarray):
    assert len(s) == 3, "Input must be a 3D vector"

    S = np.zeros((3, 3))
    S[1, 2] = -s[0]
    S[0, 2] = s[1]
    S[0, 1] = -s[2]
    S -= S.T

    assert np.allclose(S.T, -S), "Output is not skew-symmetric"
    assert S.shape == (3, 3), "Output is not 3-by-3"

    return S


def to_homogeneous(arr: np.ndarray):
    assert arr.shape[-1] in [2, 3], "Last dimension should be 2 or 3"
    shape = list(arr.shape)
    shape[-1] += 1
    arr_homo = np.empty(shape)
    arr_homo[..., -1] = 1
    arr_homo[..., :-1] = arr.copy()

    return arr_homo


def project(arr: np.ndarray):
    assert arr.shape[-1] == 3, "Must be 3D pts"

    return arr / arr[..., -1:]


def draw_epipolar_line(pts1, E, img2, K=None):
    """
    pts1 are 2d IMAGE points, which means we need to use fundamental matrix F.
    """
    if K is None:
        K = np.eye(3)

    H, W = img2.shape
    img2_c = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    for i in range(pts1.shape[0]):
        l = F @ to_homogeneous(pts1[i, :])
        x1, x2 = 0, W - 1
        y1, y2 = -(l[2] + l[0] * x1) / l[1], -(l[2] + l[0] * x2) / l[1]
        # print(f"{(x1, y1), (x2, y2)}")
        cv2.line(img2_c, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(img2_c, cv2.COLOR_BGR2RGB), cmap="gray")
    plt.show()
