import numpy as np
import matplotlib.pyplot as plt

from .utils import to_homogeneous, vec2skew, skew2vec, project


# problem: alternative optimization doesn't converge to the correct solution
# 3D reconstruction step is not clear using only alpha

def compute_3d(pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, T: np.ndarray):
    """
    pts1, pts2 are arrays of 2d camera coordinate points.

    Returns
    -------
    X: np.ndarray
        array of reconstructed 3D pts, (N, 3)
    lamda: np.ndarray
        lamda1's of the world frame, (N,)
    gamma: float
        scaler of T
    """
    assert pts1.shape[0] == pts2.shape[0], "Number of points must match"
    num_pts = pts1.shape[0]
    M = np.zeros((num_pts * 3, num_pts + 1))

    for i in range(num_pts):
        x1, x2 = to_homogeneous(pts1[i, :]), to_homogeneous(pts2[i, :])
        x2_skew = vec2skew(x2)
        M[3 * i : 3 * i + 3, i] = x2_skew @ (R @ x1)
        M[3 * i : 3 * i + 3, -1] = x2_skew @ T

    _, _, V_h = np.linalg.svd(M, full_matrices=False)
    cand = V_h[-1, :]
    if cand[0] < 0:
       cand = -cand
    lamda, gamma = cand[:-1], cand[-1]

    X = lamda[..., np.newaxis] * to_homogeneous(pts1) @ R.T + gamma * T

    return X, lamda, gamma


def _check_essential(pts1, pts2, R, T):
    """
    pts1, pts2 are 2d camera coordinate points.
    """
    X, lamda1, gamma = compute_3d(pts1, pts2, R, T)
    if np.all(lamda1 > 0) and np.all(X[:, -1] > 0):  # X[-1] = lamda2
        return True

    return False


def estimate_essential(pts1: np.ndarray, pts2: np.ndarray):
    """
    pts1, pts2 are 2d camera coordinate points.

    Returns E, R, T
    """
    assert len(pts1) == len(pts2), "Number of points must match"
    num_pts = pts1.shape[0]
    X = np.empty((num_pts, 9))

    for i in range(num_pts):
        x1, x2 = to_homogeneous(pts1[i, :]), to_homogeneous(pts2[i, :])
        X[i, :] = np.kron(x1, x2)

    # print(X)
    _, _, V_h = np.linalg.svd(X, full_matrices=False)
    E = V_h[-1, :].reshape((3, 3), order="F")

    for E_cand in [E, -E]:
        U, s, V_h = np.linalg.svd(E_cand, full_matrices=True)
        # print(s)

        for R_z in [np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])]:
            R = U @ R_z @ V_h
            T_skew = U @ R_z.T @ np.diag([1, 1, 0]) @ U.T
            T = skew2vec(T_skew)

            if not np.allclose(np.linalg.det(R), 1):
                continue

            if _check_essential(pts1, pts2, R, T):
                return E_cand, R, T

    raise ValueError("No solution")


def recons_multi(x, eps=1e-2, max_iter=200, if_plot=False):
    """
    Parameters
    ----------
    x: np.ndarray
        Of shape (m, n, 2) where m = #frames, n = #pts
    eps: float
        Rel error tol

    Returns
    -------
    depth_list, R_list, T_list: list
    """
    num_frames, num_pts = x.shape[:2]
    num_total_pts = num_frames * num_pts

    depth_list = _init_multi(x)
    depth_list /= depth_list[0]
    R_list, T_list = None, None
    diff = float("inf")
    diff_doc = []
    # x_norm = np.linalg.norm(x)
    diff_criterion = eps

    count = 0
    eval_inter = 10
    while diff > diff_criterion and count < max_iter:
        R_list, T_list = _solve_for_motion(x, depth_list)
        depth_list = _solve_for_depth(x, R_list, T_list)
        # T_list = np.array(T_list) * depth_list[0]
        depth_list /= depth_list[0]
        print(f"{count}, depth_list:\n{depth_list}")
        x_hat = _project_multi_view(x[0, ...], 1 / depth_list, R_list, T_list)
        diff = np.linalg.norm(x_hat - x) / num_total_pts
        diff_doc.append(diff)
        count += 1
        if count % eval_inter == 0:
            print(f"{count} iter: diff is {diff} / {diff_criterion}")

    if if_plot:
        fig, axis = plt.subplots(figsize=(10.8, 4.8))
        axis.plot(diff_doc)
        axis.grid(True)
        plt.show()

    return depth_list, R_list, T_list


def _init_multi(x):
    """
    x is 2d camera coord pts.

    Returns depth_list
    """
    assert x.shape[-1] == 2, "Input should be 2D camera coord pts"
    _, R, T = estimate_essential(x[0, ...], x[1, ...])

    return _solve_for_depth(x[:2, ...], [np.eye(3), R], [np.zeros(3), T])


def _solve_for_motion(x, depth_list):
    """
    x is 2d camera coord pts.

    Returns R_list and T_list.
    """
    x = to_homogeneous(x)  # (m, n, 3)
    R_list, T_list = [np.eye(3)], [np.zeros((3))]
    num_frames, num_pts, _ = x.shape

    for i in range(1, num_frames):
        X = np.empty((3 * num_pts, 12))
        for j in range(num_pts):
            xi_skew = vec2skew(x[i, j, :])
            x1 = x[0, j, :]
            alpha = depth_list[j]
            X[3 * j : 3 * (j + 1), :] = np.block([np.kron(x1, xi_skew), alpha * xi_skew])
            # print(f"np.block(.):\n{np.block([np.kron(x1, xi_skew), alpha * xi_skew])}")

        _, _, Vh = np.linalg.svd(X, full_matrices=False)
        R_hat, T_hat = Vh[-1, :-3].reshape((3, 3), order="F"), Vh[-1, -3:]
        U, s, Vh = np.linalg.svd(R_hat)
        # print(s)
        R = U @ Vh
        sign = 1 if np.linalg.det(R) > 0 else -1
        R *= sign
        T = sign / (np.prod(s) ** (1 / 3)) * T_hat

        assert np.allclose(R @ R.T, np.eye(3)), "error in R"

        R_list.append(R)
        T_list.append(T)

    return R_list, T_list


def _solve_for_depth(x, R_list, T_list):
    """
    x is 2d camera coord pts.

    Returns depth_list.
    """
    x = to_homogeneous(x)  # (m, n, 3)
    depth_list = []
    num_frames, num_pts, _ = x.shape

    for j in range(num_pts):
        x1 = x[0, j, :]
        # num, den = 0, 0
        # for i in range(1, num_frames):
        #     xi_skew = vec2skew(x[i, j, :])
        #     xi_skew_Ti = xi_skew @ T_list[i]
        #     num += xi_skew_Ti @ (xi_skew @ (R_list[i] @ x1))
        #     den += np.linalg.norm(xi_skew_Ti) ** 2

        Mp1 = np.concatenate([vec2skew(x[i, j, :]) @ R_list[i] @ x1 for i in range(1, num_frames)], axis=-1)
        Mp2 = np.concatenate([vec2skew(x[i, j, :]) @ T_list[i] for i in range(1, num_frames)], axis=-1)
        depth = np.linalg.lstsq(Mp2[:, np.newaxis], Mp1[:, np.newaxis], rcond=None)[0]
        # print(depth)

        # depth_list.append(num / den)
        depth_list.append(depth[0][0])

    depth_list = np.array(depth_list)

    return -depth_list
    # return depth_list


def _project_multi_view(x1, depth_list, R_list, T_list):
    """
    x1 = x[0, ...], which is n-by-2.

    Returns
    -------
    x_hat: np.ndarray
        Projected estimated camera coord points, of shape (m, n, 2)
    """
    x1 = to_homogeneous(x1)  # (n, 3)
    num_pts = len(depth_list)
    num_frames = len(R_list)
    x_hat = np.empty((num_frames, num_pts, 3))
    x_hat[0, ...] = x1

    for i in range(1, num_frames):
        # x_hat[i, ...] = depth_list[:, np.newaxis] * x1 @ R_list[i].T + T_list[i]
        x_hat[i, ...] = x1 @ R_list[i].T + T_list[i]

    return project(x_hat)[..., :-1]
