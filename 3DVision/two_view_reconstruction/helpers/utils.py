import numpy as np
import matplotlib.pyplot as plt
import cv2


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


def to_homogenous(pt: np.ndarray):
    assert len(pt) in [2, 3], "Must be 2D or 3D point"

    return np.append(pt, [1], axis=-1)


def project(pt):
    assert len(pt) == 3, "Must be 3D point"

    return pt / pt[2]


def compute_3d(pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, T: np.ndarray):
    """
    pts1, pts2 are array of 2d camera coordinate points.

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
        x1, x2 = to_homogenous(pts1[i, :]), to_homogenous(pts2[i, :])
        x2_skew = vec2skew(x2)
        M[3 * i : 3 * i + 3, i] = x2_skew @ (R @ x1)
        M[3 * i : 3 * i + 3, -1] = x2_skew @ T

    _, _, V_h = np.linalg.svd(M, full_matrices=False)
    cand = V_h[-1, :]
    if cand[0] < 0:
       cand = -cand
    lamda, gamma = cand[:-1], cand[-1]

    X = np.empty((num_pts, 3))
    for i in range(num_pts):
        x1 = to_homogenous(pts1[i, :])
        X[i, :] = lamda[i] * R @ x1 + gamma * T

    return X, lamda, gamma


def _check_essential(pts1, pts2, R, T):
    """
    pts1, pts2 are 2d camera coordinate points.
    """
    X, lamda1, gamma = compute_3d(pts1, pts2, R, T)
    # print(f"lambda1:\n{lamda1}")
    # print(f"gamma: {gamma}")
    # print(f"X:\n{X}")
    # print("-" * 50)
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
        x1, x2 = to_homogenous(pts1[i, :]), to_homogenous(pts2[i, :])
        X[i, :] = np.kron(x1, x2)

    # print(X)
    _, _, V_h = np.linalg.svd(X, full_matrices=False)
    E = V_h[-1, :].reshape((3, 3), order="F")

    for E_cand in [E, -E]:
        U, s, V_h = np.linalg.svd(E_cand, full_matrices=True)
        # print(s)

        for R_z in [np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                    np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])]:
            R = U @ R_z.T @ V_h
            T_skew = U @ R_z @ np.diag([1, 1, 0]) @ U.T
            T = skew2vec(T_skew)
            # print(f"det(R): {np.linalg.det(R)}")
            # print(f"R:\n{R}\nT:\n{T}")
            if _check_essential(pts1, pts2, R, T) and np.allclose(np.linalg.det(R), 1):
                return E_cand, R, T

    raise ValueError("No solution")


def compute_epipolar_line(pts1, E, img2, K=None):
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
        l = F @ to_homogenous(pts1[i, :])
        x1, x2 = 0, W - 1
        y1, y2 = -(l[2] + l[0] * x1) / l[1], -(l[2] + l[0] * x2) / l[1]
        # print(f"{(x1, y1), (x2, y2)}")
        cv2.line(img2_c, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(img2_c, cv2.COLOR_BGR2RGB), cmap="gray")
    plt.show()


def project_3d(X, R, T, K):
    """
    X is 3d world / camera points.
    """
    pi = K @ np.block([R, T[:, np.newaxis]])
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    # print(pi)
    x_temp = (pi @ X.T).T

    return x_temp / x_temp[:, -1:]


def compute_homography(pts1: np.ndarray, pts2: np.ndarray):
    """
    pts1, pts2 are 2d camera coordinate pts whose 3d correspondences are on a plane.
    """
    assert len(pts1) == len(pts2), "Number of points must match"
    num_pts = pts1.shape[0]
    X = np.empty((3 * num_pts, 9))

    for i in range(num_pts):
        x1, x2 = to_homogenous(pts1[i, :]), to_homogenous(pts2[i, :])
        x2_skew = vec2skew(x2)
        X[i * 3 : i * 3 + 3] = np.kron(x1, x2_skew)

    _, s, V_h = np.linalg.svd(X, full_matrices=False)
    H = V_h[-1, :].reshape((3, 3), order="F")
    sign = to_homogenous(pts1[0, :]) @ (H @ to_homogenous(pts2[0, :]))
    if sign < 0:
        H = -H

    # recover R, T and N
    _, s, V_h = np.linalg.svd(H)
    H /= s[1]
    s /= s[1]
    v1, v2, v3 = V_h[0, :], V_h[1, :], V_h[2, :]
    u1 = (np.sqrt(1 - s[2] ** 2) * v1 + np.sqrt(s[0] ** 2 - 1) * v3) / np.sqrt(s[0] ** 2 - s[2] ** 2)
    u2 = (np.sqrt(1 - s[2] ** 2) * v1 - np.sqrt(s[0] ** 2 - 1) * v3) / np.sqrt(s[0] ** 2 - s[2] ** 2)
    v2_skew = vec2skew(v2)
    Hv2_skew = vec2skew(H @ v2)
    U1 = np.concatenate([v2.reshape(-1, 1), u1.reshape(-1, 1), (v2_skew @ u1).reshape(-1, 1)], axis=1)
    U2 = np.concatenate([v2.reshape(-1, 1), u2.reshape(-1, 1), (v2_skew @ u2).reshape(-1, 1)], axis=1)
    W1 = np.concatenate([(H @ v2).reshape(-1, 1), (H @ u1).reshape(-1, 1), (Hv2_skew @ (H @ u1)).reshape(-1, 1)],
                        axis=1)
    W2 = np.concatenate([(H @ v2).reshape(-1, 1), (H @ u2).reshape(-1, 1), (Hv2_skew @ (H @ u2)).reshape(-1, 1)],
                        axis=1)

    N1 = v2_skew @ u1
    sign1 = 1 if N1[2] > 0 else -1
    N1 *= sign1
    R1 = W1 @ U1.T
    T1 = (H - R1) @ N1

    N2 = v2_skew @ u2
    sign2 = 1 if N2[2] > 0 else 0-1
    N2 *= sign2
    R2 = W2 @ U2.T
    T2 = (H - R2) @ N2

    return [(H, R1, T1, N1), (H, R2, T2, N2)]



if __name__ == '__main__':
    # read in two images and mark points
    def adjust_text_pos(x, y, img_size: tuple, x_offset=50, y_offset=50):
        H, W = img_size[:2]
        if x < 0:
            x += x_offset
        if x + x_offset >= W - 1:
            x -= x_offset
        if y < 0:
            y += y_offset
        if y + y_offset >= H - 1:
            y -= y_offset

        return x, y

    def on_mouse(event, x, y, flags, params):
        img = params.get("img")
        filename = params.get("log", None)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            x_pos, y_pos = adjust_text_pos(x, y, img.shape)
            cv2.putText(img, f"({x}, {y})", (x_pos, y_pos), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
            cv2.imshow("img", img)
            if filename is not None:
                with open(filename, "a") as wf:
                    wf.write(f"({x}, {y})\n")


    path1, path2 = "../images/desk_W.png", "../images/desk_C.png"
    log_file = "../images/log.txt"
    with open(log_file, "w") as wf:
        pass

    for path in [path1, path2]:
        delimiter_ind = path.find(".png")
        new_path = path[:delimiter_ind] + "_orig" + path[delimiter_ind:]

        img = read_and_resize(path, fx=0.2, fy=0.2, if_save=True, new_path=new_path)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("img", on_mouse, param=dict(img=img_color, log=log_file))
        cv2.imshow("img", img_color)
        # print(img.shape)
        key = cv2.waitKey(0)

        if key == ord("s"):
            delimiter_ind = path.find(".png")
            save_path = path[:delimiter_ind] + "_out" + path[delimiter_ind:]
            print(img_color.shape)
            # plt.imshow(img_color)
            # plt.show()
            cv2.imwrite(save_path, (img_color * 255).astype(np.uint8))
            cv2.destroyAllWindows()

        if key == ord("q"):
            cv2.destroyAllWindows()
