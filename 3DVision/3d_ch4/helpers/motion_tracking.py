import numpy as np
import matplotlib.pyplot as plt
import cv2

from .harris_corner import compute_features


def optical_flow_with_harris_corners(img_prev: np.ndarray, img_next: np.ndarray,
                                     win_size=11, num_features=20, if_show=True, **kwargs):
    """
    img_prev and img_next should be [0, 1] ranged and of dtype np.float32
    """
    pts_x, pts_y, st = compute_features(img_prev, win_size=win_size, if_show=True, max_num_pts=num_features,
                                        if_return_st=True, markerSize=7, thickness=1)
    Ix, Iy, Ixx, Iyy, Ixy = st
    It = img_next - img_prev
    Ixt = cv2.GaussianBlur(Ix * It, ksize=(win_size, win_size), sigmaX=0)
    Iyt = cv2.GaussianBlur(Iy * It, ksize=(win_size, win_size), sigmaX=0)

    ux, uy = np.zeros_like(pts_x, dtype=np.float32), np.zeros_like(pts_y, dtype=np.float32)

    for i, (x, y) in enumerate(zip(pts_x, pts_y)):
        A = np.array([[Ixx[y, x], Ixy[y, x]],
                      [Ixy[y, x], Iyy[y, x]]])
        b = -np.array([Ixt[y, x], Iyt[y, x]])

        u = np.linalg.inv(A) @ b
        ux[i], uy[i] = u

    if if_show:
        fig, axis = plt.subplots()
        axis.imshow(img_prev, cmap="gray")
        axis.quiver(pts_x, pts_y, ux, uy, color="red", units="dots", width=1, angles="xy")
        plt.show()

    return pts_x, pts_y, ux, uy


def extract_line_features(img: np.ndarray, if_show=False,
                          if_save_lines=False, lines_path=None, lines_path_hough_P=None, **kwargs):
    """
    Parameters
    ----------
    img: np.ndarray
        Grayscale image in [0, 1], np.float32
    if_show: bool
    if_save_lines: bool
    lines_path, lines_path_hough_P: str
    """
    win_size = kwargs.get("ksize", 5)
    img_blur = cv2.GaussianBlur(img, ksize=(win_size, win_size), sigmaX=0)
    th1, th2 = kwargs.get("th1", 50), kwargs.get("th2", 150)
    edges = cv2.Canny((img_blur * 255).astype(np.uint8), th1, th2, edges=None, apertureSize=3)  # binary, 8-bit

    if if_show:
        plt.imshow(edges, cmap="gray")
        plt.colorbar()
        plt.show()

    threshold = kwargs.get("threshold", 300)

    if if_show:
        min_line_length = kwargs.get("minLineLength", 100)
        max_line_gap = kwargs.get("maxLineGap", 50)
        end_pts = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=threshold,
                                  minLineLength=min_line_length, maxLineGap=max_line_gap)  # (N, 1, 4)
        end_pts = end_pts.squeeze()  # (N, 4)
        img_c = draw_lines(img, end_pts)
        if if_save_lines:
            assert lines_path is not None
            cv2.imwrite(lines_path_hough_P, img_c)

    # (N, 1, 2), theta: in rad
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=threshold, srn=0, stn=0)
    lines = lines.squeeze()  # (N, 2)

    if if_show:
        end_pts = convert_rho_theta_to_xy(img.shape, lines[:, 0], lines[:, 1])
        img_c = draw_lines(img, end_pts)
        if if_save_lines:
            assert lines_path is not None
            cv2.imwrite(lines_path, img_c)

    return lines


def convert_rho_theta_to_xy(img_shape: tuple, rho: np.ndarray, theta:np.ndarray):
    """
    Using the formula: rho = x * cos(theta) + y * sin(theta), where theta is in [0, pi]

    Returns
    -------
    end_pts: np.ndarray
        [[x_start, y_start, x_end, y_end]...]
    """
    H, W = img_shape
    end_pts = np.empty((len(rho), 4))

    sine = np.sin(theta)
    cosine = np.cos(theta)
    x0 = rho * cosine
    y0 = rho * sine
    x0, y0 = x0, y0
    shift_x, shift_y = W * 10, H * 10
    # line direction: [-sin(theta), cos(theta)]
    x1 = x0 + shift_x * sine
    y1 = y0 - shift_y * cosine
    x2 = x0 - shift_x * sine
    y2 = y0 + shift_y * cosine

    end_pts = np.concatenate([x1.reshape((-1, 1)), y1.reshape((-1, 1)), x2.reshape((-1, 1)), y2.reshape((-1, 1))],
                             axis=1)

    return end_pts.astype(int)


def draw_lines(img: np.ndarray, end_pts: np.ndarray):
    """
    Parameters
    ----------
    img: np.ndarray
        Grayscale image to draw on
    end_pts: np.ndarray
        [[x_start, y_start, x_end, y_end]...]
    """
    img_c = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 255).astype(np.uint8)
    for x1, y1, x2, y2 in end_pts:
        img_c = cv2.line(img_c, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
    plt.show()

    return img_c
