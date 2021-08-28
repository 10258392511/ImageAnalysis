import numpy as np
import cv2
import matplotlib.pyplot as plt


def display_image(path: str, if_resize=True, resize_shape=(512, 512)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if if_resize:
        img = cv2.resize(img, dst=None, dsize=resize_shape, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img", img)
    key = cv2.waitKey(0)
    if key == ord("q"):
        cv2.destroyAllWindows()


def point_tracker(path: str, if_resize=True, resize_shape=(512, 512)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if if_resize:
        img = cv2.resize(img, dst=None, dsize=resize_shape, interpolation=cv2.INTER_LINEAR)

    def callback(event, x, y, flags, params: dict):
        tmp_file = params.get("tmp_file", "./tmp_file.txt")
        if event == cv2.EVENT_LBUTTONDOWN:
            with open(tmp_file, "a") as wf:
                wf.write(f"{x},{y}\n")

    window = cv2.namedWindow("img")
    cv2.setMouseCallback("img", callback, param={})

    while True:
        cv2.imshow("img", img)
        if cv2.waitKey(0) == ord("q"):
            break
        cv2.destroyAllWindows()


def read_and_resize(path, resize_shape=(512, 512), if_display=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=resize_shape, dst=None)
    if if_display:
        plt.imshow(img, cmap="gray")
        plt.show()

    return img


def warp_homography_remap(src, H_mat):
    H, W = src.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    grid = np.concatenate([xx.ravel().reshape((1, -1)), yy.ravel().reshape((1, -1))], axis=0)  # (2, N)
    grid = np.concatenate([grid, np.ones((1, W * H))], axis=0)  # (3, N)
    src_map = np.linalg.inv(H_mat) @ grid
    src_map /= src_map[-1:, :]
    src_map = src_map.astype(np.float32)
    # map_x, map_y begins from (0, 0) and follows raster scan order
    dest = cv2.remap(src, src_map[0, :].reshape((H, W)), src_map[1, :].reshape((H, W)), interpolation=cv2.INTER_LINEAR)

    return dest


def sift_homography(img1, img2, if_draw_keypoints=False, if_draw_matches=False):
    sift = cv2.xfeatures2d.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(img1, None)
    kpts2, desc2 = sift.detectAndCompute(img2, None)

    if if_draw_keypoints:
        img1_kpts = cv2.drawKeypoints(img1, kpts1, None)
        img2_kpts = cv2.drawKeypoints(img2, kpts2, None)
        plt.imshow(img1_kpts, cmap="gray")
        plt.imshow(img2_kpts, cmap="gray")
        plt.show()

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(desc1, desc2, k=2)  # query, train

    ratio = 0.7
    sel_matches = []
    for m1, m2 in matches:
        if m1.distance < m2.distance * ratio:
            sel_matches.append(m1)

    if if_draw_matches:
        img_matches = cv2.drawMatches(img1, kpts1, img2, kpts2, sel_matches, None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches, cmap="gray")
        plt.show()

    img1_features, img2_features = [], []
    for match in sel_matches:
        img1_features.append(kpts1[match.queryIdx].pt)
        img2_features.append(kpts2[match.trainIdx].pt)
    img1_features = np.array(img1_features)
    img2_features = np.array(img2_features)

    H, mask = cv2.findHomography(img1_features, img2_features, method=cv2.RANSAC, ransacReprojThreshold=3)
    mask = mask.ravel()
    sel_matches_final = [sel_matches[i] for i in range(len(mask)) if mask[i]]

    if if_draw_matches:
        img_matches = cv2.drawMatches(img1, kpts1, img2, kpts2, sel_matches_final, None,
                                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches, cmap="gray")
        plt.show()

    return H


def extract_first_frames(path, num_frs=2):
    cap = cv2.VideoCapture(path)
    frames = []

    count = 0
    while cap.isOpened():
        if count == num_frs:
            break

        ret, frame = cap.read()
        count += 1

        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break

    return frames


def generate_grid(img: np.ndarray):
    H, W = img.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    return np.concatenate([xx.reshape((-1, 1)), yy.reshape((-1, 1))], axis=1)


def mark_kpts(img, kpts):
    kpts = kpts.reshape((-1, 2))
    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for kpt in kpts:
        img_c = cv2.drawMarker(img_c, tuple(map(int, kpt)), color=(0, 0, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=20, thickness=5, line_type=cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
    plt.show()


def compute_optical_flow(prev_img, next_img, if_visualize=False, if_dense=False):
    """
    Returns
    -------
    sparse features:
        kpts: np.ndarray
            Of shape (N, 1, 2), np.float32
        flow: np.ndarray
            Of shape (N, 1, 2), np.float32

    dense features:
        xx, yy: np.ndarray
            Of dtype np.float32
        flow: np.ndarray
            Of shape (H, W, 2), np.float32
    """
    if not if_dense:
        # use sparse LK
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        kpts = cv2.goodFeaturesToTrack(prev_img, **feature_params)

        if if_visualize:
            mark_kpts(prev_img, kpts)

        next_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_img, next_img, kpts, nextPts=None,
                                                           **lk_params)  # kpts: (N, 1, 2)
        motion = next_pts - kpts  # (N, 1, 2)

        if if_visualize:
            fig, axis = plt.subplots(figsize=(7.2, 4.8))
            axis.imshow(prev_img, cmap="gray")
            X, Y = kpts[:, 0, 0], kpts[:, 0, 1]
            U, V = motion[:, 0, 0], motion[:, 0, 1]
            axis.quiver(X, Y, U, V, color="red", units="dots", width=1, angles="xy")
            plt.show()

        return kpts, motion

    else:
        dense_params = dict(pyr_scale=0.5, levels=3, winsize=15,
                            iterations=5, poly_n=5, poly_sigma=1.2, flags=0)
        flow = cv2.calcOpticalFlowFarneback(prev_img, next_img, flow=None, **dense_params)  # np.float32, (H, W, 2)

        H, W = prev_img.shape
        if if_visualize:
            hsv = np.zeros((H, W, 3), dtype=prev_img.dtype)  # np.uint8
            hsv[..., 1] = 255
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2  # opencv has H value in {0, 1, ..., 179}
            hsv[..., 2] = cv2.normalize(mag, dst=None, alpha=0, beta=255,
                                        norm_type=cv2.NORM_MINMAX)  # special norm_type for min-max normalization
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            plt.show()

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))

        return xx.astype(np.float32), yy.astype(np.float32), flow


if __name__ == '__main__':
    path = "../images/homography_2.png"
    # display_image(path)
    for path in [f"../images/homography_{i}.png" for i in range(1, 3)]:
        point_tracker(path)
