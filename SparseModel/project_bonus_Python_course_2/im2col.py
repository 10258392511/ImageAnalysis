import numpy as np

def im2col(A, patch_size, stepsize=1):
    # Parameters
    M,N = A.shape
    col_extent = N - patch_size[1] + 1
    row_extent = M - patch_size[0] + 1

    # Get Starting block indices
    start_idx = np.arange(patch_size[0])[:, None]*N + np.arange(patch_size[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel()[::stepsize])