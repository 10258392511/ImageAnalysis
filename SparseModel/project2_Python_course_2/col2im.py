import numpy as np


def col2im(patches, patch_size, im_size):
    # COL_TO_IM Rearrange matrix columns into an image of size MXN
    #
    # Inputs:
    #  patches - A matrix of size p * q, where p is the patch flatten size (height * width = m * n), and q is number of patches.
    #  patch_size - The size of the patch [height width] = [m n]
    #  im_size    - The size of the image we aim to build [height width] = [M N]
    #
    # Output:
    #  im - The reconstructed image, computed by returning the patches in
    #       'patches' to their original locations, followed by a
    #       patch-averaging over the overlaps

    num_im = np.zeros((im_size[0], im_size[1]))
    denom_im = np.zeros((im_size[0], im_size[1]))

    for i in range(im_size[0] - patch_size[0] + 1):
        for j in range(im_size[1] - patch_size[1] + 1):
            # rebuild current patch
            num_of_curr_patch = i * (im_size[1] - patch_size[1] + 1) + (j + 1)
            last_row = i + patch_size[0]
            last_col = j + patch_size[1]
            curr_patch = patches[:, num_of_curr_patch - 1]
            curr_patch = np.reshape(curr_patch, (patch_size[0], patch_size[1]))

            # update 'num_im' and 'denom_im' w.r.t. 'curr_patch'
            num_im[i:last_row, j:last_col] = num_im[i:last_row, j:last_col] + curr_patch
            denom_im[i:last_row, j:last_col] = denom_im[i:last_row, j:last_col] + np.ones(curr_patch.shape)


    # Averaging
    im = num_im / denom_im

    return im