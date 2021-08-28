import numpy as np
from compute_stat import compute_stat
from im2col import im2col
from col2im import col2im
from batch_thresholding import batch_thresholding


def dct_image_denoising(noisy_im, D_DCT, epsilon):
    # DCT_IMAGE_DENOISING Denoise an image via the DCT transform
    # 
    # Inputs:
    #   noisy_im - The input noisy image
    #   D_DCT    - A column-normalized DCT dictionary
    #   epsilon  - The noise-level in a PATCH, 
    #              used as the stopping criterion of the pursuit
    #
    # Output:
    #  est_dct - The denoised image
    #
    
    # TODO: Get the patch size [height, width] from D_DCT
    # Write your code here... patch_size = ???
    side_DCT = int(np.ceil((np.sqrt(D_DCT.shape[0]))))
    patch_size = (side_DCT, side_DCT)

    # Divide the noisy image into fully overlapping patches
    patches = im2col(noisy_im, patch_size, stepsize=1)  ## (100, N)
 
    # TODO: Step 1: Compute the representation of each noisy patch using the 
    # Thresholding pursuit
    # Write your code here... [est_patches, est_coeffs] = batch_thresholding(?, ?, ?)
    est_patches, est_coeffs = batch_thresholding(D_DCT, patches, epsilon)

    # TODO: Step 2: Reconstruct the image using 'col_to_im' function
    # Write your code here... est_dct = col2im(?, ?, ?)
    est_dct = col2im(est_patches, patch_size, noisy_im.shape)

    # Compute and display the statistics
    print('DCT dictionary: ', end=' ')
    compute_stat(est_patches, patches, est_coeffs)
 
    return  est_dct
