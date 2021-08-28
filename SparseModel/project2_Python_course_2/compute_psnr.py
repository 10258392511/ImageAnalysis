import numpy as np
import math

def compute_psnr(y_original, y_estimated):
    
    # COMPUTE_PSNR Computes the PSNR between two images
    #
    # Input:
    #  y_original  - The original image
    #  y_estimated - The estimated image
    #
    # Output:
    #  psnr_val - The Peak Signal to Noise Ratio (PSNR) score

    y_original = np.reshape(y_original,(-1))
    y_estimated = np.reshape(y_estimated,(-1))

    # TODO: Compute the dynamic range
    # Write your code here... dynamic_range = ????
    dynamic_range = 255.0

    # TODO: Compute the Mean Squared Error (MSE)
    # Write your code here... mse_val = ????
    mse_val = (1/len(y_original)) * np.sum((y_original - y_estimated)**2)

    # TODO: Compute the PSNR
    # Write your code here... psnr_val = ????
    psnr_val = 10*math.log10(dynamic_range**2 / mse_val)

    return psnr_val

