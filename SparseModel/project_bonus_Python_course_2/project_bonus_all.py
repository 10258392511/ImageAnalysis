import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage
from compute_psnr import compute_psnr
from build_dct_unitary_dictionary import build_dct_unitary_dictionary
from dct_image_denoising import dct_image_denoising

#%% Part A: Data Construction

# Read an image
orig_im = np.array(Image.open('foreman.png'))

# Define the blur kernel
h_blur = np.ones((9,9))/81
# Blur the image 
blurred_im = scipy.ndimage.correlate(orig_im, h_blur, mode='reflect')

# Set the global noise standard-deviation
sigma = np.sqrt(2)
# Add noise to the blurry image
y = blurred_im + sigma*np.random.randn(np.shape(orig_im)[0],np.shape(orig_im)[1])

# Compute and print the PSNR of the degraded image
psnr_input = compute_psnr(orig_im, y)
print('PSNR of the blurred and noisy image is %.3f\n\n' % psnr_input)

# Show the original and degraded images
# plt.figure(figsize=[10,3])
# plt.subplot(1,3,1)
# plt.imshow(orig_im,'gray')
# plt.axis('off')
# plt.title('Original')
# plt.subplot(1, 3, 2)
# plt.imshow(blurred_im, "gray")
# plt.axis("off")
# plt.title("Blurry")
# plt.subplot(1,3,3)
# plt.imshow(y,'gray')
# plt.axis('off')
# plt.title('Input: PSNR = %.3f' % psnr_input)
# plt.show()

plt.figure(0,figsize=[10,3])
plt.subplot(1,3,1) 
plt.imshow(orig_im,'gray')
plt.axis('off')
plt.title('Original')
plt.subplot(1,3,2) 
plt.imshow(y,'gray')
plt.axis('off')
plt.title('Input: PSNR = %.3f' % psnr_input)


#%% Part B: Deblurring via Regualrization by Denoising (RED)

# In this part we aim to tackle the image deblurring problem.
# To this end, we will use RED, suggest minimizing
# \min_x 1/(2*sigma**2) * ||H*x - y ||_2**2 + lambda/2 * x'*(x - f(x)),
# where f(x) is a denoising algorithm that operates on x, and the matrix
# H is a known blur kernel.
# We will use the Fixed-Point scheme to minimize the above objective, where
# the update rule is given by
# x_new = [1/sigma**2 H'*H + lambda*I]^(-1) * [1/sigma**2 * H'*y + lambda * f(x)]


# Choose the parameters of the DCT denoiser

# Set the patch dimensions [height, width]
patch_size = 6,6

# TODO: Create a unitary DCT dictionary
# Write your code here D = build_dct_unitary_dictionary( ? )
D = build_dct_unitary_dictionary(patch_size)

# TODO: Set the noise-level in a PATCH for the pursuit
# Write your code here... epsilon = ???;
epsilon = np.sqrt(1.1) * patch_size[0] * sigma

# Set the number of outer FP iterations
K = 50

# Set the number of inner iterations for solving the linear system Az=b
# using Richardson algorithm
T = 50

# Set the step size of each step of the Richardson method
mu = 1

# Set the FP parameter lambda that controls the importance of the prior
lambd = 0.1

# Allocate a vector that stores the PSNR per iteration
psnr_red = np.zeros(K)

# Initialize the solution x with the input image
x = y

# Run the fixed-point algorithm for num_outer_iters
for outer_iter in range(K):
    print(f"current: k = {outer_iter + 1}/{K}")

    # Stage 1:
    # TODO: Compute fx = f(x_prev), by running our DCT image denoising 
    # algorithm
    # Write your code here... fx = dct_image_denoising(?, ?, ?)
    fx = dct_image_denoising(x, D, epsilon)

    # Stage 2:
    # Solve the linear system Az=b, where
    # A = 1/sigma**2*H'H + lambda*I, and b = 1/sigma**2*H'y + lambda*x.
    # This is done using the (iterative) Richardson method,
    # where its update rule is given by
    # z_new = z_old - mu*(A*z_old - b)

    # TODO: Initialize the the previous solution 'z_old' of the Richardson
    # method to be the denoised image 'fx'
    # Write your code here... z_old = ???
    z_old = fx

    # Compute b = 1/sigma**2*H'y + lambda*x. The multiplication
    # by H or H.T is done by filtering the image we operate on with
    # 'h_blur'. Notice that H is symmetric, therefore the multiplication by
    # H or H' is done in the very same way.
    HT_y = scipy.ndimage.correlate(y, h_blur, mode='reflect')
    b = (1/sigma**2) * HT_y + lambd*fx

    # Repeat z_new = z_old - step_size*(A*z_old - b) for num_inner_iters
    for inner_iter in range(T):

        # TODO: Compute H*z_old by convolving the image 'z_old' with the 
        # filter 'h_blur'
        # Write your code here... H_z_old = ???
        H_z_old = scipy.ndimage.correlate(z_old, h_blur, mode="reflect")

        # TODO: Compute H'*H_z_old by convolving the image 'H_z_old' with 
        # the filter 'h_blur' (in our case H is symmetric)
        # Write your code here... HTH_z_old = ???
        HTH_z_old = scipy.ndimage.correlate(H_z_old, h_blur, mode="reflect")

        # TODO: Compute the image A*z_old which is nothing but
        # 1/sigma^2*H'*H*z_old + lambda*z_old
        # Write your code here... A_z_old = ???
        A_z_old = (1 / sigma ** 2) * HTH_z_old + lambd * z_old

        # TODO: Compute z_new = z_old - step_size*(A*z_old - b)
        # Write your code here... z_new = ???
        z_new = z_old - mu * (A_z_old - b)

        # TODO: Update z_old to be the new z
        # Write your code here... z_old = ???
        z_old = z_new

    # TODO: Update x to be z_new
    # Write your code here... x = ???
    x = z_new

    # Compute the PSNR of the restored image
    psnr_red[outer_iter] = compute_psnr(orig_im, x)
    print('RED: Fixed-Point Iter %02d, PSNR %.3f\n' % (outer_iter+1, psnr_red[outer_iter]))


#%% Present the results

# Show the restored image obtained by RED
plt.figure(0)
plt.subplot(1,3,3) 
plt.imshow(x, 'gray')
plt.axis('off')
plt.title('RED: PSNR = %.3f' % psnr_red[-1])

# Plot the PSNR of the RED as a function of the iterations
plt.figure(1) 
plt.plot(np.arange(K), psnr_red)
plt.grid(True)
plt.title('RED: PSNR vs. Iterations')
plt.ylabel('PSNR')
plt.xlabel('Fixed-Point Iteration')

plt.show()
