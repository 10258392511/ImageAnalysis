import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from compute_psnr import compute_psnr
from build_dct_unitary_dictionary import build_dct_unitary_dictionary
from show_dictionary import show_dictionary
from dct_image_denoising import dct_image_denoising
from unitary_image_denoising import unitary_image_denoising

#%% Part A: Data construction
 
# Read an image
orig_im = np.array(Image.open('barbara.png'))
# Crop the image
orig_im = orig_im[:256, 251:251+256]

# TODO: Set the seed for the random generator
# Write your code here... seed = ???
seed = 10

 
# Set a fixed random seed to reproduce the results
np.random.seed(seed)
 
# TODO: Set the standard-deviation of the Gaussian noise
# Write your code here... sigma = ???
sigma = 20

# TODO: Add noise to the input image
# Write your code here... noisy_im = ???
noisy_im = orig_im + np.random.randn(*orig_im.shape) * sigma


# Compute the PSNR of the noisy image and print its value
psnr_noisy = compute_psnr(orig_im, noisy_im)
print('PSNR of the noisy image is %.3f\n\n\n' % psnr_noisy)
 
# Show the original and noisy images
plt.figure(0, figsize=[10,10])
plt.subplot(2,2,1)
plt.imshow(orig_im,'gray')
plt.axis('off')
plt.title('Original')
plt.subplot(2,2,2)
plt.imshow(noisy_im,'gray')
plt.axis('off')
plt.title('Noisy PSNR = %.3f' % psnr_noisy)
 
#%% Part B: Patch-Based Image Denoising
 
# Set the patch dimensions [height, width]
patch_size = (10,10)
 
# TODO: Create a unitary DCT dictionary
# Write your code here... D_DCT = build_dct_unitary_dictionary( ? )
D_DCT = build_dct_unitary_dictionary(patch_size)

# Show the unitary DCT dictionary
plt.figure(1,figsize=[8,4])
plt.subplot(1, 2, 1)
show_dictionary(D_DCT)
plt.title('Unitary DCT Dictionary') 
 
#%% Part B-1: Unitary DCT denoising
 
# TODO: Set the noise-level of a PATCH for the pursuit,
# multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
# Write your code here... epsilon_dct = ???
epsilon_dct = np.sqrt(1.1) * patch_size[0] * sigma

# Denoise the input image via the DCT transform
est_dct = dct_image_denoising(noisy_im, D_DCT, epsilon_dct)
 
# Compute and print the PSNR
psnr_dct = compute_psnr(orig_im, est_dct)
print('DCT dictionary: PSNR %.3f\n\n\n' % psnr_dct)
 
# Show the resulting image
plt.figure(0)
plt.subplot(2,2,3)
plt.imshow(est_dct,'gray')
plt.axis('off')
plt.title('DCT: $\epsilon$ = %.3f PSNR = %.3f' % (epsilon_dct,psnr_dct))
 
#%% Part B-2: Unitary dictionary learning for image denoising
 
# TODO: Set the number of training iterations for the learning algorithm
# Write your code here... T = ???
T = 20

# TODO: Set the noise-level of a PATCH for the pursuit,
# multiplied by some constant (e.g. sqrt(1.1)) to improve the restoration
# Write your code here... epsilon_learning = ???;
epsilon_learning = np.sqrt(1.1) * patch_size[0] * sigma
 
# Denoise the image using unitary dictionary learning
est_learning, D_learned, mean_error, mean_cardinality = unitary_image_denoising(\
    noisy_im, D_DCT, T, epsilon_learning)
 
#% Show the dictionary
plt.figure(1)
plt.subplot(1,2,2)
show_dictionary(D_learned) 
plt.title('Learned Unitary Dictioanry')
 
#% Show the representation error and the cardinality as a function of the
#% learning iterations
plt.figure(2, figsize=[8,4])
plt.subplot(1,2,1) 
plt.plot(np.arange(T), mean_error, linewidth=2.0)
plt.ylim((0, 5*int(np.sqrt(patch_size[0]*patch_size[1]))*sigma))
plt.ylabel('Average Representation Error')
plt.xlabel('Learning Iteration')
plt.subplot(1,2,2)
plt.plot(np.arange(T),mean_cardinality, linewidth=2.0)
plt.ylabel('Average Number of Non-Zeros')
plt.xlabel('Learning Iteration')
 
#% Compute and print the PSNR
psnr_unitary = compute_psnr(orig_im, est_learning)
print('Unitary dictionary: PSNR %.3f\n\n\n' % psnr_unitary)
 
#% Show the results
plt.figure(0)
plt.subplot(2,2,4)
plt.imshow(est_learning,'gray')
plt.axis('off')
plt.title('Unitary: $\epsilon$ = %.3f PSNR = %.3f' % (epsilon_learning, psnr_unitary))
 
#%% SOS Boosting
 
# TODO: Set the strengthening factor
# Write your code here... rho = ???
rho = 1

# TODO: Set the noise-level in a PATCH for the pursuit. 
# A common choice is a slightly higher noise-level than the one set 
# in epsilon_learning, e.g. 1.1*epsilon_learning;
# Write your code here... epsilon_sos = ???
epsilon_sos = 1.1 * epsilon_learning

# TODO: Init D_sos to be D_learned 
# Write your code here... D_sos = ???
D_sos = D_learned

# TODO: Compute the signal-strengthen image by adding to 'noisy_im' the 
# denoised image 'est_learning', multiplied by an appropriate 'rho'
# Write your code here... s_im = ???
s_im = noisy_im + rho * est_learning

# TODO: Operate the denoiser on the signal-strengthened image
# Write your code here... est_learning_sos = unitary_image_denoising(?, ?, ?, ?)
est_learning_sos = unitary_image_denoising(s_im, D_sos, T, epsilon_sos)[0]

# TODO: Subtract from 'est_learning_sos' image the previous estimate
# 'est_learning', multiplied by the strengthening factor
# Write your code here... est_learning_sos = ???
est_learning_sos -= rho * est_learning

# Compute and print the PSNR
psnr_unitary_sos = compute_psnr(orig_im, est_learning_sos)
print('SOS Boosting: epsilon %.3f, rho %.2f, PSNR %.3f\n\n\n' % \
      (epsilon_sos,rho,psnr_unitary_sos))


fig, axis = plt.subplots()
axis.imshow(est_learning_sos,'gray')
axis.set_title(r'SOS Boosting: $\epsilon$ = %.3f, $\rho$ = %.3f, PSNR %.3f' % (epsilon_sos, rho, psnr_unitary_sos))
plt.axis("off")

plt.show()
