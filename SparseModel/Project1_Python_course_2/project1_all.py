import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import view_as_windows as viewW
from build_dct_unitary_dictionary import build_dct_unitary_dictionary
from show_dictionary import show_dictionary
from batch_thresholding import batch_thresholding
from compute_stat import compute_stat
from unitary_dictionary_learning import unitary_dictionary_learning

#%% Part A: Data Construction and Parameter-Setting

# Read an image
im = np.array(Image.open('barbara.png'))

# Show the image
plt.figure(0)
plt.imshow(im,'gray') 
plt.title('Original image')
plt.show()

# Patch dimensions [height, width]
patch_size = [6,6]

# TODO: Divide the image into FULLY overlapping patches using skimage viewW function
# Write your code here... all_patches = ???
patches_block = viewW(im, patch_size)
all_patches = patches_block.reshape((-1, np.prod(patch_size)))  # (N, 36)

# Number of patches to train on
num_train_patches = 10000
# Number of patches to test on
num_test_patches = 5000

# TODO: Set the seed for the random generator
# Write your code here... seed = ???
seed = 10

# Set a fixed random seed to reproduce the results
np.random.seed(seed)

# TODO: Create a training set by choosing a random subset 
# of 'num_train_patches' taken from 'all_patches'
# Write your code here... train_patches = ???
inds_selected = np.random.randint(0, all_patches.shape[0], (num_train_patches + num_test_patches,))
selected_cols = all_patches[inds_selected, :]
train_patches = all_patches[:num_train_patches, :].T  # (36, N_tr)

# TODO: Create a test set by choosing another random subset 
# of 'num_test_patches' taken from the remaining patches
# Write your code here... test_patches = ???
test_patches = all_patches[num_train_patches : num_train_patches + num_test_patches, :].T  # (36, N_te)

# TODO: Initialize the dictionary
# Write your code here... D_DCT = build_dct_unitary_dictionary( ? )
D_DCT = build_dct_unitary_dictionary(patch_size)

# Show the unitary DCT dictionary
plt.figure(1)
plt.subplot(1, 2, 1)
show_dictionary(D_DCT)
plt.title('Unitary DCT Dictionary')

# TODO: Set K - the cardinality of the solution.
# This will serve us later as the stopping criterion of the pursuit
# Write your code here... K = ???
K = 4

#%% Part B: Compute the Representation Error Obtained by the DCT Dictionary
 
# Compute the representation of each patch that belongs to the training set using Thresholding
est_train_patches_dct, est_train_coeffs_dct = batch_thresholding(D_DCT, train_patches, K)

# Compute the representation of each patch that belongs to the test set using Thresholding
est_test_patches_dct, est_test_coeffs_dct = batch_thresholding(D_DCT, test_patches, K)
 
# Compute and display the statistics
print('\n\nDCT dictionary: Training set, ')
compute_stat(est_train_patches_dct, train_patches, est_train_coeffs_dct)
print('DCT dictionary: Testing  set, ')
compute_stat(est_test_patches_dct, test_patches, est_test_coeffs_dct)
print('\n\n')
 
#%% Part C: Procrustes Dictionary Learning

# TODO: Set the number of training iterations
# Write your code here... T = ???
T = 20

# Train a dictionary via Procrustes analysis
D_learned, mean_error, mean_cardinality = unitary_dictionary_learning(train_patches, D_DCT, T, K)

# Show the dictionary
plt.figure(1)
plt.subplot(1, 2, 2)
show_dictionary(D_learned)
plt.title("Learned Unitary Dictionary")
plt.show()

# Show the representation error and the cardinality as a function of the learning iterations
plt.figure(2)
plt.subplot(1, 2, 1) 
plt.plot(np.arange(T), mean_error, linewidth=2.0)
plt.ylabel("Average Representation Error")
plt.xlabel("Learning Iteration")
plt.subplot(1, 2, 2) 
plt.plot(np.arange(T), mean_cardinality, linewidth=2.0)
plt.ylabel('Average Number of Non-Zeros') 
plt.ylim((K-1, K+1))
plt.xlabel('Learning Iteration')
plt.show()

# Compute the representation of each signal that belong to the training set using Thresholding
est_train_patches_learning, est_train_coeffs_learning = batch_thresholding(D_learned, train_patches, K)

# Compute the representation of each signal that belong to the testing set using Thresholding
est_test_patches_learning, est_test_coeffs_learning = batch_thresholding(D_learned, test_patches, K)
 
# Compute and display the statistics
print('\n\nLearned dictionary: Training set, ')
compute_stat(est_train_patches_learning, train_patches, est_train_coeffs_learning)
print('Learned dictionary: Testing  set, ')
compute_stat(est_test_patches_learning, test_patches, est_test_coeffs_learning)
print('\n\n')
