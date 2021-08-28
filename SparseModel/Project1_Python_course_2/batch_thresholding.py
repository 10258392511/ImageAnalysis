import numpy as np


def batch_thresholding(D, Y, K):
    
    # BATCH_THRESHOLDING Solve the pursuit problem via Thresholding using 
    # a fixed cardinality as the stopping criterion.
    # 
    # Solves the following problem:
    #   min_{alpha_i} \sum_i ||y_i - D alpha_i||_2^2 
    #                   s.t. || alpha_i ||_0 = K for all i,
    # where D is a dictionary of size n X n, y_i are the input signals of
    # length n, and K stands for the number of nonzeros allows per each alpha_i
    # 
    # The above can be equally written as follows:
    #   min_{A} \sum_i ||Y - DA||_F^2 
    #             s.t. || alpha_i ||_0 = K for all i,
    # where Y is a matrix that contains the y_i patches as its columns.
    # Similarly, and the matrix A contains the representations alpha_i 
    # as its columns
    #
    # The solution is returned in the matrix A, along with the denoised signals
    # given by  X = DA.

    # TODO: Compute the inner products between the dictionary atoms and the
    # input patches (hint: the result should be a matrix of size n X N)
    # Write your code here... inner_products = ???
    inner_products = D.T @ Y  # (n^2, N)

    # TODO: Compute the absolute value of the inner products
    # Write your code here... abs_inner_products = ???
    abs_inner_products = np.abs(inner_products)
 
    # TODO: Sort the absolute value of the inner products in a descend order.
    # Notice that one can sort each column independently
    # Write your code here... mat_sorted = np.sort(?, ?)
    mat_sorted = np.sort(abs_inner_products, axis=0)
    mat_sorted = mat_sorted[::-1, :]

    # TODO: Compute the threshold value per patch, by picking the K largest entry in each column of 'mat_sorted'
    # Write your code here... vec_thresh = mat_sorted[?,?]
    vec_thresh = mat_sorted[K - 1, :]

    # Replicate the vector of thresholds and create a matrix of the same size as 'inner_products'
    mat_thresh = np.tile(np.expand_dims(vec_thresh, axis=0), (np.shape(abs_inner_products)[0], 1))

    # TODO: Given the thresholds, initialize the matrix of coeffecients to be equal to 'inner_products' matrix
    # Write your code here... A = ???
    A = inner_products.copy()

    # TODO: Set the entries in A that are smaller than 'mat_thresh' (in absolute value) to zero
    # Write your code here... A[ ??? ] = 0
    A[abs_inner_products < mat_thresh] = 0

    #TODO: Compute the estimated patches given their representations in 'A'
    # Write your code here... X = ???
    X = D @ A

    return X, A
