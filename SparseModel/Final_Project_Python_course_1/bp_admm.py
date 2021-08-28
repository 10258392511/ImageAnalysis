import numpy as np
from scipy.linalg import solve_triangular
import pywt

def bp_admm(CA, b, lmbda):

    # BP_ADMM Solve Basis Pursuit problem via ADMM
    #
    # Solves the following problem:
    #   min_x 1/2*||b - CAx||_2^2 + lambda*|| x ||_1
    #
    # The solution is returned in the vector v.

    # Set the accuracy tolerance of ADMM, run for at most max_admm_iters
    tol_admm = 1e-4
    max_admm_iters = 100

    # TODO: Compute the vector of inner products between the atoms and the signal
    CA_T_b = CA.T @ b

    # In the x-update step of the ADMM we use the Cholesky factorization for
    # solving efficiently a given linear system Ax=b. The idea of this
    # factorization is to decompose a symmetric positive-definite matrix A
    # by A = L*L^T = L*U, where L is a lower triangular matrix and U is
    # its transpose. Given L and U, we can solve Ax = b by first solving
    # Ly = b for y by forward substitution, and then solving Ux = y
    # for x by back substitution.
    # To conclude, given A and b, where A is symmetric and positive-definite,
    # we first compute L using Numpy's command L = np.linalg.cholesky( A, 'lower' );
    # and get U by setting U = L'; Then, we obtain x via x = U \ (L \ b);
    # Note that the matrix A is fixed along the iterations of the ADMM
    # (and so as L and U). Therefore, in order to reduce computations,
    # we compute its decomposition once.

    # TODO: Compute the Cholesky factorization of M = CA'*CA + I for fast computation
    # of the x-update. Use Numpy's `linalg.cholesky` function and produce a lower triangular
    # matrix L, satisfying the equation M = L*L'
    M = CA.T @ CA
    M += np.eye(M.shape[0])
    L = np.linalg.cholesky(M)

    U = L.T

    # TODO: Initialize v
    v = np.zeros((CA.shape[1]))

    # TODO: Initialize u, the dual variable of ADMM
    u = np.zeros((CA.shape[1]))

    # TODO: Initialize the previous estimate of v, used for convergence test
    v_prev = v.copy()


    # main loop
    for i in range(max_admm_iters):
        # TODO: x-update via Cholesky factorization. Solve the linear system
        # (CA'*CA + I)x = (CAtb + v - u) using the `solve_triangular` function
        # Write your code here... x = ????   
        y = solve_triangular(L, CA_T_b + (v - u), lower=True)
        x = solve_triangular(U, y)

        # TODO: v-update via soft thresholding using the `pewit.threshold` function
        v = pywt.threshold(x + u, lmbda)

        # TODO: u-update according to the ADMM formula
        u += x - v

        # Check if converged
        if np.linalg.norm(v):
            if (np.linalg.norm(v - v_prev) / np.linalg.norm(v)) < tol_admm:
                break

        # TODO: Save the previous estimate in v_prev
        v_prev = v.copy()
 
    return v
