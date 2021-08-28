import numpy as np


def omp(A, b, k):
    # OMP Solve the P0 problem via OMP
    #
    # Solves the following problem:
    #   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
    #
    # The solution is returned in the vector x

    # Initialize the vector x
    x = np.zeros(np.shape(A)[1])

    # TODO: Implement the OMP algorithm
    support = np.zeros_like(x, dtype=bool)
    r = b
    xs = None
    for _ in range(k):
        j = np.argmax(np.abs(r.T @ A))
        support[j] = True
        As = A[:, support]
        xs = np.linalg.lstsq(As, b, rcond=None)[0]
        r = As @ xs - b

    x[support] = xs
    # return the obtained x
    return x
