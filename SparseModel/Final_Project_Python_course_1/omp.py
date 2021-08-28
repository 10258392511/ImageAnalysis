import numpy as np


def omp(CA, b, k):
    
    # OMP Solve the sparse coding problem via OMP
    #
    # Solves the following problem:
    #   min_x ||b - CAx||_2^2 s.t. ||x||_0 <= k
    #
    # The solution is returned in the vector x

    # Initialize the vector x
    x = np.zeros(np.shape(CA)[1])

    # TODO: Implement the OMP algorithm
    # Write your code here... x = ????
    support = np.zeros_like(x, dtype=bool)
    r = b.copy()

    xs = None
    for i in range(k):
        inner_prod = r.T @ CA
        sel_ind = np.argmax(np.abs(inner_prod))
        support[sel_ind] = True
        CAs = CA[:, support]
        xs = np.linalg.lstsq(CAs, b, rcond=None)[0]
        r = b - CAs @ xs

    x[support] = xs

    return x
