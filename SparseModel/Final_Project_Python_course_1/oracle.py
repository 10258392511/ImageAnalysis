import numpy as np


def oracle(CA, b, s):
    # ORACLE Implementation of the Oracle estimator
    #
    # Solves the following problem:
    #   min_x ||b - CAx||_2^2 s.t. supp{x} = s
    # where s is a vector containing the support of the true sparse vector
    #
    # The solution is returned in the vector x

    # Initialize the vector x
    x = np.zeros(np.shape(CA)[1])

    # TODO: Implement the Oracle estimator
    # Write your code here... x = ????
    CA_sampled = CA[:, s]
    x_sampled = np.linalg.lstsq(CA_sampled, b, rcond=None)[0]
    x[s] = x_sampled

    return x
