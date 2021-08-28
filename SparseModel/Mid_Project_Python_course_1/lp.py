import numpy as np
from scipy.optimize import linprog


def lp(A, b, tol):
    # LP Solve Basis Pursuit via linear programing
    #
    # Solves the following problem:
    #   min_x || x ||_1 s.t. b = Ax
    #
    # The solution is returned in the vector x.

    # Set the options to be used by the linprog solver
    opt = {"tol": tol, "disp": False}

    # TODO: Use scipy.optimize linprog function to solve the BP problem
    # Write your code here ... x = ????

    n, m = A.shape
    c = np.ones((2 * m,))
    A_tilda = np.block([A, -A])
    z = linprog(c, A_eq=A_tilda, b_eq=b, options=opt).x
    x = z[:m] - z[m:]

    return x
