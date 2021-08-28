import numpy as np
import matplotlib.pyplot as plt

from .utils import compute_error, dict_sim


def _omp_solver(A, y, k0):
    r = y
    x = np.zeros((A.shape[1],))
    x_mask = np.zeros((A.shape[1],), dtype=bool)
    xs = None
    for k in range(k0):
        # print(A.shape, r.shape)
        inner_prod = np.abs(A.T @ r)
        new_support = np.argmax(inner_prod)
        x_mask[new_support] = True
        As = A[:, x_mask]
        xs = np.linalg.lstsq(As, y, rcond=None)[0]
        r = y - As @ xs


    x[x_mask] = xs

    return x


def omp(A, Y, k0):
    assert np.allclose(np.linalg.norm(A, axis=0), 1), "A must be normalized first"
    X = np.zeros((A.shape[1], Y.shape[1]))
    for i in range(Y.shape[1]):
        y = Y[:, i]
        x = _omp_solver(A, y, k0)
        X[:, i] = x

    return X


def _update_dict_mod(X, Y):
    # print(X.shape)
    # A = Y @ X.T @ np.linalg.inv(X @ X.T)
    A = Y @ np.linalg.pinv(X)
    A /= np.linalg.norm(A, axis=0)

    return A


def _update_dict_k_svd(A, X, Y, eps=1e-10):
    for i in range(A.shape[1]):
        a = A[:, i]
        x = X[i, :]
        E = Y - A @ X + a[:, np.newaxis] @ x[np.newaxis, :]
        non_zeros_mask = (np.abs(x) > eps)
        if non_zeros_mask.sum() == 0:
            continue
        E = E[:, non_zeros_mask]
        U, s, Vh = np.linalg.svd(E, full_matrices=False)
        A[:, i] = U[:, 0]
        X[i, :][non_zeros_mask] = s[0] * Vh[0, :]

    return A, X


def update_dict(A, X, Y, method="K-SVD", **kwargs):
    assert method in ["MOD", "K-SVD"], "Method not supported"

    if method == "MOD":
        A = _update_dict_mod(X, Y)

    elif method == "K-SVD":
        A, X = _update_dict_k_svd(A, X, Y, kwargs.get("eps", 1e-10))

    return A, X


def eval(A, Y_train, Y_test, A_ref=None, k0=4):
    rec_percent = None
    if A_ref is not None:
        rec_percent = dict_sim(A_ref, A)

    X_train = omp(A, Y_train, k0)
    X_test = omp(A, Y_test, k0)
    return rec_percent, compute_error(A, X_train, Y_train), compute_error(A, X_test, Y_test)


def train(A_init, Y_train, Y_test, k0=4, method="K-SVD", **kwargs):
    assert np.allclose(np.linalg.norm(A_init, axis=0), 1), "A must be normalized first"

    # pull out settings
    stop_eps = kwargs.get("stop_eps", 1e-4)
    A_ref = kwargs.get("A_ref")
    A = A_init.copy()
    max_iters = kwargs.get("max_iters", 150)
    if A_ref is not None:
        log_recovered_per = np.empty((max_iters,))
    log_train_error = np.empty((max_iters,))
    log_test_error = np.empty((max_iters,))
    val_interval = kwargs.get("val_interval", 10)

    for iter in range(max_iters):
        X = omp(A, Y_train, k0)
        A, X = update_dict(A, X, Y_train, method)

        # evaluation
        rec_percent, train_error, test_error = eval(A, Y_train, Y_test, A_ref, k0=k0)
        if A_ref is not None:
            log_recovered_per[iter] = rec_percent
        log_train_error[iter] = train_error
        log_test_error[iter] = test_error

        if iter % val_interval == 0:
            print(f"Current iteration: {iter}/{max_iters}")
            print(f"training error: {train_error}, test error: {test_error}")

        if iter >=1:
            if abs(log_train_error[iter] - log_train_error[iter - 1]) / log_train_error[iter - 1] < stop_eps:
                break

    if A_ref is None:
        log_recovered_per = None
    else:
        log_recovered_per = log_recovered_per[:iter + 1]

    log_dict = {"recovered_atoms_percent": log_recovered_per,
                "train_error": log_train_error[:iter + 1],
                "test_error": log_test_error[:iter + 1]}

    return A, log_dict
