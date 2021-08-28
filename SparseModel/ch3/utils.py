import numpy as np
import matplotlib.pyplot as plt
from solvers import *


def norm_rel_error(x_true: np.ndarray, x: np.ndarray) -> float:
    return np.linalg.norm(x_true - x) / np.linalg.norm(x_true)


def support_rel_error(x_true: np.ndarray, x: np.ndarray, eps=1e-9) -> float:
    support_hat = np.argwhere(np.abs(x_true) > eps).ravel()
    support = np.argwhere(np.abs(x) > eps).ravel()
    support_hat = set(support_hat)
    support = set(support)
    return 1 - len(support.intersection(support_hat)) / max(len(support), len(support_hat))


def generate_x_true(M: int, sup: int) -> np.ndarray:
    x_true = np.zeros(M)
    non_zeros = np.zeros((sup,))
    for i in range(len(non_zeros)):
        if np.random.rand() > 0.5:
            non_zeros[i] = np.random.rand() + 1
        else:
            non_zeros[i] = -np.random.rand() - 1

    # print(f"number of nonzeros: {len(non_zeros)}")
    inds = np.random.choice(np.arange(M), (sup,), replace=False)
    x_true[inds] = non_zeros

    return x_true


def run(num_runs: int, mode: str = None, solvers: list = None, *args, **kwargs) -> None:
    assert mode in ["MP", "CVX", None], "invalid mode"

    N, M = 10, 30
    support_upper = (M - 1) // 2

    if solvers is None:
        if mode == "MP":
            solvers = [OMP(), LSOMP(), MP(), WMP(), TH()]
        else:
            solvers = [OMP(), IRLS(), LP()]

    doc = {solver.__str__(): {"norm": np.zeros((num_runs, support_upper + 1)),
                              "support": np.zeros((num_runs, support_upper + 1))} for solver in solvers}
    for run_itr in range(num_runs):
        print_interval = num_runs // 10
        if run_itr % print_interval == 0:
            print(f"current: {run_itr + 1}/{num_runs}")

        for sup in range(1, support_upper + 1):
            A = np.random.randn(N, M)
            x_true = generate_x_true(M, sup)
            b = A @ x_true
            prob = Problem(A, b)

            for solver in solvers:
                # print(solver)
                # print(f"number of support: {sup}")
                solver.prob = prob
                x = solver.solve(num_support=sup)
                # print(f"x: {x}\nx_true: {x_true}")
                # print(f"support: {np.where(np.abs(x) > 1e-10)}\nsupport_true: {np.where(np.abs(x_true) > 1e-10)}")
                # print("-" * 50)
                doc[solver.__str__()]["norm"][run_itr, sup] = norm_rel_error(x_true, x)
                doc[solver.__str__()]["support"][run_itr, sup] = support_rel_error(x_true, x)

    plot_doc(doc, **kwargs)


def plot_doc(doc: dict, *args, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    axes[0].set_title("rel norm error")
    axes[1].set_title("support set distance")

    for key in doc:
        norm_doc = doc[key]["norm"].mean(axis=0)
        support_doc = doc[key]["support"].mean(axis=0)
        xx = np.arange(1, norm_doc.shape[0])
        axes[0].plot(xx, norm_doc[1:], label=key)
        if "norm_y_lim" in kwargs:
            axes[0].set_ylim(kwargs["norm_y_lim"])
        axes[1].plot(xx, support_doc[1:], label=key)

    for axis in axes:
        axis.grid(True)
        axis.legend()

    plt.show()


if __name__ == '__main__':
    # # test norm_rel_error(.) & support_rel_error(.)
    # N = 10
    # x_true = np.random.randn(N)
    # x = x_true.copy()
    # x[0] = 0
    # print(f"rel error: {norm_rel_error(x_true, x)}")
    # print(f"rel support dist: {support_rel_error(x_true, x)}")

    # test generate_x_true(.)
    M = 30
    sup = 3
    print(f"x_true: {generate_x_true(M, sup)}")
