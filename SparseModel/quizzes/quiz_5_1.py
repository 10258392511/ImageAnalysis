import numpy as np


def omp(A, b, k):
    # Initialize the vector x
    x = np.zeros(np.shape(A)[1])

    # TODO: Implement the OMP algorithm
    support = np.zeros_like(x, dtype=bool)
    r = b
    xs = None
    for i in range(k):
        j = np.argmax(np.abs(r.T @ A))
        support[j] = True
        As = A[:, support]
        xs = np.linalg.lstsq(As, b, rcond=None)[0]
        r = As @ xs - b
        print(f"iter {i}: choosing {j}, r = {r}, ||r|| = {np.linalg.norm(r)}")

    x[support] = xs
    # return the obtained x
    return x


def ls_omp(A, b, k):
    # Initialize the vector x
    x = np.zeros(np.shape(A)[1])

    # TODO: Implement the OMP algorithm
    support = np.zeros_like(x, dtype=bool)
    r = b
    xs = None

    for i in range(k):
        j_out = -1
        r_norm_out = float("inf")

        for j in range(len(support)):
            if support[j]:
                continue

            support_copy = support.copy()
            support_copy[j] = True
            As = A[:, support_copy]
            xs = np.linalg.lstsq(As, b, rcond=None)[0]
            r_cand = b - As @ xs
            r_cand_norm = np.linalg.norm(r_cand)
            # print(f"{i}, {j}, {r_cand}")

            if r_cand_norm < r_norm_out:
                r_norm_out = r_cand_norm
                j_out = j

        j = j_out
        support[j] = True
        As = A[:, support]
        xs = np.linalg.lstsq(As, b, rcond=None)[0]
        r = As @ xs - b
        print(f"iter {i}: choosing {j}, r = {r}, ||r|| = {np.linalg.norm(r)}")

    x[support] = xs
    # return the obtained x
    return x


if __name__ == '__main__':
    A = np.array([[0.1817, 0.5394, -0.1197, 0.6404],
                  [0.6198, 0.1994, 0.00946, -0.3121],
                  [-0.7634, -0.8181, 0.9883, 0.7018]])

    b = np.array([1.1862, -0.1158, -0.1093])
    N, M = A.shape

    # OMP
    x_omp = omp(A, b, M)
    print("-" * 30)

    # # THR
    # A_norm = A / np.linalg.norm(A, axis=0, keepdims=True)
    # inner_prods = np.abs(b.T @ A_norm)
    # print(inner_prods)
    # print(np.argsort(inner_prods)[::-1])
    #
    # # WMP
    # t = 0.5
    # print(t * np.linalg.norm(b))

    # LS-OMP
    x_ls_omp = ls_omp(A, b, M)
