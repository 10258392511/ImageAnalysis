import numpy as np
import cvxpy as cp


class Problem(object):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b

    def __repr__(self):
        return f"Prob(A: {self.A.shape}, b: {self.b.shape})"


class Solver(object):
    def __init__(self, prob: Problem = None):
        self.prob = prob

    def __str__(self):
        if self.prob is None:
            raise ValueError("problem not initiated")

        raise NotImplementedError

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def insert_index(inds: list, new_ind: int) -> None:
        """
        O(n)
        """
        j = -1
        for i in range(len(inds)):
            if inds[i] >= new_ind:
                inds.insert(i, new_ind)
                j = i
                break

        if j == -1:
            inds.append(new_ind)

    @staticmethod
    def find_best_normalized_col(*args, **kwargs):
        raise NotImplementedError


# TODO: OMP, LSOMP, MP, WMP, TH, IRLS, LP
class OMP(Solver):
    def __str__(self):
        return "OMP"

    def solve(self, num_support, eps=1e-10):
        A, b = self.prob.A, self.prob.b
        N, M = A.shape
        support = []
        x = np.zeros(M)
        r = b.copy()
        A_norm = np.linalg.norm(A, axis=0, keepdims=True)
        A = A.copy() / A_norm

        x_best = None
        for i in range(num_support):
            j = self.find_best_normalized_col(A, r)
            self.insert_index(support, j)
            As = self.prob.A[:, support]
            x_best = np.linalg.lstsq(As, b, rcond=None)[0]
            r = b - As @ x_best
            # if np.linalg.norm(r) < eps:
            #     break

        x[support] = x_best
        return x

    @staticmethod
    def find_best_normalized_col(A: np.ndarray, r: np.ndarray):
        inner_prods = np.abs(A.T @ r)

        return np.argmax(inner_prods)


class LSOMP(Solver):
    def __str__(self):
        return "LSOMP"

    def solve(self, num_support, eps=1e-10):
        A, b = self.prob.A, self.prob.b
        N, M = A.shape
        support = []
        x = np.zeros(M)
        r = b.copy()
        A_norm = np.linalg.norm(A, axis=0, keepdims=True)
        A = A.copy() / A_norm

        x_best = None
        for i in range(num_support):
            j = self.find_best_normalized_col(self.prob.A, b, support)
            self.insert_index(support, j)
            # As = self.prob.A[:, support]
            # x_best = np.linalg.lstsq(As, b, rcond=None)[0]
            # r = b - As @ x_best
            # if np.linalg.norm(r) < eps:
            #     break

        As = self.prob.A[:, support]
        x_best = np.linalg.lstsq(As, b, rcond=None)[0]
        x[support] = x_best
        return x

    @staticmethod
    def find_best_normalized_col(A: np.ndarray, b: np.ndarray, support: list):
        As0 = A[:, support]
        N, M = A.shape
        j = -1
        r = float("inf")
        for i in range(M):
            if i in support:
                continue
            As = np.concatenate([As0, A[:, i : i + 1]], axis=1)
            x_best = np.linalg.lstsq(As, b, rcond=None)[0]
            r_cand = np.linalg.norm(b - As @ x_best)
            if r_cand < r:
                r = r_cand
                j = i

        return j


class WMP(Solver):
    def __str__(self):
        return "WMP"

    def solve(self, num_support, t=0.7, eps=1e-10):
        A, b = self.prob.A, self.prob.b
        N, M = A.shape
        support = []
        x = np.zeros(M)
        r = b.copy()
        A_norm = np.linalg.norm(A, axis=0, keepdims=True)
        A = A.copy() / A_norm

        for i in range(num_support):
            j = self.find_best_normalized_col(A, r, support, t)
            self.insert_index(support, j)
            z_opt = self.prob.A[:, j] @ b / (A_norm.ravel()[j] ** 2)
            x[j] = z_opt
            r -= z_opt * self.prob.A[:, j]
            # if np.linalg.norm(r) < eps:
            #     break

        return x

    @staticmethod
    def find_best_normalized_col(A: np.ndarray, r: np.ndarray, support: list, t: float):
        j = -1
        z_best = -float("inf")
        for i in range(A.shape[1]):
            if i in support:
                continue
            z_opt = abs(A[:, i] @ r)
            if z_opt > np.linalg.norm(r) * t:
                return i
            else:
                if z_opt > z_best:
                    j = i
                    z_best = z_opt

        return j


# class MP(Solver):
#     def __str__(self):
#         return "MP"
#
#     def solve(self, num_support, eps=1e-10):
#         A, b = self.prob.A, self.prob.b
#         N, M = A.shape
#         support = []
#         x = np.zeros(M)
#         r = b.copy()
#         A_norm = np.linalg.norm(A, axis=0, keepdims=True)
#         A = A.copy() / A_norm
#
#         for i in range(num_support):
#             j = self.find_best_normalized_col(A, r, support)
#             self.insert_index(support, j)
#             z_opt = self.prob.A[:, j] @ b / (A_norm.ravel()[j] ** 2)
#             x[j] = z_opt
#             r -= z_opt * self.prob.A[:, j]
#             # if np.linalg.norm(r) < eps:
#             #     break
#
#         return x
#
#     @staticmethod
#     def find_best_normalized_col(A: np.ndarray, r: np.ndarray, support: list):
#         inner_prods = np.abs(A.T @ r)
#         inner_prods[support] = 0
#
#         return np.argmax(inner_prods)


class MP(WMP):
    def __str__(self):
        return "MP"

    @staticmethod
    def find_best_normalized_col(A: np.ndarray, r: np.ndarray, support: list, t: float):
        # t is not used
        inner_prods = np.abs(A.T @ r)
        inner_prods[support] = 0

        return np.argmax(inner_prods)


class TH(Solver):
    def __str__(self):
        return "TH"

    def solve(self, num_support, eps=1e-10, if_eps=False):
        A, b = self.prob.A, self.prob.b
        A_norm = np.linalg.norm(A, axis=0, keepdims=True)
        A = A.copy() / A_norm
        inner_prods = np.abs(A.T @ b)
        inds = np.argsort(inner_prods)[::-1]
        # print(f"inner prods: {inner_prods}\nsorted: {inner_prods[inds]}")
        N, M = A.shape
        x = np.zeros(M)

        if if_eps:
            for i in range(1, num_support + 1):
                inds_sel = inds[:i]
                As = self.prob.A[:, inds_sel]
                x_best, r = np.linalg.lstsq(As, b, rcond=None)[:2]
                if np.linalg.norm(r) < eps:
                    x[inds_sel] = x_best
                    return x
        else:
            inds_sel = inds[:num_support]
            # print(f"inds_sel: {inds_sel}")
            As = self.prob.A[:, inds_sel]
            x_best = np.linalg.lstsq(As, b, rcond=None)[0]
            # print(f"res: {np.linalg.norm(b - As @ x_best)}")
            x[inds_sel] = x_best
            return x

    @staticmethod
    def find_best_normalized_col(*args, **kwargs):
        pass


class IRLS(Solver):
    def __str__(self):
        return "IRLS"

    def solve(self, num_support=None, max_iters=1000, p=1, eps=1e-6):
        A, b = self.prob.A.copy(), self.prob.b.copy()
        N, M = A.shape
        # x_prev = None
        x_next = np.ones((M,))
        X = np.eye(M)
        for k in range(max_iters):
            # print(f"current {k + 1}/{max_iters}")
            x_prev = x_next
            X_sq = X ** 2
            A_pseudo_b = np.linalg.lstsq(A @ X_sq @ A.T, b, rcond=None)[0]
            x_next = X_sq @ A.T @ A_pseudo_b
            X = np.diag(np.abs(x_next) ** (1 - 0.5 * p))
            # print(f"{x_prev}\n{x_next}")
            if np.linalg.norm(x_next - x_prev) < eps:
                break

        return x_next


class LP(Solver):
    def __str__(self):
        return "LP"

    def solve(self, num_support=None):
        A, b = self.prob.A.copy(), self.prob.b.copy()
        N, M = A.shape
        z = cp.Variable((2 * M,))
        A_norm = np.linalg.norm(A, axis=0, keepdims=True)
        A *= A_norm
        prob = cp.Problem(cp.Minimize(np.ones((2 * M,)).T @ z),
                          [A @ np.concatenate([np.eye(M), -np.eye(M)], axis=1) @ z == b,
                           z >= 0])
        prob.solve()
        # print(z.value)
        z_opt = z.value
        x = z_opt[:M] - z_opt[M:]
        # print(f"res: {np.linalg.norm(A @ x - b)}")
        # print(f"A norm: {A_norm}")

        return x * A_norm.ravel()
