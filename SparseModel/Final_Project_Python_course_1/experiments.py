import numpy as np
import matplotlib.pyplot as plt
import time

from construct_data import construct_data
from compute_psnr import compute_psnr
from compute_effective_dictionary import compute_effective_dictionary
from omp import omp
from bp_admm import bp_admm
from oracle import oracle


n = 40
m = 2 * n ** 2
p = 0.4
sigma = 0.05  # 0.05
true_k = 10
base_seed = 10
num_experiments = 10


def construct_A(use_seed=False):
    A = np.zeros((n ** 2, m))

    # In this part we construct A by creating its atoms one by one, where
    # each atom is a rectangle of random size (in the range 5-20 pixels),
    # position (uniformly spread in the area of the image), and sign.
    # Lastly we will normalize each atom to a unit norm.
    for i in range(A.shape[1]):

        # Choose a specific random seed to reproduce the results
        if use_seed:
            np.random.seed(i + base_seed)

        empty_atom_flag = 1

        while empty_atom_flag:

            # TODO: Create a rectangle of random size and position
            # Write your code here... atom = ????
            atom = np.zeros((n, n))
            start_x, start_y = np.random.randint(0, n, (2,))
            len_x, len_y = np.random.randint(5, n // 2 + 1, (2,))
            end_x = min(start_x + len_x, n - 1)
            end_y = min(start_y + len_y, n - 1)
            atom[start_x: end_x + 1, start_y: end_y + 1] = np.random.choice([1, -1])

            # Reshape the atom to a 1D vector
            atom = np.reshape(atom, (-1))

            # Verify that the atom is not empty or nearly so
            if np.sqrt(np.sum(atom ** 2)) > 1e-5:
                empty_atom_flag = 0

                # TODO: Normalize the atom
                # Write your code here... atom = ????
                atom /= np.linalg.norm(atom)

                # Assign the generated atom to the matrix A
                A[:, i] = atom

    return A


def display_gray_img(img, reshape_size=(n, n), if_save=False, save_path=None):
    plt.imshow(img.reshape(reshape_size), cmap="gray")
    plt.colorbar()
    if if_save:
        assert save_path is not None
        plt.savefig(save_path)

    plt.show()


def test_construct_data(if_save=False):
    A = construct_A()
    x0, b0, noise_std, b0_noisy, C, b = construct_data(A, p, sigma, true_k)
    print(f"{b0.shape}, {b0_noisy.shape}, {b.shape}")

    time_stamp = None
    if if_save:
        time_stamp = f"{time.time()}".replace(".", "_")

    display_gray_img(b0, (n, n), if_save=if_save, save_path=f"imgs/orig_{time_stamp}.png")
    display_gray_img(b0_noisy, (n, n), if_save=if_save, save_path=f"imgs/noisy_{time_stamp}.png")
    display_gray_img(C.T @ b, (n, n), if_save=if_save, save_path=f"imgs/sparse_{time_stamp}.png")


def test_oracle_estimator():
    PSNR_oracle = np.zeros(num_experiments)
    A = construct_A(True)

    # Loop over num_experiments
    for experiment in range(num_experiments):
        # Choose a specific random seed to reproduce the results
        np.random.seed(experiment + 1 + base_seed)

        # Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)

        # TODO: Compute the subsampled dictionary
        # Write your code here... A_eff = ????
        A_eff = C @ A

        # TODO: Compute the oracle estimation
        # Write your code here... x_oracle = oracle(?,?,?)
        support = (x0 != 0)
        x_oracle = oracle(A_eff, b, support)

        # Compute the estimated image
        b_oracle = A @ x_oracle

        # Compute the PSNR
        PSNR_oracle[experiment] = compute_psnr(b0, b_oracle)

        # Print some statistics
        print('Oracle experiment %d/%d, PSNR: %.3f' % (experiment + 1, num_experiments, PSNR_oracle[experiment]))

    # Display the average PSNR of the oracle
    print('Oracle: Average PSNR = %.3f\n' % np.mean(PSNR_oracle))
    time_stamp = f"{time.time()}".replace(".", "_")
    paths = [f"imgs/part3/{path}_{time_stamp}.png" for path in ["orig", "noisy", "sparse", "recons"]]
    for img, path in zip([b0, b0_noisy, C.T @ b, b_oracle], paths):
        display_gray_img(img, if_save=True, save_path=path)


def test_OMP():
    A = construct_A(True)

    # We will sweep over k = 1 up-to k = max_k and pick the best result
    max_k = min(2 * true_k, m)

    # Allocate a vector to store the PSNR estimations per each k
    PSNR_omp = np.zeros((num_experiments, max_k))

    best_b_omp = None
    # Loop over the different realizations
    for experiment in range(num_experiments):

        # Choose a specific random seed to reproduce the results
        np.random.seed(experiment + 1 + base_seed)

        # Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)

        # Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)

        # Run the OMP for various values of k and pick the best results
        for k_ind in range(max_k):

            # Compute the OMP estimation
            x_omp = omp(A_eff_normalized, b, k_ind + 1)

            # Un-normalize the coefficients
            x_omp = x_omp / atoms_norm

            # Compute the estimated image
            b_omp = A @ x_omp

            # Compute the current PSNR
            PSNR_omp[experiment, k_ind] = compute_psnr(b0, b_omp)

            # Save the best result of this realization, we will present it later
            if PSNR_omp[experiment, k_ind] == max(PSNR_omp[experiment, :]):
                best_b_omp = b_omp

            # Print some statistics
            print('OMP experiment %d/%d, cardinality %d, PSNR: %.3f' % (
            experiment + 1, num_experiments, k_ind, PSNR_omp[experiment, k_ind]))

    # Compute the best PSNR, computed for different values of k
    PSNR_omp_best_k = np.max(PSNR_omp, axis=-1)

    # Display the average PSNR of the OMP (obtained by the best k per image)
    print('OMP: Average PSNR = %.3f\n' % np.mean(PSNR_omp_best_k))

    # Plot the average PSNR vs. k
    psnr_omp_k = np.mean(PSNR_omp, 0)
    print(psnr_omp_k)

    best_k = np.argmax(psnr_omp_k) + 1

    k_scope = np.arange(1, max_k + 1)
    plt.figure(1)
    plt.plot(k_scope, psnr_omp_k, '-r*')
    plt.xlabel("k", fontsize=16)
    plt.ylabel("PSNR [dB]", fontsize=16)
    plt.title("OMP: PSNR vs. k, True Cardinality = " + str(true_k) + ", and best_k = " + str(best_k))

    time_stamp = f"{time.time()}".replace(".", "_")
    plt.savefig(f"imgs/part4/omp_k_psnr_{time_stamp}.png")
    plt.show()

    paths = [f"imgs/part4/{path}_{time_stamp}.png" for path in ["orig", "noisy", "sparse", "recons"]]
    for img, path in zip([b0, b0_noisy, C.T @ b, best_b_omp], paths):
        display_gray_img(img, if_save=True, save_path=path)


def test_admm():
    A = construct_A(True)

    # We will sweep over various values of lambda
    num_lambda_values = 10

    # Allocate a vector to store the PSNR results obtained for the best lambda
    # num_experiments = 1
    PSNR_admm_best_lambda = np.zeros(num_experiments)

    # Loop over num_experiments
    for experiment in range(num_experiments):

        # Choose a specific random seed to reproduce the results
        np.random.seed(experiment + 1 + base_seed)

        # Construct data
        [x0, b0, noise_std, b0_noisy, C, b] = construct_data(A, p, sigma, true_k)

        # Compute the effective subsampled dictionary
        [A_eff_normalized, atoms_norm] = compute_effective_dictionary(C, A)

        # Run the BP for various values of lambda and pick the best result
        lambda_max = np.linalg.norm(A_eff_normalized.T @ b, np.inf)
        lambda_vec = np.logspace(-5, 0, num_lambda_values) * lambda_max
        psnr_admm_lambda = np.zeros(num_lambda_values)

        # Loop over various values of lambda
        for lambda_ind in range(num_lambda_values):

            # Compute the BP estimation
            x_admm = bp_admm(A_eff_normalized, b, lambda_vec[lambda_ind])

            # Un-normalize the coefficients
            x_admm = x_admm / atoms_norm

            # Compute the estimated image
            b_admm = A @ x_admm

            # Compute the current PSNR
            psnr_admm_lambda[lambda_ind] = compute_psnr(b0, b_admm)

            # Save the best result of this realization, we will present it later
            if psnr_admm_lambda[lambda_ind] == max(psnr_admm_lambda):
                best_b_admm = b_admm

            # print some statistics
            print('BP experiment %d/%d, lambda %d/%d, PSNR %.3f' % \
                  (experiment + 1, num_experiments, lambda_ind + 1, num_lambda_values, psnr_admm_lambda[lambda_ind]))

        # Save the best PSNR
        PSNR_admm_best_lambda[experiment] = max(psnr_admm_lambda)

    # Plot the PSNR vs. lambda of the last realization
    plt.figure()
    plt.semilogx(lambda_vec, psnr_admm_lambda, '-*r')
    plt.xlabel(r'$\lambda$', fontsize=16)
    plt.ylabel("PSNR [dB]", fontsize=16)
    plt.title("BP via ADMM: PSNR vs. " + r'$\lambda$')

    time_stamp = f"{time.time()}".replace(".", "_")
    plt.savefig(f"imgs/part5/lambda_psnr_{time_stamp}.png")
    plt.show()

    paths = [f"imgs/part5/{path}_{time_stamp}.png" for path in ["orig", "noisy", "sparse", "recons"]]
    # Display the average PSNR of the BP
    print("BP via ADMM: Average PSNR = ", np.mean(PSNR_admm_best_lambda), "\n")
    for img, path in zip([b0, b0_noisy, C.T @ b, best_b_admm], paths):
        display_gray_img(img, if_save=True, save_path=path)



if __name__ == '__main__':
    # test_construct_data()

    # test_oracle_estimator()

    # test_OMP()

    test_admm()

    pass
