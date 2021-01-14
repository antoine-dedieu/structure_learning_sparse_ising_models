import itertools
import math

import numba as nb
import numpy as np


def generate_train_val_data(N, P, rho, type_matrix, seed=0):
    # Train and validation generation for smaller graphs
    np.random.seed(seed)
    W = generate_W(P, rho, type_matrix)
    X, cumsum_probas = compute_probas(W)

    data_train = generate_samples(N, X, cumsum_probas)
    data_val = generate_samples(N, X, cumsum_probas)
    return W, data_train, data_val


def generate_train_val_data_gibbs(N, P, rho, type_matrix, seed=0):
    # Train and validation generation for larger graphs
    np.random.seed(seed)
    W = generate_W(P, rho, type_matrix)

    data_train = gibbs_sampling_from_W(W, N)
    data_val = gibbs_sampling_from_W(W, N)
    return W, data_train, data_val


def generate_W(P, rho, type_matrix):
    # Generate the interaction matrix W
    W = np.zeros((P, P))

    # Periodic graph
    if type_matrix in ['periodic_graph_uniform_sign', 'periodic_graph_random_sign']:
        degree = 4
        grid_size = int(math.sqrt(P))
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                idx_up = (i - 1) % grid_size * grid_size + j
                idx_down = (i + 1) % grid_size * grid_size + j
                idx_left = i * grid_size + (j - 1) % grid_size
                idx_right = i * grid_size + (j + 1) % grid_size

                for idx_neighbor in [idx_up, idx_down, idx_left, idx_right]:
                    if type_matrix == 'periodic_graph_uniform_sign':
                        W[idx, idx_neighbor] = rho
                    elif type_matrix == "periodic_graph_random_sign":
                        W_ij = np.sign(np.random.uniform() - 0.5) * rho

    # Random graph
    elif type_matrix in [
        'random_graph_uniform_sign',
        'random_graph_random_sign',
        'random_graph_uniform_sign_uniform_val',
        'random_graph_random_sign_uniform_val',
    ]:
        degree = 3

        # Repeat main
        success = False
        it_main = 0
        while not success and it_main < 100:
            success = True
            it_main += 1
            count_degree = {idx: [] for idx in range(P)}

            W = np.zeros((P, P))

            for _ in range(int(degree * P / 2)):
                found1 = False
                subset1 = [x for x in range(P)]
                while not found1 and len(subset1) > 0:
                    idx_start = np.random.randint(len(subset1))
                    idx_start = subset1[idx_start]
                    found1 = len(count_degree[idx_start]) < degree
                    subset1.remove(idx_start)

                found2 = False
                subset2 = [x for x in range(P)]
                while not found2 and len(subset2) > 0:
                    idx_end = np.random.randint(len(subset2))
                    idx_end = subset2[idx_end]
                    found2 = (
                        len(count_degree[idx_end]) < degree
                        and idx_end not in count_degree[idx_start]
                        and idx_start != idx_end
                    )
                    subset2.remove(idx_end)

                if type_matrix == "random_graph_uniform_sign":
                    W_ij = rho
                elif type_matrix == "random_graph_random_sign":
                    W_ij = np.sign(np.random.uniform() - 0.5) * rho
                elif type_matrix == "random_graph_uniform_sign_uniform_val":
                    W_ij = np.random.uniform(rho, rho + 0.2)
                elif type_matrix == "random_graph_random_sign_uniform_val":
                    W_ij = np.sign(np.random.uniform() - 0.5) * np.random.uniform(
                        rho, rho + 0.3
                    )

                W[idx_start, idx_end] = W_ij
                W[idx_end, idx_start] = W_ij

                count_degree[idx_start].append(idx_end)
                count_degree[idx_end].append(idx_start)
                if not (found1 and found2):
                    success = False
                    break
        assert success
        for idx in range(P):
            assert len(np.where(W[idx] != 0)[0]) == degree

    assert (np.diag(W) == 0).all()
    assert (W == W.T).all()
    print("W generated")
    return W


@nb.njit()
def ising_model(x, W):
    return np.exp(0.5 * x.reshape(1, -1).dot(W).dot(x.reshape(-1, 1)))  # factor 2 in LR


def compute_probas(W):
    P = W.shape[0]
    X = np.array(list(itertools.product([-1, 1], repeat=P))).astype(float)
    assert X.shape[0] == 2 ** P

    probas = np.array([float(ising_model(X[i], W)) for i in range(X.shape[0])])
    probas /= probas.sum()
    cumsum_probas = np.cumsum(probas)
    return X, cumsum_probas


def generate_samples(N, X, cumsum_probas):
    samples = []
    for _ in range(N):
        u = np.random.uniform()
        for idx in range(X.shape[0]):
            if u <= cumsum_probas[idx]:
                break
        samples.append(X[idx])
    samples = np.array(samples)
    return samples


@nb.njit()
def gibbs_sampling_from_W(W, N_samples, N_steps=1000, refresh=100):
    # The result is reflected in S, which is updated in place
    d = W.shape[0]
    S = 2 * (np.random.rand(N_samples, d) < 0.5).astype(np.float64) - 1

    b = np.zeros((d, 1))
    assert W.shape == (d, d)
    assert (np.diag(W) == 0).all()
    assert (W == W.T).all()

    g = S @ W.T + b.T  # size N_samples x d, g_ij = x^{(i)}^T w_j
    for step in range(N_steps):
        for i in np.random.permutation(d):
            delta = -2 * g[:, i : i + 1] * S[:, i : i + 1]
            threshold = sigmoid(delta)  # p(switch_i | x_j) = sigmoid(2 * x_i * g_i)

            flip = (np.random.rand(N_samples, 1) < threshold).astype(np.float64)
            S[:, i : i + 1] = (1 - 2 * flip) * S[:, i : i + 1]
            if step % refresh == 0:
                g = S @ W.T + b.T
            else:
                g += flip * 2 * S[:, i : i + 1] * W[i : i + 1]  # update g
    return S


@nb.vectorize(["f8(f8)", "f4(f4)"])
def sigmoid(x):
    if x > 0:
        x = np.exp(x)
        return x / (1 + x)
    else:
        return 1 / (1 + np.exp(-x))


