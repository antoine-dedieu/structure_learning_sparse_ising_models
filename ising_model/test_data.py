import itertools
import numpy as np

from ising_model.data import generate_samples, generate_train_val_data_gibbs, ising_model

def test_generate_samples():
    # Test whether random sampler empirical frequencies match the ground truth
    N = 10000
    from collections import Counter
    probas = np.array([0.5, 0.1, 0.3, 0.1])
    s = generate_samples(N, np.array([0, 1, 2, 3]), np.cumsum(probas))
    print("Ground truth: {} Counts: {}".format(probas, Counter(s)))


def test_gibbs_sampling_from_W():
    # Test whether gibbs sampler empirical frequencies match the ground truth
    N = 10000
    P = 16
    rho = 0.4
    type_matrix = 'random_graph_uniform_sign'
    P = 4
    W, data_train, data_val = generate_train_val_data_gibbs(N, P, rho, type_matrix)
    print(W)

    P = W.shape[0]
    X = np.array(list(itertools.product([-1, 1], repeat=P))).astype(float)
    assert X.shape[0] == 2 ** P
    probas = np.array([float(ising_model(X[i], W)) for i in range(X.shape[0])])
    probas /= probas.sum()
    print("Ground truth: ", probas.round(3))

    counters = [0 for idx in range(X.shape[0])]
    for row_idx in range(data_train.shape[0]):
        row = data_train[row_idx]
        for idx in range(X.shape[0]):
            if np.all(row == X[idx]):
                counters[idx] += 1
    print("Gibbs sampler frequencies: ", np.array(counters) / N)


if __name__ == "__main__":
    test_generate_samples()
    test_gibbs_sampling_from_W()
