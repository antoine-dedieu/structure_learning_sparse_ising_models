import math
import numpy as np

from ising_model.l1_ise import misc_loss_lik


class L0L2Constrained_ISE:
    def __init__(self, data_train, data_val, W_init=None):
        assert (-1 <= data_train).all() and (data_train <= 1).all()
        self.data_train = data_train
        self.data_val = data_val
        self.best_val_lik = 0

        self.N, self.P = data_train.shape

        if W_init is not None:
            self.W = W_init
            self.l1_norm = 2 * np.sum(np.abs(W_init), axis=1).max()
        else:
            self.W = np.zeros((self.P, self.P))
            self.l1_norm = 1e3

    def estimate_W(
        self,
        validate=False,
        n_steps=20,
        estimate_on_support=True,
        K=None
    ):
        # The continuation heuristic allows us to not tune the regularization parameter
        assert K is not None

        # Loop accross nodes
        for pred_idx in range(self.P):
            print("\nNODE {}".format(pred_idx))
            features_idx = np.array(
                list(range(0, pred_idx)) + list(range(pred_idx + 1, self.P))
            )
            best_beta = None
            best_val_lik = np.inf

            # Train data
            X_train = self.data_train[:, features_idx]
            y_train = self.data_train[:, pred_idx]
            beta_start = self.W[pred_idx, features_idx]

            # Val data
            X_val = self.data_val[:, features_idx]
            y_val = self.data_val[:, pred_idx]

            assert not validate
            alpha_list = [self.l1_norm / math.sqrt(K) + 1e-9]

            for alpha in alpha_list:
                beta = train_L0L2_ISE(X_train, y_train, alpha, K, beta_start=beta_start)

                if estimate_on_support:  # no penalty
                    support = np.where(beta != 0)[0]
                    beta_support = train_L0L2_ISE(
                        X_train[:, support], y_train, np.inf, K, beta_start=beta[support]
                    )
                    beta = np.zeros(X_train.shape[1])
                    beta[support] = beta_support

                val_accu, val_loss, val_lik = misc_loss_lik(X_val, y_val, beta)
                print(
                    "Alpha: {}, (Neg. normalized) val lik:{}, val accu: {}".format(
                        round(alpha, 4), round(val_lik, 4), round(val_accu, 4)
                    )
                )
                if validate:
                    if val_lik < best_val_lik:
                        best_beta = beta
                        best_val_lik = val_lik
                else:
                    best_beta = beta
                    best_val_lik = val_lik

            self.W[pred_idx, features_idx] = best_beta
            self.best_val_lik += best_val_lik


def train_L0L2_ISE(X, y, alpha, K, eta=1e-3, T_max=300, beta_start=None):
    N, P = X.shape
    old_beta = np.ones(P)

    if beta_start is None:
        beta_m = np.zeros(P, dtype=float)
    else:
        beta_m = beta_start

    Lipchtiz_coeff = float(np.linalg.norm(X, ord='fro') ** 2)

    it = 0
    while np.linalg.norm(beta_m - old_beta) > eta and it < T_max:
        it += 1

        aux = y * np.exp(-y * np.dot(X, beta_m))
        gradient = -np.dot(X.T, aux)

        # Gradient descent
        old_beta = beta_m.copy()
        grad = beta_m - gradient / Lipchtiz_coeff

        # L0 thresholding
        coefs_sorted = np.abs(grad).argsort()[::-1]
        for idx in coefs_sorted[K:]:
            grad[idx] = 0
        beta_m = grad

        # L2 projection
        l2_norm = np.linalg.norm(beta_m, 2)
        if l2_norm >= alpha:
            beta_m *= alpha / l2_norm
    return beta_m.astype(np.float32)
