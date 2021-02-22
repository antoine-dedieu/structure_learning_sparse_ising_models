import math
import numpy as np
from sklearn.linear_model import LogisticRegression


class L1_LogReg:
    def __init__(self, data_train, data_val=None):
        assert (-1 <= data_train).all() and (data_train <= 1).all()
        self.data_train = data_train
        self.data_val = data_val
        self.best_val_lik = 0

        self.N, self.P = data_train.shape
        self.W = np.zeros((self.P, self.P))

    def estimate_W(
        self,
        use_scikit=False,
        validate=True,
        n_steps=20,
        estimate_on_support=True,
    ):
        # Loop accross nodes
        for pred_idx in range(self.P):
            print("\nNODE {}".format(pred_idx))
            features_idx = np.array(
                list(range(0, pred_idx)) + list(range(pred_idx + 1, self.P))
            )
            best_beta = None
            best_val_lik = np.inf
            best_accu_lik = np.inf

            # Train data
            X_train = 2 * self.data_train[:, features_idx]  # scale the entries
            y_train = self.data_train[:, pred_idx]

            # Val data
            X_val = 2 * self.data_val[:, features_idx]  # scale the entries
            y_val = self.data_val[:, pred_idx]

            alpha_max = np.max(np.sum(np.abs(X_train), axis=0))

            if validate:
                alpha_list = [alpha_max * 0.5 ** i for i in range(n_steps)]
            else:
                alpha = 0.2 * math.sqrt(
                    math.log(self.P ** 2 / 0.05) * self.N
                )  # from Science paper
                alpha_list = [alpha]

            for alpha in alpha_list:
                if use_scikit:
                    model = LogisticRegression(
                        penalty='l1',
                        C=1.0 / alpha,
                        fit_intercept=False,
                        dual=False,
                        solver='liblinear',
                    ).fit(X_train, y_train)

                    beta = model.coef_.reshape(-1)
                else:
                    beta = train_L1_LogReg(X_train, y_train, alpha)

                if estimate_on_support:  # no penalty
                    support = np.where(beta != 0)[0]
                    beta_support = train_L1_LogReg(
                        X_train[:, support], y_train, 0, beta_start=beta[support]
                    ) # scikit does not support alpha = 0

                    beta = np.zeros(X_train.shape[1])
                    beta[support] = beta_support

                val_accu, val_lik = misc_loss(X_val, y_val, beta)
                print(
                    "Alpha: {}, (Neg. normalized) val lik:{}, val accu: {}".format(
                        round(alpha, 4), round(val_lik, 4), round(val_accu, 4)
                    )
                )
                if validate:
                    if val_lik < best_val_lik:
                        best_beta = beta
                        best_accu_lik = val_accu
                        best_val_lik = val_lik
                else:
                    best_beta = beta

            self.W[pred_idx, features_idx] = best_beta
            self.best_val_lik += best_val_lik


def train_L1_LogReg(X, y, alpha, eta=1e-3, T_max=300, beta_start=None):
    N, P = X.shape
    old_beta = np.ones(P)

    if beta_start is None:
        beta_m = np.zeros(P, dtype=float)
    else:
        beta_m = beta_start

    # Parameters for Accelerated GD
    t_AGD_old = 1
    t_AGD = 1
    eta_m_old = beta_m
    Lipchtiz_coeff = 0.25 * float(np.linalg.norm(X, ord='fro') ** 2)

    it = 0
    while np.linalg.norm(beta_m - old_beta) > eta and it < T_max:
        it += 1

        aux = y / (1 + np.exp(y * np.dot(X, beta_m)))
        gradient = -np.dot(X.T, aux)

        # Gradient descent
        old_beta = beta_m
        grad = beta_m - gradient / Lipchtiz_coeff

        # L1 thresholding
        eta_m = np.array(
            [soft_thresholding(grad[i], alpha / Lipchtiz_coeff) for i in range(P)]
        )

        # AGD update
        t_AGD = (1 + math.sqrt(1 + 4 * t_AGD_old ** 2)) / 2
        aux_t_AGD = (t_AGD_old - 1) / t_AGD
        beta_m = eta_m + aux_t_AGD * (eta_m - eta_m_old)

        t_AGD_old = t_AGD
        eta_m_old = eta_m
    return beta_m


def misc_loss(X, y, beta):
    # Returns misclassification rate, pseudo-likelihood loss
    N = float(X.shape[0])
    aux = y * (np.dot(X, beta))
    return np.sum(aux <= 0) / N, np.log(1 + np.exp(-aux)).sum() / N  # entries have been scaled


def soft_thresholding(c, llambda):
    if llambda >= abs(c):
        return 0
    else:
        if c >= 0:
            return c - llambda
        else:
            return c + llambda
