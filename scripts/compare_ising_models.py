import math
import os
import sys

import numpy as np

import torch
from ising_model import (
    data,
    l0_l2constrained_ise,
    l0_l2constrained_logreg,
    l1_constrained_logreg,
    l1_ise,
    l1_logreg,
    metrics,
)

# Parameters
current_id = sys.argv[1]
dataset_root = sys.argv[2]
N = int(sys.argv[3])
P = int(sys.argv[4])
rho = float(sys.argv[5])

type_matrix = sys.argv[6]
assert type_matrix in [
    'periodic_graph_uniform_sign',
    'periodic_graph_random_sign',
    'random_graph_uniform_sign',
    'random_graph_random_sign',
    'random_graph_uniform_sign_uniform_val',
    'random_graph_random_sign_uniform_val',
]

# Folder for results
string = type_matrix + '_N_{}_P_{}_rho_{}'.format(N, P, rho)

folder = os.path.join(dataset_root, string)
if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)
savetofile = os.path.join(dataset_root, string, current_id + '.npz')


# Data generation
if P <= 16:
    W, data_train, data_val = data.generate_train_val_data(
        N, P, rho, type_matrix, seed=int(current_id)
    )
else:
    W, data_train, data_val = data.generate_train_val_data_gibbs(
        N, P, rho, type_matrix, seed=int(current_id)
    )
Ks = [7, 6, 5, 4, 3, 2, 1]


#############################
############# LR ############
#############################

# L1 LR with scikit, no validation
L1LogReg = l1_logreg.L1_LogReg(data_train, data_val)
L1LogReg.estimate_W(validate=False, use_scikit=True)
W_L1_LR = L1LogReg.W
W_L1_LR += W_L1_LR.T
W_L1_LR /= 2
n_vertex_L1_LR_scikit, recovered_L1_LR_scikit = metrics.recover_graph(W, W_L1_LR, rho)

# L1 LR with scikit, validation: performs better in practice
L1LogReg = l1_logreg.L1_LogReg(data_train, data_val)
L1LogReg.estimate_W(validate=True, use_scikit=True)
W_L1_LR_val = L1LogReg.W
W_L1_LR_val += W_L1_LR_val.T
W_L1_LR_val /= 2
n_vertex_L1_LR_scikit_val, recovered_L1_LR_scikit_val = metrics.recover_graph(
    W, W_L1_LR_val, rho
)


#############################
###### LR Constrained #######
#############################

# L1 consrained LR with scikit, validation
L1ConstrainedLogReg = l1_constrained_logreg.L1Constrained_LogReg(data_train, data_val)
L1ConstrainedLogReg.estimate_W(validate=True)
W_L1_constrained_LR = L1ConstrainedLogReg.W
W_L1_constrained_LR += W_L1_constrained_LR.T
W_L1_constrained_LR /= 2
n_vertex_L1_constrained_LR, recovered_L1_constrained_LR = metrics.recover_graph(
    W, W_L1_constrained_LR, rho
)

#############################
######### L0-L2 LR ##########
#############################

# L0-L2 LR with warm-start
vals_L0L2_LR = []
bics_L0L2_LR = []
n_vertices_L0L2_LR = []
recovered_L0L2_LR = []
Ws_L0L2_LR = []

W_init = W_L1_LR_val.copy()
for K in Ks:
    L0L2ConstrainedLogReg = l0_l2constrained_logreg.L0L2Constrained_LogReg(
        data_train, data_val, W_init=W_init
    )
    # The continuation heuristic allows us to not tune the regularization parameter
    L0L2ConstrainedLogReg.estimate_W(validate=False, K=K)

    W_L0L2_LR = L0L2ConstrainedLogReg.W
    W_L0L2_LR += W_L0L2_LR.T
    W_L0L2_LR /= 2
    n_vertices_K, recovered_K = metrics.recover_graph(W, W_L0L2_LR, 0)  # rho is unused for L0-L2 LR

    val = L0L2ConstrainedLogReg.best_val_lik
    sparsity = len(np.where(W_L0L2_LR != 0)[0])
    bic = 2 * N * val + sparsity * math.log(N)

    vals_L0L2_LR.append(val)
    bics_L0L2_LR.append(bic)
    n_vertices_L0L2_LR.append(n_vertices_K)
    recovered_L0L2_LR.append(recovered_K)
    Ws_L0L2_LR.append(W_L0L2_LR)

    W_init = W_L0L2_LR.copy()


#############################
############# ISE ###########
#############################

# ISE, no validation
L1ISE = l1_ise.L1_ISE(data_train, data_val)
L1ISE.estimate_W(validate=False)
W_L1_ISE = L1ISE.W
W_L1_ISE += W_L1_ISE.T
W_L1_ISE /= 2
n_vertex_L1_ISE, recovered_L1_ISE = metrics.recover_graph(W, W_L1_ISE, rho)

# ISE, validation: performs better in practice
L1ISE = l1_ise.L1_ISE(data_train, data_val)
L1ISE.estimate_W(validate=True)
W_L1_ISE_val = L1ISE.W
W_L1_ISE_val += W_L1_ISE_val.T
W_L1_ISE_val /= 2
n_vertex_L1_ISE_val, recovered_L1_ISE_val = metrics.recover_graph(W, W_L1_ISE_val, rho)


#############################
######### L0-L2 ISE #########
#############################

# L0-L2 ISE, warm-start
W_init = W_L1_ISE_val.copy()
vals_L0L2_ISE = []
bics_L0L2_ISE = []
n_vertices_L0L2_ISE = []
recovered_L0L2_ISE = []
Ws_L0L2_ISE = []

for K in Ks:
    L0L2ISE = l0_l2constrained_ise.L0L2Constrained_ISE(
        data_train, data_val, W_init=W_init
    )
    # The continuation heuristic allows us to not tune the regularization parameter
    L0L2ISE.estimate_W(validate=False, K=K)

    W_L0L2_ISE = L0L2ISE.W
    W_L0L2_ISE += W_L0L2_ISE.T
    W_L0L2_ISE /= 2
    n_vertices_K, recovered_K = metrics.recover_graph(W, W_L0L2_ISE, 0)  # rho is unused for L0-L2 ISE

    val = L0L2ISE.best_val_lik
    sparsity = len(np.where(W_L0L2_ISE != 0)[0])
    bic = 2 * N * val + sparsity * math.log(N)

    vals_L0L2_ISE.append(val)
    bics_L0L2_ISE.append(bic)
    n_vertices_L0L2_ISE.append(n_vertices_K)
    recovered_L0L2_ISE.append(recovered_K)
    Ws_L0L2_ISE.append(W_L0L2_ISE)

    W_init = W_L0L2_ISE.copy()


# Save results
np.savez_compressed(
    savetofile,
    Ks=Ks,
    W_gt=W,
    # L1-LR
    n_vertex_L1_LR_scikit=n_vertex_L1_LR_scikit,
    recovered_L1_LR_scikit=recovered_L1_LR_scikit,
    W_L1_LR=W_L1_LR,
    n_vertex_L1_LR_scikit_val=n_vertex_L1_LR_scikit_val,
    recovered_L1_LR_scikit_val=recovered_L1_LR_scikit_val,
    W_L1_LR_val=W_L1_LR_val,
    # L1-constrained LR
    W_L1_constrained_LR=W_L1_constrained_LR,
    n_vertex_L1_constrained_LR=n_vertex_L1_constrained_LR,
    recovered_L1_constrained_LR=recovered_L1_constrained_LR,
    # L0-L2 LR
    vals_L0L2_LR=vals_L0L2_LR,
    bics_L0L2_LR=bics_L0L2_LR,
    n_vertices_L0L2_LR=n_vertices_L0L2_LR,
    recovered_L0L2_LR=recovered_L0L2_LR,
    Ws_L0L2_LR=Ws_L0L2_LR,
    # L1-ISE
    n_vertex_L1_ISE=n_vertex_L1_ISE,
    recovered_L1_ISE=recovered_L1_ISE,
    W_L1_ISE=W_L1_ISE,
    n_vertex_L1_ISE_val=n_vertex_L1_ISE_val,
    recovered_L1_ISE_val=recovered_L1_ISE_val,
    W_L1_ISE_val=W_L1_ISE_val,
    # L0-L2 ISE
    vals_L0L2_ISE=vals_L0L2_ISE,
    bics_L0L2_ISE=bics_L0L2_ISE,
    n_vertices_L0L2_ISE=n_vertices_L0L2_ISE,
    recovered_L0L2_ISE=recovered_L0L2_ISE,
    Ws_L0L2_ISE=Ws_L0L2_ISE,
)
