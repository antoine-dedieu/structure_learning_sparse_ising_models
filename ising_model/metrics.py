import math
import numpy as np

def recover_graph(W_gt, W_est, rho):
    # Computes the number of edges recovered and the binary full recovery 
    print("\n\nMETRICS")
    P = W_gt.shape[0]
    W_est_support = np.abs(W_est) > rho / 2

    edges_recovered = []
    for idx in range(W_est.shape[0]):
        gt_indexes = np.where(W_gt[idx])[0]
        est_indexes = np.where(W_est_support[idx])[0]
        print('Node {}, neighbors {}'.format(idx, est_indexes))

        edge_recovered = set(gt_indexes) == set(est_indexes)
        edges_recovered.append(edge_recovered)
    return sum(edges_recovered), sum(edges_recovered) == P
