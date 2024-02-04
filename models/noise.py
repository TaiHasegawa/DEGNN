import random
import numpy as np
import torch
import scipy.sparse as sp


def add_noise(adj, features, edge_noise_ratio, node_noise_ratio):
    """
    Add noise.
    """
    noisy_adj = add_random_noise_edge(adj, edge_noise_ratio)
    noisy_features = add_gaussian_noise_node(features, node_noise_ratio)
    return noisy_adj, noisy_features

def add_random_noise_edge(adj, noise_ratio):
    """
    Add random noise to the adjacency matrix.
    """
    n_nodes = adj.shape[0]
    if np.array_equal(adj.todense(), adj.todense().T):
        adj_triu = sp.triu(adj - sp.eye(n_nodes)).A
        row, col = adj_triu.nonzero()
        n_edges = row.shape[0]
        remove_num = int(n_edges * noise_ratio)
        add_num =  int(n_edges * noise_ratio)
        # remove
        remove_inds = random.sample(range(n_edges), k=remove_num)
        remove_edges_mat = sp.csr_matrix((np.ones(remove_num), (row[remove_inds], col[remove_inds])), shape=(n_nodes, n_nodes))
        # inject
        zero_row, zero_col = sp.triu(1 - adj.A).nonzero()
        n_nonedge = zero_row.shape[0]
        add_inds = random.sample(range(n_nonedge), k=add_num)
        add_edges_mat = sp.csr_matrix((np.ones(add_num), (zero_row[add_inds], zero_col[add_inds])), shape=(n_nodes, n_nodes))

        noisy_adj = adj - remove_edges_mat - remove_edges_mat.T + add_edges_mat + add_edges_mat.T
        noisy_adj.setdiag(1)
    else:
        adj_temp = adj.todense()
        adj_temp = adj_temp - sp.eye(n_nodes)
        # remove
        total_edges = np.sum(adj_temp)
        edges_to_remove = int(noise_ratio * total_edges)
        indices = np.where(adj_temp == 1)
        random_indices = np.random.choice(len(indices[0]), edges_to_remove, replace=False)
        adj_temp[indices[0][random_indices], indices[1][random_indices]] = 0
        # add
        edges_to_add = edges_to_remove
        zero_indices = np.where(adj_temp == 0)
        random_indices = np.random.choice(len(zero_indices[0]), edges_to_add, replace=False)
        adj_temp[zero_indices[0][random_indices], zero_indices[1][random_indices]] = 1
        noisy_adj = adj_temp
        noisy_adj = np.array(noisy_adj)
        noisy_adj = sp.csr_matrix(noisy_adj)
        noisy_adj.setdiag(1)
    return noisy_adj

def add_gaussian_noise_node(features, noise_ratio):
    """
    Add independent Gaussian noise.
    """
    r = torch.max(features.clone().detach(), dim=1).values.mean().item()
    noisy_features = features.clone().detach()
    eps = torch.randn(size=features.shape)
    noise = noise_ratio * r * eps
    noisy_features += noise
    return noisy_features