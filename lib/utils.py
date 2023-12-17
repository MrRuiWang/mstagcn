import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import torch
from scipy.sparse import linalg
from scipy.sparse.linalg import eigs


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    """
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx / 1000, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def calculate_normalized_laplacian(adj, ):
    """
    # L = D ^ -0.5 * (A + I) * D ^ -0.5
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj += sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)

    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_laplacian = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
    normalized_laplacian = normalized_laplacian.astype(np.float32)

    return normalized_laplacian


def calculate_topalogy_laplacian(adj, k=2, undirected=True):
    """TAGN"""
    if undirected:
        adj = np.maximum.reduce([adj, adj.T])
    L = []
    L_1 = calculate_normalized_laplacian(adj).toarray()

    L.append(sp.coo_matrix(L_1).tocoo())

    for i in range(2, k + 1):
        L_temp = np.power(L_1, i)
        L.append(sp.coo_matrix(L_1).tocoo())

    return L

def calculate_random_walk_matrix(adj_mx):
    """
    # D^-1 * A
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def scaled_Laplacian(W):
    """
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    """

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    # 拉普拉斯矩阵定义：L = D - W(邻接矩阵)
    L = D - W

    # shift变换将特征值对角矩阵值转换到[-1, 1]
    # 为什么值区间转换成[-1, 1]?
    # Chebyshev多项式作为GCN卷积核有什么好处？
    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    """

    N = L_tilde.shape[0]

    # T_0: I, T_1: scaled_Laplacian matrix
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    # T_k = 2 * L_tilde * T_{k-1} - T_{k-2}
    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials
