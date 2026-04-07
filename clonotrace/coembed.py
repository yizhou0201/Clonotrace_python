"""
coembed.py - Clone-aware cell coembedding

Python port of R/coembed.R from Clonotrace_yuntian.
"""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from .auxiliary import (
    knn_flat, embedding2knn, long_symmetry, long2sparse
)


def cell_knn_matrix_mutiplication(knn, cell_feature_mat, feature_feature_mat):
    """Matrix multiplication for cell-feature weighted edges.

    Parameters
    ----------
    knn : pd.DataFrame  columns node1 (i), node2 (j)  (1-indexed)
    cell_feature_mat : np.ndarray  (N, F)
    feature_feature_mat : np.ndarray or sp.spmatrix  (F, F)

    Returns
    -------
    np.ndarray  scores per edge
    """
    i = (knn.iloc[:, 0].values - 1).astype(int)
    j = (knn.iloc[:, 1].values - 1).astype(int)

    if sp.issparse(feature_feature_mat):
        intermediate = feature_feature_mat.dot(cell_feature_mat.T)  # (F, N)
    else:
        intermediate = feature_feature_mat @ cell_feature_mat.T  # (F, N)

    cell_subset = cell_feature_mat[i]  # (E, F)
    inter_subset = intermediate[:, j].T  # (E, F)

    if sp.issparse(inter_subset):
        inter_subset = inter_subset.toarray()
    if sp.issparse(cell_subset):
        cell_subset = cell_subset.toarray()

    return np.sum(cell_subset * inter_subset, axis=1)


def cell_knn_matrix_multiplication_parallel(knn, cell_feature_mat, feature_feature_mat,
                                             chunk_size=5000):
    """Parallelized matrix multiplication for kNN edge weighting.

    Parameters
    ----------
    knn : pd.DataFrame  kNN edges (columns node1, node2)
    cell_feature_mat : np.ndarray  (N, F)
    feature_feature_mat : sp.spmatrix  (F, F)
    chunk_size : int

    Returns
    -------
    np.ndarray  scores per edge
    """
    if sp.issparse(feature_feature_mat):
        feature_feature_mat = sp.csr_matrix(feature_feature_mat)

    n_edges = len(knn)
    chunks = [list(range(s, min(s + chunk_size, n_edges)))
              for s in range(0, n_edges, chunk_size)]

    def _process_chunk(chunk_idx):
        sub_knn = knn.iloc[chunk_idx]
        return cell_knn_matrix_mutiplication(sub_knn, cell_feature_mat, feature_feature_mat)

    results = Parallel(n_jobs=1)(delayed(_process_chunk)(c) for c in chunks)
    return np.concatenate(results)


def cell_clone_coembed(cell_embedding, clone_embedding, cell_clone_prob,
                       cell_k=30, clone_k=15):
    """Construct cell-cell distance matrix via clone-aware coembedding.

    Parameters
    ----------
    cell_embedding : np.ndarray  (N_cells, D)
    clone_embedding : np.ndarray  (N_clones, D2)
    cell_clone_prob : np.ndarray or sp.spmatrix  (N_cells, N_clones)
    cell_k : int  cell kNN neighbors
    clone_k : int  clone kNN neighbors

    Returns
    -------
    sp.csr_matrix  symmetric distance matrix (N_cells, N_cells)
    """
    cell_embedding = np.asarray(cell_embedding)
    n_cells = len(cell_embedding)

    # Cell kNN (with self, deduplicated)
    cell_knn_flat = knn_flat(cell_embedding, k=cell_k,
                              if_dedup=True, symmetric=False, if_self=True)

    # Clone kNN (binary adjacency, with self)
    clone_knn = embedding2knn(clone_embedding, k=clone_k, mode="connectivity")
    # Binarize clone kNN
    clone_knn_bin = clone_knn.copy()
    clone_knn_bin.data = np.ones_like(clone_knn_bin.data)

    if sp.issparse(cell_clone_prob):
        cell_clone_prob = cell_clone_prob.toarray()
    cell_clone_prob = np.asarray(cell_clone_prob, dtype=float)

    # Edge weights via clone-informed matrix multiplication
    weights = cell_knn_matrix_multiplication_parallel(
        cell_knn_flat, cell_clone_prob, clone_knn_bin, chunk_size=5000
    )
    cell_knn_flat = cell_knn_flat.copy()
    cell_knn_flat["weight"] = weights

    # Symmetrize
    cell_knn_mat = long_symmetry(cell_knn_flat, row_names_from="node1",
                                  col_names_from="node2")

    # Filter and compute distance
    import pandas as pd
    distance = cell_knn_mat[cell_knn_mat["weight"] > 0.1].copy()
    distance["dis"] = distance["dist"] / distance["weight"]
    distance = (distance
                .groupby("node1", group_keys=False)
                .apply(lambda g: g.nsmallest(20, "dis")))

    mat = long2sparse(
        distance,
        row_names_from="node1",
        col_names_from="node2",
        values_from="dis",
        unique_rows=list(range(1, n_cells + 1)),
        unique_cols=list(range(1, n_cells + 1)),
        symmetric=True,
    )

    # Zero diagonal
    mat = mat.tolil()
    mat.setdiag(0)
    mat = mat.tocsr()
    mat.eliminate_zeros()
    return mat
