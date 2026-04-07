"""
pseudotime.py - Pseudotime inference

Python port of R/pseudotime.R from Clonotrace_yuntian.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve

from .auxiliary import embedding2knn


def acct(T_mat):
    """Compute accumulated commute time (ACT) matrix from transition matrix.

    Parameters
    ----------
    T_mat : sp.spmatrix or np.ndarray  (N, N) transition matrix

    Returns
    -------
    np.ndarray  (N, N) ACT matrix
    """
    T_mat = sp.csr_matrix(T_mat, dtype=float)
    N = T_mat.shape[0]

    # Dominant eigenvector
    vals, vecs = eigsh(T_mat, k=1, which="LM")
    phi0 = np.array(vecs[:, 0], dtype=float).ravel()

    # A = I + phi0 * phi0^T - T_mat
    I = sp.eye(N, format="csr", dtype=float)
    outer = sp.csr_matrix(np.outer(phi0, phi0))
    A = I + outer - T_mat

    # Solve A @ M = I  => M = A^{-1} - I
    M = np.zeros((N, N))
    e_i = np.zeros(N)
    for i in range(N):
        e_i[:] = 0.0
        e_i[i] = 1.0
        col, _ = sp.linalg.cg(A, e_i, maxiter=500)
        M[:, i] = col
    M = M - np.eye(N)
    return M


def DPT_T(T_mat, start):
    """Diffusion pseudotime from ACT matrix.

    Parameters
    ----------
    T_mat : sp.spmatrix or np.ndarray
    start : int  root cell index (0-based)

    Returns
    -------
    np.ndarray  (N,) pseudotime values
    """
    M = acct(T_mat)
    diff = M - M[start]
    return np.sqrt(np.sum(diff ** 2, axis=1))


def dpt(T_mat, root, k=30):
    """Diffusion pseudotime via eigendecomposition.

    Parameters
    ----------
    T_mat : sp.spmatrix or np.ndarray
    root : int  root cell index (0-based)
    k : int  number of eigenvectors

    Returns
    -------
    np.ndarray  (N,) normalized pseudotime in [0, 1]
    """
    T_mat = sp.csr_matrix(T_mat, dtype=float)
    k_actual = min(k, T_mat.shape[0] - 2)

    eigenvalues, eigenvectors = eigsh(T_mat, k=k_actual, which="LM")
    # Sort by descending eigenvalue
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Filter eigenvalues < 0.9999
    flag = eigenvalues < 0.9999
    eigenvalues = eigenvalues[flag]
    eigenvectors = eigenvectors[:, flag]

    if len(eigenvalues) == 0:
        return np.zeros(T_mat.shape[0])

    # Scale: lambda / (1 - lambda)
    scale = eigenvalues / np.maximum(1 - eigenvalues, 1e-12)
    eigenvectors = eigenvectors * scale[np.newaxis, :]

    diff = eigenvectors - eigenvectors[root]
    pseudotime = np.sqrt(np.sum(diff ** 2, axis=1))
    max_pt = pseudotime.max()
    if max_pt > 0:
        pseudotime = pseudotime / max_pt
    return pseudotime


def embedding2dpt(embedding, nn_k, root, dpt_k=30):
    """Compute diffusion pseudotime from embedding.

    Parameters
    ----------
    embedding : np.ndarray  (N, D)
    nn_k : int  kNN neighbors
    root : int  root cell index (0-based)
    dpt_k : int  eigenvectors for DPT

    Returns
    -------
    np.ndarray  (N,) pseudotime
    """
    knn = embedding2knn(embedding, nn_k)
    # Row-normalize
    row_sums = np.array(knn.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1e-12)
    D_inv = sp.diags(1.0 / row_sums)
    T_mat = D_inv.dot(knn)
    return dpt(T_mat, root=root, k=dpt_k)


def clone_root(clones, cell_meta, clone_col, cluster_col, start_cluster):
    """Identify root clone from cluster enrichment.

    Parameters
    ----------
    clones : list  clone identifiers
    cell_meta : pd.DataFrame  with clone_col and cluster_col
    clone_col : str
    cluster_col : str
    start_cluster : str or int

    Returns
    -------
    str  name of the most enriched clone in start_cluster
    """
    import pandas as pd

    sub = cell_meta[cell_meta[clone_col].isin(clones)]
    summary = (sub.groupby(clone_col)
               .apply(lambda g: pd.Series({
                   "ratio": (g[cluster_col] == start_cluster).mean(),
                   "count": len(g),
               }))
               .reset_index()
               .sort_values(["ratio", "count"], ascending=[False, False]))
    return str(summary.iloc[0][clone_col])


def clone_dpt(clone_embedding, cell_meta, clone_col, cluster_col, start_cluster,
              k=10, dpt_k=30, clone_names=None):
    """Clone-level diffusion pseudotime.

    Parameters
    ----------
    clone_embedding : np.ndarray or pd.DataFrame  (N_clones, D)
        If DataFrame, row index is used as clone names.
    cell_meta : pd.DataFrame
    clone_col, cluster_col : str
    start_cluster : str or int
    k, dpt_k : int
    clone_names : list or None
        Explicit clone names; inferred from DataFrame index if not provided.

    Returns
    -------
    np.ndarray  (N_clones,) pseudotime
    """
    import pandas as pd

    if isinstance(clone_embedding, pd.DataFrame):
        clone_names = clone_embedding.index.tolist()
        clone_embedding = clone_embedding.values
    else:
        clone_embedding = np.asarray(clone_embedding)
        if clone_names is None:
            clone_names = list(range(clone_embedding.shape[0]))
        else:
            clone_names = list(clone_names)

    knn = embedding2knn(clone_embedding, k)
    row_sums = np.array(knn.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1e-12)
    D_inv = sp.diags(1.0 / row_sums)
    T_mat = D_inv.dot(knn)

    root_clone = clone_root(clone_names, cell_meta, clone_col, cluster_col, start_cluster)
    root_idx = clone_names.index(root_clone) if root_clone in clone_names else 0

    return dpt(T_mat, root=root_idx, k=dpt_k)
