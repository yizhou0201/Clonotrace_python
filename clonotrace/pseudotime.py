"""
pseudotime.py - Pseudotime inference

Python port of R/pseudotime.R from Clonotrace_yuntian.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, spsolve, LinearOperator

from .auxiliary import embedding2knn

# Threshold above which we avoid materializing the dense N×N outer product
_LINOP_THRESHOLD = 5000


def _build_acct_operator(T_mat):
    """Build the ACT linear operator A = I + phi0*phi0^T - T from transition matrix.

    For N <= _LINOP_THRESHOLD, returns a sparse matrix (compatible with splu).
    For N > _LINOP_THRESHOLD, returns a LinearOperator to avoid the dense
    N×N outer-product allocation.

    Parameters
    ----------
    T_mat : sp.spmatrix or np.ndarray  (N, N) transition matrix

    Returns
    -------
    A : sp.spmatrix or LinearOperator  the linear operator
    phi0 : np.ndarray  dominant eigenvector
    """
    T_mat = sp.csr_matrix(T_mat, dtype=float)
    N = T_mat.shape[0]

    # Use deterministic start vector for reproducible eigenvectors
    v0 = np.ones(N, dtype=float) / np.sqrt(N)
    vals, vecs = eigsh(T_mat, k=1, which="LM", v0=v0)
    phi0 = np.array(vecs[:, 0], dtype=float).ravel()

    if N <= _LINOP_THRESHOLD:
        I = sp.eye(N, format="csr", dtype=float)
        outer = sp.csr_matrix(np.outer(phi0, phi0))
        A = I + outer - T_mat
    else:
        # Avoid N×N dense outer product: represent A as a LinearOperator
        def matvec(x):
            return x + phi0 * np.dot(phi0, x) - T_mat.dot(x)
        A = LinearOperator((N, N), matvec=matvec, dtype=float)

    return A, phi0


def acct(T_mat):
    """Compute accumulated commute time (ACT) matrix from transition matrix.

    Parameters
    ----------
    T_mat : sp.spmatrix or np.ndarray  (N, N) transition matrix

    Returns
    -------
    np.ndarray  (N, N) ACT matrix
    """
    A, phi0 = _build_acct_operator(T_mat)
    N = phi0.shape[0]

    # Solve A @ M = I  => M = A^{-1} - I
    if sp.issparse(A):
        try:
            from scipy.sparse.linalg import splu
            lu = splu(sp.csc_matrix(A))
            M = lu.solve(np.eye(N))
        except Exception:
            # Dense fallback
            M = np.linalg.solve(A.toarray(), np.eye(N))
    else:
        # A is a LinearOperator (large N) — must use iterative CG
        M = np.zeros((N, N))
        e_i = np.zeros(N)
        for i in range(N):
            e_i[:] = 0.0
            e_i[i] = 1.0
            M[:, i], _ = sp.linalg.cg(A, e_i, maxiter=500)

    M = M - np.eye(N)
    return M


def DPT_T(T_mat, start):
    """Diffusion pseudotime from ACT matrix.

    Memory-efficient: streams columns of A^{-1} and accumulates running
    sums for the distance computation, using O(N) memory instead of O(N²).

    For N > _LINOP_THRESHOLD, also avoids the dense outer-product
    allocation by using a LinearOperator.

    Parameters
    ----------
    T_mat : sp.spmatrix or np.ndarray
    start : int  root cell index (0-based)

    Returns
    -------
    np.ndarray  (N,) pseudotime values
    """
    A, phi0 = _build_acct_operator(T_mat)
    N = phi0.shape[0]

    # We need ||M[i,:] - M[start,:]||² for each i, where M = A⁻¹ - I.
    # Expanding: sum_j (M[i,j] - M[start,j])² = row_sq[i] - 2*cross[i] + row_sq[start]
    # where row_sq[i] = sum_j M[i,j]² and cross[i] = sum_j M[i,j]*M[start,j].
    # We accumulate these by streaming one column of M at a time: O(N) memory.
    row_sq = np.zeros(N)
    cross = np.zeros(N)

    try:
        if sp.issparse(A):
            from scipy.sparse.linalg import splu
            lu = splu(sp.csc_matrix(A))
            e_j = np.zeros(N)
            for j in range(N):
                e_j[:] = 0.0
                e_j[j] = 1.0
                col_j = lu.solve(e_j)
                col_j[j] -= 1.0  # M = A^{-1} - I
                row_sq += col_j * col_j
                cross += col_j * col_j[start]
        else:
            raise RuntimeError("use CG path")
    except Exception:
        # Fallback: iterative CG (works with LinearOperator)
        e_j = np.zeros(N)
        for j in range(N):
            e_j[:] = 0.0
            e_j[j] = 1.0
            col_j, _ = sp.linalg.cg(A, e_j, maxiter=500)
            col_j[j] -= 1.0
            row_sq += col_j * col_j
            cross += col_j * col_j[start]

    dpt_sq = row_sq - 2.0 * cross + row_sq[start]
    return np.sqrt(np.maximum(dpt_sq, 0.0))


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
