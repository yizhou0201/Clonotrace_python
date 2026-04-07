"""
auxiliary.py - Utility functions for Clonotrace

Python port of R/auxiliary.R from Clonotrace_yuntian.
"""

import math
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

def long2wide(long, row_names_from, col_names_from, values_from, symmetric=False):
    """Convert long-format DataFrame to wide-format DataFrame.

    Parameters
    ----------
    long : pd.DataFrame
    row_names_from, col_names_from, values_from : str  column names
    symmetric : bool  if True, add reversed pairs and deduplicate

    Returns
    -------
    pd.DataFrame  wide matrix with row/col names
    """
    long = long[[row_names_from, col_names_from, values_from]].copy()
    if symmetric:
        rlong = long[[col_names_from, row_names_from, values_from]].copy()
        rlong.columns = [row_names_from, col_names_from, values_from]
        long = pd.concat([long, rlong], ignore_index=True).drop_duplicates()

    out = long.pivot_table(
        index=row_names_from,
        columns=col_names_from,
        values=values_from,
        aggfunc="first",
    )
    out.columns.name = None
    if symmetric:
        nodes = sorted(out.index.tolist())
        out = out.reindex(index=nodes, columns=nodes)
    return out


def long_symmetry(long, row_names_from, col_names_from):
    """Enforce symmetry in a long-format pairwise DataFrame.

    Returns a symmetric long DataFrame with reversed entries added.
    """
    long = long.copy()
    side = [c for c in long.columns if c not in (row_names_from, col_names_from)]
    long = long[[row_names_from, col_names_from] + side]

    rlong = long[[col_names_from, row_names_from] + side].copy()
    rlong.columns = [row_names_from, col_names_from] + side

    out = pd.concat([long, rlong], ignore_index=True).drop_duplicates()
    out = out.sort_values([row_names_from, col_names_from]).reset_index(drop=True)
    return out


def long2square(long, row_names_from, col_names_from, values_from,
                symmetric=True, na_fill=np.nan, nodes=None):
    """Convert long-format DataFrame to a square numpy matrix.

    Parameters
    ----------
    symmetric : bool  enforce symmetric pairs
    na_fill : scalar  fill value for missing entries
    nodes : list or None  explicit node list

    Returns
    -------
    np.ndarray  square matrix, row/col order = sorted(nodes)
    """
    long = long[[row_names_from, col_names_from, values_from]].copy()
    if symmetric:
        long = long_symmetry(long, row_names_from, col_names_from)

    if nodes is None:
        nodes = sorted(set(long[row_names_from].tolist() + long[col_names_from].tolist()),
                       key=str)
    else:
        # Preserve caller's ordering; convert to strings but do NOT re-sort
        nodes = [str(n) for n in nodes]

    long[row_names_from] = long[row_names_from].astype(str)
    long[col_names_from] = long[col_names_from].astype(str)

    mat_df = long.pivot_table(
        index=row_names_from, columns=col_names_from,
        values=values_from, aggfunc="first"
    )
    mat_df = mat_df.reindex(index=nodes, columns=nodes)
    mat = mat_df.fillna(na_fill).values.astype(float)
    return mat


def long2sparse(long, row_names_from, col_names_from, values_from,
                unique_rows=None, unique_cols=None, symmetric=False):
    """Convert long-format DataFrame to scipy sparse matrix (CSR).

    Parameters
    ----------
    unique_rows, unique_cols : array-like or None  row/col label order
    symmetric : bool  add reversed pairs

    Returns
    -------
    sp.csr_matrix with .rownames and .colnames attributes set
    """
    long = long[[row_names_from, col_names_from, values_from]].copy()
    if symmetric:
        long = long_symmetry(long, row_names_from, col_names_from)

    if unique_rows is None:
        unique_rows = list(pd.unique(long[row_names_from]))
    if unique_cols is None:
        unique_cols = list(pd.unique(long[col_names_from]))

    row_map = {v: i for i, v in enumerate(unique_rows)}
    col_map = {v: i for i, v in enumerate(unique_cols)}

    rows = long[row_names_from].map(row_map).values
    cols = long[col_names_from].map(col_map).values
    vals = long[values_from].values.astype(float)

    mask = (~np.isnan(rows.astype(float))) & (~np.isnan(cols.astype(float)))
    mat = sp.csr_matrix(
        (vals[mask], (rows[mask].astype(int), cols[mask].astype(int))),
        shape=(len(unique_rows), len(unique_cols))
    )
    mat.rownames = list(unique_rows)
    mat.colnames = list(unique_cols)
    return mat


def wide2long(mat):
    """Convert a 2-D array to long-format DataFrame with columns i, j, value."""
    mat = np.asarray(mat)
    n_rows, n_cols = mat.shape
    ii, jj = np.meshgrid(np.arange(1, n_rows + 1),
                          np.arange(1, n_cols + 1), indexing="ij")
    return pd.DataFrame({
        "i": ii.ravel(),
        "j": jj.ravel(),
        "value": mat.ravel(),
    })


# ---------------------------------------------------------------------------
# Graph / cluster helpers
# ---------------------------------------------------------------------------

def link2cluster(link, nodes):
    """Cluster nodes based on link connectivity via matrix diffusion + DBSCAN.

    Parameters
    ----------
    link : pd.DataFrame  columns (i, j)
    nodes : array-like  all node names

    Returns
    -------
    np.ndarray  cluster labels (1-indexed)
    """
    nodes = list(nodes)
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    rows, cols = [], []
    for _, row in link.iterrows():
        ri, ci = node_idx.get(row.iloc[0]), node_idx.get(row.iloc[1])
        if ri is not None and ci is not None:
            rows += [ri, ci]
            cols += [ci, ri]
    # self-loops
    rows += list(range(n))
    cols += list(range(n))

    vals = np.ones(len(rows))
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # Matrix power via repeated squaring
    A_f = A.astype(float)
    result = A_f.copy()
    for _ in range(4):  # 2^4 = 16 ≈ 20 in R
        result = result.dot(result)
    result = result.toarray()
    dist_mat = 1.0 - result

    db = DBSCAN(eps=0, min_samples=1, metric="precomputed")
    labels = db.fit_predict(dist_mat) + 1  # 1-indexed
    return labels


def mnn_dist(dis, k):
    """Mutual nearest neighbors from a distance matrix.

    Returns
    -------
    pd.DataFrame  columns i, j  (i < j)
    """
    dis = np.asarray(dis)
    n = dis.shape[0]
    nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    nn.fit(dis)
    dists, indices = nn.kneighbors(dis)

    # Build directed edge sets
    id_df = pd.DataFrame({
        "i": np.repeat(np.arange(n), k),
        "j": indices.ravel(),
    })
    rid_df = id_df.rename(columns={"i": "j", "j": "i"})

    mnn = id_df.merge(rid_df, on=["i", "j"])
    mnn = mnn[mnn["i"] < mnn["j"]].drop_duplicates().reset_index(drop=True)
    return mnn


def nearest_knn(dis, k, top=3):
    """Top nearest-neighbor pairs by distance (order-insensitive dedup).

    Returns
    -------
    pd.DataFrame  columns i, j, dis
    """
    dis = np.asarray(dis)
    n = dis.shape[0]
    nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    nn.fit(dis)
    dists, indices = nn.kneighbors(dis)

    df = pd.DataFrame({
        "i": np.repeat(np.arange(n), k),
        "j": indices.ravel(),
        "dis": dists.ravel(),
    })
    df["_key"] = df.apply(lambda r: tuple(sorted((r["i"], r["j"]))), axis=1)
    df = df.drop_duplicates("_key").drop(columns="_key")
    df = df.sort_values("dis").reset_index(drop=True)
    return df.iloc[: min(top, len(df))]


def cluster_merge(input_list, cluster):
    """Merge list elements by cluster assignment."""
    if len(input_list) != len(cluster):
        raise ValueError("input_list and cluster must have the same length")
    from collections import defaultdict
    groups = defaultdict(list)
    for item, c in zip(input_list, cluster):
        groups[c].extend(item if hasattr(item, "__iter__") else [item])
    return list(groups.values())


def combn_dedup(combn_arr):
    """Remove duplicate (order-invariant) pairs.

    Parameters
    ----------
    combn_arr : array-like shape (N, 2)

    Returns
    -------
    np.ndarray bool  True where unique
    """
    arr = np.asarray(combn_arr)
    keys = [frozenset(row) for row in arr]
    seen = set()
    mask = []
    for k in keys:
        key_tuple = tuple(sorted(k))
        if key_tuple not in seen:
            seen.add(key_tuple)
            mask.append(True)
        else:
            mask.append(False)
    return np.array(mask)


def dismat_mst(mat):
    """Minimum spanning tree from a full distance matrix.

    Returns
    -------
    pd.DataFrame  columns from, to, weight
    """
    mat = np.asarray(mat, dtype=float)
    np.fill_diagonal(mat, 0)
    sparse = sp.csr_matrix(mat)
    mst = minimum_spanning_tree(sparse)
    mst_coo = mst.tocoo()
    return pd.DataFrame({
        "from": mst_coo.row.astype(int),
        "to": mst_coo.col.astype(int),
        "weight": mst_coo.data,
    })


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def dis_point_to_edge(point, edge_start, edge_end):
    """Distance from a point to a line segment and projection t in [0,1]."""
    P = np.asarray(point, dtype=float)
    A = np.asarray(edge_start, dtype=float)
    B = np.asarray(edge_end, dtype=float)
    AB = B - A
    AP = P - A
    denom = np.dot(AB, AB)
    t = np.dot(AP, AB) / denom if denom > 0 else 0.0
    t = float(np.clip(t, 0, 1))
    closest = A + t * AB
    distance = float(np.sqrt(np.sum((P - closest) ** 2)))
    return np.array([distance, t])


def dis_points_to_edges(points, edges):
    """Distances from multiple points to multiple edges.

    Parameters
    ----------
    points : array-like (N, D)
    edges : list of (2, D) arrays  [start; end]

    Returns
    -------
    dict with keys 'map' (N, E) and 'dis' (N, E)
    """
    points = np.asarray(points)
    n_pts = len(points)
    results = []
    for edge in edges:
        edge = np.asarray(edge)
        col = np.array([dis_point_to_edge(points[y], edge[0], edge[1])
                        for y in range(n_pts)])
        results.append(col)
    results = np.concatenate(results, axis=1)  # (N, 2*E)
    n_edges = len(edges)
    dis_map = results[:, np.arange(0, 2 * n_edges, 2)]   # distance values
    t_map = results[:, np.arange(1, 2 * n_edges, 2)]      # t projections
    return {"map": t_map, "dis": dis_map}


# ---------------------------------------------------------------------------
# kNN helpers
# ---------------------------------------------------------------------------

def knn_between_groups(distance, k):
    """k nearest neighbors for each row of a distance matrix.

    Returns
    -------
    pd.DataFrame  columns group1, group2, dist
    """
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    rows = []
    for i in range(n):
        x = distance[i]
        idx = np.argsort(x)[:k]
        for j in idx:
            rows.append((i, j, x[j]))
    return pd.DataFrame(rows, columns=["group1", "group2", "dist"])


def find_mutual_nn(distance, k, dis_thresh=None):
    """Mutual nearest neighbors between groups.

    Returns
    -------
    pd.DataFrame  columns group1, group2, dist
    """
    g1 = knn_between_groups(distance, k)
    g2 = knn_between_groups(distance.T, k)
    g2 = g2.rename(columns={"group1": "group2", "group2": "group1"})
    mnn = g1.merge(g2, on=["group1", "group2", "dist"])
    if dis_thresh is not None:
        mnn = mnn[mnn["dist"] < dis_thresh]
    return mnn.reset_index(drop=True)


def top_k(x, k):
    """Return the k smallest values of array x."""
    x = np.asarray(x)
    k = min(k, len(x))
    return np.partition(x, k - 1)[:k]


def knn_flat(x, k, input="matrix", symmetric=False, if_dedup=False, if_self=False):
    """Compute flattened kNN edges.

    Parameters
    ----------
    x : np.ndarray or distance matrix
    k : int
    input : 'matrix' or 'dist'
    symmetric : add reverse edges
    if_dedup : remove (i,j)/(j,i) duplicates
    if_self : add self-edges (i, i, 0)

    Returns
    -------
    pd.DataFrame  columns node1, node2, dist
    """
    if input == "matrix":
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(x)
        dists, indices = nn.kneighbors(x)
        n = len(x)
    else:
        x_arr = np.asarray(x)
        n = x_arr.shape[0]
        nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
        nn.fit(x_arr)
        dists, indices = nn.kneighbors(x_arr)

    flat = pd.DataFrame({
        "node1": np.repeat(np.arange(1, n + 1), k),
        "node2": (indices + 1).ravel(),
        "dist": dists.ravel(),
    })

    if symmetric:
        flat = long_symmetry(flat, "node1", "node2")
    if if_dedup:
        mask = combn_dedup(flat[["node1", "node2"]].values)
        flat = flat[mask].reset_index(drop=True)
    if if_self:
        self_df = pd.DataFrame({
            "node1": np.arange(1, n + 1),
            "node2": np.arange(1, n + 1),
            "dist": np.zeros(n),
        })
        flat = pd.concat([flat, self_df], ignore_index=True)
    return flat


def embedding2knn(embedding, k, mode="connectivity", **kwargs):
    """Build kNN sparse matrix from embedding.

    Parameters
    ----------
    mode : 'connectivity' (Gaussian kernel) or 'dist'

    Returns
    -------
    sp.csr_matrix  (N, N)
    """
    embedding = np.asarray(embedding)
    n = len(embedding)
    knn = knn_flat(embedding, k=k, input="matrix", symmetric=True, **kwargs)

    if mode == "connectivity":
        sigma_df = knn.groupby("node1")["dist"].mean().reset_index()
        sigma_df.columns = ["node1", "sigma"]
        knn = knn.merge(sigma_df, on="node1")
        sigma_j = sigma_df.rename(columns={"node1": "node2", "sigma": "sigma_j"})
        knn = knn.merge(sigma_j, on="node2")
        knn["connectivity"] = np.exp(
            -knn["dist"] ** 2 / (knn["sigma"] * knn["sigma_j"])
        )
        mat = long2sparse(
            knn, "node1", "node2", "connectivity",
            unique_rows=list(range(1, n + 1)),
            unique_cols=list(range(1, n + 1)),
            symmetric=False,
        )
    else:
        mat = long2sparse(
            knn, "node1", "node2", "dist",
            unique_rows=list(range(1, n + 1)),
            unique_cols=list(range(1, n + 1)),
            symmetric=False,
        )
    return mat


def dist2knn(embedding, k, mode="connectivity", **kwargs):
    """Build kNN sparse matrix from precomputed distance matrix."""
    embedding = np.asarray(embedding)
    n = embedding.shape[0]
    knn = knn_flat(embedding, k=k, input="dist", symmetric=True, **kwargs)

    if mode == "connectivity":
        sigma_df = knn.groupby("node1")["dist"].mean().reset_index()
        sigma_df.columns = ["node1", "sigma"]
        knn = knn.merge(sigma_df, on="node1")
        knn["connectivity"] = np.exp(-knn["dist"] ** 2 / (2 * knn["sigma"] ** 2))
        mat = long2sparse(
            knn, "node1", "node2", "connectivity",
            unique_rows=list(range(1, n + 1)),
            unique_cols=list(range(1, n + 1)),
            symmetric=False,
        )
    else:
        mat = long2sparse(
            knn, "node1", "node2", "dist",
            unique_rows=list(range(1, n + 1)),
            unique_cols=list(range(1, n + 1)),
            symmetric=False,
        )
    return mat


def sync_sparse_rows(mat, row_names):
    """Re-index sparse matrix to match row_names order (insert zeros for missing)."""
    if sp.issparse(mat):
        if hasattr(mat, "rownames"):
            existing = mat.rownames
        else:
            existing = list(range(mat.shape[0]))
        idx = [existing.index(r) if r in existing else None for r in row_names]
        data = mat.toarray()
        new_data = np.zeros((len(row_names), data.shape[1]))
        for new_i, old_i in enumerate(idx):
            if old_i is not None:
                new_data[new_i] = data[old_i]
        result = sp.csr_matrix(new_data)
        result.rownames = list(row_names)
        if hasattr(mat, "colnames"):
            result.colnames = mat.colnames
        return result
    elif isinstance(mat, pd.DataFrame):
        return mat.reindex(row_names).fillna(0)
    else:
        raise TypeError("mat must be sparse matrix or DataFrame")


# ---------------------------------------------------------------------------
# Sparse matrix utilities
# ---------------------------------------------------------------------------

def sparse_norm(mat, norm="l2"):
    """Row-normalize a sparse matrix."""
    from sklearn.preprocessing import normalize
    return normalize(mat, norm=norm, axis=1)


def sparse_manipulation(mat, func):
    """Apply func to non-zero values of sparse matrix."""
    mat = mat.tocsr().copy()
    mat.data = func(mat.data)
    return mat


def compute_transition(adj):
    """Double-stochastic transition matrix: T = Z * D^{-1} * A * D^{-1} * Z.

    Matches R compute_transition: D = diag(rowSums(A)),
    M = D^{-1} A D^{-1}, Z = diag(1/sqrt(rowSums(M))), T = Z M Z.
    """
    adj = sp.csr_matrix(adj, dtype=float)
    row_sums = np.array(adj.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1e-12)
    D_inv = sp.diags(1.0 / row_sums)
    M = D_inv.dot(adj).dot(D_inv)
    z_vals = np.array(M.sum(axis=1)).ravel()
    z_vals = np.maximum(z_vals, 1e-12)
    Z = sp.diags(1.0 / np.sqrt(z_vals))
    return Z.dot(M).dot(Z)


def filter_network(adj, n_neighbors=5):
    """Iteratively remove nodes with fewer than n_neighbors edges."""
    adj = sp.csr_matrix(adj)
    flag = np.array((adj > 0).sum(axis=1)).ravel() >= n_neighbors
    while flag.sum() < adj.shape[0]:
        adj = adj[flag][:, flag]
        flag = np.array((adj > 0).sum(axis=1)).ravel() >= n_neighbors
    return adj


def is_symmetric(mat):
    """Check if matrix is symmetric."""
    if sp.issparse(mat):
        diff = mat - mat.T
        return diff.nnz == 0 or np.allclose(diff.data, 0)
    return np.allclose(mat, mat.T)


def mat_split(mat, size, index=0):
    """Split matrix into row (index=0) or column (index=1) chunks."""
    mat = np.asarray(mat)
    cumsize = np.cumsum(size)
    chunks = []
    prev = 0
    for end in cumsize:
        if index == 0:
            chunks.append(mat[prev:end, :])
        else:
            chunks.append(mat[:, prev:end])
        prev = end
    return chunks


def mat_sparsify(mat, row_mass=0.9, col_mass=0.9):
    """Zero out entries below cumulative mass threshold (rows, then columns).

    Matches R mat_sparsify(mat, row_mass=0.9, col_mass=0.9): applies
    mass_filter rowwise first, then columnwise.
    """
    mat = np.asarray(mat, dtype=float).copy()
    for i in range(mat.shape[0]):
        mask = mass_filter(mat[i], thresh=row_mass)
        new_row = np.zeros_like(mat[i])
        new_row[mask] = mat[i, mask]
        mat[i] = new_row
    for j in range(mat.shape[1]):
        mask = mass_filter(mat[:, j], thresh=col_mass)
        new_col = np.zeros_like(mat[:, j])
        new_col[mask] = mat[mask, j]
        mat[:, j] = new_col
    return mat


def mass_filter(x, thresh=0.9):
    """Keep values contributing to top `thresh` fraction of cumulative mass."""
    x = np.asarray(x)
    idx = np.argsort(-x)
    cumsum = np.cumsum(x[idx])
    total = cumsum[-1]
    cutoff = np.searchsorted(cumsum, thresh * total)
    mask = np.zeros(len(x), dtype=bool)
    mask[idx[:cutoff + 1]] = True
    return mask


def bin_filter(x, n_bins=20, thresh=0):
    """Iterative bin-based filter."""
    x = np.asarray(x, dtype=float)
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    counts, _ = np.histogram(x, bins=bins)
    keep_bins = counts > thresh
    mask = np.zeros(len(x), dtype=bool)
    for i, keep in enumerate(keep_bins):
        if keep:
            in_bin = (x >= bins[i]) & (x < bins[i + 1])
            mask |= in_bin
    return mask


def dis2connec_sparse(dis_sparse, sigma=None):
    """Convert sparse distance matrix to Gaussian connectivity."""
    dis_sparse = dis_sparse.tocsr().copy().astype(float)
    if sigma is None:
        sigma = np.mean(dis_sparse.data)
    dis_sparse.data = np.exp(-(dis_sparse.data ** 2) / (sigma ** 2))
    return dis_sparse


def build_edges(df, id_col, order_col, feature_cols=None):
    """Build temporal edge table from time-series DataFrame."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (id_col, order_col)]
    df = df[[id_col, order_col] + feature_cols].copy()
    df = df.sort_values([id_col, order_col])

    def make_edges(g):
        g = g.copy()
        for col in feature_cols:
            g[f"Next_{col}"] = g[col].shift(-1)
        g["Next_Time"] = g[order_col].shift(-1)
        g = g.dropna(subset=["Next_Time"])
        return g

    edge_table = df.groupby(id_col, group_keys=False).apply(make_edges)
    edge_table = edge_table.rename(columns={order_col: "Start"})
    return edge_table.reset_index(drop=True)


def ceil_digit(x, n):
    """Round up to n decimal places."""
    scale = 10 ** n
    return np.ceil(np.asarray(x) * scale) / scale
