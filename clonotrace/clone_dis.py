"""
clone_dis.py - Clone distance computation

Python port of R/clone_dis.R from Clonotrace_yuntian.
"""

import os
import pickle
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

try:
    import igraph as ig
except ImportError:
    ig = None

try:
    import ot
except ImportError:
    ot = None

from .auxiliary import knn_flat, long2sparse, sync_sparse_rows


def _build_snn_graph(embedding, k):
    """Build SNN igraph weighted by exp(-shared_neighbor_count).

    Matches R's bluster::makeSNNGraph(type="number") followed by
    exp(-weight) transformation as used in R clone_disance().

    Returns igraph.Graph
    """
    if ig is None:
        raise ImportError("python-igraph is required: pip install igraph")

    n = len(embedding)
    # Request k+1 neighbors because sklearn includes self (distance=0);
    # R's bluster::makeSNNGraph(k=k) returns k actual neighbors *excluding* self.
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embedding)
    _, indices = nn.kneighbors(embedding)

    # Remove self from kNN results, keep exactly k true neighbors
    node_ids = np.arange(n)[:, None]
    not_self = indices != node_ids
    indices_no_self = np.array(
        [indices[i][not_self[i]][:k] for i in range(n)]
    )

    # Build adjacency including self in neighbor set to match R's bluster:
    # R's bluster uses set {i} ∪ kNN(i) for shared-neighbor counting
    indices_with_self = np.hstack([np.arange(n).reshape(-1, 1), indices_no_self])
    rows = np.repeat(np.arange(n), k + 1)
    cols = indices_with_self.ravel()
    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n)
    )

    # SNN: shared neighbor count = A * A^T
    shared = adj.dot(adj.T).tocoo()

    # Keep upper triangle only (avoid duplicate edges in undirected graph)
    mask = (shared.row < shared.col) & (shared.data > 0)
    shared_filtered = sp.coo_matrix(
        (shared.data[mask], (shared.row[mask], shared.col[mask])), shape=(n, n)
    )

    # Weight = exp(-shared_count), matching R's clone_disance:
    #   w <- edge_attr(cell_graph, "weight")
    #   set_edge_attr(cell_graph, "weight", value = exp(-w))
    edges = list(zip(shared_filtered.row.tolist(), shared_filtered.col.tolist()))
    weights = np.exp(-shared_filtered.data).tolist()
    G = ig.Graph(n=n, edges=edges, directed=False)
    G.es["weight"] = weights
    return G


def clone_distance(embedding, cell_clone_prob, outpath, graph_k=10,
                   overwrite=False, exact=False, **kwargs):
    """Compute clone-to-clone distances from cell embedding and clone assignments.

    Parameters
    ----------
    embedding : np.ndarray or pd.DataFrame  (cells, dims)
    cell_clone_prob : np.ndarray, sp.spmatrix, or pd.DataFrame  (cells, clones)
    outpath : str  directory to save/load intermediate results
    graph_k : int  number of SNN neighbors
    overwrite : bool
    exact : bool  use OT (True) or kNN approximate (False)

    Returns
    -------
    pd.DataFrame  columns group1, group2, dis
    """
    import pandas as pd

    os.makedirs(outpath, exist_ok=True)

    # Align rows by index if both inputs have named indices (matches R behavior)
    emb_has_index = hasattr(embedding, 'index')
    prob_has_index = hasattr(cell_clone_prob, 'index')
    if emb_has_index and prob_has_index:
        common = embedding.index.intersection(cell_clone_prob.index)
        if len(common) < len(cell_clone_prob.index):
            n_removed = len(cell_clone_prob.index) - len(common)
            warnings.warn(
                f"cell_clone_prob has {n_removed} cells not in embedding; "
                "those cells will be removed."
            )
        embedding = embedding.loc[common]
        cell_clone_prob = cell_clone_prob.loc[common]
    elif embedding.shape[0] != (cell_clone_prob.shape[0] if not sp.issparse(cell_clone_prob) else cell_clone_prob.shape[0]):
        warnings.warn(
            "embedding and cell_clone_prob have different row counts. "
            "Pass DataFrames with matching indices for automatic alignment."
        )

    embedding = np.asarray(embedding)

    graph_path = os.path.join(outpath, "cell_graph.pkl")
    if os.path.exists(graph_path) and not overwrite:
        print("cell_graph already calculated, using existing.")
        with open(graph_path, "rb") as f:
            cell_graph = pickle.load(f)
    else:
        cell_graph = _build_snn_graph(embedding, k=graph_k)
        with open(graph_path, "wb") as f:
            pickle.dump(cell_graph, f)

    dis_path = os.path.join(outpath, "clone_graph_dis.pkl")
    if os.path.exists(dis_path) and not overwrite:
        print("clone_graph_dis already calculated, using existing.")
        with open(dis_path, "rb") as f:
            dis_result = pickle.load(f)
    else:
        if exact:
            ot_keys = {"prob_thresh", "cores", "cache", "verbose"}
            call_args = {k: v for k, v in kwargs.items() if k in ot_keys}
            dis_result = graph_clone_ot(cell_graph, cell_clone_prob, **call_args)
        else:
            nn_keys = {"prob_thresh", "k", "verbose"}
            call_args = {k: v for k, v in kwargs.items() if k in nn_keys}
            dis_result = graph_clone_nn(cell_graph, cell_clone_prob, **call_args)
        with open(dis_path, "wb") as f:
            pickle.dump(dis_result, f)

    return dis_result


def clone_partition(clone_matrix, k=10, similarity_threshold=0):
    """Partition clones using farthest point sampling + greedy assignment.

    Parameters
    ----------
    clone_matrix : np.ndarray or sp.spmatrix  (cells, clones)
    k : int  number of partitions
    similarity_threshold : float

    Returns
    -------
    dict  {group_id: [clone_names]}
    """
    if sp.issparse(clone_matrix):
        bin_mat = (clone_matrix > 0).astype(float)
    else:
        bin_mat = (np.asarray(clone_matrix) > 0).astype(float)

    n_clones = bin_mat.shape[1]
    sizes = np.array(bin_mat.sum(axis=0)).ravel()

    if sp.issparse(bin_mat):
        intersection = (bin_mat.T.dot(bin_mat)).toarray()
    else:
        intersection = bin_mat.T @ bin_mat

    denom = np.tile(sizes, (n_clones, 1))
    denom = np.maximum(denom, 1e-12)
    sim = intersection / denom
    np.fill_diagonal(sim, 1.0)

    # Step 1: Farthest Point Sampling
    seeds = [int(np.argmax(sizes))]
    while len(seeds) < k:
        remaining = [i for i in range(n_clones) if i not in seeds]
        seed_sim = sim[np.ix_(remaining, seeds)]
        max_sim_to_seeds = seed_sim.max(axis=1)
        next_seed = remaining[int(np.argmin(max_sim_to_seeds))]
        seeds.append(next_seed)

    # Step 2: Greedy assignment
    group_ids = np.full(n_clones, -1, dtype=int)
    group_sizes = np.zeros(k, dtype=int)
    target_size = int(np.ceil(n_clones / k))

    clone_sim_to_seeds = sim[seeds, :]
    assign_order = np.argsort(-clone_sim_to_seeds.max(axis=0))

    for i in assign_order:
        clone_sims = clone_sim_to_seeds[:, i]
        ranked = np.argsort(-clone_sims)
        assigned = False
        for g in ranked:
            if clone_sims[g] >= similarity_threshold and group_sizes[g] < target_size:
                group_ids[i] = g
                group_sizes[g] += 1
                assigned = True
                break
        if not assigned:
            g = int(np.argmin(group_sizes))
            group_ids[i] = g
            group_sizes[g] += 1

    result = {}
    for g in range(k):
        result[g] = list(np.where(group_ids == g)[0])
    return result


def clone_2_ot(distance, group1_mass, group2_mass):
    """Optimal transport distance between two clones.

    Parameters
    ----------
    distance : np.ndarray  (n1, n2) cost matrix
    group1_mass, group2_mass : np.ndarray  mass distributions

    Returns
    -------
    float  OT distance
    """
    if ot is None:
        raise ImportError("POT library required: pip install POT")

    distance = np.asarray(distance, dtype=float)
    a = np.asarray(group1_mass, dtype=float)
    b = np.asarray(group2_mass, dtype=float)
    a = a / a.sum()
    b = b / b.sum()

    plan = ot.emd(a, b, distance)
    return float((plan * distance).sum())


def group_2_min(distance, group1, group2, k=3):
    """Average minimal pairwise distance between two groups.

    Parameters
    ----------
    distance : np.ndarray  full pairwise distance matrix
    group1 : array-like  row indices
    group2 : array-like  column indices
    k : int  number of nearest neighbors to average

    Returns
    -------
    float
    """
    distance = np.asarray(distance)
    sub = distance[np.ix_(group1, group2)]

    if sub.ndim == 0 or sub.size == 1:
        return float(sub.mean())

    def row_top_k_mean(row):
        return float(np.mean(np.partition(row, min(k, len(row)) - 1)[:k]))

    row_means = np.array([row_top_k_mean(sub[i]) for i in range(sub.shape[0])])
    col_means = np.array([row_top_k_mean(sub[:, j]) for j in range(sub.shape[1])])
    return float(np.mean([row_means.mean(), col_means.mean()]))


def graph_clone_ot_sub(graph, cell_clone_prob, target_clone=None, cache=5000, verbose=True):
    """Compute pairwise OT distances using graph-based subset strategy.

    Parameters
    ----------
    graph : igraph.Graph
    cell_clone_prob : np.ndarray  (cells, clones)
    target_clone : list or None  clone indices to process
    cache : int  max pool size
    verbose : bool

    Returns
    -------
    np.ndarray  shape (M, 3) with columns group1, group2, dis
    """
    if ig is None:
        raise ImportError("python-igraph required")

    if sp.issparse(cell_clone_prob):
        cell_clone_prob = cell_clone_prob.toarray()
    cell_clone_prob = np.asarray(cell_clone_prob, dtype=float)

    n_clones = cell_clone_prob.shape[1]
    if target_clone is None:
        target_clone = list(range(n_clones))

    target_clone = list(target_clone)
    target_clone_ident = (cell_clone_prob[:, target_clone] > 0).astype(float)
    flag = np.zeros(len(target_clone), dtype=bool)

    pool = []
    cell_dis_cache = {}
    full_ot = []

    col_sums = target_clone_ident.sum(axis=0)
    target_id = int(np.argmax(col_sums))

    while flag.sum() < len(target_clone):
        flag[target_id] = True
        global_id = target_clone[target_id]
        if verbose:
            print(f"Processing clone {global_id}")

        if global_id == n_clones - 1:
            # Update next target
            unflagged = np.where(~flag)[0]
            if len(unflagged) == 0:
                break
            target_id = unflagged[0]
            continue

        cell_id = np.where(cell_clone_prob[:, global_id] > 0)[0].tolist()

        if len(cell_id) > cache:
            raise ValueError(
                f"Clone {global_id} has {len(cell_id)} cells, exceeding cache={cache}"
            )

        append_cells = [c for c in cell_id if c not in set(pool)]
        for c in append_cells:
            dists = graph.distances(source=c, weights="weight")[0]
            cell_dis_cache[c] = np.array(dists)

        # Evict from pool if needed
        if len(append_cells) + len(pool) > cache:
            remove_n = len(append_cells) + len(pool) - cache
            pool_not_in_clone = [p for p in pool if p not in set(cell_id)]
            unflag_idx = np.where(~flag)[0]
            if len(pool_not_in_clone) > 0 and len(unflag_idx) > 0:
                freq = target_clone_ident[np.ix_(pool_not_in_clone, unflag_idx)].sum(axis=1)
                evict_idx = np.argsort(freq)[:remove_n]
                evict = [pool_not_in_clone[i] for i in evict_idx]
                pool = [p for p in pool if p not in set(evict)]
                for e in evict:
                    cell_dis_cache.pop(e, None)

        pool.extend(append_cells)

        clone_mass = cell_clone_prob[cell_id, global_id]
        row_id = [pool.index(c) for c in cell_id]

        for i in range(global_id + 1, n_clones):
            clone2_cells = np.where(cell_clone_prob[:, i] > 0)[0]
            if len(clone2_cells) == 0:
                continue
            # Build sub-distance matrix
            sub_dis = np.array([
                [cell_dis_cache[pool[r]][c2] for c2 in clone2_cells]
                for r in row_id
            ])
            dist_val = clone_2_ot(
                sub_dis, clone_mass, cell_clone_prob[clone2_cells, i]
            )
            full_ot.append([global_id, i, dist_val])

        # Choose next target: minimize unseen cells not yet in pool
        unflag_idx = np.where(~flag)[0]
        if len(unflag_idx) == 0:
            break
        pool_indicator = np.zeros(cell_clone_prob.shape[0])
        pool_indicator[pool] = 1
        remaining_out = (
            target_clone_ident[:, unflag_idx].sum(axis=0)
            - pool_indicator @ target_clone_ident[:, unflag_idx]
        )
        target_id = unflag_idx[int(np.argmin(remaining_out))]

    return np.array(full_ot) if full_ot else np.empty((0, 3))


def graph_clone_ot(graph, cell_clone_prob, prob_thresh=0.05, cache=5000,
                   cores=1, verbose=True):
    """Parallel clone-to-clone OT computation.

    Returns
    -------
    pd.DataFrame  columns group1, group2, dis
    """
    import pandas as pd

    if sp.issparse(cell_clone_prob):
        cell_clone_prob = cell_clone_prob.toarray()
    cell_clone_prob = np.asarray(cell_clone_prob, dtype=float)
    cell_clone_prob[cell_clone_prob < prob_thresh] = 0

    partition = clone_partition(cell_clone_prob, k=max(1, cores))
    if verbose:
        print(f"There are {len(partition)} clone partitions")

    def _run_partition(i, indices):
        if verbose:
            print(f"Running clone partition {i}, clones: {indices}")
        return graph_clone_ot_sub(graph, cell_clone_prob, indices, cache, verbose)

    if cores == 1:
        results = [_run_partition(i, idxs) for i, idxs in partition.items()]
    else:
        results = Parallel(n_jobs=cores)(
            delayed(_run_partition)(i, idxs)
            for i, idxs in partition.items()
        )

    combined = np.vstack([r for r in results if len(r) > 0])
    return pd.DataFrame(combined, columns=["group1", "group2", "dis"])


def graph_clone_nn(graph, cell_clone_prob, prob_thresh=0.1, k=2, verbose=False):
    """Clone-to-clone nearest-neighbor distances on a cell graph.

    Returns
    -------
    pd.DataFrame  columns group1, group2, dis
    """
    import pandas as pd

    if sp.issparse(cell_clone_prob):
        cell_clone_prob = cell_clone_prob.toarray()

    cell_group_mat = np.asarray(cell_clone_prob, dtype=float)
    cell_group_mat = (cell_group_mat >= prob_thresh).astype(float)
    n_groups = cell_group_mat.shape[1]

    def _process_clone(i):
        if verbose:
            print(f"Processing clone {i}")
        from_cells = np.where(cell_group_mat[:, i] > 0)[0].tolist()
        if not from_cells:
            return []

        to_cells = np.where(cell_group_mat[:, i + 1:].sum(axis=1) > 0)[0].tolist()
        if not to_cells:
            return []

        # Graph distances from source to target cells
        dis_i = np.array(graph.distances(source=from_cells, target=to_cells,
                                          weights="weight"))

        out = []
        for j in range(i + 1, n_groups):
            id_j = np.where(cell_group_mat[:, j] > 0)[0]
            to_in_j = [ti for ti, tc in enumerate(to_cells) if tc in set(id_j.tolist())]
            if not to_in_j:
                continue
            sub_dis = dis_i[:, to_in_j]
            d = group_2_min(sub_dis, list(range(sub_dis.shape[0])),
                            list(range(sub_dis.shape[1])), k=k)
            out.append([i, j, d])
        return out

    results = Parallel(n_jobs=1)(
        delayed(_process_clone)(i) for i in range(n_groups - 1)
    )
    rows = [r for sub in results for r in sub]
    df = pd.DataFrame(rows, columns=["group1", "group2", "dis"])
    return df
