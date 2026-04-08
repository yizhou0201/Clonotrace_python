"""
cluster.py - Clustering functions

Python port of R/cluster.R from Clonotrace_yuntian.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

try:
    import igraph as ig
except ImportError:
    ig = None

try:
    import umap
except ImportError:
    umap = None


def _build_snn_jaccard(data, k, prune_snn=0):
    """Build SNN graph with Jaccard weights from embedding or distance matrix.

    Parameters
    ----------
    data : np.ndarray  embedding (N, D)
    k : int
    prune_snn : float  threshold to prune weak edges

    Returns
    -------
    sp.csr_matrix  (N, N) Jaccard SNN matrix
    """
    n = len(data)
    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 for self
    nn.fit(data)
    _, indices = nn.kneighbors(data)
    indices = indices[:, 1:]  # remove self

    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=float
    )

    shared = adj.dot(adj.T).tocoo()
    jaccard_data = shared.data / (2 * k - shared.data)
    # Prune and remove self-loops using data arrays directly
    keep = (jaccard_data > prune_snn) & (shared.row != shared.col)
    jaccard = sp.csr_matrix(
        (jaccard_data[keep], (shared.row[keep], shared.col[keep])), shape=(n, n)
    )
    return jaccard


def _snn_from_dist(dismat, k, prune_snn=0):
    """Build SNN edge list from precomputed distance matrix.

    Matches R's ``dbscan::sNN`` + ``graph_from_data_frame(directed=FALSE)``
    behaviour: Jaccard similarity is computed only for kNN edges (i -> j
    where j is a k-nearest neighbour of i), and the directed edge list is
    returned so that ``_igraph_from_edgelist`` can build an undirected
    multigraph (parallel edges when both i->j and j->i exist).

    Returns
    -------
    starts, ends, weights : np.ndarray
        Directed kNN edge list with Jaccard weights (after pruning).
    n : int
        Number of nodes.
    """
    n = dismat.shape[0]
    nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    nn.fit(dismat)
    dists, indices = nn.kneighbors(dismat)

    # Build binary kNN adjacency for shared-neighbour counting
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    adj = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=float
    )

    # Compute shared neighbour counts only for kNN edges
    # shared_full[i,j] = |kNN(i) ∩ kNN(j)| for all pairs; extract kNN entries
    shared_full = adj.dot(adj.T).tocsr()
    row_idx = np.repeat(np.arange(n), k)
    col_idx = indices.ravel()
    shared_counts = np.asarray(shared_full[row_idx, col_idx], dtype=float).ravel()

    jaccard = shared_counts / (2 * k - shared_counts)

    # Prune weak edges
    keep = jaccard > prune_snn
    starts = rows[keep]
    ends = cols[keep]
    weights = jaccard[keep]

    return starts, ends, weights, n


def _igraph_from_sparse(jaccard):
    """Convert sparse Jaccard matrix to weighted igraph."""
    if ig is None:
        raise ImportError("python-igraph required: pip install igraph")
    coo = jaccard.tocoo()
    n = jaccard.shape[0]
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    weights = coo.data.tolist()
    G = ig.Graph(n=n, edges=edges, directed=False)
    G.es["weight"] = weights
    return G


def _igraph_from_edgelist(starts, ends, weights, n):
    """Build undirected igraph from directed kNN edge list.

    Mirrors R's ``graph_from_data_frame(directed=FALSE)``: when both
    i->j and j->i are present, they become parallel (multi-)edges in
    the undirected graph.
    """
    if ig is None:
        raise ImportError("python-igraph required: pip install igraph")
    edges = list(zip(starts.tolist(), ends.tolist()))
    G = ig.Graph(n=n, edges=edges, directed=False)
    G.es["weight"] = weights.tolist()
    return G


def leiden_embedding(data, k=30, prune_snn=0, weight="jaccard", resolution=1,
                     method="louvain"):
    """Community detection from embedding via shared nearest neighbors.

    Parameters
    ----------
    data : np.ndarray  (N, D)
    k : int
    prune_snn : float
    weight : str  currently only 'jaccard' is supported
    resolution : float
    method : str  'louvain' (default, matches R) or 'leiden'

    Returns
    -------
    np.ndarray  cluster labels (0-indexed)
    """
    if ig is None:
        raise ImportError("python-igraph required: pip install igraph")

    jaccard = _build_snn_jaccard(np.asarray(data), k, prune_snn)
    G = _igraph_from_sparse(jaccard)

    if method == "leiden":
        partition = G.community_leiden(
            weights="weight",
            resolution=resolution,
            objective_function="modularity",
            n_iterations=-1,
            beta=0.01,
        )
    else:
        partition = G.community_multilevel(weights="weight", resolution=resolution)

    return np.array(partition.membership)


def leiden_embedding_fast(data, k=30, prune_snn=0, weight="jaccard", resolution=1,
                          method="leiden"):
    """Fast community detection using sparse SNN matrix multiplication.

    Parameters
    ----------
    data : np.ndarray  (N, D)
    k : int
    prune_snn : float
    resolution : float
    method : str  'leiden' (default) or 'louvain'

    Returns
    -------
    np.ndarray  cluster labels (0-indexed)
    """
    if ig is None:
        raise ImportError("python-igraph required: pip install igraph")

    jaccard = _build_snn_jaccard(np.asarray(data), k, prune_snn)
    G = _igraph_from_sparse(jaccard)

    if method == "louvain":
        partition = G.community_multilevel(weights="weight")
    else:
        partition = G.community_leiden(
            weights="weight",
            resolution=resolution,
            objective_function="modularity",
            n_iterations=-1,
            beta=0.01,
        )
    return np.array(partition.membership)


def leiden_dis(dismat, k=10, prune_snn=0, weight="jaccard", resolution=1,
               if_umap=True, method="leiden", umap_n_epochs=200):
    """Community detection from precomputed distance matrix.

    Parameters
    ----------
    dismat : np.ndarray  (N, N) symmetric distance matrix
    k : int
    prune_snn : float
    resolution : float
    if_umap : bool  if True, also return UMAP coordinates
    method : str  'leiden' (default) or 'louvain'
    umap_n_epochs : int  UMAP optimization epochs (default 200, matching R)

    Returns
    -------
    If if_umap: pd.DataFrame with columns umap1, umap2, cluster
    Else: np.ndarray  cluster labels
    """
    import pandas as pd

    if ig is None:
        raise ImportError("python-igraph required: pip install igraph")

    dismat = np.asarray(dismat, dtype=float)

    umap_coords = None
    if if_umap:
        if umap is None:
            raise ImportError("umap-learn required: pip install umap-learn")
        reducer = umap.UMAP(metric="precomputed", n_epochs=umap_n_epochs)
        umap_coords = reducer.fit_transform(dismat)

    starts, ends, weights, nn = _snn_from_dist(dismat, k, prune_snn)
    G = _igraph_from_edgelist(starts, ends, weights, nn)

    if method == "louvain":
        partition = G.community_multilevel(weights="weight", resolution=resolution)
    else:
        partition = G.community_leiden(
            weights="weight",
            resolution=resolution,
            objective_function="modularity",
            n_iterations=-1,
            beta=0.01,
        )
    labels = np.array(partition.membership)

    if if_umap:
        df = pd.DataFrame(umap_coords, columns=["umap1", "umap2"])
        df["cluster"] = labels
        return df
    return labels
