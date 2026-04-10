"""
visualization.py - Plotting functions

Python port of R/visualization.R from Clonotrace_yuntian.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

try:
    import umap as umap_lib
except ImportError:
    umap_lib = None

try:
    from sklearn.manifold import MDS
except ImportError:
    MDS = None

from .auxiliary import filter_network


def connectivity_coord(coord, connectivity, dims=(0, 1)):
    """Convert connectivity matrix to edge coordinates for plotting.

    Parameters
    ----------
    coord : np.ndarray or pd.DataFrame  (N, D) coordinates
    connectivity : np.ndarray or sp.spmatrix  (N, N) edge weights
    dims : tuple of 2 ints  which coordinate columns to use (0-indexed)

    Returns
    -------
    pd.DataFrame  columns i, j, x (weight), i_x, i_y, j_x, j_y
    """
    coord = np.asarray(coord)[:, list(dims)]
    coord_df = pd.DataFrame(coord, columns=["x", "y"])
    coord_df["id"] = np.arange(len(coord_df))

    if sp.issparse(connectivity):
        connectivity = connectivity.copy()
    else:
        connectivity = sp.csr_matrix(connectivity)
    connectivity.setdiag(0)
    connectivity.eliminate_zeros()

    coo = connectivity.tocoo()
    edges = pd.DataFrame({
        "i": coo.row,
        "j": coo.col,
        "x": coo.data,
    })
    edges = edges.merge(coord_df.rename(columns={"x": "i_x", "y": "i_y", "id": "i"}),
                        on="i")
    edges = edges.merge(coord_df.rename(columns={"x": "j_x", "y": "j_y", "id": "j"}),
                        on="j")
    return edges


def dimplot(embedding, annot, color_by, alpha_by=None, connectivity=None,
            label=True, dims=(0, 1), connectivity_thresh=0.1,
            label_size=10, label_type="text", label_color="black",
            raster_thresh=10000, ax=None, **scatter_kwargs):
    """2D embedding scatter plot with optional connectivity and labels.

    Parameters
    ----------
    embedding : np.ndarray  (N, D)
    annot : pd.DataFrame  annotation dataframe (indexed as embedding rows)
    color_by : str  column in annot for color
    alpha_by : str or None  column for alpha transparency
    connectivity : np.ndarray or sp.spmatrix or None  (G, G) cluster connectivity
    label : bool  annotate cluster centers
    dims : tuple of 2 ints  which dims to plot
    connectivity_thresh : float
    label_size : float
    label_type : 'text' or 'label'
    label_color : str
    raster_thresh : int  rasterize if N > this value
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.get_figure()

    emb = np.asarray(embedding)[:, list(dims)]
    annot = annot[[color_by] + ([alpha_by] if alpha_by else [])].copy()

    x = emb[:, 0]
    y = emb[:, 1]

    # Background points
    ax.scatter(x, y, c="lightgrey", s=3, alpha=0.5, rasterized=(len(x) > raster_thresh))

    # Colored points — cast float-valued integers to int for clean labels
    col_series = annot[color_by]
    if col_series.dtype.kind == 'f':
        non_null = col_series.dropna()
        if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
            annot = annot.copy()
            annot[color_by] = col_series.astype("Int64")
    colors = annot[color_by].values
    unique_colors = pd.Categorical(colors).categories
    cmap = plt.get_cmap("tab20" if len(unique_colors) <= 20 else "hsv")
    color_map = {c: cmap(i / max(len(unique_colors), 1)) for i, c in enumerate(unique_colors)}
    c_vals = [color_map.get(c, "grey") for c in colors]

    alpha_vals = annot[alpha_by].values if alpha_by else None
    valid_mask = ~pd.isna(colors)

    sc = ax.scatter(
        x[valid_mask], y[valid_mask],
        c=[c_vals[i] for i in np.where(valid_mask)[0]],
        alpha=alpha_vals[valid_mask] if alpha_vals is not None else 0.8,
        rasterized=(valid_mask.sum() > raster_thresh),
        **scatter_kwargs
    )

    # Cluster centers
    if label or connectivity is not None:
        centers = (pd.DataFrame({"x": x, "y": y, color_by: colors})
                   .groupby(color_by)[["x", "y"]]
                   .median()
                   .reset_index())
        counts = pd.Series(colors).value_counts().rename("count").reset_index()
        counts.columns = [color_by, "count"]
        centers = centers.merge(counts, on=color_by)

        if connectivity is not None:
            edge_df = connectivity_coord(centers[["x", "y"]].values, connectivity)
            edge_df = edge_df[edge_df["x"] >= connectivity_thresh]
            if len(edge_df) > 0:
                segments = [[(r["i_x"], r["i_y"]), (r["j_x"], r["j_y"])]
                            for _, r in edge_df.iterrows()]
                lc = LineCollection(segments, colors="honeydew", alpha=0.75,
                                    linewidths=edge_df["x"].values * 2)
                ax.add_collection(lc)
            ax.scatter(centers["x"], centers["y"],
                       s=np.log(centers["count"]) * 10, c="black", zorder=5)

        if label:
            texts = []
            for _, row in centers.iterrows():
                t = ax.text(row["x"], row["y"], str(row[color_by]),
                            fontsize=label_size, color=label_color)
                texts.append(t)
            if adjust_text is not None:
                adjust_text(texts, ax=ax)

    ax.set_xlabel(f"Dim {dims[0] + 1}")
    ax.set_ylabel(f"Dim {dims[1] + 1}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def scatterpie(scatter_coord, composition, connectivity=None,
               connectivity_thresh=0.5, dims=("umap_1", "umap_2"),
               cluster_col="cluster", edge_color="lightgrey", edge_alpha=1,
               label_size=10, ax=None):
    """Pie chart compositions at spatial/embedding coordinates.

    Parameters
    ----------
    scatter_coord : pd.DataFrame  coordinates, one row per cluster
    composition : np.ndarray or pd.DataFrame  (N_clusters, categories)
    connectivity : np.ndarray or None  (N, N)
    connectivity_thresh : float
    dims : tuple of 2 str  coordinate column names
    cluster_col : str  column for cluster labels
    edge_color, edge_alpha : style parameters
    label_size : float
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    if not isinstance(composition, pd.DataFrame):
        composition = pd.DataFrame(composition)
    if not isinstance(scatter_coord, pd.DataFrame):
        scatter_coord = pd.DataFrame(scatter_coord)

    n = len(scatter_coord)
    x_col, y_col = dims[0], dims[1]

    # Draw connectivity edges
    if connectivity is not None:
        coord_arr = scatter_coord[[x_col, y_col]].values
        edge_df = connectivity_coord(coord_arr, connectivity)
        edge_df = edge_df[edge_df["x"] >= connectivity_thresh]
        if len(edge_df) > 0:
            segments = [[(r["i_x"], r["i_y"]), (r["j_x"], r["j_y"])]
                        for _, r in edge_df.iterrows()]
            lc = LineCollection(segments, colors=edge_color, alpha=edge_alpha,
                                linewidths=edge_df["x"].values * 2)
            ax.add_collection(lc)

    # Draw pies
    cmap = plt.get_cmap("tab20")
    cat_colors = {col: cmap(i / max(len(composition.columns), 1))
                  for i, col in enumerate(composition.columns)}

    for idx in range(n):
        xi = scatter_coord.iloc[idx][x_col]
        yi = scatter_coord.iloc[idx][y_col]
        counts = composition.iloc[idx].values.astype(float)
        total = counts.sum()
        if total == 0:
            continue
        radius = max(np.log(total + 1) / 12, 0.05)
        fracs = counts / total
        start_angle = 0
        for j, frac in enumerate(fracs):
            angle = frac * 360
            wedge = mpatches.Wedge(
                center=(xi, yi), r=radius,
                theta1=start_angle, theta2=start_angle + angle,
                color=cmap(j / max(len(composition.columns), 1))
            )
            ax.add_patch(wedge)
            start_angle += angle

    # Labels
    texts = []
    for idx in range(n):
        xi = scatter_coord.iloc[idx][x_col]
        yi = scatter_coord.iloc[idx][y_col]
        label = scatter_coord.iloc[idx].get(cluster_col, str(idx))
        t = ax.text(xi, yi, str(label), fontsize=label_size)
        texts.append(t)
    if adjust_text is not None:
        adjust_text(texts, ax=ax)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def umap_from_knn(adj, n_neighbors=5, seed=1024):
    """UMAP embedding from kNN adjacency matrix.

    Parameters
    ----------
    adj : sp.spmatrix  (N, N)
    n_neighbors : int
    seed : int

    Returns
    -------
    pd.DataFrame  columns umap_1, umap_2
    np.ndarray  boolean mask of nodes kept after filter_network
    """
    if umap_lib is None:
        raise ImportError("umap-learn required: pip install umap-learn")

    adj, kept_mask = filter_network(adj, n_neighbors=n_neighbors)

    # Convert to dense distance representation: 1 - normalized_adjacency
    adj_arr = adj.toarray()
    # Normalize to [0, 1]
    max_val = adj_arr.max()
    if max_val > 0:
        adj_arr = adj_arr / max_val
    dist_arr = 1.0 - adj_arr
    np.fill_diagonal(dist_arr, 0)

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        metric="precomputed",
        random_state=seed,
    )
    coords = reducer.fit_transform(dist_arr)
    n = adj.shape[0]
    df = pd.DataFrame(coords, columns=["umap_1", "umap_2"])
    if hasattr(adj, "rownames"):
        df.index = adj.rownames
    return df, kept_mask


def diffusion_map(dismat, maxdim=30, eps_val=None, delta=1e-5):
    """Compute diffusion map coordinates from a pairwise distance matrix.

    Replicates R's diffusionMap::diffuse().

    Parameters
    ----------
    dismat : np.ndarray  (N, N) symmetric distance matrix
    maxdim : int  number of diffusion dimensions to return
    eps_val : float or None  bandwidth for Gaussian kernel.
        If None, uses median distance to the 1%-nearest neighbor.
    delta : float  sparsification threshold

    Returns
    -------
    pd.DataFrame  (N, maxdim) columns dm1, dm2, ...
    """
    from scipy.sparse.linalg import eigsh

    D = np.asarray(dismat, dtype=float)
    n = D.shape[0]

    # Default epsilon: median distance to the 0.01*n nearest neighbor
    if eps_val is None:
        knn_k = max(1, int(np.ceil(0.01 * n)))
        nn_dists = np.sort(D, axis=1)[:, knn_k]
        eps_val = float(np.median(nn_dists) ** 2)

    # Gaussian kernel
    K = np.exp(-D ** 2 / eps_val)

    # Normalized graph Laplacian (anisotropic diffusion)
    v = np.sqrt(K.sum(axis=1))
    A = K / np.outer(v, v)

    # Sparsify
    A[A < delta] = 0
    A_sp = sp.csr_matrix(A)

    # Eigen decomposition (largest eigenvalues of symmetric matrix)
    neff = min(maxdim + 1, n - 1)
    eigenvals, eigenvecs = eigsh(A_sp, k=neff, which="LM")

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    # Normalize eigenvectors: psi = eigenvecs / eigenvecs[:, 0]
    psi = eigenvecs / eigenvecs[:, 0:1]

    # Diffusion coordinates (t=0): X = psi[:, 1:] * lambda/(1-lambda)
    lam = eigenvals[1:maxdim + 1]
    lam_scaled = lam / (1 - lam + 1e-12)
    X = psi[:, 1:maxdim + 1] * lam_scaled[np.newaxis, :]

    cols = [f"dm{i + 1}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


def mds_from_knn(adj, n_components=15):
    """MDS embedding from kNN adjacency matrix.

    Parameters
    ----------
    adj : sp.spmatrix  (N, N)
    n_components : int

    Returns
    -------
    pd.DataFrame  columns mds_1, ..., mds_n
    """
    if MDS is None:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    adj, kept_mask = filter_network(adj, n_neighbors=5)

    adj_arr = adj.toarray()
    max_val = adj_arr.max()
    if max_val > 0:
        adj_arr = adj_arr / max_val
    dist_arr = 1.0 - adj_arr
    np.fill_diagonal(dist_arr, 0)

    mds = MDS(n_components=n_components, dissimilarity="precomputed",
              random_state=42, n_init=1, max_iter=300)
    coords = mds.fit_transform(dist_arr)
    cols = [f"mds_{i + 1}" for i in range(n_components)]
    df = pd.DataFrame(coords, columns=cols)
    if hasattr(adj, "rownames"):
        df.index = adj.rownames
    return df, kept_mask
