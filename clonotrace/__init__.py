"""
Clonotrace - Python package for T cell fate and clonal dynamics analysis.

Python port of Clonotrace_yuntian (R package by Yuntian Fu).
"""

from .auxiliary import (
    long2wide,
    long_symmetry,
    long2square,
    long2sparse,
    wide2long,
    link2cluster,
    mnn_dist,
    nearest_knn,
    cluster_merge,
    combn_dedup,
    dismat_mst,
    dis_point_to_edge,
    dis_points_to_edges,
    knn_between_groups,
    find_mutual_nn,
    top_k,
    knn_flat,
    embedding2knn,
    dist2knn,
    sync_sparse_rows,
    sparse_norm,
    sparse_manipulation,
    compute_transition,
    filter_network,
    is_symmetric,
    mat_split,
    mat_sparsify,
    mass_filter,
    bin_filter,
    dis2connec_sparse,
    build_edges,
    ceil_digit,
)

from .clone_dis import (
    clone_distance,
    clone_partition,
    graph_clone_ot_sub,
    graph_clone_ot,
    clone_2_ot,
    graph_clone_nn,
    group_2_min,
)

from .cluster import (
    leiden_embedding,
    leiden_embedding_fast,
    leiden_dis,
)

from .coembed import (
    cell_clone_coembed,
    cell_knn_matrix_mutiplication,
    cell_knn_matrix_multiplication_parallel,
)

from .label_propagation import (
    label_spreading,
    label_spreading_bootstrap,
    label_spreading_blocked,
    create_hdf5_matrix,
    normalize_hdf5_rows,
    label_spreading_bootstrap_blocked,
)

from .profile_deg import (
    bin_filter_profile_mass,
    ridge_regression,
    soft_cluster_gam_fit,
    cluster_profile_mass,
    cluster_profile_enrich,
    profile_cluster_DEG_permute,
    profile_cluster_DEG,
    profile_multiclusters_DEG,
)

from .pseudotime import (
    acct,
    DPT_T,
    dpt,
    embedding2dpt,
    clone_root,
    clone_dpt,
)

from .visualization import (
    connectivity_coord,
    dimplot,
    scatterpie,
    umap_from_knn,
    mds_from_knn,
    diffusion_map,
)

__version__ = "1.1.0"
__all__ = [
    # auxiliary
    "long2wide", "long_symmetry", "long2square", "long2sparse", "wide2long",
    "link2cluster", "mnn_dist", "nearest_knn", "cluster_merge", "combn_dedup",
    "dismat_mst", "dis_point_to_edge", "dis_points_to_edges",
    "knn_between_groups", "find_mutual_nn", "top_k", "knn_flat",
    "embedding2knn", "dist2knn", "sync_sparse_rows", "sparse_norm",
    "sparse_manipulation", "compute_transition", "filter_network",
    "is_symmetric", "mat_split", "mat_sparsify", "mass_filter", "bin_filter",
    "dis2connec_sparse", "build_edges", "ceil_digit",
    # clone_dis
    "clone_distance", "clone_partition", "graph_clone_ot_sub",
    "graph_clone_ot", "clone_2_ot", "graph_clone_nn", "group_2_min",
    # cluster
    "leiden_embedding", "leiden_embedding_fast", "leiden_dis",
    # coembed
    "cell_clone_coembed", "cell_knn_matrix_mutiplication",
    "cell_knn_matrix_multiplication_parallel",
    # label_propagation
    "label_spreading", "label_spreading_bootstrap", "label_spreading_blocked",
    "create_hdf5_matrix", "normalize_hdf5_rows",
    "label_spreading_bootstrap_blocked",
    # profile_deg
    "bin_filter_profile_mass", "ridge_regression", "soft_cluster_gam_fit",
    "cluster_profile_mass", "cluster_profile_enrich",
    "profile_cluster_DEG_permute", "profile_cluster_DEG",
    "profile_multiclusters_DEG",
    # pseudotime
    "acct", "DPT_T", "dpt", "embedding2dpt", "clone_root", "clone_dpt",
    # visualization
    "connectivity_coord", "dimplot", "scatterpie", "umap_from_knn", "mds_from_knn", "diffusion_map",
]
