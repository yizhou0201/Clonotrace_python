# Clonotrace R-to-Python Function Mapping

## Summary

### Function Counts

| Module | R Functions | Python Functions |
|--------|------------|-----------------|
| auxiliary | 26 | 27 |
| clone_dis | 6 | 7 |
| label_propagation | 5 | 5 |
| cluster | 3 | 3 (+3 internal helpers) |
| pseudotime | 5 | 5 |
| profile_DEG | 7 | 7 |
| coembed | 3 | 3 |
| visualization | 5 | 5 |
| **Total** | **60** | **62** |

### Functions Present in R but Missing in Python

None -- all R functions have Python counterparts.

### Functions Present in Python but Missing in R

| Function | Module | Notes |
|----------|--------|-------|
| `_build_snn_jaccard` | cluster.py | Internal helper; R uses `dbscan::sNN` inline instead |
| `_snn_from_dist` | cluster.py | Internal helper for distance-based SNN; R uses `dbscan::sNN(as.dist(...))` inline |
| `_igraph_from_sparse` | cluster.py | Internal helper to convert sparse matrix to igraph; R calls `igraph::graph_from_data_frame` inline |

### Known Algorithmic Differences

1. **`leiden_embedding` uses Louvain instead of Leiden (BUG):** Python `leiden_embedding()` calls `G.community_multilevel(weights="weight")` which is Louvain, not Leiden. The R version also uses `igraph::cluster_louvain()`. The function name is misleading in both cases; the R code has commented-out Leiden calls. However, `leiden_embedding_fast` in Python correctly uses `G.community_leiden()`.

2. **SNN graph weighting in `_build_snn_graph` (clone_dis.py):** Python uses Jaccard similarity (`shared / (2k - shared)`), then `exp(-jaccard)` as edge weight. R uses `bluster::makeSNNGraph(type="number")` which weights edges by raw shared neighbor counts, then transforms with `exp(-weight)`. These produce different distance scales.

3. **`clone_distance` does NOT intersect cell names (BUG):** R `clone_disance()` explicitly intersects `rownames(embedding)` with `rownames(cell_clone_prob)` and warns about mismatches. Python `clone_distance()` does not perform this intersection.

4. **OT solver difference in `clone_2_ot`:** R uses `transport::transport()` (Hungarian-like solver); Python uses `ot.emd()` from the POT library. Both solve the exact Earth Mover's Distance, but implementations differ.

5. **`cell_clone_coembed` reads `cell_clone_prob` from global environment in R:** R function signature is `cell_clone_coembed(cell_embedding, clone_embedding, cell_k, clone_k)` and references `cell_clone_prob` from the calling environment. Python takes it as an explicit parameter: `cell_clone_coembed(cell_embedding, clone_embedding, cell_clone_prob, cell_k, clone_k)`.

6. **Index conventions:** R is 1-indexed, Python is 0-indexed. Python `knn_flat()` returns 1-indexed node IDs (matching R convention) but Python cluster labels from igraph are 0-indexed (R returns 1-indexed factors).

7. **UMAP backend:** R `umap_from_knn` calls a Python backend (`umap_from_knn_py`) via reticulate. Python `umap_from_knn` converts adjacency to distance (`1 - normalized_adj`) and uses `umap-learn` directly with `metric="precomputed"`.

8. **MDS backend:** R `mds_from_knn` calls `mds_from_knn_py` via reticulate. Python uses `sklearn.manifold.MDS` with precomputed dissimilarity.

9. **Parallelism:** R uses `future.apply::future_lapply` for parallel execution. Python uses `joblib.Parallel` (often with `n_jobs=1` effectively serial).

10. **Spline basis:** R uses `splines::ns()` (natural splines). Python uses `sklearn.preprocessing.SplineTransformer` (B-spline basis, not true natural splines). The number of basis functions may differ for the same `df` parameter.

11. **`sparse_norm`:** R normalizes by row (`dim=1`) or column (`dim=2`) using L1 norm. Python uses `sklearn.preprocessing.normalize` with configurable norm type (default L2), which differs from R's L1 default.

12. **`dis2connec_sparse`:** R computes per-node sigma from mean neighbor distance and uses `exp(-dist^2 / (sigma_i * sigma_j))`. Python accepts an optional global `sigma` parameter; if `None`, uses the global mean of all nonzero distances, not per-node sigma.

---

## Module: auxiliary

**R file:** `Clonotrace_yuntian/R/auxiliary.R`
**Python file:** `Clonotrace_python/clonotrace/auxiliary.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| `long2wide` | `long, row_names_from, col_names_from, values_from, symmetric=FALSE` | `long2wide` | `long, row_names_from, col_names_from, values_from, symmetric=False` | R uses `tidyr::pivot_wider`; Python uses `pd.pivot_table(aggfunc="first")` |
| `long_symmetry` | `long, row_names_from, col_names_from` | `long_symmetry` | `long, row_names_from, col_names_from` | Equivalent |
| `long2square` | `long, row_names_from, col_names_from, values_from, symmetric=TRUE, na.fill=NA, nodes=NULL` | `long2square` | `long, row_names_from, col_names_from, values_from, symmetric=True, na_fill=np.nan, nodes=None` | Parameter renamed: `na.fill` -> `na_fill`. Python casts node names to `str` for sorting |
| `long2sparse` | `long, row_names_from, col_names_from, values_from, unique_rows=NULL, unique_cols=NULL, symmetric=FALSE` | `long2sparse` | `long, row_names_from, col_names_from, values_from, unique_rows=None, unique_cols=None, symmetric=False` | R returns `dgCMatrix`; Python returns `csr_matrix` with `.rownames`/`.colnames` attributes |
| `wide2long` | `mat` | `wide2long` | `mat` | Both return i, j, value columns (1-indexed) |
| `link2cluster` | `link, nodes` | `link2cluster` | `link, nodes` | R uses matrix power `%^% 20` then `dbscan::dbscan`; Python uses repeated squaring (16 iterations) then `sklearn.cluster.DBSCAN`. R uses `long2wide` intermediary; Python builds sparse adjacency directly |
| `mnn_dist` | `dis, k` | `mnn_dist` | `dis, k` | R uses `dbscan::kNN`; Python uses `sklearn.neighbors.NearestNeighbors(metric="precomputed")` |
| `nearest_knn` | `dis, k, top=3` | `nearest_knn` | `dis, k, top=3` | Same logic, different kNN backends |
| `cluster_merge` | `input_list, cluster` | `cluster_merge` | `input_list, cluster` | R uses `lapply(unique(cluster), ...)`; Python uses `defaultdict(list)` |
| `combn_dedup` | `combn` | `combn_dedup` | `combn_arr` | Parameter renamed: `combn` -> `combn_arr`. R returns logical vector; Python returns `np.ndarray` of bools |
| `dismat_mst` | `mat` | `dismat_mst` | `mat` | R uses `igraph::mst()`; Python uses `scipy.sparse.csgraph.minimum_spanning_tree()`. R returns 1-indexed from/to; Python returns 0-indexed |
| `dis_point_to_edge` | `point, edge_start, edge_end` | `dis_point_to_edge` | `point, edge_start, edge_end` | Equivalent algorithm. R returns `c(distance, t)`; Python returns `np.array([distance, t])` |
| `dis_points_to_edges` | `points, edges` | `dis_points_to_edges` | `points, edges` | R returns `list(map=..., dis=...)` where `map` has distance and `dis` has t-values. **Python swaps the naming**: returns `{"map": t_map, "dis": dis_map}`. Note: R `map` column contains distances (odd columns) and `dis` contains t-values (even columns); Python correctly names them |
| `knn_between_groups` | `distance, k` | `knn_between_groups` | `distance, k` | R uses rank-based selection; Python uses `np.argsort`. R is 1-indexed; Python is 0-indexed |
| `find_mutual_nn` | `distance, k, dis_thresh=NULL` | `find_mutual_nn` | `distance, k, dis_thresh=None` | Equivalent logic |
| `top_k` | `v, k, decreasing=FALSE` | `top_k` | `x, k` | R has `decreasing` parameter; Python always returns k smallest (uses `np.partition`). Parameter renamed: `v` -> `x` |
| `knn_flat` | `x, k, input="matrix", symmetric=FALSE, if_dedup=FALSE, if_self=FALSE` | `knn_flat` | `x, k, input="matrix", symmetric=False, if_dedup=False, if_self=False` | R uses `dbscan::kNN`; Python uses `sklearn.neighbors.NearestNeighbors`. Both return 1-indexed node IDs |
| `embedding2knn` | `embedding, k, mode="connectivity", ...` | `embedding2knn` | `embedding, k, mode="connectivity", **kwargs` | R `if_self` not passed by default; Python passes additional kwargs to `knn_flat`. Gaussian kernel formula differs slightly in Python (uses product of sigmas vs R's same) |
| `dist2knn` | `embedding, k, mode="connectivity", ...` | `dist2knn` | `embedding, k, mode="connectivity", **kwargs` | R connectivity formula: `exp(-dist^2/2*mean(dist)^2)` (likely a precedence bug: `2*mean^2` not `2*(mean^2)`). Python: `exp(-dist^2 / (2*sigma^2))` where sigma is per-node mean. Different formulas |
| `compute_transition` | `connectivity` | `compute_transition` | `adj` | Parameter renamed: `connectivity` -> `adj`. Equivalent double-stochastic normalization: T = Z * D^{-1} * A * D^{-1} * Z |
| `sparse_norm` | `mat, dim=1` | `sparse_norm` | `mat, norm="l2"` | **Different interface**: R normalizes by dimension (row=1, col=2) using L1; Python uses `sklearn.preprocessing.normalize` with configurable norm (default L2), always row-wise |
| `sparse_manupulation` | `mat, func` | `sparse_manipulation` | `mat, func` | Typo fixed: R `sparse_manupulation` -> Python `sparse_manipulation`. R reconstructs via `long2sparse`; Python modifies `.data` in-place |
| `filter_network` | `adj, n_neighbors=5` | `filter_network` | `adj, n_neighbors=5` | Equivalent |
| `is_symmetric` | `matrix` | `is_symmetric` | `mat` | Parameter renamed. R uses `all.equal(matrix, t(matrix))`; Python handles both sparse and dense |
| `mat_split` | `mat, size, index=0` | `mat_split` | `mat, size, index=0` | Equivalent |
| `mat_sparsify` | `mat, row_mass=0.9, col_mass=0.9` | `mat_sparsify` | `mat, row_mass=0.9, col_mass=0.9` | R `mass_filter` returns modified values (zeros non-contributing); Python `mass_filter` returns boolean mask. Python `mat_sparsify` applies mask correctly |
| `mass_filter` | `mass, thresh=0.9` | `mass_filter` | `x, thresh=0.9` | **Different return type**: R returns the vector with small values zeroed out; Python returns a boolean mask. Parameter renamed: `mass` -> `x` |
| `bin_filter` | `x, col, thresh=10, breaks=100` | `bin_filter` | `x, n_bins=20, thresh=0` | **Different interface**: R takes a data frame + column name, iteratively filters; Python takes a 1D array. Different defaults: R `thresh=10, breaks=100`; Python `n_bins=20, thresh=0`. R iterates until stable; Python does single pass |
| `dis2connec_sparse` | `D` | `dis2connec_sparse` | `dis_sparse, sigma=None` | R computes per-node sigma from mean neighbor distance. Python accepts optional global `sigma`; if `None`, uses global mean of all non-zero distances. **Different bandwidth computation** |
| `build_edges` | `df, id_col, order_col, feature_cols=NULL` | `build_edges` | `df, id_col, order_col, feature_cols=None` | R uses `dplyr::lead`; Python uses `shift(-1)` |
| `ceil_digit` | `x, n` | `ceil_digit` | `x, n` | Equivalent |
| `sync_sparse_rows` | `sparse_mat, target_rownames` | `sync_sparse_rows` | `mat, row_names` | R requires `dgCMatrix`; Python handles sparse and DataFrame. Parameter renamed |
| -- | -- | `top_k` | `x, k` | Python also has this (R has it too; listed above) |

---

## Module: clone_dis

**R file:** `Clonotrace_yuntian/R/clone_dis.R`
**Python file:** `Clonotrace_python/clonotrace/clone_dis.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| -- | -- | `_build_snn_graph` | `embedding, k` | **Python-only internal.** Builds SNN with Jaccard weights: `shared/(2k-shared)` then `exp(-jaccard)`. R uses `bluster::makeSNNGraph(type="number")` which weights by raw shared count, then `exp(-weight)`. **Different weighting schemes** |
| `clone_disance` | `embedding, cell_clone_prob, outpath, graph_k=10, overwrite=FALSE, exact=FALSE, ...` | `clone_distance` | `embedding, cell_clone_prob, outpath, graph_k=10, overwrite=False, exact=False, **kwargs` | Typo fixed: R `clone_disance` -> Python `clone_distance`. **R intersects cell names** between embedding and cell_clone_prob rows; **Python does NOT** (bug). R saves `.rds`; Python saves `.pkl`. R uses `bluster::makeSNNGraph`; Python uses internal `_build_snn_graph` with Jaccard |
| `clone_partition` | `clone_matrix, k=10, similarity_threshold=0` | `clone_partition` | `clone_matrix, k=10, similarity_threshold=0` | R returns named list of clone name vectors; Python returns dict of clone index lists (0-indexed) |
| `graph_clone_ot_sub` | `graph, cell_clone_prob, target_clone=NULL, cache=5000, verbose=TRUE` | `graph_clone_ot_sub` | `graph, cell_clone_prob, target_clone=None, cache=5000, verbose=True` | R `target_clone` defaults to `1:ncol(...)` (1-indexed); Python defaults to `range(n_clones)` (0-indexed). R skips the last clone (`ncol(cell_clone_prob)`); Python skips clone at `n_clones-1`. Both use dict/cache for distance storage |
| `graph_clone_ot` | `graph, cell_clone_prob, prob_thresh=0.05, cache=5000, cores=1, verbose=TRUE` | `graph_clone_ot` | `graph, cell_clone_prob, prob_thresh=0.05, cache=5000, cores=1, verbose=True` | R uses `future.apply::future_lapply`; Python uses `joblib.Parallel`. Both partition clones then run `graph_clone_ot_sub` per partition |
| `clone_2_ot` | `distance, group1_mass, group2_mass` | `clone_2_ot` | `distance, group1_mass, group2_mass` | **R uses `transport::transport()`; Python uses `ot.emd()` (POT library).** Both normalize masses to sum to 1. R extracts plan components `from`, `to`, `mass`, `cost`; Python multiplies transport plan matrix element-wise with cost |
| `graph_clone_nn` | `graph, cell_clone_prob, prob_thresh=0.1, k=2, verbose=FALSE` | `graph_clone_nn` | `graph, cell_clone_prob, prob_thresh=0.1, k=2, verbose=False` | R uses `future.apply` for parallelism; Python uses `joblib.Parallel(n_jobs=1)` (serial). Both delegate to `group_2_min` |
| `group_2_min` | `distance, group1, group2, k=3` | `group_2_min` | `distance, group1, group2, k=3` | R uses `apply` + custom `top_k`; Python uses `np.partition` for top-k. Equivalent logic |

---

## Module: label_propagation

**R file:** `Clonotrace_yuntian/R/label_propagation.R`
**Python file:** `Clonotrace_python/clonotrace/label_propagation.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| `label_spreading` | `adj, labels, label_n=NULL, alpha=0.9, max_iter=100, tol=1e-3, epsilon=0, verbose=TRUE` | `label_spreading` | `adj, labels, label_n=None, alpha=0.9, max_iter=100, tol=1e-3, epsilon=0, verbose=True` | R uses `message()`; Python uses `print()`. Both apply `as.numeric(as.factor(labels))` / `pd.factorize` to convert labels. Equivalent algorithm |
| `label_spreading_bootstrap` | `adj, labels, refer=NULL, alpha=0.8, sample_rate=0.8, sample_n=50, ...` | `label_spreading_bootstrap` | `adj, labels, refer=None, alpha=0.8, sample_rate=0.8, sample_n=50, **kwargs` | R uses `future_lapply` for parallel bootstrap; Python uses `joblib.Parallel(n_jobs=1)` (serial). R returns `list(prob=..., deviance=...)`; Python returns `dict` |
| `label_spreading_blocked` | `adj, labels, label_n=NULL, alpha=0.9, max_iter=100, tol=1e-3, block_size=128, outfile=NULL, verbose=TRUE, epsilon=0` | `label_spreading_blocked` | `adj, labels, label_n=None, alpha=0.9, max_iter=100, tol=1e-3, block_size=128, outfile=None, verbose=True, epsilon=0` | R uses `rhdf5` for HDF5 I/O; Python uses `h5py`. Equivalent blocked iteration logic |
| `create_hdf5_matrix` | `h5file, dataset="prob", nrow, ncol, chunk_rows=4096, chunk_cols=128` | `create_hdf5_matrix` | `h5file, dataset="prob", nrow=None, ncol=None, chunk_rows=4096, chunk_cols=128` | Python `nrow`/`ncol` default to `None` (must be provided). R uses `rhdf5`; Python uses `h5py` |
| `normalize_hdf5_rows` | `h5file, dataset="prob", block_cols=256, add_eps=TRUE` | `normalize_hdf5_rows` | `h5file, dataset="prob", block_cols=256, add_eps=True` | R uses `rhdf5::h5read/h5write`; Python uses `h5py` context manager |
| `label_spreading_bootstrap_blocked` | `adj, labels, alpha=0.8, sample_rate=0.8, sample_n=50, block_size=128, tol=1e-3, max_iter=100, epsilon=0, refer_h5=NULL, tmpdir=tempdir(), verbose=TRUE` | `label_spreading_bootstrap_blocked` | `adj, labels, alpha=0.8, sample_rate=0.8, sample_n=50, block_size=128, tol=1e-3, max_iter=100, epsilon=0, refer_h5=None, tmpdir=None, verbose=True` | R `tmpdir` defaults to `tempdir()`; Python defaults to `None` -> `tempfile.gettempdir()`. R uses `future_lapply`; Python uses `joblib.Parallel(n_jobs=1)` |

---

## Module: cluster

**R file:** `Clonotrace_yuntian/R/cluster.R`
**Python file:** `Clonotrace_python/clonotrace/cluster.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| -- | -- | `_build_snn_jaccard` | `data, k, prune_snn=0` | **Python-only internal.** Builds SNN Jaccard matrix. R uses `dbscan::sNN` inline |
| -- | -- | `_snn_from_dist` | `dismat, k, prune_snn=0` | **Python-only internal.** Same as above for distance input |
| -- | -- | `_igraph_from_sparse` | `jaccard` | **Python-only internal.** Converts sparse matrix to igraph |
| `leiden_embedding` | `data, k=30, prune.snn=0, weight="jaccard", resolution=1` | `leiden_embedding` | `data, k=30, prune_snn=0, weight="jaccard", resolution=1` | Parameter renamed: `prune.snn` -> `prune_snn`. **Both use Louvain, not Leiden** (R: `igraph::cluster_louvain`; Python: `G.community_multilevel`). R builds SNN via `dbscan::sNN` then `igraph::graph_from_data_frame`; Python builds Jaccard matrix via `_build_snn_jaccard` then `_igraph_from_sparse`. R returns 1-indexed factor; Python returns 0-indexed numpy array. **`resolution` parameter is ignored in Python** (not passed to `community_multilevel`) |
| `leiden_embedding_fast` | `data, k=30, prune.snn=0, weight="jaccard", resolution=1` | `leiden_embedding_fast` | `data, k=30, prune_snn=0, weight="jaccard", resolution=1` | Parameter renamed: `prune.snn` -> `prune_snn`. R uses `RANN::nn2` + sparse matrix multiplication for SNN; Python uses `_build_snn_jaccard`. **Both correctly use Leiden** (R: `igraph::cluster_leiden`; Python: `G.community_leiden`). R returns 1-indexed factor; Python returns 0-indexed numpy array |
| `leiden_dis` | `dismat, k=10, prune.snn=0, weight="jaccard", resolution=1, if_umap=TRUE` | `leiden_dis` | `dismat, k=10, prune_snn=0, weight="jaccard", resolution=1, if_umap=True` | Parameter renamed: `prune.snn` -> `prune_snn`. R UMAP uses `umap::umap(input="dist")`; Python uses `umap.UMAP(metric="precomputed")`. Both use Leiden clustering. R returns 1-indexed cluster factor; Python returns 0-indexed cluster labels |

---

## Module: pseudotime

**R file:** `Clonotrace_yuntian/R/pseudotime.R`
**Python file:** `Clonotrace_python/clonotrace/pseudotime.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| `acct` | `T_mat` | `acct` | `T_mat` | R uses `RSpectra::eigs(k=1)` + `Matrix::solve(..., method="CG")`; Python uses `scipy.sparse.linalg.eigsh(k=1)` + column-by-column `scipy.sparse.linalg.cg`. Python loops over columns explicitly which may be slower for large matrices |
| `DPT_T` | `T_mat, start` | `DPT_T` | `T_mat, start` | R `start` is 1-indexed; Python `start` is 0-indexed. Equivalent algorithm |
| `dpt` | `T_mat, root, k=30` | `dpt` | `T_mat, root, k=30` | R uses `RSpectra::eigs`; Python uses `scipy.sparse.linalg.eigsh`. R `root` is 1-indexed; Python `root` is 0-indexed. Python adds safety: `k_actual = min(k, N-2)` and handles zero eigenvalues |
| `embedding2dpt` | `embedding, nn_k, root, dpt_k=30` | `embedding2dpt` | `embedding, nn_k, root, dpt_k=30` | R `root` is 1-indexed; Python `root` is 0-indexed. Both build kNN, row-normalize, then call `dpt` |
| `clone_root` | `clones, cell_meta, clone_col, cluster_col, start_cluster` | `clone_root` | `clones, cell_meta, clone_col, cluster_col, start_cluster` | R uses `dplyr::filter_at/group_by_at/summarise/arrange`; Python uses `groupby.apply` with `pd.Series`. Equivalent logic |
| `clone_dpt` | `clone_embedding, cell_meta, clone_col, cluster_col, start_cluster, k=10, dpt_k=30` | `clone_dpt` | `clone_embedding, cell_meta, clone_col, cluster_col, start_cluster, k=10, dpt_k=30, clone_names=None` | Python adds `clone_names` parameter. R gets clone names from `rownames(clone_embedding)`; Python infers from DataFrame index or explicit parameter. R `root` is 1-indexed; Python is 0-indexed |

---

## Module: profile_DEG

**R file:** `Clonotrace_yuntian/R/profile_DEG.R`
**Python file:** `Clonotrace_python/clonotrace/profile_deg.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| `bin_filter_profile_mass` | `mass, time, thresh=5, binsize=0.005` | `bin_filter_profile_mass` | `mass, time, thresh=5, binsize=0.005` | R uses `dplyr::group_by/summarise/filter`; Python uses `pd.cut` + `groupby`. Equivalent logic |
| `ridge_regression` | `X, G, lambda=1e-4` | `ridge_regression` | `X, G, lam=1e-4` | Parameter renamed: `lambda` -> `lam` (Python reserved word). R uses `crossprod/solve`; Python uses `np.linalg.solve`. Equivalent closed-form solution |
| `soft_cluster_gam_fit` | `G, t, P, df=5, lambda=1e-4, test="F"` | `soft_cluster_gam_fit` | `G, t, P, df=5, lam=1e-4, test="F"` | Parameter renamed: `lambda` -> `lam`. **Spline basis differs**: R uses `splines::ns()` (natural cubic splines); Python uses `sklearn.preprocessing.SplineTransformer` (B-splines, not natural splines). R full model: `model.matrix(~ 0 + B:P)`; Python: column-stack of `B * P[:, k]`. R uses `pf()`; Python uses `scipy.stats.f.sf()` |
| `cluster_profile_mass` | `cell_profile_prob, cluster_label` | `cluster_profile_mass` | `cell_profile_prob, cluster_label` | R returns named matrix with cluster rownames; Python returns numpy array (rownames lost) |
| `cluster_profile_enrich` | `cell_profile_prob, cluster_label, permute_n=300` | `cluster_profile_enrich` | `cell_profile_prob, cluster_label, permute_n=300` | R uses `abind::abind` for 3D array; Python uses `np.stack`. Equivalent permutation logic |
| `profile_cluster_DEG_permute` | `P, G, dpt, n=50` | `profile_cluster_DEG_permute` | `P, G, dpt, n=50` | Equivalent permutation of rows + columns of P |
| `profile_cluster_DEG` | `profile, cluster, exprs, cell_meta, cell_profile_prob, cluster_col="cluster", pseudotime_col="cell_t", permute_n=50` | `profile_cluster_DEG` | `profile, cluster, exprs, cell_meta, cell_profile_prob, cluster_col="cluster", pseudotime_col="cell_t", permute_n=50` | R uses `stats::p.adjust(method="fdr")`; Python uses `statsmodels.stats.multitest.multipletests(method="fdr_bh")` (falls back to Bonferroni if statsmodels unavailable). R gene filter: `rowSums(G != 0) > 30`; Python same |
| `profile_multiclusters_DEG` | `profile, exprs, cell_meta, cell_profile_prob, clusters=NULL, cluster_col="cluster", pseudotime_col="cell_t", mass_thresh=100` | `profile_multiclusters_DEG` | `profile, exprs, cell_meta, cell_profile_prob, clusters=None, cluster_col="cluster", pseudotime_col="cell_t", mass_thresh=100` | R uses `future_lapply`; Python uses `joblib.Parallel(n_jobs=1)`. R returns named list; Python returns dict. R filters null results with `Filter(Negate(is.null), ...)`; Python filters in dict comprehension |
| -- | -- | `_natural_spline_basis` | `t, df=5, include_intercept=True` | **Python-only internal.** Approximates R's `ns()` using `SplineTransformer`. Not a true natural spline -- different knot placement and boundary behavior |

---

## Module: coembed

**R file:** `Clonotrace_yuntian/R/coembed.R`
**Python file:** `Clonotrace_python/clonotrace/coembed.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| `cell_clone_coembed` | `cell_embedding, clone_embedding, cell_k=30, clone_k=15` | `cell_clone_coembed` | `cell_embedding, clone_embedding, cell_clone_prob, cell_k=30, clone_k=15` | **R reads `cell_clone_prob` from the calling environment (implicit global).** Python takes it as an explicit parameter. R uses `clone_knn@x = rep(1,...)` to binarize; Python uses `clone_knn_bin.data = np.ones_like(...)`. R filters `weight > 0.1` then takes top 20 per node via `slice_min`; Python same via `nsmallest(20, "dis")` |
| `cell_knn_matrix_mutiplication` | `knn, cell_feature_mat, feature_feature_mat` | `cell_knn_matrix_mutiplication` | `knn, cell_feature_mat, feature_feature_mat` | Typo preserved: `mutiplication`. R `knn` has columns `i`, `j`; Python `knn` has columns `node1`, `node2` (1-indexed, converted to 0-indexed inside). Equivalent matrix algebra |
| `cell_knn_matrix_multiplication_parallel` | `knn, cell_feature_mat, feature_feature_mat, chunk_size=5000` | `cell_knn_matrix_multiplication_parallel` | `knn, cell_feature_mat, feature_feature_mat, chunk_size=5000` | R uses `future_lapply` with `future.packages = c("Matrix")`; Python uses `joblib.Parallel(n_jobs=1)` |

---

## Module: visualization

**R file:** `Clonotrace_yuntian/R/visualization.R`
**Python file:** `Clonotrace_python/clonotrace/visualization.py`

| R Function | R Parameters (with defaults) | Python Function | Python Parameters (with defaults) | Key Differences |
|---|---|---|---|---|
| `connectivity_coord` | `coord, connectivity, dims=c(1,2)` | `connectivity_coord` | `coord, connectivity, dims=(0,1)` | **R `dims` is 1-indexed; Python `dims` is 0-indexed.** R uses `dplyr::left_join`; Python uses `pd.merge`. R extracts from sparse via `summary()`; Python via `.tocoo()` |
| `dimplot` | `embedding, annot, color_by, alpha_by=NULL, connectivity=NULL, label=TRUE, dims=c(1,2), connectivity_thresh=0.1, label_size=5, label_type="text", label_color="black", box.padding=0.25, point.padding=1e-6, raster_thresh=10000, ...` | `dimplot` | `embedding, annot, color_by, alpha_by=None, connectivity=None, label=True, dims=(0,1), connectivity_thresh=0.1, label_size=10, label_type="text", label_color="black", raster_thresh=10000, ax=None, **scatter_kwargs` | **R uses ggplot2; Python uses matplotlib.** R `dims` 1-indexed; Python 0-indexed. Default `label_size`: R=5, Python=10. R has `box.padding`/`point.padding` params (ggrepel); Python has `ax` param (matplotlib Axes). R uses `ggrepel::geom_text_repel`; Python uses `adjustText` if available. R uses `ggrastr::rasterize`; Python uses matplotlib `rasterized=True` |
| `scatterpie` | `scatter_coord, composition, connectivity=NULL, connectivity_thresh=0.5, dims=c("umap_1","umap_2"), cluster_col="cluster", edge_color="lightgrey", edge_alpha=1, label_size=5` | `scatterpie` | `scatter_coord, composition, connectivity=None, connectivity_thresh=0.5, dims=("umap_1","umap_2"), cluster_col="cluster", edge_color="lightgrey", edge_alpha=1, label_size=10, ax=None` | **R uses `scatterpie::geom_scatterpie` (ggplot2 layer list); Python draws `matplotlib.patches.Wedge` manually.** Default `label_size`: R=5, Python=10. Python has `ax` param. R returns list of ggplot layers; Python returns `(fig, ax)` tuple |
| `umap_from_knn` | `adj, n_neighbors=5, seed=1024` | `umap_from_knn` | `adj, n_neighbors=5, seed=1024` | **R calls `umap_from_knn_py()` via reticulate** (Python backend). Python converts adjacency to distance matrix (`1 - normalized_adj`) and uses `umap.UMAP(metric="precomputed")` directly. R calls `filter_network` then the Python function; Python calls `filter_network` then UMAP |
| `mds_from_knn` | `adj, n_components=15` | `mds_from_knn` | `adj, n_components=15` | **R calls `mds_from_knn_py()` via reticulate.** Python uses `sklearn.manifold.MDS(dissimilarity="precomputed")` directly with `random_state=42, n_init=1, max_iter=300`. R hardcodes `n_neighbors=5` for filtering; Python same |
