#!/usr/bin/env python3
"""
12_run_python_pipeline.py - Run full Python pipeline on real hematopoiesis data
and save all intermediate outputs for comparison with R.

Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/
Prerequisite: Run 10_export_real_data.R first
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.manifold import MDS

sys.path.insert(0, "/Users/yizhouw/Desktop/packages/Clonotrace_python")
import clonotrace as ct

os.chdir("/Users/yizhouw/Desktop/packages/Clonotrace_python")
os.makedirs("tests/real_outputs_python", exist_ok=True)

print("=== Loading exported data ===")
pca = pd.read_csv("tests/real_data/pca.csv", index_col=0).values.astype(float)
cell_meta = pd.read_csv("tests/real_data/cell_meta.csv", index_col=0)
cell_names = pd.read_csv("tests/real_data/cell_names.csv")["cell_name"].tolist()
clone_names = pd.read_csv("tests/real_data/clone_names.csv")["clone_name"].tolist()
triplets = pd.read_csv("tests/real_data/cell_clone_binary_triplets.csv")
label_df = pd.read_csv("tests/real_data/cell_clone_labels.csv")

n_cells = len(cell_names)
n_clones = len(clone_names)
print(f"PCA: {pca.shape[0]} x {pca.shape[1]} | Clones: {n_clones} | Cells: {n_cells}")

# Reconstruct binary cell-clone probability matrix (sparse)
rows = triplets["cell_idx"].values.astype(int)
cols = triplets["clone_idx"].values.astype(int)
vals = triplets["value"].values.astype(float)
cell_clone_binary = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_clones))

# ── Step 1: kNN + transition ────────────────────────────────────────────────
print("\n=== Step 1: kNN + transition ===")
cell_knn = ct.embedding2knn(pca, k=30, mode="connectivity")
T_mat = ct.compute_transition(cell_knn)

# Save sample
T_sub = T_mat[:200, :].tocoo()
T_df = pd.DataFrame({"row": T_sub.row + 1, "col": T_sub.col + 1,
                      "value": T_sub.data})
T_df.to_csv("tests/real_outputs_python/cell_knn_sample.csv", index=False)
print(f"T_mat[:200,:] nnz: {len(T_df)}")

# ── Step 2: Label spreading (bootstrap) ─────────────────────────────────────
print("\n=== Step 2: Label spreading (bootstrap, sample_n=48) ===")
np.random.seed(42)

clone_label_map = dict(zip(label_df["cell"], label_df["clone"]))
clone_labels_all = np.array([
    clone_label_map.get(cell_names[i], np.nan)
    for i in range(n_cells)
], dtype=object)

start_time = time.time()
clone_spread = ct.label_spreading_bootstrap(
    adj=T_mat,
    labels=clone_labels_all,
    alpha=0.6,
    sample_rate=0.8,
    sample_n=48
)
elapsed = (time.time() - start_time) / 60
print(f"Label spreading time: {elapsed:.1f} min")

cell_clone_prob_raw = clone_spread["prob"]
deviance = clone_spread["deviance"]

# Save deviance
pd.DataFrame({"cell": cell_names, "deviance": deviance}).to_csv(
    "tests/real_outputs_python/deviance.csv", index=False)

# Save sample
prob_sample = pd.DataFrame(
    cell_clone_prob_raw[:500, :50],
    index=cell_names[:500]
)
prob_sample.to_csv("tests/real_outputs_python/cell_clone_prob_sample.csv")
print(f"cell_clone_prob: {cell_clone_prob_raw.shape}")

# ── Step 3: Filter and sparsify ─────────────────────────────────────────────
print("\n=== Step 3: Filter by deviance + sparsify ===")
keep_mask = deviance < 0.3
cell_clone_prob = cell_clone_prob_raw[keep_mask]
row_sums = cell_clone_prob.sum(axis=1, keepdims=True)
row_sums = np.maximum(row_sums, 1e-12)
cell_clone_prob = cell_clone_prob / row_sums

cell_clone_prob = ct.mat_sparsify(cell_clone_prob, row_mass=0.9, col_mass=0.9)
row_sums = cell_clone_prob.sum(axis=1, keepdims=True)
row_sums = np.maximum(row_sums, 1e-12)
cell_clone_prob = cell_clone_prob / row_sums
cell_clone_prob_sparse = sp.csr_matrix(cell_clone_prob)

kept_cell_names = [cell_names[i] for i in range(n_cells) if keep_mask[i]]
print(f"After filtering: {cell_clone_prob.shape[0]} cells x {cell_clone_prob.shape[1]} clones")

# Save filtered cell_clone_prob as triplets
ccp_coo = cell_clone_prob_sparse.tocoo()
pd.DataFrame({
    "row": ccp_coo.row + 1,
    "col": ccp_coo.col + 1,
    "value": ccp_coo.data
}).to_csv("tests/real_outputs_python/cell_clone_prob_filtered_triplets.csv",
          index=False)
pd.DataFrame({"cell": kept_cell_names}).to_csv(
    "tests/real_outputs_python/cell_clone_prob_filtered_cells.csv", index=False)

# ── Step 4: Clone NN distance (full) ────────────────────────────────────────
print("\n=== Step 4: Clone NN distance (full, approximate) ===")
# Use PCA for kept cells only
kept_indices = [i for i in range(n_cells) if keep_mask[i]]
pca_kept = pca[kept_indices]

start_time = time.time()
clone_nn_dis = ct.clone_distance(
    embedding=pca_kept,
    cell_clone_prob=cell_clone_prob_sparse,
    outpath="tests/real_outputs_python/clone_nn_cache/",
    graph_k=10,
    overwrite=True,
    exact=False
)
elapsed = (time.time() - start_time) / 60
print(f"Clone NN distance time: {elapsed:.1f} min")
clone_nn_dis.to_csv("tests/real_outputs_python/clone_nn_dis.csv", index=False)

# Convert to square matrix
clone_dis_sq = ct.long2square(
    long=clone_nn_dis,
    row_names_from="group1",
    col_names_from="group2",
    values_from="dis",
    symmetric=True,
    na_fill=0.0,
    nodes=list(range(n_clones))
)
np.fill_diagonal(clone_dis_sq, 0)

# Save first 50x50
sq50 = pd.DataFrame(clone_dis_sq[:50, :50],
                     index=clone_names[:50], columns=clone_names[:50])
sq50.to_csv("tests/real_outputs_python/clone_nn_dis_sq50.csv")

# ── Step 5: Clone OT distance — SKIPPED (too slow on real data) ──────────────
print("\n=== Step 5: Clone OT distance — SKIPPED ===")
print("OT on real data takes hours. Use existing test data (tests/02-04) for OT validation.")

# ── Step 6: Clone MDS (30 dims) ─────────────────────────────────────────────
print("\n=== Step 6: Clone MDS (30 dims) ===")
np.random.seed(42)
# Use metric MDS init, then refine with non-metric.
# sklearn's non-metric SMACOF degenerates at k>=5 with random init.
mds_metric = MDS(n_components=30, dissimilarity="precomputed", metric=True,
                 random_state=42, n_init=1, max_iter=300, n_jobs=1)
init_embedding = mds_metric.fit_transform(clone_dis_sq)
mds = MDS(n_components=30, dissimilarity="precomputed", metric=False,
          random_state=42, n_init=1, max_iter=300, n_jobs=1)
clone_embedding = mds.fit_transform(clone_dis_sq, init=init_embedding)
clone_embedding_df = pd.DataFrame(
    clone_embedding,
    index=clone_names,
    columns=[f"mds_{i+1}" for i in range(30)]
)
clone_embedding_df.to_csv("tests/real_outputs_python/clone_mds.csv")

# ── Step 7: Leiden clustering ────────────────────────────────────────────────
print("\n=== Step 7: Leiden clustering ===")
clone_cluster_df = ct.leiden_dis(
    dismat=clone_dis_sq, k=20, resolution=0.5, if_umap=True
)
clone_cluster_df.index = clone_names
clone_cluster_df.to_csv("tests/real_outputs_python/clone_cluster.csv")
print(f"Clusters: {clone_cluster_df['cluster'].nunique()} unique")

# ── Step 8: Clone pseudotime ────────────────────────────────────────────────
print("\n=== Step 8: Clone pseudotime ===")
clone_t = ct.clone_dpt(
    clone_embedding=clone_embedding_df,
    cell_meta=cell_meta,
    clone_col="clone",
    cluster_col="cluster",
    start_cluster="0"
)
pd.DataFrame({"clone": clone_names, "dpt": clone_t}).to_csv(
    "tests/real_outputs_python/clone_t.csv", index=False)
print(f"Clone pseudotime range: [{clone_t.min():.3f}, {clone_t.max():.3f}]")

# ── Step 9: Cell profile probabilities ───────────────────────────────────────
print("\n=== Step 9: Cell profile probabilities ===")
cluster_order = sorted(clone_cluster_df["cluster"].unique(), key=str)
clone_profile_mat = np.zeros((n_clones, len(cluster_order)), dtype=float)
cluster_idx_map = {c: i for i, c in enumerate(cluster_order)}
for ci, cl_label in enumerate(clone_cluster_df["cluster"].values):
    clone_profile_mat[ci, cluster_idx_map[cl_label]] = 1.0

cell_profile_prob = cell_clone_prob_sparse.toarray() @ clone_profile_mat
row_sums = cell_profile_prob.sum(axis=1, keepdims=True)
row_sums = np.maximum(row_sums, 1e-12)
cell_profile_prob_norm = cell_profile_prob / row_sums

cell_profile_df = pd.DataFrame(
    cell_profile_prob_norm,
    index=kept_cell_names,
    columns=[str(c) for c in cluster_order]
)

# Cell pseudotime
cell_t = (cell_clone_prob_sparse.toarray() @ clone_t).ravel()
cell_meta_sub = cell_meta.loc[kept_cell_names].copy()
cell_meta_sub["cell_t"] = cell_t
cell_meta_sub.to_csv("tests/real_outputs_python/cell_meta_with_t.csv")
cell_profile_df.to_csv("tests/real_outputs_python/cell_profile_prob.csv")

# ── Step 10: Profile enrichment ──────────────────────────────────────────────
print("\n=== Step 10: Profile enrichment ===")
cell_meta_sub["cluster"] = cell_meta_sub["cluster"].astype(str)
cell_clusters = cell_meta_sub["cluster"].values

enrich = ct.cluster_profile_enrich(
    cell_profile_prob=cell_profile_prob_norm,
    cluster_label=cell_clusters,
    permute_n=300
)
pd.DataFrame(enrich["prob"]).to_csv("tests/real_outputs_python/enrich_mass.csv")
pd.DataFrame(enrich["pval"]).to_csv("tests/real_outputs_python/enrich_pval.csv")
print(f"Enrichment: {enrich['prob'].shape[0]} clusters x {enrich['prob'].shape[1]} profiles")

# ── Step 11: Profile DEG ────────────────────────────────────────────────────
print("\n=== Step 11: Profile DEG ===")
exprs_path = "tests/real_data/exprs_cluster4.csv"
if os.path.exists(exprs_path):
    exprs_cl4 = pd.read_csv(exprs_path, index_col=0)
    profile_name = str(cluster_order[0])
    print(f"Testing profile: {profile_name}")

    deg_result = ct.profile_cluster_DEG(
        profile=profile_name,
        cluster="4",
        exprs=exprs_cl4,
        cell_meta=cell_meta_sub,
        cell_profile_prob=cell_profile_df,
        cluster_col="cluster",
        pseudotime_col="cell_t"
    )

    if deg_result is not None:
        deg_result["stat"].to_csv("tests/real_outputs_python/DEG_stats.csv")
        print(f"DEG result: {len(deg_result['stat'])} genes")
    else:
        print("DEG returned None (insufficient cells)")
else:
    print("Expression data not found. Run 10_export_real_data.R with Seurat object.")

# ── Step 12: SNN graph comparison ────────────────────────────────────────────
print("\n=== Step 12: Export SNN graph weights for comparison ===")
from clonotrace.clone_dis import _build_snn_graph
G = _build_snn_graph(pca[:500], k=10)
edges = G.get_edgelist()
weights = G.es["weight"]
snn_df = pd.DataFrame({
    "from": [e[0] + 1 for e in edges],  # 1-indexed to match R
    "to": [e[1] + 1 for e in edges],
    "weight_exp": weights
})
snn_df.to_csv("tests/real_outputs_python/snn_graph_weights.csv", index=False)
print(f"SNN graph edges: {len(snn_df)}")

print("\n=== All Python pipeline steps complete ===")
print("Outputs saved to tests/real_outputs_python/")
