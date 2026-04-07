#!/usr/bin/env python3
"""
03_test_python.py - Run Python vignette pipeline and save outputs for comparison
Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/
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
os.makedirs("tests/outputs_python", exist_ok=True)

print("=== Loading exported data ===")
pca         = pd.read_csv("tests/data/pca.csv", index_col=0).values.astype(float)
cell_meta   = pd.read_csv("tests/data/cell_meta.csv", index_col=0)
cell_names  = pd.read_csv("tests/data/cell_names.csv")["cell_name"].tolist()
clone_names = pd.read_csv("tests/data/clone_names.csv")["clone_name"].tolist()
triplets    = pd.read_csv("tests/data/cell_clone_binary_triplets.csv")
label_df    = pd.read_csv("tests/data/cell_clone_labels.csv")
exprs_cl4   = pd.read_csv("tests/data/exprs_cluster4.csv", index_col=0)

n_cells  = len(cell_names)
n_clones = len(clone_names)
print(f"PCA: {pca.shape[0]} x {pca.shape[1]} | Clones: {n_clones} | Cells: {n_cells}")

# Reconstruct binary cell-clone probability matrix (sparse)
rows = triplets["cell_idx"].values.astype(int)
cols = triplets["clone_idx"].values.astype(int)
vals = triplets["value"].values.astype(float)
cell_clone_prob = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_clones))
print(f"cell_clone_prob: {cell_clone_prob.shape} (sparse, nnz={cell_clone_prob.nnz})")

# ── Step 1: kNN + transition ──────────────────────────────────────────────────
print("\n=== Step 1: kNN + transition ===")
cell_knn = ct.embedding2knn(pca, k=30, mode="connectivity")
T_mat = ct.compute_transition(cell_knn)

# Save first 200 cells' non-zero entries of T_mat
T_sub = T_mat[:200, :].tocoo()
T_df = pd.DataFrame({"row": T_sub.row + 1, "col": T_sub.col + 1, "value": T_sub.data})
T_df.to_csv("tests/outputs_python/cell_knn_sample.csv", index=False)
print(f"T_mat[:200,:] non-zero entries: {len(T_df)}")

# ── Step 2: Label spreading (small subset) ───────────────────────────────────
print("\n=== Step 2: Label spreading (subset: 2000 cells, 20 clones) ===")
sub_cells = list(range(2000))
sub_clones = clone_names[:20]
T_sub20 = T_mat[:2000, :2000]

clone_label_map = dict(zip(label_df["cell"], label_df["clone"]))
clone_labels_sub = np.array([
    clone_label_map.get(cell_names[i], np.nan)
    for i in sub_cells
], dtype=object)
# Replace clones not in sub_clones with NaN
clone_labels_sub = np.where(
    pd.Series(clone_labels_sub).isin(sub_clones).values,
    clone_labels_sub,
    np.nan
).astype(float) if False else [  # use string labels
    (v if v in sub_clones else float("nan"))
    for v in clone_labels_sub
]
clone_labels_sub = np.array(clone_labels_sub, dtype=object)

spread_result = ct.label_spreading_bootstrap(
    adj=T_sub20,
    labels=clone_labels_sub,
    alpha=0.6, sample_rate=0.8, sample_n=5
)
prob_sub = spread_result["prob"]
prob_df = pd.DataFrame(prob_sub, index=[cell_names[i] for i in sub_cells])
prob_df.to_csv("tests/outputs_python/label_spread_small.csv")
print(f"Label spread result: {prob_df.shape}")

# ── Step 3: Clone NN distance (full 802 clones) ──────────────────────────────
print("\n=== Step 3: Clone NN distance (full, 802 clones) ===")
start_time = time.time()
clone_nn_dis = ct.clone_distance(
    embedding=pca,
    cell_clone_prob=cell_clone_prob,
    outpath="tests/outputs_python/clone_dis_cache/",
    graph_k=10, overwrite=True, exact=False
)
elapsed = (time.time() - start_time) / 60
print(f"Clone NN distance time: {elapsed:.1f} min")
clone_nn_dis.to_csv("tests/outputs_python/clone_nn_dis.csv", index=False)
print(f"Clone NN pairs: {len(clone_nn_dis)}")

# Convert to square matrix with 0-indexed nodes
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
# Save named square matrix for comparison (first 50x50)
sq50 = pd.DataFrame(clone_dis_sq[:50, :50],
                    index=clone_names[:50], columns=clone_names[:50])
sq50.to_csv("tests/outputs_python/clone_nn_dis_sq50.csv")

# ── Step 4: Clone OT distance (cells from first 5 clones only) ───────────────
# Note: R OT uses igraph::distances() to ALL cells (O(N^2)).
# Use only cells belonging to first 5 clones (~552 cells) for feasibility.
print("\n=== Step 4: Clone OT distance (cells from first 5 clones) ===")
cc_sub5 = cell_clone_prob[:, :5]
sub_cell_mask = np.array(cc_sub5.sum(axis=1)).ravel() > 0
sub_cell_idx = np.where(sub_cell_mask)[0]
pca_sub_ot = pca[sub_cell_idx, :]
cell_clone_sub_ot = cc_sub5[sub_cell_idx, :]
print(f"OT subset: {cell_clone_sub_ot.shape[0]} cells x {cell_clone_sub_ot.shape[1]} clones")
start_time = time.time()
clone_ot_dis = ct.clone_distance(
    embedding=pca_sub_ot,
    cell_clone_prob=cell_clone_sub_ot,
    outpath="tests/outputs_python/clone_ot_cache/",
    graph_k=10, overwrite=True, exact=True
)
elapsed = (time.time() - start_time) / 60
print(f"Clone OT distance time: {elapsed:.1f} min")
clone_ot_dis.to_csv("tests/outputs_python/clone_ot_dis_subset.csv", index=False)
print(f"Clone OT pairs: {len(clone_ot_dis)}")

# ── Step 5: Clone MDS (30 dims, non-metric to match MASS::isoMDS) ────────────
print("\n=== Step 5: Clone MDS (30 dims) ===")
np.random.seed(42)
mds = MDS(n_components=30, dissimilarity="precomputed", metric=False,
          random_state=42, n_init=1, max_iter=300, n_jobs=1)
clone_embedding = mds.fit_transform(clone_dis_sq)
clone_embedding_df = pd.DataFrame(
    clone_embedding,
    index=list(range(n_clones)),
    columns=[f"mds_{i+1}" for i in range(30)]
)
# Rename index to clone_names
clone_embedding_df.index = clone_names
clone_embedding_df.to_csv("tests/outputs_python/clone_mds.csv")
print(f"Clone MDS: {clone_embedding_df.shape}")

# ── Step 6: Leiden clustering ─────────────────────────────────────────────────
print("\n=== Step 6: Leiden clustering ===")
clone_cluster_df = ct.leiden_dis(
    dismat=clone_dis_sq, k=20, resolution=0.5, if_umap=True
)
clone_cluster_df.index = clone_names
clone_cluster_df.to_csv("tests/outputs_python/clone_cluster.csv")
print(f"Clone clusters: {clone_cluster_df['cluster'].nunique()} unique, {len(clone_cluster_df)} clones")

# ── Step 7: Clone pseudotime ─────────────────────────────────────────────────
print("\n=== Step 7: Clone pseudotime ===")
# clone_dpt accepts DataFrame with index as clone names
clone_t = ct.clone_dpt(
    clone_embedding=clone_embedding_df,
    cell_meta=cell_meta,
    clone_col="clone",
    cluster_col="cluster",
    start_cluster="0"
)
clone_t_df = pd.DataFrame({"clone": clone_names, "dpt": clone_t})
clone_t_df.to_csv("tests/outputs_python/clone_t.csv", index=False)
print(f"Clone pseudotime range: [{clone_t.min():.3f}, {clone_t.max():.3f}]")

# ── Step 8: Cell profile probabilities ───────────────────────────────────────
print("\n=== Step 8: Cell profile probabilities ===")
# Build clone -> leiden cluster mapping
cluster_order = sorted(clone_cluster_df["cluster"].unique(), key=str)
clone_profile_mat = np.zeros((n_clones, len(cluster_order)), dtype=float)
cluster_idx_map = {c: i for i, c in enumerate(cluster_order)}
for ci, cl_label in enumerate(clone_cluster_df["cluster"].values):
    clone_profile_mat[ci, cluster_idx_map[cl_label]] = 1.0

# cell_profile_prob = cell_clone_prob @ clone_profile_mat
cell_profile_prob = cell_clone_prob.toarray() @ clone_profile_mat
row_sums = cell_profile_prob.sum(axis=1, keepdims=True)
row_sums = np.maximum(row_sums, 1e-12)
cell_profile_prob_norm = cell_profile_prob / row_sums

cell_profile_df = pd.DataFrame(
    cell_profile_prob_norm,
    index=cell_names,
    columns=[str(c) for c in cluster_order]
)

# Cell pseudotime
cell_t = (cell_clone_prob.toarray() @ clone_t).ravel()
cell_meta["cell_t"] = cell_t
cell_meta.to_csv("tests/outputs_python/cell_meta_with_t.csv")
cell_profile_df.to_csv("tests/outputs_python/cell_profile_prob.csv")

# ── Step 9: Profile enrichment ───────────────────────────────────────────────
print("\n=== Step 9: Profile enrichment ===")
cell_meta["cluster"] = cell_meta["cluster"].astype(str)  # ensure string for comparisons
cell_clusters = cell_meta["cluster"].values
enrich = ct.cluster_profile_enrich(
    cell_profile_prob=cell_profile_prob_norm,
    cluster_label=cell_clusters,
    permute_n=300
)
pd.DataFrame(enrich["prob"]).to_csv("tests/outputs_python/enrich_mass.csv")
pd.DataFrame(enrich["pval"]).to_csv("tests/outputs_python/enrich_pval.csv")
print(f"Enrichment result: {enrich['prob'].shape[0]} clusters x {enrich['prob'].shape[1]} profiles")

# ── Step 10: Profile DEG ─────────────────────────────────────────────────────
print("\n=== Step 10: Profile DEG (profile=first cluster, cluster=4) ===")
profile_name = str(cluster_order[0])
print(f"Testing profile: {profile_name}")

# exprs_cl4 is (genes, cells); profile_cluster_DEG expects (genes, cells)
deg_result = ct.profile_cluster_DEG(
    profile=profile_name,
    cluster="4",
    exprs=exprs_cl4.values,   # (genes, cells)
    cell_meta=cell_meta,
    cell_profile_prob=cell_profile_df,
    cluster_col="cluster",
    pseudotime_col="cell_t"
)

if deg_result is not None:
    deg_result["stat"].to_csv("tests/outputs_python/DEG_stats.csv")
    print(f"DEG result: {len(deg_result['stat'])} genes")
else:
    print("DEG returned None (insufficient cells)")
    pd.DataFrame({"note": ["None result"]}).to_csv("tests/outputs_python/DEG_stats.csv", index=False)

print("\n=== All Python pipeline steps complete ===")
print("Outputs saved to tests/outputs_python/")
