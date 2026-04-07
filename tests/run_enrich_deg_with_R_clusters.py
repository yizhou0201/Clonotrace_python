#!/usr/bin/env python3
"""
Run steps 9-11 (profile enrichment + DEG) using R's Leiden clustering,
to isolate these tests from Leiden stochasticity.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "/Users/yizhouw/Desktop/packages/Clonotrace_python")
import clonotrace as ct

os.chdir("/Users/yizhouw/Desktop/packages/Clonotrace_python")

# === Load data ===
print("=== Loading data ===")
cell_meta = pd.read_csv("tests/real_data/cell_meta.csv", index_col=0)
cell_names = pd.read_csv("tests/real_data/cell_names.csv")["cell_name"].tolist()
clone_names = pd.read_csv("tests/real_data/clone_names.csv")["clone_name"].tolist()
n_clones = len(clone_names)

# Load R's clone clustering
r_cluster = pd.read_csv("tests/real_outputs_R/clone_cluster.csv", index_col=0)
print(f"R clusters: {r_cluster['cluster'].nunique()} unique")

# Load R's clone pseudotime
r_pt = pd.read_csv("tests/real_outputs_R/clone_t.csv")
clone_t = r_pt["dpt"].values
print(f"R pseudotime range: [{clone_t.min():.3f}, {clone_t.max():.3f}]")

# Load filtered cell_clone_prob from Python steps 1-3
triplets = pd.read_csv("tests/real_outputs_python/cell_clone_prob_filtered_triplets.csv")
kept_cell_names = pd.read_csv("tests/real_outputs_python/cell_clone_prob_filtered_cells.csv")["cell"].tolist()
n_kept = len(kept_cell_names)

rows = triplets["row"].values.astype(int) - 1
cols = triplets["col"].values.astype(int) - 1
vals = triplets["value"].values.astype(float)
cell_clone_prob_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(n_kept, n_clones))
print(f"cell_clone_prob: {cell_clone_prob_sparse.shape}, nnz={cell_clone_prob_sparse.nnz}")

# === Step 9: Cell profile probabilities (using R's clustering) ===
print("\n=== Step 9: Cell profile probabilities (R clusters) ===")
cluster_order = sorted(r_cluster["cluster"].unique(), key=str)
clone_profile_mat = np.zeros((n_clones, len(cluster_order)), dtype=float)
cluster_idx_map = {c: i for i, c in enumerate(cluster_order)}
for ci, cname in enumerate(clone_names):
    cl_label = r_cluster.loc[cname, "cluster"]
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

# Cell pseudotime using R's clone_t
cell_t = (cell_clone_prob_sparse.toarray() @ clone_t).ravel()
cell_meta_sub = cell_meta.loc[kept_cell_names].copy()
cell_meta_sub["cell_t"] = cell_t

cell_meta_sub.to_csv("tests/real_outputs_python/cell_meta_with_t_Rcluster.csv")
cell_profile_df.to_csv("tests/real_outputs_python/cell_profile_prob_Rcluster.csv")
print(f"Cell profile prob: {cell_profile_df.shape}")

# === Step 10: Profile enrichment ===
print("\n=== Step 10: Profile enrichment (R clusters) ===")
cell_meta_sub["cluster"] = cell_meta_sub["cluster"].astype(str)
cell_clusters = cell_meta_sub["cluster"].values

enrich = ct.cluster_profile_enrich(
    cell_profile_prob=cell_profile_prob_norm,
    cluster_label=cell_clusters,
    permute_n=300
)
pd.DataFrame(enrich["prob"]).to_csv("tests/real_outputs_python/enrich_mass_Rcluster.csv")
pd.DataFrame(enrich["pval"]).to_csv("tests/real_outputs_python/enrich_pval_Rcluster.csv")
print(f"Enrichment: {enrich['prob'].shape[0]} clusters x {enrich['prob'].shape[1]} profiles")

# === Step 11: Profile DEG ===
print("\n=== Step 11: Profile DEG (R clusters) ===")
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
        deg_result["stat"].to_csv("tests/real_outputs_python/DEG_stats_Rcluster.csv")
        print(f"DEG result: {len(deg_result['stat'])} genes")
    else:
        print("DEG returned None (insufficient cells)")
else:
    print("Expression data not found.")

# === Compare with R ===
print("\n" + "=" * 70)
print("  Comparison: Python (R clusters) vs R")
print("=" * 70)

from scipy import stats

def safe_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(stats.pearsonr(a[mask], b[mask])[0])

def safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(a[mask], b[mask])[0])

# Enrichment
r_ep = pd.read_csv("tests/real_outputs_R/enrich_pval.csv", index_col=0)
py_ep = pd.read_csv("tests/real_outputs_python/enrich_pval_Rcluster.csv", index_col=0)
print(f"\nEnrichment pval: R {r_ep.shape}, Py {py_ep.shape}")
r_vals = r_ep.values.ravel()
py_vals = py_ep.values.ravel()
n = min(len(r_vals), len(py_vals))
sc = safe_pearson(r_vals[:n], py_vals[:n])
print(f"  Pearson r: {sc:.4f}")

# Enrichment mass
r_em = pd.read_csv("tests/real_outputs_R/enrich_mass.csv", index_col=0)
py_em = pd.read_csv("tests/real_outputs_python/enrich_mass_Rcluster.csv", index_col=0)
print(f"\nEnrichment mass: R {r_em.shape}, Py {py_em.shape}")
r_vals = r_em.values.ravel()
py_vals = py_em.values.ravel()
n = min(len(r_vals), len(py_vals))
sc = safe_pearson(r_vals[:n], py_vals[:n])
print(f"  Pearson r: {sc:.4f}")

# DEG
r_deg = pd.read_csv("tests/real_outputs_R/DEG_stats.csv", index_col=0)
py_deg = pd.read_csv("tests/real_outputs_python/DEG_stats_Rcluster.csv", index_col=0)
common_genes = r_deg.index.intersection(py_deg.index)
print(f"\nDEG: R {len(r_deg)} genes, Py {len(py_deg)} genes, common {len(common_genes)}")
if len(common_genes) > 10:
    sc_stat = safe_spearman(r_deg.loc[common_genes, "stat"], py_deg.loc[common_genes, "stat"])
    sc_cohen = safe_spearman(r_deg.loc[common_genes, "cohen"], py_deg.loc[common_genes, "cohen"])
    sc_mean = safe_spearman(r_deg.loc[common_genes, "mean_diff"], py_deg.loc[common_genes, "mean_diff"])
    print(f"  F-stat Spearman r:    {sc_stat:.4f}")
    print(f"  Cohen's d Spearman r: {sc_cohen:.4f}")
    print(f"  mean_diff Spearman r: {sc_mean:.4f}")

print("\n=== Done ===")
