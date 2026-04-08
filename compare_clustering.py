"""
Compare Leiden vs Louvain clustering on real hematopoiesis dataset.
Tests: speed, cluster count, and assignment similarity.
"""
import time
import numpy as np
import pandas as pd

# Load clone distance matrix
print("Loading clone distance matrix...")
clone_dis_raw = pd.read_csv("real_data/clone_graph_dis.tsv", sep="\t", index_col=0)
from clonotrace.auxiliary import long2square
clone_dis = long2square(clone_dis_raw, row_names_from="group1",
                         col_names_from="group2", values_from="dis", symmetric=True)
np.fill_diagonal(clone_dis, 0)
max_dis = np.nanmax(clone_dis)
clone_dis = np.nan_to_num(clone_dis, nan=max_dis)
N = clone_dis.shape[0]
print(f"  Clone distance matrix: {N}x{N}")

from clonotrace.cluster import leiden_dis

# ================================================================
# Benchmark: Louvain vs Leiden (multiple seeds for stability)
# ================================================================
print("\n" + "=" * 60)
print("SPEED COMPARISON")
print("=" * 60)

for method in ["louvain", "leiden"]:
    times = []
    for trial in range(3):
        np.random.seed(1230 + trial)
        t0 = time.time()
        result = leiden_dis(clone_dis, k=20, resolution=0.5,
                            if_umap=True, method=method)
        t = time.time() - t0
        times.append(t)
        n_clusters = len(result["cluster"].unique())
    print(f"  {method:8s}: {np.mean(times):.3f}s ± {np.std(times):.3f}s "
          f"(clusters: {n_clusters})")

# ================================================================
# Cluster comparison at seed=1230
# ================================================================
print("\n" + "=" * 60)
print("CLUSTER COMPARISON (seed=1230, k=20, resolution=0.5)")
print("=" * 60)

np.random.seed(1230)
t0 = time.time()
res_louvain = leiden_dis(clone_dis, k=20, resolution=0.5,
                          if_umap=True, method="louvain")
t_louvain = time.time() - t0

np.random.seed(1230)
t0 = time.time()
res_leiden = leiden_dis(clone_dis, k=20, resolution=0.5,
                         if_umap=True, method="leiden")
t_leiden = time.time() - t0

labels_l = res_louvain["cluster"].values
labels_d = res_leiden["cluster"].values

n_l = len(np.unique(labels_l))
n_d = len(np.unique(labels_d))

print(f"  Louvain: {n_l} clusters, {t_louvain:.3f}s")
print(f"  Leiden:  {n_d} clusters, {t_leiden:.3f}s")
print(f"  Speedup: {t_louvain / t_leiden:.2f}x")

# Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(labels_l, labels_d)
nmi = normalized_mutual_info_score(labels_l, labels_d)
print(f"\n  Adjusted Rand Index (ARI): {ari:.4f}")
print(f"  Normalized Mutual Info (NMI): {nmi:.4f}")

# Exact agreement
exact = (labels_l == labels_d).mean()
print(f"  Exact label match: {100*exact:.1f}%")

# Cluster size distributions
print(f"\n  Louvain cluster sizes: {sorted(np.bincount(labels_l), reverse=True)}")
print(f"  Leiden  cluster sizes: {sorted(np.bincount(labels_d), reverse=True)}")

# Cross-tabulation
ct = pd.crosstab(pd.Series(labels_l, name="Louvain"),
                  pd.Series(labels_d, name="Leiden"))
print(f"\n  Cross-tabulation (Louvain rows × Leiden cols):")
print(ct.to_string())

# ================================================================
# Breakdown: UMAP vs clustering time
# ================================================================
print("\n" + "=" * 60)
print("TIMING BREAKDOWN")
print("=" * 60)

# UMAP only
np.random.seed(1230)
import umap as umap_module
t0 = time.time()
reducer = umap_module.UMAP(metric="precomputed")
_ = reducer.fit_transform(clone_dis)
t_umap = time.time() - t0
print(f"  UMAP:     {t_umap:.3f}s")

# Clustering only (no UMAP)
np.random.seed(1230)
t0 = time.time()
_ = leiden_dis(clone_dis, k=20, resolution=0.5,
               if_umap=False, method="louvain")
t_louvain_only = time.time() - t0
print(f"  Louvain:  {t_louvain_only:.3f}s (without UMAP)")

np.random.seed(1230)
t0 = time.time()
_ = leiden_dis(clone_dis, k=20, resolution=0.5,
               if_umap=False, method="leiden")
t_leiden_only = time.time() - t0
print(f"  Leiden:   {t_leiden_only:.3f}s (without UMAP)")
print(f"  Clustering speedup (Leiden/Louvain): {t_louvain_only / t_leiden_only:.2f}x")
print(f"  UMAP dominates: {100*t_umap/(t_umap + max(t_louvain_only, t_leiden_only)):.0f}% of total time")

# ================================================================
# Resolution sweep
# ================================================================
print("\n" + "=" * 60)
print("RESOLUTION SWEEP (no UMAP)")
print("=" * 60)
for res in [0.3, 0.5, 0.8, 1.0, 1.5]:
    np.random.seed(1230)
    rl = leiden_dis(clone_dis, k=20, resolution=res, if_umap=False, method="louvain")
    np.random.seed(1230)
    rd = leiden_dis(clone_dis, k=20, resolution=res, if_umap=False, method="leiden")
    ari = adjusted_rand_score(rl, rd)
    print(f"  res={res:.1f}: Louvain {len(np.unique(rl))} clusters, "
          f"Leiden {len(np.unique(rd))} clusters, ARI={ari:.4f}")
