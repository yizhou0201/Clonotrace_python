"""
Benchmark clone_distance computation on the real hematopoiesis dataset.
Tests both exact (OT) and approximate (NN) methods.
"""
import time
import os
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import pandas as pd

print("Loading data...")
pca = pd.read_csv("real_data/pca.csv", index_col=0)
cell_clone_prob = mmread("real_data/cell_clone_prob.mtx").tocsr()
prob_rows = open("real_data/cell_clone_prob_rows.txt").read().splitlines()
prob_cols = open("real_data/cell_clone_prob_cols.txt").read().splitlines()

# Align PCA to cell_clone_prob rows
pca_aligned = pca.loc[prob_rows]
N, C = cell_clone_prob.shape
print(f"  Cells: {N}, Clones: {C}")
print(f"  PCA dims: {pca_aligned.shape[1]}")

from clonotrace.clone_dis import clone_distance, _build_snn_graph, graph_clone_nn, graph_clone_ot

# ================================================================
# Step 1: SNN graph construction
# ================================================================
print("\n" + "=" * 60)
print("STEP 1: SNN graph construction (k=10)")
print("=" * 60)

t0 = time.time()
cell_graph = _build_snn_graph(pca_aligned.values, k=10)
t_graph = time.time() - t0
print(f"  Time: {t_graph:.2f}s")
print(f"  Nodes: {cell_graph.vcount()}, Edges: {cell_graph.ecount()}")

# ================================================================
# Step 2: Approximate method (graph_clone_nn)
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Clone distance — NN approximate (default)")
print("=" * 60)

t0 = time.time()
dis_nn = graph_clone_nn(cell_graph, cell_clone_prob, prob_thresh=0.1, k=2,
                         verbose=False, n_jobs=-1)
t_nn = time.time() - t0
print(f"  Time: {t_nn:.2f}s")
print(f"  Clone pairs: {len(dis_nn)}")
print(f"  Distance range: [{dis_nn['dis'].min():.4f}, {dis_nn['dis'].max():.4f}]")

# Also single-core
t0 = time.time()
dis_nn_1 = graph_clone_nn(cell_graph, cell_clone_prob, prob_thresh=0.1, k=2,
                            verbose=False, n_jobs=1)
t_nn_1 = time.time() - t0
print(f"  Time (n_jobs=1): {t_nn_1:.2f}s")
print(f"  Parallel speedup: {t_nn_1 / t_nn:.1f}x")

# ================================================================
# Step 3: Exact method (graph_clone_ot) — may be very slow
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Clone distance — OT exact")
print("=" * 60)

# First check if POT is available
try:
    import ot
    has_pot = True
except ImportError:
    has_pot = False
    print("  POT not installed, skipping exact OT benchmark")

if has_pot:
    # Try with a small subset first (10 clones) to estimate time
    print("  Testing with 10 clones to estimate full time...")
    small_prob = cell_clone_prob[:, :10].toarray()
    t0 = time.time()
    dis_ot_small = graph_clone_ot(cell_graph, small_prob, prob_thresh=0.05,
                                   cache=5000, cores=1, verbose=False)
    t_small = time.time() - t0
    n_pairs_small = len(dis_ot_small)
    n_pairs_full = C * (C - 1) // 2
    # Estimate: time scales roughly with number of pairs
    t_estimate = t_small * (n_pairs_full / max(n_pairs_small, 1))
    print(f"  10-clone OT: {t_small:.2f}s ({n_pairs_small} pairs)")
    print(f"  Estimated full ({C} clones, {n_pairs_full} pairs): {t_estimate:.0f}s")

    if t_estimate < 600:  # Only run if estimated < 10 min
        print(f"\n  Running full OT (estimated {t_estimate:.0f}s)...")
        t0 = time.time()
        dis_ot = graph_clone_ot(cell_graph, cell_clone_prob, prob_thresh=0.05,
                                 cache=5000, cores=1, verbose=True)
        t_ot = time.time() - t0
        print(f"  Full OT time: {t_ot:.2f}s")
        print(f"  Clone pairs: {len(dis_ot)}")
    else:
        print(f"  Skipping full OT (would take ~{t_estimate/60:.0f} min)")

# ================================================================
# Step 4: Compare NN distances with pre-computed OT distances
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: Compare NN vs pre-computed OT distances")
print("=" * 60)

from clonotrace.auxiliary import long2square

# Load pre-computed OT distances
ot_raw = pd.read_csv("real_data/clone_graph_dis.tsv", sep="\t", index_col=0)
ot_dis = long2square(ot_raw, row_names_from="group1", col_names_from="group2",
                      values_from="dis", symmetric=True)
np.fill_diagonal(ot_dis, 0)

# Build NN distance matrix
nn_dis = long2square(dis_nn, row_names_from="group1", col_names_from="group2",
                      values_from="dis", symmetric=True)
np.fill_diagonal(nn_dis, 0)

# Both should be C×C
print(f"  OT matrix: {ot_dis.shape}, NN matrix: {nn_dis.shape}")

# Correlation of upper triangle
mask = np.triu(np.ones_like(ot_dis, dtype=bool), k=1)
ot_vals = ot_dis[mask]
nn_vals = nn_dis[mask]
# Remove NaN
valid = np.isfinite(ot_vals) & np.isfinite(nn_vals)
ot_v = ot_vals[valid]
nn_v = nn_vals[valid]

from scipy.stats import spearmanr, pearsonr
pearson_r, _ = pearsonr(ot_v, nn_v)
spearman_r, _ = spearmanr(ot_v, nn_v)
print(f"  Valid pairs: {valid.sum()}")
print(f"  Pearson correlation (OT vs NN): {pearson_r:.4f}")
print(f"  Spearman correlation (OT vs NN): {spearman_r:.4f}")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  SNN graph construction: {t_graph:.2f}s")
print(f"  NN approximate (parallel): {t_nn:.2f}s")
print(f"  NN approximate (single):   {t_nn_1:.2f}s")
print(f"  Total (graph + NN parallel): {t_graph + t_nn:.2f}s")
if has_pot:
    print(f"  OT exact (10 clones): {t_small:.2f}s")
    print(f"  OT exact (estimated full): ~{t_estimate:.0f}s")
