"""
Benchmark clone_distance computation on the real hematopoiesis dataset.
Profile the bottleneck: graph.distances() calls.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import pandas as pd

print("Loading data...")
pca = pd.read_csv("real_data/pca.csv", index_col=0)
cell_clone_prob = mmread("real_data/cell_clone_prob.mtx").tocsr()
prob_rows = open("real_data/cell_clone_prob_rows.txt").read().splitlines()
prob_cols = open("real_data/cell_clone_prob_cols.txt").read().splitlines()

pca_aligned = pca.loc[prob_rows]
N, C = cell_clone_prob.shape
print(f"  Cells: {N}, Clones: {C}")

from clonotrace.clone_dis import _build_snn_graph, graph_clone_nn

# ================================================================
# Step 1: SNN graph construction
# ================================================================
print("\n--- SNN graph construction (k=10) ---")
t0 = time.time()
cell_graph = _build_snn_graph(pca_aligned.values, k=10)
t_graph = time.time() - t0
print(f"  Time: {t_graph:.2f}s")
print(f"  Nodes: {cell_graph.vcount()}, Edges: {cell_graph.ecount()}")

# ================================================================
# Step 2: Profile graph.distances() cost
# ================================================================
print("\n--- Profile: graph.distances() cost ---")

# Single source, all targets
t0 = time.time()
d = cell_graph.distances(source=[0], weights="weight")
t_single = time.time() - t0
print(f"  Single source → all targets: {t_single:.3f}s")

# 10 sources, all targets
t0 = time.time()
d = cell_graph.distances(source=list(range(10)), weights="weight")
t_10 = time.time() - t0
print(f"  10 sources → all targets: {t_10:.3f}s ({t_10/10:.3f}s/source)")

# 10 sources, 100 targets
t0 = time.time()
d = cell_graph.distances(source=list(range(10)), target=list(range(100)), weights="weight")
t_10_100 = time.time() - t0
print(f"  10 sources → 100 targets: {t_10_100:.3f}s")

# Estimate for NN method: each clone has ~39 cells on average (31262/802)
# For each clone i, it queries from_cells → to_cells
# from_cells ≈ cells in clone i, to_cells ≈ cells in clones i+1..802
prob_dense = cell_clone_prob.toarray()
prob_binary = (prob_dense >= 0.1).astype(float)
cells_per_clone = prob_binary.sum(axis=0)
print(f"\n  Cells per clone: mean={cells_per_clone.mean():.0f}, "
      f"max={cells_per_clone.max():.0f}, min={cells_per_clone.min():.0f}")

# Count total graph.distances() calls and source cells
total_sources = 0
for i in range(C):
    from_cells = np.where(prob_binary[:, i] > 0)[0]
    total_sources += len(from_cells)
print(f"  Total source cells across all clones: {total_sources}")
print(f"  Estimated NN time (serial): {total_sources * t_single:.0f}s")

# ================================================================
# Step 3: Small-scale NN benchmark (first 20 clones)
# ================================================================
print("\n--- NN benchmark: first 20 clones ---")
small_prob = cell_clone_prob[:, :20]

t0 = time.time()
dis_nn_20 = graph_clone_nn(cell_graph, small_prob, prob_thresh=0.1, k=2,
                            verbose=False, n_jobs=1)
t_nn_20_1 = time.time() - t0
print(f"  20 clones (n_jobs=1): {t_nn_20_1:.2f}s, pairs: {len(dis_nn_20)}")

t0 = time.time()
dis_nn_20p = graph_clone_nn(cell_graph, small_prob, prob_thresh=0.1, k=2,
                              verbose=False, n_jobs=-1)
t_nn_20_p = time.time() - t0
print(f"  20 clones (n_jobs=-1): {t_nn_20_p:.2f}s")

# Estimate full
n_pairs_20 = 20 * 19 // 2
n_pairs_full = C * (C - 1) // 2
# The bottleneck scales with the number of source cells × graph size,
# not just number of pairs
print(f"\n  Estimated full 802 clones (serial): ~{t_nn_20_1 * C / 20:.0f}s")
print(f"  Estimated full 802 clones (parallel): ~{t_nn_20_p * C / 20:.0f}s")

# ================================================================
# Step 4: Small-scale OT benchmark (first 10 clones)
# ================================================================
print("\n--- OT benchmark: first 10 clones ---")
try:
    import ot
    from clonotrace.clone_dis import graph_clone_ot

    small_prob_ot = cell_clone_prob[:, :10]
    t0 = time.time()
    dis_ot_10 = graph_clone_ot(cell_graph, small_prob_ot, prob_thresh=0.05,
                                cache=5000, cores=1, verbose=False)
    t_ot_10 = time.time() - t0
    print(f"  10 clones OT: {t_ot_10:.2f}s, pairs: {len(dis_ot_10)}")
    print(f"  Estimated full 802 clones: ~{t_ot_10 * (C/10)**2:.0f}s")
except ImportError:
    print("  POT not installed, skipping OT benchmark")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  SNN graph construction:          {t_graph:.2f}s")
print(f"  graph.distances() per source:    {t_single:.3f}s")
print(f"  NN 20 clones (serial):           {t_nn_20_1:.2f}s")
print(f"  NN 20 clones (parallel):         {t_nn_20_p:.2f}s")
print(f"  NN 802 clones (est. parallel):   ~{t_nn_20_p * C / 20:.0f}s")
print(f"  Key bottleneck: igraph Dijkstra on {N}-node, {cell_graph.ecount()}-edge graph")
