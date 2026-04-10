"""
Verify optimized graph_clone_nn produces identical results to original,
then benchmark on full dataset.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import pandas as pd

# ================================================================
# Original implementation (copy for comparison)
# ================================================================
def graph_clone_nn_old(graph, cell_clone_prob, prob_thresh=0.1, k=2):
    """Original per-clone igraph.distances() implementation."""
    if sp.issparse(cell_clone_prob):
        cell_clone_prob = cell_clone_prob.toarray()
    cell_group_mat = np.asarray(cell_clone_prob, dtype=float)
    cell_group_mat = (cell_group_mat >= prob_thresh).astype(float)
    n_groups = cell_group_mat.shape[1]

    from clonotrace.clone_dis import group_2_min

    rows = []
    for i in range(n_groups - 1):
        from_cells = np.where(cell_group_mat[:, i] > 0)[0].tolist()
        if not from_cells:
            continue
        to_cells = np.where(cell_group_mat[:, i + 1:].sum(axis=1) > 0)[0].tolist()
        if not to_cells:
            continue
        dis_i = np.array(graph.distances(source=from_cells, target=to_cells,
                                          weights="weight"))
        for j in range(i + 1, n_groups):
            id_j = np.where(cell_group_mat[:, j] > 0)[0]
            to_in_j = [ti for ti, tc in enumerate(to_cells) if tc in set(id_j.tolist())]
            if not to_in_j:
                continue
            sub_dis = dis_i[:, to_in_j]
            d = group_2_min(sub_dis, list(range(sub_dis.shape[0])),
                            list(range(sub_dis.shape[1])), k=k)
            rows.append([i, j, d])
    return pd.DataFrame(rows, columns=["group1", "group2", "dis"])


# ================================================================
# Load data
# ================================================================
print("Loading data...")
pca = pd.read_csv("real_data/pca.csv", index_col=0)
cell_clone_prob = mmread("real_data/cell_clone_prob.mtx").tocsr()
prob_rows = open("real_data/cell_clone_prob_rows.txt").read().splitlines()
pca_aligned = pca.loc[prob_rows]
N, C = cell_clone_prob.shape
print(f"  Cells: {N}, Clones: {C}")

from clonotrace.clone_dis import _build_snn_graph, graph_clone_nn

print("Building SNN graph...")
cell_graph = _build_snn_graph(pca_aligned.values, k=10)

# ================================================================
# TEST 1: Equivalence on 20-clone subset
# ================================================================
print("\n" + "=" * 60)
print("TEST 1: Equivalence on 20-clone subset")
print("=" * 60)

small_prob = cell_clone_prob[:, :20]

print("  Running OLD (igraph, per-clone)...")
t0 = time.time()
dis_old = graph_clone_nn_old(cell_graph, small_prob, prob_thresh=0.1, k=2)
t_old = time.time() - t0
print(f"  Time: {t_old:.2f}s, pairs: {len(dis_old)}")

print("  Running NEW (scipy, cached)...")
t0 = time.time()
dis_new = graph_clone_nn(cell_graph, small_prob, prob_thresh=0.1, k=2, verbose=True)
t_new = time.time() - t0
print(f"  Time: {t_new:.2f}s, pairs: {len(dis_new)}")

# Merge and compare
merged = dis_old.merge(dis_new, on=["group1", "group2"], suffixes=("_old", "_new"))
max_diff = (merged["dis_old"] - merged["dis_new"]).abs().max()
corr = merged["dis_old"].corr(merged["dis_new"])

print(f"\n  Pairs matched: {len(merged)}")
print(f"  Max absolute diff: {max_diff:.2e}")
print(f"  Correlation: {corr:.10f}")
print(f"  Speedup: {t_old / t_new:.1f}x")

if max_diff < 1e-10:
    print("  RESULT: EXACT MATCH")
else:
    print(f"  RESULT: DIFFERS by {max_diff:.2e}")

# ================================================================
# TEST 2: Full dataset benchmark
# ================================================================
print("\n" + "=" * 60)
print("TEST 2: Full dataset (802 clones)")
print("=" * 60)

print("  Running optimized graph_clone_nn...")
t0 = time.time()
dis_full = graph_clone_nn(cell_graph, cell_clone_prob, prob_thresh=0.1, k=2, verbose=True)
t_full = time.time() - t0
print(f"  Time: {t_full:.2f}s")
print(f"  Clone pairs: {len(dis_full)}")
print(f"  Distance range: [{dis_full['dis'].min():.4f}, {dis_full['dis'].max():.4f}]")

# ================================================================
# Summary
# ================================================================
t_graph = 0.97  # from earlier benchmark
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  SNN graph:                  {t_graph:.2f}s")
print(f"  20 clones - old (igraph):   {t_old:.2f}s")
print(f"  20 clones - new (scipy):    {t_new:.2f}s  ({t_old/t_new:.1f}x faster)")
print(f"  802 clones - new (scipy):   {t_full:.2f}s")
print(f"  Total (graph + NN):         {t_graph + t_full:.2f}s")
print(f"  Est. old 802 clones:        ~{t_old * C / 20:.0f}s")
print(f"  Overall speedup:            ~{(t_old * C / 20) / (t_graph + t_full):.1f}x")
print(f"  Equivalence:                {'EXACT' if max_diff < 1e-10 else 'CLOSE'}")
