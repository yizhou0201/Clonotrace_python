"""Quick profile: how many iterations does warm start save?"""
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd

print("Loading data...")
pca = pd.read_csv("real_data/pca.csv", index_col=0).values
cell_meta = pd.read_csv("real_data/cell_meta.csv", index_col=0)
from clonotrace.auxiliary import embedding2knn, compute_transition
from clonotrace.label_propagation import _prepare_labels, _build_P, _build_Y

cell_knn = embedding2knn(pca, k=30)
cell_knn = compute_transition(cell_knn)

clone_size = cell_meta.groupby("clone").size().reset_index(name="count")
expanded = clone_size[clone_size["count"] >= 10]["clone"].values
cell_clone = cell_meta["clone"].copy()
cell_clone[~cell_clone.isin(expanded)] = np.nan

labels, valid_mask = _prepare_labels(cell_clone.values)
N = len(labels)
C = int(np.nanmax(labels))
P = _build_P(cell_knn)
Y_ref = _build_Y(labels, valid_mask, N, C, epsilon=0)

# Cold start reference
print(f"\nCold start iteration (N={N}, C={C}):")
F = Y_ref.copy()
alpha = 0.6
Y_scaled = (1 - alpha) * Y_ref
for it in range(1, 101):
    F_new = alpha * P.dot(F) + Y_scaled
    diff = float(np.max(np.abs(F_new - F)))
    if it <= 5 or it % 5 == 0 or diff < 1e-3:
        print(f"  Iter {it:3d}: diff = {diff:.4e}")
    F = F_new
    if diff < 1e-3:
        break
F_ref = F.copy()

# Warm start bootstrap
labeled_idx = np.where(valid_mask)[0]
label_sample_count = max(1, round(len(labeled_idx) * 0.8))

np.random.seed(42)
sub_labels = np.full(N, np.nan)
sampled = np.random.choice(labeled_idx, size=label_sample_count, replace=False)
sub_labels[sampled] = labels[sampled]
Y_b = _build_Y(sub_labels, np.isfinite(sub_labels), N, C, epsilon=0)
Y_b_scaled = (1 - alpha) * Y_b

print(f"\nWarm start iteration (from F_ref):")
F = F_ref.copy()
for it in range(1, 101):
    F_new = alpha * P.dot(F) + Y_b_scaled
    diff = float(np.max(np.abs(F_new - F)))
    print(f"  Iter {it:3d}: diff = {diff:.4e}")
    F = F_new
    if diff < 1e-3:
        break

# Time single iteration
print(f"\nTiming single P.dot(F) where F is ({N}, {C})...")
times = []
for _ in range(5):
    t0 = time.time()
    _ = P.dot(F_ref)
    times.append(time.time() - t0)
print(f"  Per iteration: {np.mean(times):.4f}s")

# Time batched iteration
for bs in [1, 2, 4, 8, 16, 48]:
    F_wide = np.tile(F_ref, (1, bs))
    times = []
    for _ in range(3):
        t0 = time.time()
        _ = P.dot(F_wide)
        times.append(time.time() - t0)
    per_sample = np.mean(times) / bs
    print(f"  Batch {bs:2d}: {np.mean(times):.4f}s total, {per_sample:.4f}s/sample")
