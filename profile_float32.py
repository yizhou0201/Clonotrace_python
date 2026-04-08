"""Quick test: float32 vs float64 and relaxed tolerance."""
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd

print("Loading data...")
pca = pd.read_csv("real_data/pca.csv", index_col=0).values
cell_meta = pd.read_csv("real_data/cell_meta.csv", index_col=0)
from clonotrace.auxiliary import embedding2knn, compute_transition
from clonotrace.label_propagation import _prepare_labels, _build_P, _build_Y, _label_spreading_iterative

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
Y = _build_Y(labels, valid_mask, N, C, epsilon=0)

# Test 1: float64 with different tolerances
for tol in [1e-3, 5e-3, 1e-2]:
    t0 = time.time()
    F = _label_spreading_iterative(P, Y, 0.6, 100, tol)
    t = time.time() - t0
    print(f"  float64 tol={tol}: {t:.3f}s")

# Test 2: float32
P32 = P.astype(np.float32)
Y32 = Y.astype(np.float32)

for tol in [1e-3, 5e-3, 1e-2]:
    t0 = time.time()
    F32 = _label_spreading_iterative(P32, Y32, 0.6, 100, tol)
    t = time.time() - t0
    print(f"  float32 tol={tol}: {t:.3f}s")

# Compare float32 vs float64 accuracy
F64 = _label_spreading_iterative(P, Y, 0.6, 100, 1e-3)
F32_ref = _label_spreading_iterative(P32, Y32, 0.6, 100, 1e-3)
diff = np.max(np.abs(F64 - F32_ref.astype(np.float64)))
print(f"\n  float32 vs float64 max diff: {diff:.2e}")

# Test 3: Batched group timing with float32
print(f"\nBatched P.dot(F) timing (groups of 8):")
for dtype_name, P_test, Y_test in [("float64", P, Y), ("float32", P32, Y32)]:
    F_wide = np.tile(Y_test, (1, 8))
    times = []
    for _ in range(3):
        t0 = time.time()
        _ = P_test.dot(F_wide)
        times.append(time.time() - t0)
    print(f"  {dtype_name} batch=8: {np.mean(times):.3f}s/iter")

# Test 4: Check if threading helps (GIL release)
print("\nThread test (concurrent sparse matmul):")
from concurrent.futures import ThreadPoolExecutor

F_group = np.tile(Y, (1, 8))
def do_matmul(F):
    return P.dot(F)

# Sequential
t0 = time.time()
for _ in range(4):
    do_matmul(F_group)
t_seq = time.time() - t0
print(f"  Sequential (4 groups): {t_seq:.3f}s")

# Threaded
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(do_matmul, F_group) for _ in range(4)]
    [f.result() for f in futures]
t_thread = time.time() - t0
print(f"  Threaded (4 groups, 4 threads): {t_thread:.3f}s")
print(f"  Thread speedup: {t_seq / t_thread:.1f}x")
