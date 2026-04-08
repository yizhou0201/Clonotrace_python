"""
Benchmark: Can Parquet/PyArrow improve Clonotrace?
====================================================
Tests three potential areas:
  1. DataFrame operations (merge, groupby, concat) with PyArrow backend
  2. Disk I/O: Parquet vs pickle vs HDF5 for caching
  3. In-memory: PyArrow compute vs pandas/numpy for common patterns
"""

import os
import time
import tempfile
import tracemalloc
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq

np.random.seed(42)

def timed(func, *args, repeats=5, **kwargs):
    """Run func, return (result, median_time, peak_memory_bytes)."""
    times = []
    result = None
    peak_mem = 0
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        _, pm = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(t1 - t0)
        peak_mem = max(peak_mem, pm)
    return result, float(np.median(times)), peak_mem

print("=" * 70)
print("PARQUET/PYARROW BENCHMARK FOR CLONOTRACE")
print("=" * 70)

# =====================================================================
# AREA 1: DataFrame operations — PyArrow backend vs default pandas
# =====================================================================
print("\n" + "=" * 70)
print("AREA 1: DataFrame Operations (PyArrow backend vs pandas default)")
print("=" * 70)

# --- 1a. knn_flat output: merge + groupby (mimics embedding2knn flow) ---
print("\n--- 1a. Merge + GroupBy (embedding2knn pattern, N=50K edges) ---")

n_edges = 50000
n_nodes = 5000
knn_data = {
    "node1": np.random.randint(1, n_nodes + 1, n_edges),
    "node2": np.random.randint(1, n_nodes + 1, n_edges),
    "dist": np.random.rand(n_edges),
}

def pandas_merge_groupby():
    knn = pd.DataFrame(knn_data)
    sigma_df = knn.groupby("node1")["dist"].mean().reset_index()
    sigma_df.columns = ["node1", "sigma"]
    knn = knn.merge(sigma_df, on="node1")
    sigma_j = sigma_df.rename(columns={"node1": "node2", "sigma": "sigma_j"})
    knn = knn.merge(sigma_j, on="node2")
    knn["connectivity"] = np.exp(-knn["dist"] ** 2 / (knn["sigma"] * knn["sigma_j"]))
    return knn

def arrow_merge_groupby():
    knn = pd.DataFrame(knn_data).convert_dtypes(dtype_backend="pyarrow")
    sigma_df = knn.groupby("node1")["dist"].mean().reset_index()
    sigma_df.columns = ["node1", "sigma"]
    knn = knn.merge(sigma_df, on="node1")
    sigma_j = sigma_df.rename(columns={"node1": "node2", "sigma": "sigma_j"})
    knn = knn.merge(sigma_j, on="node2")
    knn["connectivity"] = np.exp(
        -knn["dist"].to_numpy() ** 2 / (knn["sigma"].to_numpy() * knn["sigma_j"].to_numpy())
    )
    return knn

def numpy_direct():
    """Current optimized approach (no DataFrames at all)."""
    node1 = knn_data["node1"]
    node2 = knn_data["node2"]
    dist = knn_data["dist"]
    # Compute sigma per node
    sigma = np.zeros(n_nodes + 1)
    counts = np.zeros(n_nodes + 1)
    np.add.at(sigma, node1, dist)
    np.add.at(counts, node1, 1)
    counts = np.maximum(counts, 1)
    sigma = sigma / counts
    # Gaussian kernel
    connectivity = np.exp(-dist ** 2 / (sigma[node1] * sigma[node2]))
    return connectivity

_, t_pd, m_pd = timed(pandas_merge_groupby)
_, t_pa, m_pa = timed(arrow_merge_groupby)
_, t_np, m_np = timed(numpy_direct)

print(f"  pandas default:   {t_pd:.4f}s  mem={m_pd/1024:.0f}KB")
print(f"  pyarrow backend:  {t_pa:.4f}s  mem={m_pa/1024:.0f}KB")
print(f"  numpy direct:     {t_np:.4f}s  mem={m_np/1024:.0f}KB")

# --- 1b. long_symmetry pattern: concat + dedup + sort ---
print("\n--- 1b. Concat + Dedup + Sort (long_symmetry, 100K rows) ---")

n_rows = 100000
sym_data = pd.DataFrame({
    "node1": np.random.randint(0, 5000, n_rows),
    "node2": np.random.randint(0, 5000, n_rows),
    "weight": np.random.rand(n_rows),
})

def pandas_symmetry(df):
    rdf = df[["node2", "node1", "weight"]].copy()
    rdf.columns = ["node1", "node2", "weight"]
    out = pd.concat([df, rdf], ignore_index=True).drop_duplicates()
    out = out.sort_values(["node1", "node2"]).reset_index(drop=True)
    return out

def arrow_symmetry(df):
    df_a = df.convert_dtypes(dtype_backend="pyarrow")
    rdf = df_a[["node2", "node1", "weight"]].copy()
    rdf.columns = ["node1", "node2", "weight"]
    out = pd.concat([df_a, rdf], ignore_index=True).drop_duplicates()
    out = out.sort_values(["node1", "node2"]).reset_index(drop=True)
    return out

_, t_pd, m_pd = timed(pandas_symmetry, sym_data)
_, t_pa, m_pa = timed(arrow_symmetry, sym_data)

print(f"  pandas default:   {t_pd:.4f}s  mem={m_pd/1024:.0f}KB")
print(f"  pyarrow backend:  {t_pa:.4f}s  mem={m_pa/1024:.0f}KB")

# --- 1c. Pivot table (long2wide pattern) ---
print("\n--- 1c. Pivot Table (long2wide, 10K rows) ---")

pivot_data = pd.DataFrame({
    "row": np.random.choice([f"r{i}" for i in range(200)], 10000),
    "col": np.random.choice([f"c{i}" for i in range(200)], 10000),
    "val": np.random.rand(10000),
})

def pandas_pivot(df):
    return df.pivot_table(index="row", columns="col", values="val", aggfunc="first")

def arrow_pivot(df):
    return df.convert_dtypes(dtype_backend="pyarrow").pivot_table(
        index="row", columns="col", values="val", aggfunc="first"
    )

_, t_pd, m_pd = timed(pandas_pivot, pivot_data)
_, t_pa, m_pa = timed(arrow_pivot, pivot_data)

print(f"  pandas default:   {t_pd:.4f}s  mem={m_pd/1024:.0f}KB")
print(f"  pyarrow backend:  {t_pa:.4f}s  mem={m_pa/1024:.0f}KB")

# --- 1d. GroupBy + nsmallest (coembed pattern) ---
print("\n--- 1d. GroupBy + nsmallest (coembed pattern, 200K rows) ---")

coembed_data = pd.DataFrame({
    "node1": np.random.randint(1, 5001, 200000),
    "dis": np.random.rand(200000),
    "weight": np.random.rand(200000) + 0.01,
    "dist": np.random.rand(200000),
})

def pandas_groupby_nsmallest(df):
    return df.groupby("node1", group_keys=False).apply(lambda g: g.nsmallest(20, "dis"))

def arrow_groupby_nsmallest(df):
    df_a = df.convert_dtypes(dtype_backend="pyarrow")
    return df_a.groupby("node1", group_keys=False).apply(lambda g: g.nsmallest(20, "dis"))

_, t_pd, m_pd = timed(pandas_groupby_nsmallest, coembed_data, repeats=3)
_, t_pa, m_pa = timed(arrow_groupby_nsmallest, coembed_data, repeats=3)

print(f"  pandas default:   {t_pd:.4f}s  mem={m_pd/1024:.0f}KB")
print(f"  pyarrow backend:  {t_pa:.4f}s  mem={m_pa/1024:.0f}KB")


# =====================================================================
# AREA 2: Disk I/O — Parquet vs Pickle vs HDF5
# =====================================================================
print("\n" + "=" * 70)
print("AREA 2: Disk I/O (Parquet vs Pickle vs HDF5)")
print("=" * 70)

tmpdir = tempfile.mkdtemp()

# --- 2a. DataFrame serialization (clone_distance results) ---
print("\n--- 2a. DataFrame write+read (clone_distance results, 50K rows) ---")

df_cache = pd.DataFrame({
    "group1": np.random.randint(0, 100, 50000),
    "group2": np.random.randint(0, 100, 50000),
    "dis": np.random.rand(50000),
})

pkl_path = os.path.join(tmpdir, "cache.pkl")
pq_path = os.path.join(tmpdir, "cache.parquet")

def pickle_write_read(df, path):
    with open(path, "wb") as f:
        pickle.dump(df, f)
    with open(path, "rb") as f:
        return pickle.load(f)

def parquet_write_read(df, path):
    df.to_parquet(path, engine="pyarrow")
    return pd.read_parquet(path, engine="pyarrow")

_, t_pkl, m_pkl = timed(pickle_write_read, df_cache, pkl_path)
_, t_pq, m_pq = timed(parquet_write_read, df_cache, pq_path)

pkl_size = os.path.getsize(pkl_path)
pq_size = os.path.getsize(pq_path)

print(f"  pickle:   {t_pkl:.4f}s  mem={m_pkl/1024:.0f}KB  file={pkl_size/1024:.0f}KB")
print(f"  parquet:  {t_pq:.4f}s  mem={m_pq/1024:.0f}KB  file={pq_size/1024:.0f}KB")

# --- 2b. Large matrix serialization (label_propagation HDF5 replacement) ---
print("\n--- 2b. Dense matrix write+read (5000 x 500 label matrix) ---")
import h5py

N, C = 5000, 500
big_matrix = np.random.rand(N, C).astype(np.float64)

h5_path = os.path.join(tmpdir, "prob.h5")
pq_mat_path = os.path.join(tmpdir, "prob.parquet")
pkl_mat_path = os.path.join(tmpdir, "prob.pkl")
npy_path = os.path.join(tmpdir, "prob.npy")

def hdf5_write_read(mat, path):
    with h5py.File(path, "w") as f:
        f.create_dataset("prob", data=mat, compression="gzip", compression_opts=7)
    with h5py.File(path, "r") as f:
        return f["prob"][:]

def parquet_mat_write_read(mat, path):
    df = pd.DataFrame(mat)
    df.to_parquet(path, engine="pyarrow")
    return pd.read_parquet(path, engine="pyarrow").values

def pickle_mat_write_read(mat, path):
    with open(path, "wb") as f:
        pickle.dump(mat, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "rb") as f:
        return pickle.load(f)

def npy_write_read(mat, path):
    np.save(path, mat)
    return np.load(path)

_, t_h5, m_h5 = timed(hdf5_write_read, big_matrix, h5_path, repeats=3)
_, t_pq, m_pq = timed(parquet_mat_write_read, big_matrix, pq_mat_path, repeats=3)
_, t_pkl, m_pkl = timed(pickle_mat_write_read, big_matrix, pkl_mat_path, repeats=3)
_, t_npy, m_npy = timed(npy_write_read, big_matrix, npy_path, repeats=3)

h5_size = os.path.getsize(h5_path)
pq_size = os.path.getsize(pq_mat_path)
pkl_size = os.path.getsize(pkl_mat_path)
npy_size = os.path.getsize(npy_path)

print(f"  HDF5 gzip:  {t_h5:.4f}s  mem={m_h5/1024:.0f}KB  file={h5_size/1024:.0f}KB")
print(f"  parquet:    {t_pq:.4f}s  mem={m_pq/1024:.0f}KB  file={pq_size/1024:.0f}KB")
print(f"  pickle:     {t_pkl:.4f}s  mem={m_pkl/1024:.0f}KB  file={pkl_size/1024:.0f}KB")
print(f"  numpy .npy: {t_npy:.4f}s  mem={m_npy/1024:.0f}KB  file={npy_size/1024:.0f}KB")

# --- 2c. Blocked column read (label_propagation pattern) ---
print("\n--- 2c. Blocked column read (5000 x 500 matrix, 128-col blocks) ---")

# Write once
with h5py.File(h5_path, "w") as f:
    f.create_dataset("prob", data=big_matrix, chunks=(min(N, 4096), 128),
                     compression="gzip", compression_opts=7)

# Write parquet column-chunked
table = pa.table({f"c{i}": big_matrix[:, i] for i in range(C)})
pq.write_table(table, pq_mat_path)

block_size = 128

def hdf5_blocked_read(path):
    result = np.zeros(N)
    with h5py.File(path, "r") as f:
        for s in range(0, C, block_size):
            e = min(s + block_size, C)
            block = f["prob"][:, s:e]
            result += block.sum(axis=1)
    return result

def parquet_blocked_read(path):
    result = np.zeros(N)
    pf = pq.ParquetFile(path)
    for s in range(0, C, block_size):
        e = min(s + block_size, C)
        cols = [f"c{i}" for i in range(s, e)]
        block = pf.read(columns=cols).to_pandas().values
        result += block.sum(axis=1)
    return result

_, t_h5, m_h5 = timed(hdf5_blocked_read, h5_path, repeats=3)
_, t_pq, m_pq = timed(parquet_blocked_read, pq_mat_path, repeats=3)

print(f"  HDF5 blocked:    {t_h5:.4f}s  mem={m_h5/1024:.0f}KB")
print(f"  parquet blocked: {t_pq:.4f}s  mem={m_pq/1024:.0f}KB")


# =====================================================================
# AREA 3: In-memory compute — PyArrow compute vs numpy
# =====================================================================
print("\n" + "=" * 70)
print("AREA 3: In-Memory Compute (PyArrow vs NumPy)")
print("=" * 70)

# --- 3a. Gaussian kernel computation ---
print("\n--- 3a. Gaussian kernel (100K values) ---")

n = 100000
dists = np.random.rand(n)
sigma_i = np.random.rand(n) + 0.1
sigma_j = np.random.rand(n) + 0.1

def numpy_gaussian(d, si, sj):
    return np.exp(-d ** 2 / (si * sj))

def arrow_gaussian(d, si, sj):
    d_a = pa.array(d)
    si_a = pa.array(si)
    sj_a = pa.array(sj)
    import pyarrow.compute as pc
    neg_sq = pc.negate(pc.multiply(d_a, d_a))
    denom = pc.multiply(si_a, sj_a)
    ratio = pc.divide(neg_sq, denom)
    # PyArrow doesn't have exp, need to go back to numpy
    return np.exp(ratio.to_numpy())

_, t_np, m_np = timed(numpy_gaussian, dists, sigma_i, sigma_j)
_, t_pa, m_pa = timed(arrow_gaussian, dists, sigma_i, sigma_j)

print(f"  numpy:    {t_np:.4f}s  mem={m_np/1024:.0f}KB")
print(f"  pyarrow:  {t_pa:.4f}s  mem={m_pa/1024:.0f}KB")

# --- 3b. Unique + dedup pattern ---
print("\n--- 3b. Unique key dedup (combn_dedup, 500K pairs) ---")

n_pairs = 500000
pairs_i = np.random.randint(0, 10000, n_pairs)
pairs_j = np.random.randint(0, 10000, n_pairs)

def numpy_dedup(pi, pj):
    keys = np.where(pi <= pj, pi * 10000 + pj, pj * 10000 + pi)
    _, idx = np.unique(keys, return_index=True)
    return idx

def pandas_dedup(pi, pj):
    df = pd.DataFrame({"i": pi, "j": pj})
    df["_key"] = np.where(pi <= pj, pi * 10000 + pj, pj * 10000 + pi)
    return df.drop_duplicates("_key").index.values

def arrow_dedup(pi, pj):
    keys = np.where(pi <= pj, pi * 10000 + pj, pj * 10000 + pi)
    t = pa.table({"key": keys, "idx": np.arange(len(keys))})
    # PyArrow doesn't have native drop_duplicates; use pandas
    return t.to_pandas().drop_duplicates("key")["idx"].values

_, t_np, m_np = timed(numpy_dedup, pairs_i, pairs_j)
_, t_pd, m_pd = timed(pandas_dedup, pairs_i, pairs_j)
_, t_pa, m_pa = timed(arrow_dedup, pairs_i, pairs_j)

print(f"  numpy:    {t_np:.4f}s  mem={m_np/1024:.0f}KB")
print(f"  pandas:   {t_pd:.4f}s  mem={m_pd/1024:.0f}KB")
print(f"  pyarrow:  {t_pa:.4f}s  mem={m_pa/1024:.0f}KB")

# --- 3c. Sparse matrix construction from long format ---
print("\n--- 3c. Sparse matrix from long format (100K entries, 5K x 5K) ---")

n_entries = 100000
n_dim = 5000
rows_arr = np.random.randint(0, n_dim, n_entries)
cols_arr = np.random.randint(0, n_dim, n_entries)
vals_arr = np.random.rand(n_entries)

def scipy_sparse_direct(r, c, v, n):
    return sp.csr_matrix((v, (r, c)), shape=(n, n))

def via_pandas_long2sparse(r, c, v, n):
    df = pd.DataFrame({"row": r, "col": c, "val": v})
    rows = df["row"].values
    cols = df["col"].values
    vals = df["val"].values
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

def via_arrow_long2sparse(r, c, v, n):
    t = pa.table({"row": r, "col": c, "val": v})
    rows = t.column("row").to_numpy()
    cols = t.column("col").to_numpy()
    vals = t.column("val").to_numpy()
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

_, t_sp, m_sp = timed(scipy_sparse_direct, rows_arr, cols_arr, vals_arr, n_dim)
_, t_pd, m_pd = timed(via_pandas_long2sparse, rows_arr, cols_arr, vals_arr, n_dim)
_, t_pa, m_pa = timed(via_arrow_long2sparse, rows_arr, cols_arr, vals_arr, n_dim)

print(f"  scipy direct:   {t_sp:.4f}s  mem={m_sp/1024:.0f}KB")
print(f"  via pandas:     {t_pd:.4f}s  mem={m_pd/1024:.0f}KB")
print(f"  via pyarrow:    {t_pa:.4f}s  mem={m_pa/1024:.0f}KB")


# =====================================================================
# AREA 4: End-to-end function tests with PyArrow backend
# =====================================================================
print("\n" + "=" * 70)
print("AREA 4: End-to-End Function Benchmarks (current vs PyArrow)")
print("=" * 70)

# Test knn_flat with PyArrow-backed output
print("\n--- 4a. knn_flat (N=3000, k=15) ---")
from clonotrace.auxiliary import knn_flat

emb = np.random.randn(3000, 20)

def knn_flat_default():
    return knn_flat(emb, k=15, input="matrix", symmetric=True)

def knn_flat_to_arrow():
    df = knn_flat(emb, k=15, input="matrix", symmetric=True)
    return df.convert_dtypes(dtype_backend="pyarrow")

_, t_default, m_default = timed(knn_flat_default, repeats=3)
_, t_arrow, m_arrow = timed(knn_flat_to_arrow, repeats=3)

print(f"  pandas default:  {t_default:.4f}s  mem={m_default/1024:.0f}KB")
print(f"  convert to arrow: {t_arrow:.4f}s  mem={m_arrow/1024:.0f}KB")

# Test long_symmetry
print("\n--- 4b. long_symmetry (90K edges) ---")
from clonotrace.auxiliary import long_symmetry

test_df = pd.DataFrame({
    "node1": np.random.randint(1, 3001, 90000),
    "node2": np.random.randint(1, 3001, 90000),
    "dist": np.random.rand(90000),
})

def symmetry_default():
    return long_symmetry(test_df, "node1", "node2")

def symmetry_arrow():
    df_a = test_df.convert_dtypes(dtype_backend="pyarrow")
    return long_symmetry(df_a, "node1", "node2")

_, t_default, m_default = timed(symmetry_default, repeats=3)
_, t_arrow, m_arrow = timed(symmetry_arrow, repeats=3)

print(f"  pandas default:  {t_default:.4f}s  mem={m_default/1024:.0f}KB")
print(f"  pyarrow backend: {t_arrow:.4f}s  mem={m_arrow/1024:.0f}KB")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Cleanup
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

print("Done!")
