"""
Verification & Benchmarking Script for Clonotrace Optimizations
================================================================
Compares old vs new implementations for:
  - Numerical correctness (outputs match within tolerance)
  - Speed (wall-clock timing)
  - Memory usage (peak RSS via tracemalloc)
"""

import os
import sys
import json
import time
import tracemalloc
import numpy as np
import pandas as pd
import scipy.sparse as sp

np.random.seed(42)

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timed(func, *args, repeats=3, **kwargs):
    """Run func, return (result, median_time_seconds, peak_memory_bytes)."""
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


def save_results(results, label):
    """Append results dict to JSON file under label."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            data = json.load(f)
    else:
        data = {}
    data[label] = results
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_embedding(n=2000, d=20):
    return np.random.randn(n, d)

def make_distance_matrix(n=200):
    x = np.random.randn(n, 10)
    d = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(axis=2))
    return d

def make_sparse_adj(n=500, k=10):
    from sklearn.neighbors import NearestNeighbors
    x = np.random.randn(n, 10)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x)
    dists, indices = nn.kneighbors(x)
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    vals = np.exp(-dists.ravel())
    adj = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)
    return adj

def make_transition_matrix(n=200):
    adj = make_sparse_adj(n, k=10)
    row_sums = np.array(adj.sum(axis=1)).ravel()
    row_sums = np.maximum(row_sums, 1e-12)
    D_inv = sp.diags(1.0 / row_sums)
    return D_inv.dot(adj)

def make_labels(n=500, n_classes=5, frac_labeled=0.3):
    labels = np.full(n, np.nan)
    labeled = np.random.choice(n, size=int(n * frac_labeled), replace=False)
    labels[labeled] = np.random.randint(1, n_classes + 1, size=len(labeled))
    return labels

def make_clone_prob(n_cells=200, n_clones=20):
    prob = np.zeros((n_cells, n_clones))
    for i in range(n_cells):
        clone = np.random.randint(0, n_clones)
        prob[i, clone] = np.random.rand()
    return prob

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_dis_points_to_edges():
    from clonotrace.auxiliary import dis_points_to_edges
    points = np.random.randn(5000, 3)
    edges = [np.random.randn(2, 3) for _ in range(50)]
    result, t, mem = timed(dis_points_to_edges, points, edges)
    return {
        "time": t,
        "memory_bytes": mem,
        "output_shape": list(result["dis"].shape),
        "output_checksum": float(np.sum(result["dis"])),
    }

def bench_mat_sparsify():
    from clonotrace.auxiliary import mat_sparsify
    mat = np.random.rand(300, 300)
    result, t, mem = timed(mat_sparsify, mat, 0.9, 0.9)
    return {
        "time": t,
        "memory_bytes": mem,
        "nonzero_frac": float(np.count_nonzero(result) / result.size),
        "output_checksum": float(np.sum(result)),
    }

def bench_embedding2knn():
    from clonotrace.auxiliary import embedding2knn
    emb = make_embedding(n=3000, d=20)
    result, t, mem = timed(embedding2knn, emb, k=15, mode="connectivity", repeats=2)
    return {
        "time": t,
        "memory_bytes": mem,
        "nnz": int(result.nnz),
        "output_checksum": float(result.sum()),
    }

def bench_snn_from_dist():
    from clonotrace.cluster import _snn_from_dist
    dismat = make_distance_matrix(n=500)
    result, t, mem = timed(_snn_from_dist, dismat, k=15)
    starts, ends, weights, n = result
    return {
        "time": t,
        "memory_bytes": mem,
        "n_edges": len(starts),
        "output_checksum": float(np.sum(weights)),
    }

def bench_link2cluster():
    from clonotrace.auxiliary import link2cluster
    n = 500
    # Create a few connected components
    links_i = list(range(0, 200)) + list(range(250, 400))
    links_j = list(range(1, 201)) + list(range(251, 401))
    link = pd.DataFrame({"i": links_i, "j": links_j})
    nodes = list(range(n))
    result, t, mem = timed(link2cluster, link, nodes)
    n_clusters = len(np.unique(result))
    return {
        "time": t,
        "memory_bytes": mem,
        "n_clusters": n_clusters,
        "output_checksum": float(np.sum(result)),
    }

def bench_nearest_knn():
    from clonotrace.auxiliary import nearest_knn
    dis = make_distance_matrix(n=300)
    result, t, mem = timed(nearest_knn, dis, k=10, top=20)
    return {
        "time": t,
        "memory_bytes": mem,
        "n_rows": len(result),
    }

def bench_sync_sparse_rows():
    from clonotrace.auxiliary import sync_sparse_rows
    n, m = 1000, 500
    data = np.random.rand(100)
    rows_idx = np.random.randint(0, n, 100)
    cols_idx = np.random.randint(0, m, 100)
    mat = sp.csr_matrix((data, (rows_idx, cols_idx)), shape=(n, m))
    mat.rownames = list(range(n))
    # Reorder and add some missing
    new_names = list(range(200, n + 200))
    result, t, mem = timed(sync_sparse_rows, mat, new_names)
    return {
        "time": t,
        "memory_bytes": mem,
        "is_sparse": sp.issparse(result),
        "nnz": int(result.nnz) if sp.issparse(result) else -1,
    }

def bench_DPT_T():
    from clonotrace.pseudotime import DPT_T
    T_mat = make_transition_matrix(n=200)
    result, t, mem = timed(DPT_T, T_mat, start=0, repeats=1)
    return {
        "time": t,
        "memory_bytes": mem,
        "output_shape": list(result.shape),
        "output_checksum": float(np.sum(result)),
    }

def bench_acct():
    from clonotrace.pseudotime import acct
    T_mat = make_transition_matrix(n=200)
    result, t, mem = timed(acct, T_mat, repeats=1)
    return {
        "time": t,
        "memory_bytes": mem,
        "output_shape": list(result.shape),
        "output_checksum": float(np.sum(result)),
    }

def bench_label_spreading():
    from clonotrace.label_propagation import label_spreading
    adj = make_sparse_adj(n=500, k=10)
    labels = make_labels(n=500, n_classes=5)
    result, t, mem = timed(label_spreading, adj, labels, alpha=0.9, max_iter=50,
                           verbose=False)
    return {
        "time": t,
        "memory_bytes": mem,
        "output_shape": list(result.shape),
        "output_checksum": float(np.sum(result)),
    }

def bench_clone_partition():
    from clonotrace.clone_dis import clone_partition
    clone_prob = make_clone_prob(n_cells=500, n_clones=50)
    result, t, mem = timed(clone_partition, clone_prob, k=5)
    total = sum(len(v) for v in result.values())
    return {
        "time": t,
        "memory_bytes": mem,
        "n_groups": len(result),
        "total_clones": total,
    }

def bench_cluster_profile_enrich():
    from clonotrace.profile_deg import cluster_profile_enrich
    prob = np.random.rand(500, 4)
    prob = prob / prob.sum(axis=1, keepdims=True)
    labels = np.random.choice([0, 1, 2, 3], size=500)
    result, t, mem = timed(cluster_profile_enrich, prob, labels, permute_n=100)
    return {
        "time": t,
        "memory_bytes": mem,
        "pval_shape": list(result["pval"].shape),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = {
    "dis_points_to_edges": bench_dis_points_to_edges,
    "mat_sparsify": bench_mat_sparsify,
    "embedding2knn": bench_embedding2knn,
    "snn_from_dist": bench_snn_from_dist,
    "link2cluster": bench_link2cluster,
    "nearest_knn": bench_nearest_knn,
    "sync_sparse_rows": bench_sync_sparse_rows,
    "DPT_T": bench_DPT_T,
    "acct": bench_acct,
    "label_spreading": bench_label_spreading,
    "clone_partition": bench_clone_partition,
    "cluster_profile_enrich": bench_cluster_profile_enrich,
}

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    print(f"=== Running benchmarks [{label}] ===\n")
    results = {}
    for name, func in ALL_BENCHMARKS.items():
        try:
            print(f"  {name} ... ", end="", flush=True)
            r = func()
            results[name] = r
            print(f"{r['time']:.4f}s  mem={r['memory_bytes']/1024:.0f}KB")
        except Exception as e:
            print(f"FAILED: {e}")
            results[name] = {"error": str(e)}
    save_results(results, label)
    print(f"\nResults saved to {RESULTS_FILE}")
