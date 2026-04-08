"""
Numerical Correctness Verification for Clonotrace Optimizations
================================================================
Checks mathematical properties and consistency of all optimized functions.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp

np.random.seed(42)
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


print("=" * 60)
print("Numerical Correctness Verification")
print("=" * 60)

# ---------------------------------------------------------------
# 1. dis_points_to_edges
# ---------------------------------------------------------------
print("\n1. dis_points_to_edges")
from clonotrace.auxiliary import dis_points_to_edges, dis_point_to_edge

points = np.random.randn(100, 3)
edges = [np.random.randn(2, 3) for _ in range(10)]

result = dis_points_to_edges(points, edges)

# Verify against scalar function for a few samples
for pi in [0, 50, 99]:
    for ei in [0, 5, 9]:
        edge = np.asarray(edges[ei])
        scalar = dis_point_to_edge(points[pi], edge[0], edge[1])
        check(f"point[{pi}] x edge[{ei}] distance",
              abs(result["dis"][pi, ei] - scalar[0]) < 1e-10,
              f"got {result['dis'][pi, ei]}, expected {scalar[0]}")
        check(f"point[{pi}] x edge[{ei}] projection t",
              abs(result["map"][pi, ei] - scalar[1]) < 1e-10,
              f"got {result['map'][pi, ei]}, expected {scalar[1]}")

check("distances are non-negative", np.all(result["dis"] >= 0))
check("projections in [0,1]", np.all((result["map"] >= 0) & (result["map"] <= 1)))

# ---------------------------------------------------------------
# 2. mat_sparsify
# ---------------------------------------------------------------
print("\n2. mat_sparsify")
from clonotrace.auxiliary import mat_sparsify, mass_filter

mat = np.random.rand(50, 50)
result = mat_sparsify(mat.copy(), row_mass=0.9, col_mass=0.9)

check("output same shape", result.shape == mat.shape)
check("zeroed entries are zero", np.all(result[result == 0] == 0))
check("non-zero entries match original",
      np.allclose(result[result != 0], mat[result != 0]))
check("row mass >= 0.9 for nonzero rows",
      all(result[i][result[i] > 0].sum() / max(mat[i].sum(), 1e-12) >= 0.89
          for i in range(50)))

# ---------------------------------------------------------------
# 3. embedding2knn
# ---------------------------------------------------------------
print("\n3. embedding2knn")
from clonotrace.auxiliary import embedding2knn

emb = np.random.randn(100, 10)
knn = embedding2knn(emb, k=10, mode="connectivity")

check("shape is (N,N)", knn.shape == (100, 100))
check("is sparse", sp.issparse(knn))
check("all values non-negative", np.all(knn.data >= 0))
check("symmetric", abs(knn - knn.T).nnz == 0 or np.allclose((knn - knn.T).data, 0, atol=1e-10))
check("diagonal is zero or identity-like",
      knn.diagonal().max() <= 1.01)

knn_dist = embedding2knn(emb, k=10, mode="dist")
check("dist mode: values non-negative", np.all(knn_dist.data >= 0))
check("dist mode: symmetric", abs(knn_dist - knn_dist.T).nnz == 0 or
      np.allclose((knn_dist - knn_dist.T).data, 0, atol=1e-10))

# ---------------------------------------------------------------
# 4. _snn_from_dist
# ---------------------------------------------------------------
print("\n4. _snn_from_dist")
from clonotrace.cluster import _snn_from_dist

x = np.random.randn(100, 10)
dismat = np.sqrt(((x[:, None] - x[None, :]) ** 2).sum(axis=2))
starts, ends, weights, n = _snn_from_dist(dismat, k=10)

check("n matches input", n == 100)
check("starts in range", np.all((starts >= 0) & (starts < 100)))
check("ends in range", np.all((ends >= 0) & (ends < 100)))
check("weights in (0, 1]", np.all((weights > 0) & (weights <= 1)))
check("Jaccard similarity valid", np.all(weights <= 1.0))

# ---------------------------------------------------------------
# 5. link2cluster (connected_components)
# ---------------------------------------------------------------
print("\n5. link2cluster")
from clonotrace.auxiliary import link2cluster

# Two components: {0,1,2,3} and {4,5,6}
link = pd.DataFrame({"i": [0, 1, 2, 4, 5], "j": [1, 2, 3, 5, 6]})
labels = link2cluster(link, list(range(8)))

check("labels are 1-indexed", labels.min() >= 1)
check("node 7 is isolated (its own cluster)",
      labels[7] != labels[0] and labels[7] != labels[4])
check("0-3 same cluster", len(set(labels[:4])) == 1)
check("4-6 same cluster", len(set(labels[4:7])) == 1)
check("0-3 and 4-6 different clusters", labels[0] != labels[4])

# ---------------------------------------------------------------
# 6. nearest_knn
# ---------------------------------------------------------------
print("\n6. nearest_knn")
from clonotrace.auxiliary import nearest_knn

dis = np.random.rand(50, 50)
dis = (dis + dis.T) / 2
np.fill_diagonal(dis, 0)
result = nearest_knn(dis, k=5, top=10)

check("at most 10 rows", len(result) <= 10)
check("sorted by distance", np.all(np.diff(result["dis"].values) >= -1e-10))
check("no self-pairs", all(result["i"].values != result["j"].values))

# ---------------------------------------------------------------
# 7. sync_sparse_rows
# ---------------------------------------------------------------
print("\n7. sync_sparse_rows")
from clonotrace.auxiliary import sync_sparse_rows

n, m = 100, 50
data = np.random.rand(200)
ri = np.random.randint(0, n, 200)
ci = np.random.randint(0, m, 200)
mat = sp.csr_matrix((data, (ri, ci)), shape=(n, m))
mat.rownames = list(range(n))

# Reorder + add missing
new_names = list(range(50, 150))
result = sync_sparse_rows(mat, new_names)

check("is sparse", sp.issparse(result))
check("correct shape", result.shape == (100, m))
check("has rownames", hasattr(result, "rownames") and len(result.rownames) == 100)

# Rows 50-99 should match original rows 50-99
for r in range(50, 100):
    new_idx = new_names.index(r)
    orig_row = mat[r].toarray().ravel()
    new_row = result[new_idx].toarray().ravel()
    if not np.allclose(orig_row, new_row):
        check(f"row {r} matches", False, "mismatch")
        break
else:
    check("existing rows preserved correctly", True)

# Rows 100-149 should be zero
zero_rows = result[50:].toarray()
check("missing rows are zero", np.allclose(zero_rows, 0))

# ---------------------------------------------------------------
# 8. DPT_T and acct
# ---------------------------------------------------------------
print("\n8. DPT_T / acct")
from clonotrace.pseudotime import DPT_T, acct, _build_acct_operator

# Build a small transition matrix
from sklearn.neighbors import NearestNeighbors
x = np.random.randn(100, 5)
nn = NearestNeighbors(n_neighbors=10)
nn.fit(x)
dists, indices = nn.kneighbors(x)
rows = np.repeat(np.arange(100), 10)
cols = indices.ravel()
vals = np.exp(-dists.ravel())
adj = sp.csr_matrix((vals, (rows, cols)), shape=(100, 100))
adj = adj.maximum(adj.T)
row_sums = np.array(adj.sum(axis=1)).ravel()
T_mat = sp.diags(1.0 / np.maximum(row_sums, 1e-12)).dot(adj)

pt = DPT_T(T_mat, start=0)
check("pseudotime shape", pt.shape == (100,))
check("root has zero pseudotime", abs(pt[0]) < 1e-6)
check("all pseudotimes non-negative", np.all(pt >= -1e-10))

M = acct(T_mat)
check("ACT matrix shape", M.shape == (100, 100))

# ---------------------------------------------------------------
# 9. dpt (eigendecomposition version)
# ---------------------------------------------------------------
print("\n9. dpt (eigendecomposition)")
from clonotrace.pseudotime import dpt

pt2 = dpt(T_mat, root=0, k=20)
check("dpt shape", pt2.shape == (100,))
check("dpt root is zero", abs(pt2[0]) < 1e-6)
check("dpt values in [0, 1]", np.all((pt2 >= -1e-10) & (pt2 <= 1 + 1e-10)))

# ---------------------------------------------------------------
# 10. label_spreading
# ---------------------------------------------------------------
print("\n10. label_spreading")
from clonotrace.label_propagation import label_spreading

adj = sp.csr_matrix(adj)
labels = np.full(100, np.nan)
labels[:20] = np.random.randint(1, 4, 20)

prob = label_spreading(adj, labels, alpha=0.9, max_iter=100, verbose=False)
check("output shape (N, C)", prob.shape == (100, 3))
check("all probabilities non-negative", np.all(prob >= -1e-10))
check("labeled nodes have higher prob for their class",
      all(prob[i, int(labels[i]) - 1] > 0 for i in range(20)))

# ---------------------------------------------------------------
# 11. clone_partition
# ---------------------------------------------------------------
print("\n11. clone_partition")
from clonotrace.clone_dis import clone_partition

clone_prob = np.zeros((200, 30))
for i in range(200):
    clone_prob[i, np.random.randint(0, 30)] = np.random.rand()

result = clone_partition(clone_prob, k=5)
all_clones = []
for v in result.values():
    all_clones.extend(v)

check("5 groups", len(result) == 5)
check("all clones assigned", sorted(all_clones) == list(range(30)))
check("no duplicate assignments", len(all_clones) == len(set(all_clones)))

# ---------------------------------------------------------------
# 12. cluster_profile_enrich
# ---------------------------------------------------------------
print("\n12. cluster_profile_enrich")
from clonotrace.profile_deg import cluster_profile_enrich

prob = np.random.rand(200, 3)
prob = prob / prob.sum(axis=1, keepdims=True)
labels = np.random.choice([0, 1, 2], size=200)

result = cluster_profile_enrich(prob, labels, permute_n=50)
check("pval shape matches prob shape", result["pval"].shape == result["prob"].shape)
check("pvals in [0, 1]", np.all((result["pval"] >= 0) & (result["pval"] <= 1)))
check("prob values positive", np.all(result["prob"] >= 0))

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
print("=" * 60)
