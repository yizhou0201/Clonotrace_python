#!/usr/bin/env python3
"""
06_test_extra_python.py - Additional function tests with synthetic / real data
Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.manifold import MDS

sys.path.insert(0, "/Users/yizhouw/Desktop/packages/Clonotrace_python")
import clonotrace as ct
os.chdir("/Users/yizhouw/Desktop/packages/Clonotrace_python")

np.random.seed(42)

print("=== Loading data ===")
pca         = pd.read_csv("tests/data/pca.csv", index_col=0).values.astype(float)
cell_names  = pd.read_csv("tests/data/cell_names.csv")["cell_name"].tolist()
clone_names = pd.read_csv("tests/data/clone_names.csv")["clone_name"].tolist()
triplets    = pd.read_csv("tests/data/cell_clone_binary_triplets.csv")

n_cells  = len(cell_names)
n_clones = len(clone_names)

# Reconstruct binary cell_clone_prob
rows = triplets["cell_idx"].values.astype(int)
cols = triplets["clone_idx"].values.astype(int)
vals = triplets["value"].values.astype(float)
cell_clone_prob = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_clones))

# Build cell kNN for auxiliary function tests
print("Building cell kNN (500 cells)...")
cell_knn = ct.embedding2knn(pca[:500], k=15, mode="connectivity")
T_mat_sub = ct.compute_transition(cell_knn)

# ─────────────────────────────────────────────────────────────────────────────
# Group A: Format conversion functions
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Group A: Format Conversions ===")

long_df = pd.DataFrame({
    "from":  ["A","A","B"],
    "to":    ["B","C","C"],
    "value": [1.0, 2.0, 3.0],
})

# A1: long2square
sq = ct.long2square(long_df, row_names_from="from", col_names_from="to",
                    values_from="value", symmetric=True, na_fill=0.0)
nodes = sorted(set(long_df["from"].tolist() + long_df["to"].tolist()))
sq_df = pd.DataFrame(sq, index=nodes, columns=nodes)
sq_df.to_csv("tests/outputs_python/long2square_out.csv")
print(f"long2square: {sq.shape}")

# A2: long2sparse
sp_mat = ct.long2sparse(long_df, row_names_from="from", col_names_from="to",
                         values_from="value", symmetric=True)
sp_coo = sp_mat.tocoo()
sp_rows = sp_coo.row + 1  # 1-indexed to match R
sp_cols = sp_coo.col + 1
sp_df = pd.DataFrame({"row": sp_rows, "col": sp_cols, "value": sp_coo.data})
sp_df.to_csv("tests/outputs_python/long2sparse_out.csv", index=False)
print(f"long2sparse nnz: {len(sp_df)}")

# A3: long2wide
wide_df = ct.long2wide(long_df, row_names_from="from", col_names_from="to",
                        values_from="value", symmetric=True)
wide_df.to_csv("tests/outputs_python/long2wide_out.csv")
print(f"long2wide: {wide_df.shape}")

# A4: wide2long
wide_mat = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
wl = ct.wide2long(wide_mat)
wl.to_csv("tests/outputs_python/wide2long_out.csv", index=False)
print(f"wide2long: {wl.shape}")

# A5: long_symmetry
sym_df = ct.long_symmetry(long_df, row_names_from="from", col_names_from="to")
sym_df.to_csv("tests/outputs_python/long_symmetry_out.csv", index=False)
print(f"long_symmetry: {sym_df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Group B: Core Algorithms
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Group B: Core Algorithms ===")

# B1: compute_transition
T_sub = T_mat_sub[:200, :].tocoo()
T_df = pd.DataFrame({"row": T_sub.row + 1, "col": T_sub.col + 1, "value": T_sub.data})
T_df.to_csv("tests/outputs_python/compute_transition_out.csv", index=False)
print(f"compute_transition T[:200,:] nnz: {len(T_df)}")

# B2: mat_sparsify
np.random.seed(42)
test_mat = np.abs(np.random.randn(20, 20))
test_mat = test_mat / test_mat.sum(axis=1, keepdims=True)
sparse_mat = ct.mat_sparsify(test_mat, row_mass=0.9, col_mass=0.9)
pd.DataFrame(sparse_mat).to_csv("tests/outputs_python/mat_sparsify_out.csv")
nz_frac = np.sum(sparse_mat != 0) / sparse_mat.size
print(f"mat_sparsify: non-zero fraction = {nz_frac:.3f}")

# B3: dis2connec_sparse
np.random.seed(42)
dis_dense_test = np.random.uniform(0.1, 2.0, (15, 15))
dis_sparse_test = sp.csr_matrix(dis_dense_test)
dis_sparse_test.setdiag(0)
dis_sparse_test.eliminate_zeros()
connec = ct.dis2connec_sparse(dis_sparse_test)
coo = connec.tocoo()
coo_df = pd.DataFrame({"row": coo.row + 1, "col": coo.col + 1, "value": coo.data})
coo_df.to_csv("tests/outputs_python/dis2connec_out.csv", index=False)
print(f"dis2connec nnz: {len(coo_df)}")

# B4: dismat_mst
np.random.seed(42)
dis_dense = np.random.uniform(0.5, 3.0, (10, 10))
dis_dense = (dis_dense + dis_dense.T) / 2
np.fill_diagonal(dis_dense, 0)
mst_edges = ct.dismat_mst(dis_dense)
mst_edges.to_csv("tests/outputs_python/dismat_mst_out.csv", index=False)
print(f"MST edges: {len(mst_edges)} (should be n-1 = {len(dis_dense)-1})")

# B5: dist2knn
knn_from_dist = ct.dist2knn(dis_dense, k=3)
knn_coo = knn_from_dist.tocoo()
knn_df = pd.DataFrame({"row": knn_coo.row + 1, "col": knn_coo.col + 1, "value": knn_coo.data})
knn_df.to_csv("tests/outputs_python/dist2knn_out.csv", index=False)
print(f"dist2knn nnz: {len(knn_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# Group C: Co-embedding
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Group C: Co-embedding ===")

n_sub_cells  = 200
n_sub_clones = 50
sub_pca  = pca[:n_sub_cells]
sub_prob = cell_clone_prob[:n_sub_cells, :n_sub_clones]

# Build clone embedding from co-occurrence (matches R approach)
sub_prob_dense = sub_prob.toarray()
clone_overlap = sub_prob_dense.T @ sub_prob_dense
max_val = clone_overlap.max()
dist_mat = 1 - clone_overlap / (max_val + 1e-6) + 1e-6
np.fill_diagonal(dist_mat, 0)

np.random.seed(42)
mds_clone = MDS(n_components=10, dissimilarity="precomputed",
                metric=False, random_state=42, n_init=1, max_iter=300)
clone_emb = mds_clone.fit_transform(dist_mat)
clone_emb_df = pd.DataFrame(clone_emb, columns=[f"dim{i+1}" for i in range(10)])

coembed_dis = ct.cell_clone_coembed(
    cell_embedding=sub_pca,
    clone_embedding=clone_emb,
    cell_clone_prob=sub_prob_dense,
    cell_k=15, clone_k=10
)

# Save as dense 200x10 block (distances to first 10 cells) for Pearson comparison
co_dense = coembed_dis[:200, :10].toarray()
pd.DataFrame(co_dense).to_csv("tests/outputs_python/coembed_out.csv")
print(f"coembed_out saved: {co_dense.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# Group D: Visualization embeddings (first 200 cells for speed)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Group D: Visualization Embeddings ===")

knn_sub200 = ct.embedding2knn(pca[:200], k=10, mode="connectivity")

# D1: umap_from_knn
umap_coords, _ = ct.umap_from_knn(knn_sub200, n_neighbors=5, seed=1024)
umap_coords.to_csv("tests/outputs_python/umap_coords.csv")
print(f"umap_coords: {umap_coords.shape}")

# D2: mds_from_knn
mds_coords, _ = ct.mds_from_knn(knn_sub200, n_components=5)
mds_coords.to_csv("tests/outputs_python/mds_knn_coords.csv")
print(f"mds_knn_coords: {mds_coords.shape}")

print("\n=== All extra Python tests complete ===")
print("Outputs saved to tests/outputs_python/")
