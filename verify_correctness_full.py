"""
Comprehensive correctness verification of optimized label_spreading.

Compares old (float64, sequential, per-sample iteration) vs new
(float32, batched, threaded) on the full 34K-cell hematopoiesis dataset.

Tests:
1. Single label_spreading call: old vs new (should be identical)
2. Bootstrap deviance: old vs new (checks impact of float32 + relaxed tol)
3. Downstream filtering: does deviance<0.3 select similar cells?
4. Cell-clone probability matrix: structural similarity
"""
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd

# ================================================================
# Old implementation (exact copy of original code before optimization)
# ================================================================
def _label_spreading_old(adj, labels, label_n=None, alpha=0.9,
                          max_iter=100, tol=1e-3, epsilon=0):
    """Original iterative implementation (float64, tol=1e-3)."""
    from pandas import factorize, isna
    labels = np.asarray(labels, dtype=object)
    valid_mask = np.array([not isna(v) for v in labels])
    if valid_mask.any():
        codes, _ = factorize(labels[valid_mask], sort=True)
        labels_num = np.full(len(labels), np.nan)
        labels_num[valid_mask] = codes + 1
        labels = labels_num
    else:
        labels = np.zeros(len(labels))

    N = len(labels)
    C = int(label_n) if label_n is not None else int(np.nanmax(labels))

    valid_labels = labels[valid_mask].astype(int)
    label_counts = np.bincount(valid_labels, minlength=C + 1)[1:]
    label_weights = np.where(
        label_counts > 0,
        np.log2(np.maximum(label_counts, 1)) / np.maximum(label_counts, 1),
        0.0,
    )
    Y = np.full((N, C), epsilon / C)
    labeled_idx = np.where(valid_mask)[0]
    Y[labeled_idx, :] = 0.0
    col_idx = valid_labels - 1
    Y[labeled_idx, col_idx] = label_weights[col_idx]

    adj = sp.csr_matrix(adj, dtype=float)
    degrees = np.array(adj.sum(axis=1)).ravel()
    degrees = np.maximum(degrees, 1e-6)
    D_inv = sp.diags(1.0 / degrees)
    P = D_inv.dot(adj)

    F = Y.copy()
    for it in range(1, max_iter + 1):
        F_new = alpha * P.dot(F) + (1 - alpha) * Y
        diff = float(np.max(np.abs(F_new - F)))
        F = F_new
        if diff < tol:
            break
    return F


def _bootstrap_old(adj, labels_raw, alpha=0.6, sample_rate=0.8, sample_n=48):
    """Original per-sample bootstrap (sequential, float64)."""
    from pandas import isna, factorize
    labels = np.asarray(labels_raw, dtype=object)
    valid_mask = np.array([not isna(v) for v in labels])
    if valid_mask.any():
        codes, _ = factorize(labels[valid_mask], sort=True)
        labels_num = np.full(len(labels), np.nan)
        labels_num[valid_mask] = codes + 1
        labels = labels_num
    else:
        labels = np.zeros(len(labels))

    refer = _label_spreading_old(adj, labels_raw, alpha=alpha)
    row_sums = np.maximum(refer.sum(axis=1, keepdims=True), 1e-12)
    refer_norm = refer / row_sums

    labeled_idx = np.where(valid_mask)[0]
    label_sample_count = max(1, round(len(labeled_idx) * sample_rate))
    label_n = int(np.nanmax(labels))
    full_label_flag = labels.copy()
    full_label_flag[np.isnan(full_label_flag)] = 0

    deviance_arr = []
    flag_arr = []
    for b in range(sample_n):
        sub_labels = np.full(len(labels), np.nan)
        sampled = np.random.choice(labeled_idx, size=label_sample_count, replace=False)
        sub_labels[sampled] = labels[sampled]

        prob_mat = _label_spreading_old(adj, sub_labels, label_n=label_n,
                                         alpha=alpha, epsilon=0)
        prob_mat = prob_mat + 1e-12
        prob_mat = prob_mat / prob_mat.sum(axis=1, keepdims=True)

        sample_flag = sub_labels.copy()
        sample_flag[np.isnan(sample_flag)] = 0
        same_flag = (~np.logical_xor(
            full_label_flag.astype(bool), sample_flag.astype(bool)
        )).astype(float)

        L1 = np.sum(np.abs(prob_mat - refer_norm), axis=1)
        deviance_arr.append(L1)
        flag_arr.append(same_flag)

    deviance_arr = np.column_stack(deviance_arr)
    flag_arr = np.column_stack(flag_arr)
    deviance_norm = (deviance_arr * flag_arr).sum(axis=1) / np.maximum(flag_arr.sum(axis=1), 1)
    return {"prob": refer_norm, "deviance": deviance_norm}


# ================================================================
# Load data
# ================================================================
print("Loading real dataset (34K cells, 802 clones)...")
pca = pd.read_csv("real_data/pca.csv", index_col=0).values
cell_meta = pd.read_csv("real_data/cell_meta.csv", index_col=0)

from clonotrace.auxiliary import embedding2knn, compute_transition

print("Building kNN graph (k=30)...")
cell_knn = embedding2knn(pca, k=30)
cell_knn = compute_transition(cell_knn)
N = pca.shape[0]

clone_size = cell_meta.groupby("clone").size().reset_index(name="count")
expanded = clone_size[clone_size["count"] >= 10]["clone"].values
cell_clone = cell_meta["clone"].copy()
cell_clone[~cell_clone.isin(expanded)] = np.nan
labels_raw = cell_clone.values
n_expanded = len(expanded)
print(f"  N={N}, expanded clones={n_expanded}")

# ================================================================
# TEST 1: Single label_spreading call (should be identical)
# ================================================================
print("\n" + "=" * 65)
print("TEST 1: Single label_spreading (full precision, no bootstrap)")
print("=" * 65)

from clonotrace.label_propagation import label_spreading

print("  Running old (float64, iterative)...")
t0 = time.time()
F_old = _label_spreading_old(cell_knn, labels_raw, alpha=0.6)
t_old = time.time() - t0

print("  Running new (refactored, same precision)...")
t0 = time.time()
F_new = label_spreading(cell_knn, labels_raw, alpha=0.6, verbose=False)
t_new = time.time() - t0

max_diff = np.max(np.abs(F_old - F_new))
rel_diff = max_diff / (np.max(np.abs(F_old)) + 1e-15)
print(f"  Old time: {t_old:.2f}s, New time: {t_new:.2f}s")
print(f"  Max absolute diff: {max_diff:.2e}")
print(f"  Max relative diff: {rel_diff:.2e}")

if max_diff < 1e-12:
    print("  RESULT: EXACT MATCH")
elif max_diff < 1e-6:
    print("  RESULT: MATCH (within float64 precision)")
else:
    print("  RESULT: DIFFERS — investigating...")

# ================================================================
# TEST 2: Full bootstrap — old vs new
# ================================================================
print("\n" + "=" * 65)
print("TEST 2: Full bootstrap (48 samples) — old vs new")
print("=" * 65)

from clonotrace.label_propagation import label_spreading_bootstrap

print("  Running OLD bootstrap (float64, sequential, per-sample)...")
np.random.seed(42)
t0 = time.time()
res_old = _bootstrap_old(cell_knn, labels_raw, alpha=0.6,
                          sample_rate=0.8, sample_n=48)
t_old_boot = time.time() - t0
print(f"  Time: {t_old_boot:.1f}s")

print("  Running NEW bootstrap (float32, batched, threaded)...")
np.random.seed(42)
t0 = time.time()
res_new = label_spreading_bootstrap(cell_knn, labels_raw, alpha=0.6,
                                     sample_rate=0.8, sample_n=48, n_jobs=-1)
t_new_boot = time.time() - t0
print(f"  Time: {t_new_boot:.1f}s")
print(f"  Speedup: {t_old_boot / t_new_boot:.1f}x")

# ================================================================
# TEST 3: Reference probability matrix comparison
# ================================================================
print("\n" + "=" * 65)
print("TEST 3: Reference probability matrix (refer_norm)")
print("=" * 65)

prob_max_diff = np.max(np.abs(res_old["prob"] - res_new["prob"]))
prob_rel_diff = prob_max_diff / (np.max(np.abs(res_old["prob"])) + 1e-15)
prob_corr = np.corrcoef(res_old["prob"].ravel(), res_new["prob"].ravel())[0, 1]
print(f"  Max absolute diff: {prob_max_diff:.2e}")
print(f"  Max relative diff: {prob_rel_diff:.2e}")
print(f"  Correlation: {prob_corr:.10f}")

if prob_max_diff < 1e-10:
    print("  RESULT: EXACT MATCH")
else:
    print(f"  RESULT: DIFFERS by {prob_max_diff:.2e}")

# ================================================================
# TEST 4: Deviance comparison
# ================================================================
print("\n" + "=" * 65)
print("TEST 4: Deviance values")
print("=" * 65)

dev_old = res_old["deviance"]
dev_new = res_new["deviance"]

dev_max_diff = np.max(np.abs(dev_old - dev_new))
dev_corr = np.corrcoef(dev_old, dev_new)[0, 1]

print(f"  Old range: [{dev_old.min():.4f}, {dev_old.max():.4f}], mean={dev_old.mean():.4f}")
print(f"  New range: [{dev_new.min():.4f}, {dev_new.max():.4f}], mean={dev_new.mean():.4f}")
print(f"  Max absolute diff: {dev_max_diff:.4f}")
print(f"  Pearson correlation: {dev_corr:.6f}")
print(f"  Spearman correlation: {pd.Series(dev_old).corr(pd.Series(dev_new), method='spearman'):.6f}")

# Percentile comparison
for p in [10, 25, 50, 75, 90]:
    print(f"  {p}th percentile: old={np.percentile(dev_old, p):.4f}, "
          f"new={np.percentile(dev_new, p):.4f}")

# ================================================================
# TEST 5: Downstream filtering — deviance < 0.3
# ================================================================
print("\n" + "=" * 65)
print("TEST 5: Downstream filtering (deviance < 0.3)")
print("=" * 65)

mask_old = dev_old < 0.3
mask_new = dev_new < 0.3
n_old = mask_old.sum()
n_new = mask_new.sum()
n_agree = (mask_old == mask_new).sum()
n_both = (mask_old & mask_new).sum()
jaccard = n_both / ((mask_old | mask_new).sum())

print(f"  Old: {n_old} cells pass filter")
print(f"  New: {n_new} cells pass filter")
print(f"  Agreement: {n_agree}/{N} ({100*n_agree/N:.1f}%)")
print(f"  Both pass: {n_both}")
print(f"  Jaccard similarity: {jaccard:.4f}")
print(f"  Old-only: {(mask_old & ~mask_new).sum()}, New-only: {(~mask_old & mask_new).sum()}")

# Also check at other thresholds
for thresh in [0.2, 0.25, 0.3, 0.35, 0.4]:
    m_o = (dev_old < thresh).sum()
    m_n = (dev_new < thresh).sum()
    both = ((dev_old < thresh) & (dev_new < thresh)).sum()
    union = ((dev_old < thresh) | (dev_new < thresh)).sum()
    j = both / union if union > 0 else 1.0
    print(f"  thresh={thresh}: old={m_o}, new={m_n}, Jaccard={j:.4f}")

# ================================================================
# TEST 6: Cell-clone probability structure
# ================================================================
print("\n" + "=" * 65)
print("TEST 6: Cell-clone probability structure")
print("=" * 65)

# For cells passing filter, check top-clone assignment
prob_old = res_old["prob"]
prob_new = res_new["prob"]

# Top clone per cell
top_old = np.argmax(prob_old, axis=1)
top_new = np.argmax(prob_new, axis=1)

# Among filtered cells
filtered_cells = mask_old & mask_new  # cells passing in both
n_filtered = filtered_cells.sum()
top_agree = (top_old[filtered_cells] == top_new[filtered_cells]).sum()
print(f"  Cells passing both filters: {n_filtered}")
print(f"  Top-clone agreement: {top_agree}/{n_filtered} ({100*top_agree/n_filtered:.2f}%)")

# Top-2 clones
top2_old = np.argsort(-prob_old, axis=1)[:, :2]
top2_new = np.argsort(-prob_new, axis=1)[:, :2]
top2_agree = sum(
    set(top2_old[i].tolist()) == set(top2_new[i].tolist())
    for i in np.where(filtered_cells)[0]
)
print(f"  Top-2 clone agreement: {top2_agree}/{n_filtered} ({100*top2_agree/n_filtered:.2f}%)")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Single call: {'EXACT MATCH' if prob_max_diff < 1e-10 else f'diff={prob_max_diff:.2e}'}")
print(f"  Bootstrap speedup: {t_old_boot:.1f}s → {t_new_boot:.1f}s ({t_old_boot/t_new_boot:.1f}x)")
print(f"  Deviance correlation: {dev_corr:.4f}")
print(f"  Filter agreement (dev<0.3): {100*n_agree/N:.1f}%")
print(f"  Top-clone agreement (filtered): {100*top_agree/n_filtered:.2f}%")

all_good = (
    prob_max_diff < 1e-6 and
    dev_corr > 0.95 and
    n_agree / N > 0.90 and
    top_agree / n_filtered > 0.99
)
print(f"\n  OVERALL: {'PASS — optimizations preserve equivalence' if all_good else 'NEEDS REVIEW'}")
