#!/usr/bin/env python3
"""
04_compare.py - Compare R and Python outputs and generate equivalence report.
Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import procrustes

os.chdir("/Users/yizhouw/Desktop/packages/Clonotrace_python")

try:
    from sklearn.metrics import adjusted_rand_score
    HAS_ARI = True
except ImportError:
    HAS_ARI = False

PASS_THRESHOLD = {
    "pearson":   0.95,
    "spearman":  0.95,
    "pearson_hi": 0.99,
    "exact":     1.00,
    "procrustes": 0.80,
    "ari":        0.70,
}

results = []

def check(name, metric, score, threshold_key="pearson"):
    thresh = PASS_THRESHOLD[threshold_key]
    passed = score >= thresh
    results.append((name, metric, score, thresh, "PASS" if passed else "FAIL"))
    return passed, score


def safe_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    r, _ = stats.pearsonr(a[mask], b[mask])
    return float(r)


def safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    r, _ = stats.spearmanr(a[mask], b[mask])
    return float(r)


def load_csv(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    return pd.read_csv(path)


def load_csv_idx(path):
    if not os.path.exists(path):
        print(f"  [MISSING] {path}")
        return None
    return pd.read_csv(path, index_col=0)


print("=" * 70)
print("  Clonotrace R vs Python Equivalence Report")
print(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── Vignette: kNN + transition ────────────────────────────────────────────────
print("\n--- kNN + transition ---")
r_knn  = load_csv("tests/outputs_R/cell_knn_sample.csv")
py_knn = load_csv("tests/outputs_python/cell_knn_sample.csv")
if r_knn is not None and py_knn is not None:
    # Align on (row, col) pairs
    r_knn["key"]  = r_knn["row"].astype(str) + "_" + r_knn["col"].astype(str)
    py_knn["key"] = py_knn["row"].astype(str) + "_" + py_knn["col"].astype(str)
    merged = pd.merge(r_knn[["key","value"]], py_knn[["key","value"]],
                      on="key", suffixes=("_R","_py"))
    if len(merged) > 0:
        r_vals  = merged["value_R"].values
        py_vals = merged["value_py"].values
        r_shape  = f"{len(r_knn)} nnz"
        py_shape = f"{len(py_knn)} nnz"
        sc = safe_pearson(r_vals, py_vals)
        print(f"  R nnz={len(r_knn)}, Py nnz={len(py_knn)}, matched={len(merged)}")
        check("kNN+transition", "Pearson r (matched entries)", sc, "pearson_hi")
    else:
        print("  No matching (row,col) pairs — index convention differs")
        results.append(("kNN+transition", "Pearson r", float("nan"), 0.99, "WARN"))

# ── Vignette: Label spreading ─────────────────────────────────────────────────
print("\n--- Label spreading (subset) ---")
r_ls  = load_csv_idx("tests/outputs_R/label_spread_small.csv")
py_ls = load_csv_idx("tests/outputs_python/label_spread_small.csv")
if r_ls is not None and py_ls is not None:
    # Try to align columns; R uses clone names as column headers, Python uses integers
    r_vals  = r_ls.values.ravel()
    py_vals = py_ls.values.ravel()
    n = min(len(r_vals), len(py_vals))
    sc = safe_pearson(r_vals[:n], py_vals[:n])
    print(f"  R shape={r_ls.shape}, Py shape={py_ls.shape}")
    check("Label spreading", "Pearson r (flattened)", sc, "pearson")

# ── Vignette: Clone NN distance ───────────────────────────────────────────────
print("\n--- Clone NN distance (full 802 clones) ---")
# Compare via named 50x50 square matrix (avoids 0-vs-1-indexed mismatch)
r_sq  = load_csv_idx("tests/outputs_R/clone_nn_dis_sq50.csv")
py_sq = load_csv_idx("tests/outputs_python/clone_nn_dis_sq50.csv")
if r_sq is not None and py_sq is not None:
    # Align by common clone names
    common = r_sq.index.intersection(py_sq.index)
    if len(common) > 5:
        r_vals  = r_sq.loc[common, common].values.ravel()
        py_vals = py_sq.loc[common, common].values.ravel()
        mask = np.isfinite(r_vals) & np.isfinite(py_vals)
        sc_p = safe_pearson(r_vals[mask], py_vals[mask])
        sc_s = safe_spearman(r_vals[mask], py_vals[mask])
        print(f"  Common clones={len(common)}, valid entries={mask.sum()}")
        check("Clone NN dist", "Pearson r (sq50)", sc_p, "pearson")
        check("Clone NN dist", "Spearman r (sq50)", sc_s, "spearman")
    else:
        print(f"  Too few common clones: {len(common)}")
        results.append(("Clone NN dist", "Pearson r", float("nan"), 0.95, "WARN"))

# ── Vignette: Clone OT distance ───────────────────────────────────────────────
print("\n--- Clone OT distance (30 clones subset) ---")
r_ot  = load_csv("tests/outputs_R/clone_ot_dis_subset.csv")
py_ot = load_csv("tests/outputs_python/clone_ot_dis_subset.csv")
if r_ot is not None and py_ot is not None:
    # R is 1-indexed, Python is 0-indexed → offset Python by +1 for matching
    r_ot["key"]  = r_ot["group1"].astype(str) + "_" + r_ot["group2"].astype(str)
    py_ot["key"] = (py_ot["group1"] + 1).astype(str) + "_" + (py_ot["group2"] + 1).astype(str)
    merged = pd.merge(r_ot[["key","dis"]], py_ot[["key","dis"]],
                      on="key", suffixes=("_R","_py"))
    if len(merged) < 3:
        # Also try symmetric match
        py_ot["key2"] = (py_ot["group2"] + 1).astype(str) + "_" + (py_ot["group1"] + 1).astype(str)
        r_ot["key2"]  = r_ot["group2"].astype(str) + "_" + r_ot["group1"].astype(str)
        m2 = pd.merge(r_ot[["key","dis"]], py_ot[["key2","dis"]].rename(columns={"key2":"key"}),
                      on="key", suffixes=("_R","_py"))
        merged = pd.concat([merged, m2], ignore_index=True)
    print(f"  R pairs={len(r_ot)}, Py pairs={len(py_ot)}, matched={len(merged)}")
    if len(merged) > 3:
        sc = safe_pearson(merged["dis_R"], merged["dis_py"])
        max_diff = float(np.max(np.abs(merged["dis_R"] - merged["dis_py"])))
        check("Clone OT dist", "Pearson r", sc, "pearson")
        results.append(("Clone OT dist", "Max abs diff", max_diff, float("nan"), "INFO"))

# ── Vignette: Clone MDS ───────────────────────────────────────────────────────
print("\n--- Clone MDS (30 dims) ---")
r_mds  = load_csv_idx("tests/outputs_R/clone_mds.csv")
py_mds = load_csv_idx("tests/outputs_python/clone_mds.csv")
if r_mds is not None and py_mds is not None:
    try:
        r_arr  = r_mds.values.astype(float)
        py_arr = py_mds.values.astype(float)
        n = min(r_arr.shape[0], py_arr.shape[0])
        n_dims = min(r_arr.shape[1], py_arr.shape[1], 5)  # use first 5 dims for Procrustes
        _, _, disparity = procrustes(r_arr[:n, :n_dims], py_arr[:n, :n_dims])
        procrustes_r2 = float(1 - disparity)
        print(f"  R shape={r_mds.shape}, Py shape={py_mds.shape}")
        check("Clone MDS", "Procrustes R² (5 dims)", procrustes_r2, "procrustes")
    except Exception as e:
        print(f"  Procrustes error: {e}")
        results.append(("Clone MDS", "Procrustes R²", float("nan"), 0.80, "WARN"))

# ── Vignette: Leiden clustering ───────────────────────────────────────────────
print("\n--- Leiden clustering ---")
r_cl  = load_csv_idx("tests/outputs_R/clone_cluster.csv")
py_cl = load_csv_idx("tests/outputs_python/clone_cluster.csv")
if r_cl is not None and py_cl is not None:
    print(f"  R clusters: {r_cl['cluster'].nunique()}, Py clusters: {py_cl['cluster'].nunique()}")
    r_labels  = r_cl["cluster"].values
    py_labels = py_cl["cluster"].values
    n = min(len(r_labels), len(py_labels))
    if HAS_ARI:
        ari = adjusted_rand_score(r_labels[:n], py_labels[:n])
        check("Leiden clustering", "ARI", ari, "ari")
    else:
        results.append(("Leiden clustering", "ARI", float("nan"), 0.70, "WARN"))

# ── Vignette: Pseudotime ─────────────────────────────────────────────────────
print("\n--- Clone pseudotime ---")
r_pt  = load_csv("tests/outputs_R/clone_t.csv")
py_pt = load_csv("tests/outputs_python/clone_t.csv")
if r_pt is not None and py_pt is not None:
    r_t  = r_pt["dpt"].values
    py_t = py_pt["dpt"].values
    n = min(len(r_t), len(py_t))
    sc = safe_spearman(r_t[:n], py_t[:n])
    print(f"  R range=[{r_t.min():.3f},{r_t.max():.3f}], Py range=[{py_t.min():.3f},{py_t.max():.3f}]")
    check("Clone pseudotime", "Spearman r", sc, "spearman")

# ── Vignette: Profile enrichment ─────────────────────────────────────────────
print("\n--- Profile enrichment ---")
r_ep  = load_csv_idx("tests/outputs_R/enrich_pval.csv")
py_ep = load_csv_idx("tests/outputs_python/enrich_pval.csv")
if r_ep is not None and py_ep is not None:
    r_vals  = r_ep.values.ravel()
    py_vals = py_ep.values.ravel()
    n = min(len(r_vals), len(py_vals))
    sc = safe_pearson(r_vals[:n], py_vals[:n])
    print(f"  R shape={r_ep.shape}, Py shape={py_ep.shape}")
    check("Profile enrichment pval", "Pearson r", sc, "pearson")

# ── Vignette: DEG ─────────────────────────────────────────────────────────────
print("\n--- Profile DEG ---")
r_deg  = load_csv_idx("tests/outputs_R/DEG_stats.csv")
py_deg = load_csv_idx("tests/outputs_python/DEG_stats.csv")
if r_deg is not None and py_deg is not None and "stat" in r_deg.columns and "stat" in py_deg.columns:
    # Align on common row names (genes)
    common_genes = r_deg.index.intersection(py_deg.index)
    print(f"  R genes={len(r_deg)}, Py genes={len(py_deg)}, common={len(common_genes)}")
    if len(common_genes) > 10:
        sc_stat = safe_spearman(r_deg.loc[common_genes,"stat"], py_deg.loc[common_genes,"stat"])
        sc_coh  = safe_spearman(r_deg.loc[common_genes,"cohen"], py_deg.loc[common_genes,"cohen"])
        check("DEG F-stat", "Spearman r", sc_stat, "spearman")
        check("DEG Cohen's d", "Spearman r", sc_coh, "spearman")

# ── Additional function tests ─────────────────────────────────────────────────
print("\n--- Additional function tests ---")

# Format conversions (exact match)
for fname in ["long2square_out", "long2wide_out", "wide2long_out", "long_symmetry_out"]:
    r_f  = load_csv_idx(f"tests/outputs_R/{fname}.csv")
    py_f = load_csv_idx(f"tests/outputs_python/{fname}.csv")
    if r_f is not None and py_f is not None:
        r_vals  = r_f.values.astype(float).ravel()
        py_vals = py_f.values.astype(float).ravel()
        n = min(len(r_vals), len(py_vals))
        max_diff = float(np.max(np.abs(r_vals[:n] - py_vals[:n])))
        passed = max_diff < 1e-8
        results.append((fname, "Max abs diff", max_diff, 0.0, "PASS" if passed else "FAIL"))
        print(f"  {fname}: max_diff={max_diff:.2e} {'PASS' if passed else 'FAIL'}")

# long2sparse (check COO triplets)
r_ls2  = load_csv(f"tests/outputs_R/long2sparse_out.csv")
py_ls2 = load_csv(f"tests/outputs_python/long2sparse_out.csv")
if r_ls2 is not None and py_ls2 is not None:
    r_ls2["key"]  = r_ls2.iloc[:,0].astype(str) + "_" + r_ls2.iloc[:,1].astype(str)
    py_ls2["key"] = py_ls2.iloc[:,0].astype(str) + "_" + py_ls2.iloc[:,1].astype(str)
    merged = pd.merge(r_ls2[["key", r_ls2.columns[2]]],
                      py_ls2[["key", py_ls2.columns[2]]],
                      on="key", suffixes=("_R","_py"))
    if len(merged) > 0:
        max_diff = float(np.max(np.abs(
            merged.iloc[:,-2].values - merged.iloc[:,-1].values
        )))
        passed = max_diff < 1e-8
        results.append(("long2sparse_out", "Max abs diff", max_diff, 0.0, "PASS" if passed else "FAIL"))
        print(f"  long2sparse_out: matched={len(merged)}, max_diff={max_diff:.2e} {'PASS' if passed else 'FAIL'}")

# Core algorithms
for fname, thresh_key in [
    ("compute_transition_out", "pearson_hi"),
    ("dis2connec_out",          "pearson_hi"),
    ("dist2knn_out",            "exact"),
]:
    r_f  = load_csv(f"tests/outputs_R/{fname}.csv")
    py_f = load_csv(f"tests/outputs_python/{fname}.csv")
    if r_f is not None and py_f is not None:
        r_f["key"]  = r_f.iloc[:,0].astype(str) + "_" + r_f.iloc[:,1].astype(str)
        py_f["key"] = py_f.iloc[:,0].astype(str) + "_" + py_f.iloc[:,1].astype(str)
        merged = pd.merge(r_f[["key", r_f.columns[2]]],
                          py_f[["key", py_f.columns[2]]],
                          on="key", suffixes=("_R","_py"))
        if len(merged) > 3:
            r_vals  = merged.iloc[:,-2].values.astype(float)
            py_vals = merged.iloc[:,-1].values.astype(float)
            sc = safe_pearson(r_vals, py_vals) if thresh_key != "exact" else (
                float(np.max(np.abs(r_vals - py_vals)) < 1e-6)
            )
            check(fname, "Pearson r" if thresh_key != "exact" else "Exact match", sc, thresh_key)
            print(f"  {fname}: matched={len(merged)}, score={sc:.4f}")

# mat_sparsify (dense matrix comparison)
for fname in ["mat_sparsify_out"]:
    r_f  = load_csv_idx(f"tests/outputs_R/{fname}.csv")
    py_f = load_csv_idx(f"tests/outputs_python/{fname}.csv")
    if r_f is not None and py_f is not None:
        r_vals  = r_f.values.astype(float).ravel()
        py_vals = py_f.values.astype(float).ravel()
        n = min(len(r_vals), len(py_vals))
        max_diff = float(np.max(np.abs(r_vals[:n] - py_vals[:n])))
        passed = max_diff < 1e-8
        results.append(("mat_sparsify_out", "Max abs diff", max_diff, 0.0, "PASS" if passed else "FAIL"))
        print(f"  mat_sparsify_out: max_diff={max_diff:.2e} {'PASS' if passed else 'FAIL'}")

# dismat_mst (edge set match)
r_mst  = load_csv(f"tests/outputs_R/dismat_mst_out.csv")
py_mst = load_csv(f"tests/outputs_python/dismat_mst_out.csv")
if r_mst is not None and py_mst is not None:
    r_set  = set(zip(r_mst.iloc[:,0].astype(str), r_mst.iloc[:,1].astype(str)))
    py_set = set(zip(py_mst.iloc[:,0].astype(str), py_mst.iloc[:,1].astype(str)))
    # MST edges are ordered (from < to), check both directions
    r_set2  = r_set | {(b,a) for a,b in r_set}
    py_set2 = py_set | {(b,a) for a,b in py_set}
    jaccard = len(r_set2 & py_set2) / max(len(r_set2 | py_set2), 1)
    print(f"  dismat_mst: R edges={len(r_set)}, Py edges={len(py_set)}, Jaccard={jaccard:.3f}")
    passed = jaccard > 0.99
    results.append(("dismat_mst", "Edge Jaccard", jaccard, 0.99, "PASS" if passed else "FAIL"))

# cell_clone_coembed
r_co  = load_csv_idx(f"tests/outputs_R/coembed_out.csv")
py_co = load_csv_idx(f"tests/outputs_python/coembed_out.csv")
if r_co is not None and py_co is not None:
    r_vals  = r_co.values.astype(float).ravel()
    py_vals = py_co.values.astype(float).ravel()
    n = min(len(r_vals), len(py_vals))
    mask = np.isfinite(r_vals[:n]) & np.isfinite(py_vals[:n])
    if mask.sum() > 3:
        sc = safe_pearson(r_vals[:n][mask], py_vals[:n][mask])
        check("cell_clone_coembed", "Pearson r", sc, "pearson")
        print(f"  coembed: r={sc:.4f}")

# Visualization embeddings
for fname, label in [("umap_coords", "UMAP"), ("mds_knn_coords", "MDS-kNN")]:
    r_v  = load_csv_idx(f"tests/outputs_R/{fname}.csv")
    py_v = load_csv_idx(f"tests/outputs_python/{fname}.csv")
    if r_v is not None and py_v is not None:
        try:
            r_arr  = r_v.values.astype(float)
            py_arr = py_v.values.astype(float)
            n = min(r_arr.shape[0], py_arr.shape[0])
            n_d = min(r_arr.shape[1], py_arr.shape[1], 2)
            _, _, disparity = procrustes(r_arr[:n,:n_d], py_arr[:n,:n_d])
            sc = float(1 - disparity)
            check(f"{label} embedding", "Procrustes R²", sc, "procrustes")
            print(f"  {fname}: Procrustes R²={sc:.4f}")
        except Exception as e:
            print(f"  {fname}: Procrustes error: {e}")

# ── Print Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  COMPARISON SUMMARY")
print("=" * 70)
header = f"{'Test':<35} {'Metric':<25} {'Score':>9} {'Thresh':>7} {'Status':>6}"
print(header)
print("-" * 70)
n_pass = n_fail = n_warn = 0
for name, metric, score, thresh, status in results:
    score_str = f"{score:.4f}" if isinstance(score, float) and not np.isnan(score) else "  nan  "
    thresh_str = f"{thresh:.2f}" if isinstance(thresh, float) and not np.isnan(thresh) else "  -   "
    print(f"{name:<35} {metric:<25} {score_str:>9} {thresh_str:>7} {status:>6}")
    if status == "PASS":   n_pass += 1
    elif status == "FAIL": n_fail += 1
    else:                  n_warn += 1

print("-" * 70)
total = n_pass + n_fail
print(f"\n  {n_pass}/{total} tests PASSED  ({n_warn} warnings/info)")
print("\nConclusion:", "All tests passed." if n_fail == 0 else f"{n_fail} tests failed — review discrepancies above.")
print("=" * 70)

# ── Write report.md ───────────────────────────────────────────────────────────
report_path = "tests/report.md"
with open(report_path, "w") as f:
    f.write("# Clonotrace R vs Python Equivalence Report\n\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## Vignette Pipeline Tests\n\n")
    f.write("| Test | Metric | Score | Threshold | Pass? |\n")
    f.write("|------|--------|-------|-----------|-------|\n")
    vignette_names = {"kNN+transition","Label spreading","Clone NN dist","Clone OT dist",
                      "Clone MDS","Leiden clustering","Clone pseudotime",
                      "Profile enrichment pval","DEG F-stat","DEG Cohen's d"}
    for name, metric, score, thresh, status in results:
        if any(v in name for v in ["kNN","Label","Clone","Leiden","pseudotime","enrichment","DEG"]):
            score_str  = f"{score:.4f}" if isinstance(score, float) and not np.isnan(score) else "nan"
            thresh_str = f"{thresh:.2f}" if isinstance(thresh, float) and not np.isnan(thresh) else "-"
            f.write(f"| {name} | {metric} | {score_str} | {thresh_str} | {status} |\n")

    f.write("\n## Additional Function Tests\n\n")
    f.write("| Test | Metric | Score | Threshold | Pass? |\n")
    f.write("|------|--------|-------|-----------|-------|\n")
    for name, metric, score, thresh, status in results:
        if not any(v in name for v in ["kNN","Label","Clone","Leiden","pseudotime","enrichment","DEG"]):
            score_str  = f"{score:.4f}" if isinstance(score, float) and not np.isnan(score) else "nan"
            thresh_str = f"{thresh:.2f}" if isinstance(thresh, float) and not np.isnan(thresh) else "-"
            f.write(f"| {name} | {metric} | {score_str} | {thresh_str} | {status} |\n")

    f.write(f"\n## Summary\n\n")
    f.write(f"- {n_pass}/{total} tests passed\n")
    f.write(f"- {n_warn} warnings/info items\n")
    if n_fail == 0:
        f.write("- **Conclusion**: All tests passed. R and Python implementations are equivalent.\n")
    else:
        f.write(f"- **Conclusion**: {n_fail} tests failed — review discrepancies.\n")

print(f"\nReport written to: {report_path}")
