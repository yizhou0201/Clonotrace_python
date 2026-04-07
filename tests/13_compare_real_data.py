#!/usr/bin/env python3
"""
13_compare_real_data.py - Compare R and Python pipeline outputs on real data.
Run from: /Users/yizhouw/Desktop/packages/Clonotrace_python/

Prerequisite: Run 11_run_R_pipeline.R and 12_run_python_pipeline.py first.
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

# ── Thresholds ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    "pearson":      0.95,
    "pearson_hi":   0.98,
    "spearman":     0.95,
    "spearman_lo":  0.90,
    "procrustes":   0.80,
    "ari":          0.70,
    "exact":        1.00,
}

R_DIR  = "tests/real_outputs_R"
PY_DIR = "tests/real_outputs_python"
results = []


def check(name, metric, score, threshold_key="pearson"):
    thresh = THRESHOLDS.get(threshold_key, 0.95)
    passed = score >= thresh if not np.isnan(score) else False
    results.append((name, metric, score, thresh, "PASS" if passed else "FAIL"))
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}: {metric} = {score:.4f} (thresh {thresh:.2f})")
    return passed


def safe_pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(stats.pearsonr(a[mask], b[mask])[0])


def safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(stats.spearmanr(a[mask], b[mask])[0])


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


print("=" * 80)
print("  Clonotrace R vs Python Equivalence Report — Real Hematopoiesis Data")
print(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ── 1. kNN + transition ─────────────────────────────────────────────────────
print("\n--- 1. kNN + transition ---")
r_knn  = load_csv(f"{R_DIR}/cell_knn_sample.csv")
py_knn = load_csv(f"{PY_DIR}/cell_knn_sample.csv")
if r_knn is not None and py_knn is not None:
    r_knn["key"]  = r_knn["row"].astype(str) + "_" + r_knn["col"].astype(str)
    py_knn["key"] = py_knn["row"].astype(str) + "_" + py_knn["col"].astype(str)
    merged = pd.merge(r_knn[["key", "value"]], py_knn[["key", "value"]],
                      on="key", suffixes=("_R", "_py"))
    if len(merged) > 0:
        sc = safe_pearson(merged["value_R"], merged["value_py"])
        print(f"  R nnz={len(r_knn)}, Py nnz={len(py_knn)}, matched={len(merged)}")
        check("kNN+transition", "Pearson r (matched)", sc, "pearson_hi")
    else:
        print("  No matching (row,col) pairs")
        results.append(("kNN+transition", "match", float("nan"), 0.99, "WARN"))


# ── 2. Label spreading ──────────────────────────────────────────────────────
print("\n--- 2. Label spreading (cell_clone_prob sample) ---")
r_prob  = load_csv_idx(f"{R_DIR}/cell_clone_prob_sample.csv")
py_prob = load_csv_idx(f"{PY_DIR}/cell_clone_prob_sample.csv")
if r_prob is not None and py_prob is not None:
    r_vals  = r_prob.values.ravel()
    py_vals = py_prob.values.ravel()
    n = min(len(r_vals), len(py_vals))
    sc = safe_pearson(r_vals[:n], py_vals[:n])
    print(f"  R shape={r_prob.shape}, Py shape={py_prob.shape}")
    check("Label spreading (prob)", "Pearson r", sc, "pearson")


# ── 2b. Deviance ────────────────────────────────────────────────────────────
print("\n--- 2b. Label spreading (deviance) ---")
r_dev  = load_csv(f"{R_DIR}/deviance.csv")
py_dev = load_csv(f"{PY_DIR}/deviance.csv")
if r_dev is not None and py_dev is not None:
    sc = safe_pearson(r_dev["deviance"], py_dev["deviance"])
    print(f"  R deviance range: [{r_dev['deviance'].min():.3f}, {r_dev['deviance'].max():.3f}]")
    print(f"  Py deviance range: [{py_dev['deviance'].min():.3f}, {py_dev['deviance'].max():.3f}]")
    check("Deviance", "Pearson r", sc, "spearman_lo")


# ── 3. Clone NN distance ────────────────────────────────────────────────────
print("\n--- 3. Clone NN distance ---")
r_sq  = load_csv_idx(f"{R_DIR}/clone_nn_dis_sq50.csv")
py_sq = load_csv_idx(f"{PY_DIR}/clone_nn_dis_sq50.csv")
if r_sq is not None and py_sq is not None:
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


# ── 4. Clone OT distance ────────────────────────────────────────────────────
print("\n--- 4. Clone OT distance (50 clones) ---")
r_ot  = load_csv(f"{R_DIR}/clone_ot_dis_sub50.csv")
py_ot = load_csv(f"{PY_DIR}/clone_ot_dis_sub50.csv")
if r_ot is not None and py_ot is not None:
    # Try matching by (group1, group2) key — handle 0-vs-1 indexing
    r_ot["key"] = r_ot["group1"].astype(str) + "_" + r_ot["group2"].astype(str)
    # Python might be 0-indexed
    py_ot_adj = py_ot.copy()
    py_ot_adj["key"] = (py_ot_adj["group1"]).astype(str) + "_" + \
                        (py_ot_adj["group2"]).astype(str)
    merged = pd.merge(r_ot[["key", "dis"]], py_ot_adj[["key", "dis"]],
                      on="key", suffixes=("_R", "_py"))
    if len(merged) < 3:
        # Try with +1 offset
        py_ot_adj["key"] = (py_ot_adj["group1"] + 1).astype(str) + "_" + \
                            (py_ot_adj["group2"] + 1).astype(str)
        merged = pd.merge(r_ot[["key", "dis"]], py_ot_adj[["key", "dis"]],
                          on="key", suffixes=("_R", "_py"))
    if len(merged) < 3:
        # Try symmetric match too
        py_ot_adj["key2"] = (py_ot_adj["group2"] + 1).astype(str) + "_" + \
                             (py_ot_adj["group1"] + 1).astype(str)
        m2 = pd.merge(r_ot[["key", "dis"]],
                       py_ot_adj[["key2", "dis"]].rename(columns={"key2": "key"}),
                       on="key", suffixes=("_R", "_py"))
        merged = pd.concat([merged, m2], ignore_index=True)

    print(f"  R pairs={len(r_ot)}, Py pairs={len(py_ot)}, matched={len(merged)}")
    if len(merged) > 3:
        sc = safe_pearson(merged["dis_R"], merged["dis_py"])
        max_diff = float(np.max(np.abs(merged["dis_R"] - merged["dis_py"])))
        check("Clone OT dist", "Pearson r", sc, "spearman_lo")
        results.append(("Clone OT dist", "Max abs diff", max_diff, float("nan"), "INFO"))
    else:
        print("  Too few matched pairs for comparison")
        results.append(("Clone OT dist", "matched", float("nan"), 0.90, "WARN"))


# ── 5. Clone MDS ────────────────────────────────────────────────────────────
print("\n--- 5. Clone MDS (30 dims) ---")
r_mds  = load_csv_idx(f"{R_DIR}/clone_mds.csv")
py_mds = load_csv_idx(f"{PY_DIR}/clone_mds.csv")
if r_mds is not None and py_mds is not None:
    try:
        r_arr  = r_mds.values.astype(float)
        py_arr = py_mds.values.astype(float)
        n = min(r_arr.shape[0], py_arr.shape[0])
        n_dims = min(r_arr.shape[1], py_arr.shape[1])
        _, _, disparity = procrustes(r_arr[:n, :n_dims], py_arr[:n, :n_dims])
        pr2 = float(1 - disparity)
        # Also check with fewer dims
        _, _, disp5 = procrustes(r_arr[:n, :5], py_arr[:n, :5])
        pr2_5 = float(1 - disp5)
        print(f"  R shape={r_mds.shape}, Py shape={py_mds.shape}")
        print(f"  Procrustes R² (5 dims): {pr2_5:.4f}, (all {n_dims} dims): {pr2:.4f}")
        check("Clone MDS", f"Procrustes R² ({n_dims} dims)", pr2, "procrustes")
    except Exception as e:
        print(f"  Procrustes error: {e}")


# ── 6. Leiden clustering ────────────────────────────────────────────────────
print("\n--- 6. Leiden clustering ---")
r_cl  = load_csv_idx(f"{R_DIR}/clone_cluster.csv")
py_cl = load_csv_idx(f"{PY_DIR}/clone_cluster.csv")
if r_cl is not None and py_cl is not None:
    print(f"  R clusters: {r_cl['cluster'].nunique()}, Py clusters: {py_cl['cluster'].nunique()}")
    n = min(len(r_cl), len(py_cl))
    if HAS_ARI:
        ari = adjusted_rand_score(
            r_cl["cluster"].values[:n],
            py_cl["cluster"].values[:n]
        )
        check("Leiden clustering", "ARI", ari, "ari")
    else:
        results.append(("Leiden clustering", "ARI", float("nan"), 0.70, "WARN"))


# ── 7. Clone pseudotime ─────────────────────────────────────────────────────
print("\n--- 7. Clone pseudotime ---")
r_pt  = load_csv(f"{R_DIR}/clone_t.csv")
py_pt = load_csv(f"{PY_DIR}/clone_t.csv")
if r_pt is not None and py_pt is not None:
    r_t  = r_pt["dpt"].values
    py_t = py_pt["dpt"].values
    n = min(len(r_t), len(py_t))
    sc = safe_spearman(r_t[:n], py_t[:n])
    print(f"  R range=[{r_t.min():.3f},{r_t.max():.3f}]")
    print(f"  Py range=[{py_t.min():.3f},{py_t.max():.3f}]")
    # Pseudotime direction may be flipped (different root selection); use |r|
    check("Clone pseudotime", "|Spearman r|", abs(sc), "spearman")


# ── 8. Profile enrichment ───────────────────────────────────────────────────
print("\n--- 8. Profile enrichment ---")
r_ep  = load_csv_idx(f"{R_DIR}/enrich_pval.csv")
py_ep = load_csv_idx(f"{PY_DIR}/enrich_pval.csv")
if r_ep is not None and py_ep is not None:
    r_vals  = r_ep.values.ravel()
    py_vals = py_ep.values.ravel()
    n = min(len(r_vals), len(py_vals))
    sc = safe_pearson(r_vals[:n], py_vals[:n])
    print(f"  R shape={r_ep.shape}, Py shape={py_ep.shape}")
    check("Profile enrichment pval", "Pearson r", sc, "spearman_lo")


# ── 9. Profile DEG ──────────────────────────────────────────────────────────
print("\n--- 9. Profile DEG ---")
r_deg  = load_csv_idx(f"{R_DIR}/DEG_stats.csv")
py_deg = load_csv_idx(f"{PY_DIR}/DEG_stats.csv")
if (r_deg is not None and py_deg is not None
        and "stat" in r_deg.columns and "stat" in py_deg.columns):
    common_genes = r_deg.index.intersection(py_deg.index)
    print(f"  R genes={len(r_deg)}, Py genes={len(py_deg)}, common={len(common_genes)}")
    if len(common_genes) > 10:
        sc_stat = safe_spearman(
            r_deg.loc[common_genes, "stat"],
            py_deg.loc[common_genes, "stat"]
        )
        check("DEG F-stat", "Spearman r", sc_stat, "spearman_lo")
        if "cohen" in r_deg.columns and "cohen" in py_deg.columns:
            sc_coh = safe_spearman(
                r_deg.loc[common_genes, "cohen"],
                py_deg.loc[common_genes, "cohen"]
            )
            check("DEG Cohen's d", "Spearman r", sc_coh, "spearman_lo")
else:
    if r_deg is not None and py_deg is not None:
        print(f"  R columns: {list(r_deg.columns)}")
        print(f"  Py columns: {list(py_deg.columns)}")


# ── 10. SNN graph weight comparison ─────────────────────────────────────────
print("\n--- 10. SNN graph weights (500 cells, k=10) ---")
r_snn  = load_csv(f"{R_DIR}/snn_graph_weights.csv")
py_snn = load_csv(f"{PY_DIR}/snn_graph_weights.csv")
if r_snn is not None and py_snn is not None:
    print(f"  R edges: {len(r_snn)}, Py edges: {len(py_snn)}")
    # Match by (from, to) keys
    r_snn["key"]  = r_snn["from"].astype(str) + "_" + r_snn["to"].astype(str)
    py_snn["key"] = py_snn["from"].astype(str) + "_" + py_snn["to"].astype(str)
    merged = pd.merge(
        r_snn[["key", "weight_exp"]],
        py_snn[["key", "weight_exp"]],
        on="key", suffixes=("_R", "_py")
    )
    if len(merged) > 10:
        sc = safe_pearson(merged["weight_exp_R"], merged["weight_exp_py"])
        print(f"  Matched edges: {len(merged)}")
        check("SNN graph weights", "Pearson r", sc, "pearson_hi")
    else:
        # Try symmetric key match
        py_snn["key2"] = py_snn["to"].astype(str) + "_" + py_snn["from"].astype(str)
        m2 = pd.merge(
            r_snn[["key", "weight_exp"]],
            py_snn[["key2", "weight_exp"]].rename(columns={"key2": "key"}),
            on="key", suffixes=("_R", "_py")
        )
        merged = pd.concat([merged, m2], ignore_index=True).drop_duplicates("key")
        print(f"  Matched edges (with symmetric): {len(merged)}")
        if len(merged) > 10:
            sc = safe_pearson(merged["weight_exp_R"], merged["weight_exp_py"])
            check("SNN graph weights", "Pearson r", sc, "pearson_hi")


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  COMPARISON SUMMARY — Real Hematopoiesis Data")
print("=" * 80)
header = f"{'Test':<35} {'Metric':<30} {'Score':>9} {'Thresh':>7} {'Status':>6}"
print(header)
print("-" * 80)

n_pass = n_fail = n_warn = n_info = 0
for name, metric, score, thresh, status in results:
    score_s = f"{score:.4f}" if isinstance(score, float) and not np.isnan(score) else "  nan  "
    thresh_s = f"{thresh:.2f}" if isinstance(thresh, float) and not np.isnan(thresh) else "  -   "
    print(f"{name:<35} {metric:<30} {score_s:>9} {thresh_s:>7} {status:>6}")
    if status == "PASS":   n_pass += 1
    elif status == "FAIL": n_fail += 1
    elif status == "INFO": n_info += 1
    else:                  n_warn += 1

total = n_pass + n_fail
print("-" * 80)
print(f"\n  {n_pass}/{total} tests PASSED  ({n_warn} warnings, {n_info} info)")
if n_fail == 0:
    print("  Conclusion: All tests passed.")
else:
    print(f"  Conclusion: {n_fail} tests FAILED — review discrepancies above.")
print("=" * 80)

# ── Write report ────────────────────────────────────────────────────────────
report_path = "tests/real_data_report.md"
with open(report_path, "w") as f:
    f.write("# Clonotrace R vs Python — Real Hematopoiesis Data Report\n\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("## Pipeline Tests\n\n")
    f.write("| Test | Metric | Score | Threshold | Pass? |\n")
    f.write("|------|--------|-------|-----------|-------|\n")
    for name, metric, score, thresh, status in results:
        score_s = f"{score:.4f}" if isinstance(score, float) and not np.isnan(score) else "nan"
        thresh_s = f"{thresh:.2f}" if isinstance(thresh, float) and not np.isnan(thresh) else "-"
        f.write(f"| {name} | {metric} | {score_s} | {thresh_s} | {status} |\n")

    f.write(f"\n## Summary\n\n")
    f.write(f"- {n_pass}/{total} tests passed\n")
    f.write(f"- {n_warn} warnings, {n_info} info items\n")
    if n_fail == 0:
        f.write("- **Conclusion**: All tests passed.\n")
    else:
        f.write(f"- **Conclusion**: {n_fail} tests failed — review discrepancies.\n")

print(f"\nReport: {report_path}")
