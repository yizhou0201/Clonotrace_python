"""
Generate Updated Benchmark Plots
=================================
1. Updated three-way comparison (R, Old Python, Optimized Python) - speed
2. Aggregated summary: overall improvement across all functions
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
with open("benchmark_results.json") as f:
    py_data = json.load(f)

with open("benchmark_R_results.json") as f:
    r_data = json.load(f)

old_py = py_data["old"]
opt_py = py_data["optimized"]

# ---------------------------------------------------------------
# Display names
# ---------------------------------------------------------------
display_names = {
    "dis_points_to_edges": "dis_points_to_edges\n(5K pts, 50 edges)",
    "mat_sparsify": "mat_sparsify\n(300x300)",
    "embedding2knn": "embedding2knn\n(3K pts, k=15)",
    "snn_from_dist": "_snn_from_dist\n(500 nodes, k=15)",
    "link2cluster": "link2cluster\n(500 nodes)",
    "nearest_knn": "nearest_knn\n(300 nodes, k=10)",
    "sync_sparse_rows": "sync_sparse_rows\n(1K x 500)",
    "DPT_T": "DPT_T\n(N=200)",
    "acct": "acct\n(N=200)",
    "label_spreading": "label_spreading\n(500 cells)",
    "clone_partition": "clone_partition\n(50 clones)",
    "cluster_profile_enrich": "cluster_profile_enrich\n(500 cells)",
}

short_names = {
    "dis_points_to_edges": "dis_points_to_edges",
    "mat_sparsify": "mat_sparsify",
    "embedding2knn": "embedding2knn",
    "snn_from_dist": "_snn_from_dist",
    "link2cluster": "link2cluster",
    "nearest_knn": "nearest_knn",
    "sync_sparse_rows": "sync_sparse_rows",
    "DPT_T": "DPT_T",
    "acct": "acct",
    "label_spreading": "label_spreading",
    "clone_partition": "clone_partition",
    "cluster_profile_enrich": "cluster_profile_enrich",
}

# ---------------------------------------------------------------
# Functions present in all three datasets
# ---------------------------------------------------------------
three_way_funcs = [
    "dis_points_to_edges",
    "mat_sparsify",
    "embedding2knn",
    "snn_from_dist",
    "link2cluster",
    "nearest_knn",
    "DPT_T",
    "acct",
    "clone_partition",
]

# Filter to those actually present and without errors
three_way_funcs = [f for f in three_way_funcs
                   if f in r_data and f in old_py and f in opt_py
                   and "error" not in old_py.get(f, {}) and "error" not in opt_py.get(f, {})]

# All Python functions (old vs optimized)
all_py_funcs = [f for f in old_py
                if f in opt_py
                and "error" not in old_py[f] and "error" not in opt_py[f]]

# ---------------------------------------------------------------
# Plot 1: Three-way comparison (updated)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 7))

names_3 = [display_names.get(f, f) for f in three_way_funcs]
r_t = [r_data[f]["time"] for f in three_way_funcs]
old_t = [old_py[f]["time"] for f in three_way_funcs]
opt_t = [opt_py[f]["time"] for f in three_way_funcs]

x3 = np.arange(len(three_way_funcs))
w = 0.25

bars_r = ax.bar(x3 - w, r_t, w, label="R (original)", color="#3498db", alpha=0.85)
bars_old = ax.bar(x3, old_t, w, label="Python (before optimization)", color="#e74c3c", alpha=0.85)
bars_opt = ax.bar(x3 + w, opt_t, w, label="Python (optimized)", color="#2ecc71", alpha=0.85)
ax.set_yscale("log")
ax.set_ylabel("Time (seconds, log scale)", fontsize=12)
ax.set_title("Three-Way Speed Comparison: R vs Old Python vs Optimized Python",
             fontsize=14, fontweight="bold")
ax.set_xticks(x3)
ax.set_xticklabels(names_3, rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Annotate speedup vs R for optimized Python
for i, (rt, pt) in enumerate(zip(r_t, opt_t)):
    if pt > 0 and rt > 0:
        ratio = rt / pt
        if ratio >= 1.5:
            ax.annotate(f"{ratio:.0f}x vs R", xy=(i + w, pt),
                       xytext=(0, -15), textcoords="offset points",
                       ha="center", fontsize=7, fontweight="bold", color="#27ae60",
                       rotation=90)

plt.tight_layout()
plt.savefig("report_three_way_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_three_way_comparison.png")

# ---------------------------------------------------------------
# Plot 2: Aggregated summary - single figure with key metrics
# ---------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))

# Layout: top row = big summary numbers, bottom = per-function detail
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.5], hspace=0.35, wspace=0.3)

# --- Top-left: Overall speedup gauge ---
ax_top_l = fig.add_subplot(gs[0, 0])
total_old = sum(old_py[f]["time"] for f in all_py_funcs)
total_opt = sum(opt_py[f]["time"] for f in all_py_funcs)
speedup_total = total_old / total_opt

bar_data = [total_old, total_opt]
bar_labels = ["Before\nOptimization", "After\nOptimization"]
bar_colors = ["#e74c3c", "#2ecc71"]
bars = ax_top_l.barh([0, 1], bar_data, color=bar_colors, alpha=0.85, height=0.5,
                      edgecolor="white")
ax_top_l.text(total_old + 0.05, 0, f"{total_old:.2f}s", va="center", fontsize=12,
              fontweight="bold")
ax_top_l.text(total_opt + 0.05, 1, f"{total_opt:.2f}s", va="center", fontsize=12,
              fontweight="bold")
ax_top_l.set_yticks([0, 1])
ax_top_l.set_yticklabels(bar_labels, fontsize=11)
ax_top_l.set_xlabel("Total Time (seconds)", fontsize=10)
ax_top_l.set_title(f"Overall: {speedup_total:.0f}x Faster", fontsize=14,
                    fontweight="bold", color="#27ae60")
ax_top_l.grid(axis="x", alpha=0.3)
ax_top_l.invert_yaxis()

# --- Top-right: Key metrics summary ---
ax_top_r = fig.add_subplot(gs[0, 1])
ax_top_r.axis("off")

# Compute geometric means
speedups_all = []
mem_reds_all = []
for f in all_py_funcs:
    ot = old_py[f]["time"]; nt = opt_py[f]["time"]
    if nt > 0: speedups_all.append(ot / nt)
    if "memory_bytes" in old_py[f] and "memory_bytes" in opt_py[f]:
        om = old_py[f]["memory_bytes"]; nm = opt_py[f]["memory_bytes"]
        if nm > 0: mem_reds_all.append(om / nm)

geo_speed = np.exp(np.mean(np.log(speedups_all)))
geo_mem = np.exp(np.mean(np.log(mem_reds_all)))
max_speedup = max(speedups_all)
max_fn = all_py_funcs[speedups_all.index(max_speedup)]

# R comparison
r_funcs_common = [f for f in three_way_funcs if f in r_data and f in opt_py]
r_total = sum(r_data[f]["time"] for f in r_funcs_common)
py_total = sum(opt_py[f]["time"] for f in r_funcs_common)

metrics_text = (
    f"Geometric mean speedup:     {geo_speed:.1f}x\n"
    f"Geometric mean mem reduction: {geo_mem:.1f}x\n"
    f"Max speedup: {max_speedup:.0f}x ({short_names.get(max_fn, max_fn)})\n"
    f"\n"
    f"vs R original (common funcs):\n"
    f"  R total: {r_total:.3f}s  |  Python: {py_total:.3f}s\n"
    f"  Python is {r_total/py_total:.0f}x faster than R"
)
ax_top_r.text(0.05, 0.5, metrics_text, transform=ax_top_r.transAxes,
              fontsize=12, verticalalignment="center", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))
ax_top_r.set_title("Key Metrics", fontsize=14, fontweight="bold")

# --- Bottom: Per-function speedup + memory side by side ---
ax_bot = fig.add_subplot(gs[1, :])

# Sort by speedup
func_labels = [short_names.get(f, f) for f in all_py_funcs]
order = np.argsort(speedups_all)[::-1]
speedups_sorted = [speedups_all[i] for i in order]
mem_sorted = [mem_reds_all[i] if i < len(mem_reds_all) else 1.0 for i in order]
labels_sorted = [func_labels[i] for i in order]

y = np.arange(len(labels_sorted))
bar_h = 0.35

bars_speed = ax_bot.barh(y - bar_h/2, speedups_sorted, bar_h,
                          label="Speed improvement", color="#2ecc71", alpha=0.85)
bars_mem = ax_bot.barh(y + bar_h/2, mem_sorted, bar_h,
                        label="Memory reduction", color="#9b59b6", alpha=0.75)

ax_bot.set_yticks(y)
ax_bot.set_yticklabels(labels_sorted, fontsize=9)
ax_bot.set_xscale("log")
ax_bot.axvline(x=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
ax_bot.set_xlabel("Improvement Factor (log scale)", fontsize=11)
ax_bot.set_title("Per-Function Speed & Memory Improvement", fontsize=13, fontweight="bold")
ax_bot.legend(fontsize=10, loc="lower right")
ax_bot.grid(axis="x", alpha=0.3)

# Annotate
for i, (bar_s, val_s) in enumerate(zip(bars_speed, speedups_sorted)):
    label = f"{val_s:.0f}x" if val_s >= 10 else f"{val_s:.1f}x"
    ax_bot.text(bar_s.get_width() * 1.08, bar_s.get_y() + bar_s.get_height()/2,
                label, va="center", fontsize=8, fontweight="bold", color="#27ae60")

for i, (bar_m, val_m) in enumerate(zip(bars_mem, mem_sorted)):
    if val_m > 1.3:
        label = f"{val_m:.0f}x" if val_m >= 10 else f"{val_m:.1f}x"
        ax_bot.text(bar_m.get_width() * 1.08, bar_m.get_y() + bar_m.get_height()/2,
                    label, va="center", fontsize=8, fontweight="bold", color="#8e44ad")

# Note about scaling improvements
ax_bot.text(0.98, 0.02,
            "* DPT_T memory: 206MB -> 0.1MB at N=3000 (2060x)\n"
            "  (not visible at benchmark scale N=200)",
            transform=ax_bot.transAxes, fontsize=8, ha="right", va="bottom",
            style="italic", color="#555555",
            bbox=dict(boxstyle="round", facecolor="#f9f9f9", alpha=0.8))

plt.savefig("report_aggregate_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_aggregate_summary.png")

# ---------------------------------------------------------------
# Print summary statistics
# ---------------------------------------------------------------
print("\n=== Summary Statistics ===")
print(f"Total old Python time:       {total_old:.3f}s")
print(f"Total optimized Python time: {total_opt:.3f}s")
print(f"Overall speedup:             {speedup_total:.1f}x")

speedups = speedups_all
mem_reductions = mem_reds_all
geo_speed = np.exp(np.mean(np.log(speedups)))
geo_mem = np.exp(np.mean(np.log(mem_reductions)))
print(f"Geometric mean speedup:      {geo_speed:.1f}x")
print(f"Geometric mean mem reduction: {geo_mem:.1f}x")

# R comparison
r_funcs_common = [f for f in three_way_funcs if f in r_data and f in opt_py]
r_total = sum(r_data[f]["time"] for f in r_funcs_common)
py_total = sum(opt_py[f]["time"] for f in r_funcs_common)
print(f"\nR total (common functions):   {r_total:.3f}s")
print(f"Python opt total:             {py_total:.3f}s")
print(f"R / Python speedup:           {r_total/py_total:.1f}x")
