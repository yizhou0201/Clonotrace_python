"""
Generate benchmark plots from real dataset (hematopoiesis) timings.
Produces:
  1. report_three_way_comparison.png  - R vs Python per-step bar chart
  2. report_aggregate_summary.png     - Overall summary with key metrics
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Load real dataset timing results
# ---------------------------------------------------------------
with open("real_data_python_results.json") as f:
    py_data = json.load(f)

with open("real_data_R_results.json") as f:
    r_data = json.load(f)

# ---------------------------------------------------------------
# Common pipeline steps (in order)
# ---------------------------------------------------------------
common_steps = [
    "embedding2knn",
    "compute_transition",
    "label_spreading_bootstrap",
    "filter_sparsify",
    "compute_clone_dis",
    "clone_clustering",
    "clone_dpt",
    "profile_assignment",
    "cluster_profile_enrich",
    "profile_cluster_DEG",
]

# Nice display names
display_names = {
    "embedding2knn":               "embedding2knn\n(34K cells, k=30)",
    "compute_transition":          "compute_transition\n(row-normalize)",
    "label_spreading_bootstrap":   "label_spreading_bootstrap\n(34K cells, 802 clones, n=48)",
    "filter_sparsify":             "filter + mat_sparsify\n(deviance < 0.3)",
    "compute_clone_dis":           "compute_clone_dis\n(SNN graph + NN, 802 clones)",
    "clone_clustering":            "clone_clustering\n(Louvain, k=20)",
    "clone_dpt":                   "clone_dpt\n(pseudotime)",
    "profile_assignment":          "profile_assignment\n(cell profiles)",
    "cluster_profile_enrich":      "cluster_profile_enrich\n(300 permutations)",
    "profile_cluster_DEG":         "profile_cluster_DEG\n(50 permutations)",
}

short_names = {
    "embedding2knn":               "embedding2knn",
    "compute_transition":          "compute_transition",
    "label_spreading_bootstrap":   "label_spreading_bootstrap",
    "filter_sparsify":             "filter + sparsify",
    "compute_clone_dis":           "compute_clone_dis",
    "clone_clustering":            "clone_clustering",
    "clone_dpt":                   "clone_dpt",
    "profile_assignment":          "profile_assignment",
    "cluster_profile_enrich":      "cluster_profile_enrich",
    "profile_cluster_DEG":         "profile_cluster_DEG",
}

# Filter to steps present in both
steps = [s for s in common_steps if s in py_data and s in r_data]

r_times = [r_data[s]["time"] for s in steps]
py_times = [py_data[s]["time"] for s in steps]
names = [display_names.get(s, s) for s in steps]

# ---------------------------------------------------------------
# Plot 1: R vs Python per-step comparison
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(steps))
w = 0.35

bars_r = ax.bar(x - w/2, r_times, w, label="R (original)", color="#3498db", alpha=0.85)
bars_py = ax.bar(x + w/2, py_times, w, label="Python (optimized)", color="#2ecc71", alpha=0.85)

ax.set_yscale("log")
ax.set_ylabel("Time (seconds, log scale)", fontsize=12)
ax.set_title("Real Dataset: R vs Python (Hematopoiesis, 34K cells, 802 clones)",
             fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=11, loc="upper right")
ax.grid(axis="y", alpha=0.3)

# Annotate speedup
for i, (rt, pt) in enumerate(zip(r_times, py_times)):
    if pt > 0 and rt > 0:
        ratio = rt / pt
        if ratio >= 1.5:
            label = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"
            ax.annotate(label, xy=(i + w/2, pt),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=8, fontweight="bold", color="#27ae60")
        elif ratio < 0.67:
            label = f"{1/ratio:.1f}x slower"
            ax.annotate(label, xy=(i + w/2, pt),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=7, fontweight="bold", color="#e74c3c")

plt.tight_layout()
plt.savefig("report_three_way_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_three_way_comparison.png")

# ---------------------------------------------------------------
# Plot 2: Aggregated summary
# ---------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.5], hspace=0.35, wspace=0.3)

# Total times
total_r = sum(r_times)
total_py = sum(py_times)
speedup_total = total_r / total_py

# --- Top-left: Total time bars ---
ax_top_l = fig.add_subplot(gs[0, 0])
bars = ax_top_l.barh([0, 1], [total_r, total_py],
                      color=["#3498db", "#2ecc71"], alpha=0.85, height=0.5)
ax_top_l.text(total_r + total_r*0.02, 0, f"{total_r:.1f}s", va="center", fontsize=12,
              fontweight="bold")
ax_top_l.text(total_py + total_r*0.02, 1, f"{total_py:.1f}s", va="center", fontsize=12,
              fontweight="bold")
ax_top_l.set_yticks([0, 1])
ax_top_l.set_yticklabels(["R\n(original)", "Python\n(optimized)"], fontsize=11)
ax_top_l.set_xlabel("Total Time (seconds)", fontsize=10)
ax_top_l.set_title(f"Overall: Python is {speedup_total:.1f}x Faster",
                    fontsize=14, fontweight="bold", color="#27ae60")
ax_top_l.grid(axis="x", alpha=0.3)
ax_top_l.invert_yaxis()

# --- Top-right: Key metrics ---
ax_top_r = fig.add_subplot(gs[0, 1])
ax_top_r.axis("off")

# Per-step speedups
speedups = [r/p if p > 0 else 1.0 for r, p in zip(r_times, py_times)]
geo_mean = np.exp(np.mean(np.log([max(s, 0.01) for s in speedups])))
max_speedup = max(speedups)
max_step = steps[speedups.index(max_speedup)]

metrics_text = (
    f"Dataset: Hematopoiesis (34,782 cells)\n"
    f"Expanded clones: 802\n"
    f"\n"
    f"Geometric mean speedup: {geo_mean:.1f}x\n"
    f"Max speedup: {max_speedup:.0f}x ({short_names[max_step]})\n"
    f"Total: R {total_r:.1f}s vs Python {total_py:.1f}s"
)
ax_top_r.text(0.05, 0.5, metrics_text, transform=ax_top_r.transAxes,
              fontsize=12, verticalalignment="center", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))
ax_top_r.set_title("Key Metrics", fontsize=14, fontweight="bold")

# --- Bottom: Per-step speedup horizontal bar chart ---
ax_bot = fig.add_subplot(gs[1, :])

# Sort by speedup
order = np.argsort(speedups)[::-1]
speedups_sorted = [speedups[i] for i in order]
labels_sorted = [short_names[steps[i]] for i in order]

y = np.arange(len(labels_sorted))
colors = ["#2ecc71" if s >= 1 else "#e74c3c" for s in speedups_sorted]
bars = ax_bot.barh(y, speedups_sorted, color=colors, alpha=0.85, edgecolor="white")

ax_bot.set_yticks(y)
ax_bot.set_yticklabels(labels_sorted, fontsize=9)
ax_bot.set_xscale("log")
ax_bot.axvline(x=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
ax_bot.set_xlabel("Speedup Factor: R / Python (log scale)", fontsize=11)
ax_bot.set_title("Per-Step Speedup (Real Dataset)", fontsize=13, fontweight="bold")
ax_bot.grid(axis="x", alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, speedups_sorted)):
    if val >= 1:
        label = f"{val:.0f}x faster" if val >= 10 else f"{val:.1f}x faster"
        ax_bot.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height()/2,
                    label, va="center", fontsize=8, fontweight="bold", color="#27ae60")
    else:
        label = f"{1/val:.1f}x slower"
        ax_bot.text(max(bar.get_width() * 1.08, 0.15), bar.get_y() + bar.get_height()/2,
                    label, va="center", fontsize=8, fontweight="bold", color="#e74c3c")

plt.savefig("report_aggregate_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_aggregate_summary.png")

# ---------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------
print(f"\n{'='*55}")
print(f"{'Step':<35s} {'R':>8s} {'Python':>8s} {'Speedup':>8s}")
print(f"{'='*55}")
for s, rt, pt in zip(steps, r_times, py_times):
    ratio = rt / pt if pt > 0 else float('inf')
    print(f"  {short_names[s]:<33s} {rt:>7.2f}s {pt:>7.2f}s {ratio:>7.1f}x")
print(f"{'='*55}")
print(f"  {'TOTAL':<33s} {total_r:>7.1f}s {total_py:>7.1f}s {speedup_total:>7.1f}x")
