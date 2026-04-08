"""
Generate Benchmark Report Plots
================================
Creates comparison plots for:
  1. Old Python vs Optimized Python (speed + memory)
  2. R vs Optimized Python (speed + memory)
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
# Common functions between all three
# ---------------------------------------------------------------
common_funcs_py = [
    "dis_points_to_edges",
    "mat_sparsify",
    "embedding2knn",
    "snn_from_dist",
    "nearest_knn",
    "DPT_T",
    "acct",
    "clone_partition",
    "cluster_profile_enrich",
    "label_spreading",
    "sync_sparse_rows",
]

common_funcs_r_py = [
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

# Nice display names
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

# ---------------------------------------------------------------
# Plot 1: Old Python vs Optimized Python - Speed
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Filter to functions that exist in both and have no errors
speed_funcs = [f for f in common_funcs_py
               if f in old_py and f in opt_py
               and "error" not in old_py[f] and "error" not in opt_py[f]]

names = [display_names.get(f, f) for f in speed_funcs]
old_times = [old_py[f]["time"] for f in speed_funcs]
opt_times = [opt_py[f]["time"] for f in speed_funcs]

x = np.arange(len(speed_funcs))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2, old_times, width, label="Old Python", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + width/2, opt_times, width, label="Optimized Python", color="#2ecc71", alpha=0.85)
ax.set_yscale("log")
ax.set_ylabel("Time (seconds, log scale)", fontsize=12)
ax.set_title("Speed: Old Python vs Optimized Python", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Add speedup annotations
for i, (o, n) in enumerate(zip(old_times, opt_times)):
    if n > 0:
        speedup = o / n
        if speedup >= 1.5:
            ax.annotate(f"{speedup:.0f}x", xy=(i + width/2, n),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=8, fontweight="bold", color="#27ae60")

# Plot 2: Old Python vs Optimized Python - Memory
mem_funcs = [f for f in speed_funcs
             if "memory_bytes" in old_py[f] and "memory_bytes" in opt_py[f]]

names_mem = [display_names.get(f, f) for f in mem_funcs]
old_mem = [old_py[f]["memory_bytes"] / 1024 for f in mem_funcs]  # KB
opt_mem = [opt_py[f]["memory_bytes"] / 1024 for f in mem_funcs]

x_mem = np.arange(len(mem_funcs))

ax = axes[1]
bars1 = ax.bar(x_mem - width/2, old_mem, width, label="Old Python", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x_mem + width/2, opt_mem, width, label="Optimized Python", color="#2ecc71", alpha=0.85)
ax.set_yscale("log")
ax.set_ylabel("Peak Memory (KB, log scale)", fontsize=12)
ax.set_title("Memory: Old Python vs Optimized Python", fontsize=14, fontweight="bold")
ax.set_xticks(x_mem)
ax.set_xticklabels(names_mem, rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Add reduction annotations
for i, (o, n) in enumerate(zip(old_mem, opt_mem)):
    if o > n and o > 0:
        reduction = o / n
        if reduction >= 1.5:
            ax.annotate(f"{reduction:.1f}x\nless", xy=(i + width/2, n),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=7, fontweight="bold", color="#27ae60")

plt.tight_layout()
plt.savefig("report_python_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_python_comparison.png")

# ---------------------------------------------------------------
# Plot 3: Speedup factor bar chart (Old Python -> Optimized)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

speedups = []
func_names_speedup = []
for f in speed_funcs:
    o = old_py[f]["time"]
    n = opt_py[f]["time"]
    if n > 0:
        speedups.append(o / n)
        func_names_speedup.append(display_names.get(f, f))

# Sort by speedup
order = np.argsort(speedups)[::-1]
speedups = [speedups[i] for i in order]
func_names_speedup = [func_names_speedup[i] for i in order]

colors = ["#e74c3c" if s < 1 else "#2ecc71" for s in speedups]
bars = ax.barh(range(len(speedups)), speedups, color=colors, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(speedups)))
ax.set_yticklabels(func_names_speedup, fontsize=9)
ax.set_xscale("log")
ax.axvline(x=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xlabel("Speedup Factor (log scale)", fontsize=12)
ax.set_title("Python Optimization Speedup Factors", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Annotate bars
for i, (bar, val) in enumerate(zip(bars, speedups)):
    label = f"{val:.1f}x" if val < 10 else f"{val:.0f}x"
    ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2,
            label, va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("report_speedup_factors.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_speedup_factors.png")

# ---------------------------------------------------------------
# Plot 4: R vs Optimized Python - Speed
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

r_speed_funcs = [f for f in common_funcs_r_py if f in r_data and f in opt_py and "error" not in opt_py.get(f, {})]

names_r = [display_names.get(f, f) for f in r_speed_funcs]
r_times = [r_data[f]["time"] for f in r_speed_funcs]
py_opt_times = [opt_py[f]["time"] for f in r_speed_funcs]

x_r = np.arange(len(r_speed_funcs))

ax = axes[0]
bars1 = ax.bar(x_r - width/2, r_times, width, label="R (original)", color="#3498db", alpha=0.85)
bars2 = ax.bar(x_r + width/2, py_opt_times, width, label="Python (optimized)", color="#2ecc71", alpha=0.85)
ax.set_yscale("log")
ax.set_ylabel("Time (seconds, log scale)", fontsize=12)
ax.set_title("Speed: R (Original) vs Python (Optimized)", fontsize=14, fontweight="bold")
ax.set_xticks(x_r)
ax.set_xticklabels(names_r, rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Annotations
for i, (rt, pt) in enumerate(zip(r_times, py_opt_times)):
    if pt > 0 and rt > 0:
        ratio = rt / pt
        if ratio >= 1.5:
            ax.annotate(f"{ratio:.0f}x", xy=(i + width/2, pt),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=8, fontweight="bold", color="#27ae60")
        elif ratio < 0.67:
            ax.annotate(f"{1/ratio:.1f}x\nslower", xy=(i + width/2, pt),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=7, fontweight="bold", color="#e74c3c")

# Plot 5: R vs Optimized Python - Memory
# R memory is in MB, Python is in bytes
r_mem_funcs = [f for f in r_speed_funcs if "peak_mem_mb" in r_data[f] and "memory_bytes" in opt_py.get(f, {})]
names_r_mem = [display_names.get(f, f) for f in r_mem_funcs]
r_mem_kb = [r_data[f]["peak_mem_mb"] * 1024 for f in r_mem_funcs]
py_mem_kb = [opt_py[f]["memory_bytes"] / 1024 for f in r_mem_funcs]

x_rm = np.arange(len(r_mem_funcs))

ax = axes[1]
bars1 = ax.bar(x_rm - width/2, r_mem_kb, width, label="R (original)", color="#3498db", alpha=0.85)
bars2 = ax.bar(x_rm + width/2, py_mem_kb, width, label="Python (optimized)", color="#2ecc71", alpha=0.85)
ax.set_yscale("log")
ax.set_ylabel("Peak Memory (KB, log scale)", fontsize=12)
ax.set_title("Memory: R (Original) vs Python (Optimized)", fontsize=14, fontweight="bold")
ax.set_xticks(x_rm)
ax.set_xticklabels(names_r_mem, rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("report_r_vs_python.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_r_vs_python.png")

# ---------------------------------------------------------------
# Plot 6: R vs Python speedup factor chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

r_speedups = []
r_func_names = []
for f in r_speed_funcs:
    rt = r_data[f]["time"]
    pt = opt_py[f]["time"]
    if pt > 0 and rt > 0:
        r_speedups.append(rt / pt)
        r_func_names.append(display_names.get(f, f))

order = np.argsort(r_speedups)[::-1]
r_speedups = [r_speedups[i] for i in order]
r_func_names = [r_func_names[i] for i in order]

colors = ["#e74c3c" if s < 1 else "#2ecc71" for s in r_speedups]
bars = ax.barh(range(len(r_speedups)), r_speedups, color=colors, alpha=0.85, edgecolor="white")
ax.set_yticks(range(len(r_speedups)))
ax.set_yticklabels(r_func_names, fontsize=9)
ax.set_xscale("log")
ax.axvline(x=1, color="black", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xlabel("Speedup Factor: R / Optimized Python (log scale)", fontsize=12)
ax.set_title("R vs Optimized Python: Speedup Factors", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, r_speedups)):
    label = f"{val:.1f}x" if val < 10 else f"{val:.0f}x"
    if val >= 1:
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2,
                label + " faster", va="center", fontsize=9, fontweight="bold", color="#27ae60")
    else:
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2,
                f"{1/val:.1f}x slower", va="center", fontsize=9, fontweight="bold", color="#e74c3c")

plt.tight_layout()
plt.savefig("report_r_vs_python_speedup.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_r_vs_python_speedup.png")

# ---------------------------------------------------------------
# Plot 7: Three-way comparison (R, Old Python, Optimized Python)
# ---------------------------------------------------------------
three_way_funcs = [f for f in common_funcs_r_py
                   if f in r_data and f in old_py and f in opt_py
                   and "error" not in old_py.get(f, {}) and "error" not in opt_py.get(f, {})]

fig, ax = plt.subplots(figsize=(16, 7))

names_3 = [display_names.get(f, f) for f in three_way_funcs]
r_t = [r_data[f]["time"] for f in three_way_funcs]
old_t = [old_py[f]["time"] for f in three_way_funcs]
opt_t = [opt_py[f]["time"] for f in three_way_funcs]

x3 = np.arange(len(three_way_funcs))
w = 0.25

ax.bar(x3 - w, r_t, w, label="R (original)", color="#3498db", alpha=0.85)
ax.bar(x3, old_t, w, label="Python (old)", color="#e74c3c", alpha=0.85)
ax.bar(x3 + w, opt_t, w, label="Python (optimized)", color="#2ecc71", alpha=0.85)
ax.set_yscale("log")
ax.set_ylabel("Time (seconds, log scale)", fontsize=12)
ax.set_title("Three-Way Speed Comparison: R vs Old Python vs Optimized Python", fontsize=14, fontweight="bold")
ax.set_xticks(x3)
ax.set_xticklabels(names_3, rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("report_three_way_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: report_three_way_comparison.png")

print("\nAll plots generated successfully!")
