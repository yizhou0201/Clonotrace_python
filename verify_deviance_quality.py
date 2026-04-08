"""Check deviance quality: is the float32 + tol=5e-3 deviance still useful?"""
import numpy as np
import pandas as pd
from clonotrace.auxiliary import embedding2knn, compute_transition
from clonotrace.label_propagation import label_spreading_bootstrap

# Load full data
pca = pd.read_csv("real_data/pca.csv", index_col=0).values
cell_meta = pd.read_csv("real_data/cell_meta.csv", index_col=0)
cell_knn = embedding2knn(pca, k=30)
cell_knn = compute_transition(cell_knn)

clone_size = cell_meta.groupby("clone").size().reset_index(name="count")
expanded = clone_size[clone_size["count"] >= 10]["clone"].values
cell_clone = cell_meta["clone"].copy()
cell_clone[~cell_clone.isin(expanded)] = np.nan

# Run with default settings (float32 + tol=5e-3 bootstrap)
np.random.seed(42)
result = label_spreading_bootstrap(cell_knn, cell_clone.values, alpha=0.6,
                                    sample_rate=0.8, sample_n=48, n_jobs=-1)

# The key use of deviance is: filter cells with deviance < threshold (e.g. 0.3)
# Check if the filtering is stable
dev = result["deviance"]
print(f"Deviance stats:")
print(f"  min={dev.min():.4f}, max={dev.max():.4f}, mean={dev.mean():.4f}, median={np.median(dev):.4f}")
print(f"  Cells with dev < 0.3: {(dev < 0.3).sum()} / {len(dev)}")
print(f"  Cells with dev < 0.2: {(dev < 0.2).sum()} / {len(dev)}")
print(f"  Cells with dev < 0.1: {(dev < 0.1).sum()} / {len(dev)}")
