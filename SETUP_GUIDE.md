# Clonotrace Python -- Setup & Vignette Guide

This guide walks you through setting up a Python environment and running the
Clonotrace vignette notebook (`clonotrace_vignette.ipynb`) step by step.

No programming experience is assumed -- just follow each step in order.

---

## Prerequisites

You need **one** of these already installed on your computer:

| Tool | How to check | Install link |
|------|-------------|--------------|
| **Conda** (Anaconda or Miniconda) | Open Terminal, type `conda --version` | https://docs.conda.io/en/latest/miniconda.html |
| **VS Code** (optional, for running inside VS Code) | Open the app | https://code.visualstudio.com |

> **Which Terminal to use?**
> - **Mac**: open the built-in app called **Terminal** (search for it in Spotlight with Cmd+Space)
> - **Windows**: open **Anaconda Prompt** from the Start Menu

---

## Step 1 -- Create a conda environment

Open your Terminal and run these commands **one line at a time**.
Copy-paste each line, press Enter, and wait for it to finish before running the next.

```bash
# Create a new environment named "clonotrace" with Python 3.10
conda create -n clonotrace python=3.10 -y
```

Wait until you see `done`. Then activate it:

```bash
# Activate the environment (you must do this every time you open a new terminal)
conda activate clonotrace
```

Your terminal prompt should now show `(clonotrace)` at the beginning.

---

## Step 2 -- Install required packages

Still in the same terminal (with `(clonotrace)` showing), run:

```bash
# Install Jupyter kernel support (this is what was missing in the error you saw)
pip install ipykernel

# Register this environment as a Jupyter kernel
python -m ipykernel install --user --name clonotrace --display-name "Clonotrace (Python 3.10)"
```

Now install Clonotrace and all its dependencies:

```bash
# Navigate to the Clonotrace folder
cd /Users/yizhouw/Desktop/packages/Clonotrace_python

# Install Clonotrace in editable mode (so changes to the code take effect immediately)
pip install -e .
```

This installs numpy, scipy, pandas, scikit-learn, matplotlib, igraph, umap-learn,
and everything else the package needs. It typically takes 1-2 minutes.

---

## Step 3 -- Verify the installation

Run this quick check:

```bash
python -c "import clonotrace; print('Clonotrace installed successfully!')"
```

You should see: `Clonotrace installed successfully!`

If you see an error, re-read Step 2 and make sure every command finished without errors.

---

## Step 4 -- Open and run the notebook

You have two options:

### Option A: Run in VS Code (recommended if you already use VS Code)

1. Open VS Code
2. File > Open Folder > select the `Clonotrace_python` folder
3. Open `clonotrace_vignette.ipynb`
4. In the top-right corner of the notebook, click the **kernel picker** (it may say
   "Select Kernel" or show the current kernel name)
5. Choose **"Clonotrace (Python 3.10)"** from the list
   - If you don't see it, click "Select Another Kernel" > "Python Environments" > look for `clonotrace`
6. Click **"Run All"** (the double-play button at the top) to execute all cells

### Option B: Run in Jupyter Lab (browser-based)

```bash
# Make sure you're in the right folder and environment
cd /Users/yizhouw/Desktop/packages/Clonotrace_python
conda activate clonotrace

# Install and launch Jupyter Lab
pip install jupyterlab
jupyter lab
```

This opens a browser tab. Click on `clonotrace_vignette.ipynb` to open it, then
use **Run > Run All Cells** from the menu bar.

---

## What the notebook does

The notebook runs the full Clonotrace pipeline on a hematopoiesis dataset
(34,782 cells, 802 expanded clones). Each section produces plots:

| Section | What it does | Approx. time |
|---------|-------------|--------------|
| 1. Data overview | Cell type & cluster UMAPs | < 10 sec |
| 2. Clone label spreading | Bootstrap label propagation, deviance filter | ~30 sec |
| 3. Clone distance | Load pre-computed clone distances | < 5 sec |
| 4. Clone clustering | Louvain clustering + UMAP of clones | < 5 sec |
| 5. Pseudotime | Clone & cell-level pseudotime | < 5 sec |
| 6. Profile projection | Map clone profiles to individual cells | < 5 sec |
| 7. Co-embedding | Joint cell-clone UMAP | ~1 min |
| 8. Profile enrichment | Permutation test (1000 perms) | ~30 sec |
| 9. Differential expression | GAM-based DEG analysis | ~1 min |

**Total: ~3-4 minutes** on a modern laptop.

---

## Troubleshooting

### "No module named 'clonotrace'"
You are using the wrong kernel. Make sure you selected the **Clonotrace (Python 3.10)**
kernel (Step 4), not your base Python.

### "ModuleNotFoundError: No module named 'umap'" (or igraph, POT, etc.)
The `pip install -e .` step in Step 2 didn't complete. Re-run it:
```bash
conda activate clonotrace
cd /Users/yizhouw/Desktop/packages/Clonotrace_python
pip install -e .
```

### "Requires the ipykernel package"
Run this with the clonotrace environment active:
```bash
conda activate clonotrace
pip install ipykernel
python -m ipykernel install --user --name clonotrace --display-name "Clonotrace (Python 3.10)"
```
Then restart VS Code or Jupyter and select the kernel again.

### Plots don't show / blank figures
Make sure the first code cell containing `%matplotlib inline` runs successfully.
If using Jupyter Lab, try `%matplotlib widget` instead (requires `pip install ipympl`).

### Out of memory
The expression matrix is large (~25K genes x 35K cells). Close other applications.
You need at least 8 GB of RAM; 16 GB is recommended.

---

## For other users on a different machine

If someone else wants to run this on their own computer, they should:

1. Clone or copy the entire `Clonotrace_python` folder (including the `real_data/` subfolder)
2. Follow Steps 1-4 above, replacing the `cd` path with wherever they put the folder
3. Make sure the `real_data/` folder contains all 12 data files (csv, mtx, tsv, txt)

---

## Quick reference -- commands you'll use often

```bash
# Activate the environment (do this every time you open a new terminal)
conda activate clonotrace

# Go to the project folder
cd /Users/yizhouw/Desktop/packages/Clonotrace_python

# Launch Jupyter Lab
jupyter lab

# Update Clonotrace after code changes
pip install -e .
```
