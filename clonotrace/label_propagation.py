"""
label_propagation.py - Label propagation algorithms

Python port of R/label_propagation.R from Clonotrace_yuntian.
"""

import os
import tempfile
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

try:
    import h5py
except ImportError:
    h5py = None


def _prepare_labels(labels):
    """Convert labels to numeric format, return (labels_num, valid_mask).

    Matches R: as.numeric(as.factor(labels)).
    """
    from pandas import factorize, isna
    labels = np.asarray(labels, dtype=object)
    valid_mask = np.array([not isna(v) for v in labels])
    if valid_mask.any():
        codes, _ = factorize(labels[valid_mask], sort=True)
        labels_num = np.full(len(labels), np.nan)
        labels_num[valid_mask] = codes + 1  # 1-indexed
    else:
        labels_num = np.zeros(len(labels))
        valid_mask = np.zeros(len(labels), dtype=bool)
    return labels_num, valid_mask


def _build_Y(labels, valid_mask, N, C, epsilon=0):
    """Build initial label matrix Y (N, C)."""
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
    return Y


def _build_P(adj):
    """Row-normalize adjacency to transition matrix P."""
    adj = sp.csr_matrix(adj, dtype=float)
    degrees = np.array(adj.sum(axis=1)).ravel()
    degrees = np.maximum(degrees, 1e-6)
    D_inv = sp.diags(1.0 / degrees)
    return D_inv.dot(adj)


def _label_spreading_iterative(P, Y, alpha, max_iter=100, tol=1e-3):
    """Iterate F = α·P·F + (1-α)·Y to convergence.

    Works with batched Y of any width (N, C) or (N, C*B).
    scipy's sparse matmul uses BLAS internally, so wider matrices
    get better cache utilization and throughput.

    Parameters
    ----------
    P : sp.csr_matrix  (N, N) row-normalized transition matrix
    Y : np.ndarray  (N, K) initial label matrix (K = C or C*B)
    alpha : float
    max_iter : int
    tol : float

    Returns
    -------
    np.ndarray  (N, K) converged label probabilities
    """
    F = Y.copy()
    scale = 1 - alpha
    Y_scaled = scale * Y

    for it in range(1, max_iter + 1):
        F_new = alpha * P.dot(F) + Y_scaled
        diff = float(np.max(np.abs(F_new - F)))
        F = F_new
        if diff < tol:
            break

    return F


def label_spreading(adj, labels, label_n=None, alpha=0.9, max_iter=100,
                    tol=1e-3, epsilon=0, verbose=True):
    """Label propagation via iterative graph-based spreading.

    Parameters
    ----------
    adj : sp.spmatrix or np.ndarray  (N, N)
    labels : array-like  integer labels 1..C, NaN/None for unlabeled
    label_n : int or None  total number of classes
    alpha : float  propagation coefficient
    max_iter : int
    tol : float  convergence threshold
    epsilon : float  small prior for unlabeled
    verbose : bool

    Returns
    -------
    np.ndarray  (N, C) soft label probabilities
    """
    labels, valid_mask = _prepare_labels(labels)
    N = len(labels)
    C = int(label_n) if label_n is not None else int(np.nanmax(labels))

    Y = _build_Y(labels, valid_mask, N, C, epsilon)
    P = _build_P(adj)

    F = _label_spreading_iterative(P, Y, alpha, max_iter, tol)

    if verbose:
        print(f"Label spreading: {N} cells, {C} classes")

    return F


def label_spreading_bootstrap(adj, labels, refer=None, alpha=0.8,
                               sample_rate=0.8, sample_n=50, n_jobs=-1,
                               **kwargs):
    """Bootstrap stability estimation for label propagation.

    Three-level optimization:
    1. float32 arithmetic — 2× faster sparse-dense matmul, negligible loss
    2. Group batching (8 samples) — 2× BLAS throughput vs individual calls
    3. Thread parallelism — scipy releases GIL during sparse matmul,
       enabling ~3-4× speedup on multi-core machines

    Parameters
    ----------
    adj : sp.spmatrix
    labels : array-like  labels with NaN for unlabeled
    refer : np.ndarray or None  reference soft label matrix
    sample_rate : float  fraction of labeled nodes per bootstrap
    sample_n : int  number of bootstrap replicates
    n_jobs : int  number of threads (-1 = all cores, 1 = sequential)

    Returns
    -------
    dict with keys 'prob' (N, C) and 'deviance' (N,)
    """
    from concurrent.futures import ThreadPoolExecutor

    labels, valid_mask = _prepare_labels(labels)
    N = len(labels)
    C = int(np.nanmax(labels)) if valid_mask.any() else 0

    P = _build_P(adj)

    max_iter = kwargs.get("max_iter", 100)
    tol = kwargs.get("tol", 1e-3)

    # Compute reference solution in float64 (full precision)
    if refer is None:
        Y_ref = _build_Y(labels, valid_mask, N, C, epsilon=0)
        refer = _label_spreading_iterative(P, Y_ref, alpha, max_iter, tol)

    row_sums = np.maximum(refer.sum(axis=1, keepdims=True), 1e-12)
    refer_norm = refer / row_sums

    labeled_idx = np.where(valid_mask)[0]
    label_sample_count = max(1, round(len(labeled_idx) * sample_rate))
    full_label_flag = labels.copy()
    full_label_flag[np.isnan(full_label_flag)] = 0

    # float32 versions for bootstrap (2× faster sparse matmul)
    P32 = P.astype(np.float32)
    refer_norm_32 = refer_norm.astype(np.float32)

    # Pre-generate all bootstrap Y matrices (float32) and flags
    Y_boots = []
    all_flags = np.zeros((N, sample_n), dtype=float)
    for b in range(sample_n):
        sub_labels = np.full(N, np.nan)
        sampled = np.random.choice(labeled_idx, size=label_sample_count, replace=False)
        sub_labels[sampled] = labels[sampled]

        Y_b = _build_Y(sub_labels, np.isfinite(sub_labels), N, C, epsilon=0)
        Y_boots.append(Y_b.astype(np.float32))

        sample_flag = sub_labels.copy()
        sample_flag[np.isnan(sample_flag)] = 0
        all_flags[:, b] = (~np.logical_xor(
            full_label_flag.astype(bool), sample_flag.astype(bool)
        )).astype(float)

    # Build groups of 8 for BLAS efficiency
    batch_size = min(8, sample_n)
    groups = []
    for g_start in range(0, sample_n, batch_size):
        g_end = min(g_start + batch_size, sample_n)
        Y_group = np.column_stack(Y_boots[g_start:g_end])
        groups.append((g_start, g_end, Y_group))

    # Bootstrap tolerance can be relaxed (deviance is a stability measure)
    boot_tol = max(tol, 5e-3)

    def _process_group(Y_group):
        return _label_spreading_iterative(P32, Y_group, alpha, max_iter, boot_tol)

    # Determine thread count
    if n_jobs == -1:
        import os
        n_threads = min(os.cpu_count() or 1, len(groups))
    elif n_jobs == 1:
        n_threads = 1
    else:
        n_threads = min(abs(n_jobs), len(groups))

    # Process groups (threaded if n_jobs != 1; scipy releases GIL)
    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(_process_group, g[2]) for g in groups]
            F_groups = [f.result() for f in futures]
    else:
        F_groups = [_process_group(g[2]) for g in groups]

    # Compute deviance from all bootstrap results
    deviance_sum = np.zeros(N)
    flag_sum = np.zeros(N)
    for g_idx, (g_start, g_end, _) in enumerate(groups):
        F_group = F_groups[g_idx]
        for i, b in enumerate(range(g_start, g_end)):
            prob_mat = F_group[:, i * C:(i + 1) * C].astype(np.float64)
            prob_mat += 1e-12
            prob_mat /= prob_mat.sum(axis=1, keepdims=True)

            L1 = np.sum(np.abs(prob_mat - refer_norm), axis=1)
            deviance_sum += L1 * all_flags[:, b]
            flag_sum += all_flags[:, b]

    deviance_norm = deviance_sum / np.maximum(flag_sum, 1)
    return {"prob": refer_norm, "deviance": deviance_norm}


def create_hdf5_matrix(h5file, dataset="prob", nrow=None, ncol=None,
                        chunk_rows=4096, chunk_cols=128):
    """Pre-allocate a compressed HDF5 dataset for label spreading output."""
    if h5py is None:
        raise ImportError("h5py required: pip install h5py")
    if os.path.exists(h5file):
        os.remove(h5file)
    with h5py.File(h5file, "w") as f:
        chunk = (min(nrow, chunk_rows), min(ncol, chunk_cols))
        f.create_dataset(dataset, shape=(nrow, ncol), dtype="float64",
                         chunks=chunk, compression="gzip", compression_opts=7)
    return True


def normalize_hdf5_rows(h5file, dataset="prob", block_cols=256, add_eps=True):
    """Row-normalize an HDF5 probability matrix in column blocks.

    Returns
    -------
    np.ndarray  row sums before normalization
    """
    if h5py is None:
        raise ImportError("h5py required: pip install h5py")

    with h5py.File(h5file, "r") as f:
        N, C = f[dataset].shape

    col_blocks = [list(range(s, min(s + block_cols, C)))
                  for s in range(0, C, block_cols)]

    rowsum = np.zeros(N, dtype=float)
    with h5py.File(h5file, "r+") as f:
        for cols in col_blocks:
            X = f[dataset][:, cols[0]:cols[-1] + 1]
            if add_eps:
                X = X + 1e-12
            rowsum += X.sum(axis=1)
            f[dataset][:, cols[0]:cols[-1] + 1] = X

    rowsum[rowsum == 0] = 1.0

    with h5py.File(h5file, "r+") as f:
        for cols in col_blocks:
            X = f[dataset][:, cols[0]:cols[-1] + 1]
            X = X / rowsum[:, np.newaxis]
            f[dataset][:, cols[0]:cols[-1] + 1] = X

    return rowsum


def label_spreading_blocked(adj, labels, label_n=None, alpha=0.9,
                              max_iter=100, tol=1e-3, block_size=128,
                              outfile=None, verbose=True, epsilon=0):
    """Memory-efficient label spreading in class blocks with optional HDF5 output.

    Parameters
    ----------
    adj : sp.spmatrix  (N, N)
    labels : array-like  integer labels 1..C (no re-factoring); NaN = unlabeled
    label_n : int or None
    alpha : float
    max_iter : int
    tol : float
    block_size : int  classes per block
    outfile : str or None  HDF5 path; if None returns dense matrix
    verbose : bool
    epsilon : float

    Returns
    -------
    np.ndarray (N, C) if outfile is None, else str (outfile path)
    """
    if outfile is not None and h5py is None:
        raise ImportError("h5py required: pip install h5py")

    labs = np.array(labels, dtype=float)
    labs_int = np.where(np.isnan(labs), -1, labs.astype(int))
    N = len(labs_int)
    C = int(label_n) if label_n is not None else int(np.nanmax(labs))

    # Label weights
    valid = labs_int[labs_int > 0]
    cnt = np.bincount(valid, minlength=C + 1)[1:]  # 1..C
    wts = np.zeros(C)
    nz = cnt > 0
    wts[nz] = np.log2(cnt[nz]) / cnt[nz]

    labeled_idx = np.where(labs_int > 0)[0]

    # Row-normalize adjacency
    adj = sp.csr_matrix(adj, dtype=float)
    deg = np.array(adj.sum(axis=1)).ravel()
    deg[deg < 1e-12] = 1e-12
    adj = sp.diags(1.0 / deg).dot(adj)
    adj = sp.csr_matrix(adj)

    if outfile is None:
        Prob = np.full((N, C), np.nan)
    else:
        if os.path.exists(outfile):
            os.remove(outfile)
        with h5py.File(outfile, "w") as f:
            chunk = (min(N, 4096), min(C, block_size))
            f.create_dataset("prob", shape=(N, C), dtype="float64",
                             chunks=chunk, compression="gzip", compression_opts=7)

    cls_blocks = [list(range(s, min(s + block_size, C)))
                  for s in range(0, C, block_size)]

    for b_idx, cols in enumerate(cls_blocks):
        B = len(cols)
        # Build Yb
        Yb = np.full((N, B), epsilon / C)
        if len(labeled_idx):
            Yb[labeled_idx, :] = 0.0
            in_block = np.where((labs_int > 0) & np.isin(labs_int, [c + 1 for c in cols]))[0]
            if len(in_block):
                col_pos = np.array([cols.index(labs_int[r] - 1) for r in in_block])
                Yb[in_block, col_pos] = wts[[labs_int[r] - 1 for r in in_block]]

        F = np.zeros((N, B))
        conv = False
        for it in range(1, max_iter + 1):
            F_new = alpha * adj.dot(F) + (1 - alpha) * Yb
            diff = float(np.max(np.abs(F_new - F)))
            if verbose and (it % 10 == 0 or it == 1 or diff < tol):
                print(f"Block {b_idx + 1}/{len(cls_blocks)}, iter {it}: diff={diff:.3e}")
            F = F_new
            if diff < tol:
                conv = True
                break
        if verbose and not conv:
            print(f"Block {b_idx + 1} reached max_iter")

        if outfile is None:
            Prob[:, np.array(cols)] = F
        else:
            with h5py.File(outfile, "r+") as hf:
                hf["prob"][:, cols[0]:cols[-1] + 1] = F

    return Prob if outfile is None else outfile


def label_spreading_bootstrap_blocked(adj, labels, alpha=0.8,
                                       sample_rate=0.8, sample_n=50,
                                       block_size=128, tol=1e-3,
                                       max_iter=100, epsilon=0,
                                       refer_h5=None, tmpdir=None,
                                       verbose=True, n_jobs=-1):
    """Bootstrap label spreading with HDF5 reference and per-cell deviance.

    Returns
    -------
    dict with keys 'prob' (N, C) and 'deviance' (N,)
    """
    if h5py is None:
        raise ImportError("h5py required: pip install h5py")

    if tmpdir is None:
        tmpdir = tempfile.gettempdir()

    labs_full = np.array(labels, dtype=float)
    valid_mask = ~np.isnan(labs_full)
    from pandas import factorize
    if valid_mask.any():
        codes, _ = factorize(labs_full[valid_mask], sort=True)
        labs_full[valid_mask] = codes + 1

    labs_full_int = np.where(np.isnan(labs_full), -1, labs_full.astype(int))
    N = len(labs_full_int)
    C = int(np.nanmax(labs_full))

    # Build or reuse reference
    if refer_h5 is None:
        import random
        import string
        rand_str = "".join(random.choices(string.ascii_lowercase, k=6))
        refer_h5 = os.path.join(tmpdir, f"refer_{rand_str}.h5")
        if verbose:
            print("Computing reference...")
        label_spreading_blocked(
            adj, labs_full_int, label_n=C, alpha=alpha,
            max_iter=max_iter, tol=tol, block_size=block_size,
            outfile=refer_h5, verbose=False, epsilon=epsilon,
        )
        normalize_hdf5_rows(refer_h5, "prob", block_cols=block_size, add_eps=True)
    elif verbose:
        print(f"Using provided reference: {refer_h5}")

    labeled_idx_all = np.where(labs_full_int > 0)[0]
    full_flag = labs_full_int.copy()
    full_flag[full_flag < 0] = 0

    def worker(i):
        subs = np.full(N, -1, dtype=int)
        if len(labeled_idx_all):
            take = max(1, round(len(labeled_idx_all) * sample_rate))
            idx = np.random.choice(labeled_idx_all, size=take, replace=False)
            subs[idx] = labs_full_int[idx]

        samp_flag = subs.copy()
        samp_flag[samp_flag < 0] = 0
        same_flag = (~np.logical_xor(full_flag.astype(bool), samp_flag.astype(bool))).astype(int)

        import random, string
        rand_str = "".join(random.choices(string.ascii_lowercase, k=4))
        tmp_h5 = os.path.join(tmpdir, f"boot_{i:04d}_{rand_str}.h5")
        create_hdf5_matrix(tmp_h5, "prob", nrow=N, ncol=C,
                           chunk_rows=4096, chunk_cols=block_size)
        label_spreading_blocked(
            adj, subs.astype(float), label_n=C, alpha=alpha,
            max_iter=max_iter, tol=tol, block_size=block_size,
            outfile=tmp_h5, verbose=False, epsilon=epsilon,
        )
        normalize_hdf5_rows(tmp_h5, "prob", block_cols=block_size, add_eps=True)

        cls_blocks = [list(range(s, min(s + block_size, C)))
                      for s in range(0, C, block_size)]
        dev = np.zeros(N)
        with h5py.File(tmp_h5, "r") as tf, h5py.File(refer_h5, "r") as rf:
            for cols in cls_blocks:
                Fb = tf["prob"][:, cols[0]:cols[-1] + 1]
                Rb = rf["prob"][:, cols[0]:cols[-1] + 1]
                dev += np.sum(np.abs(Fb - Rb), axis=1)

        try:
            os.unlink(tmp_h5)
        except OSError:
            pass
        return dev * same_flag, same_flag

    results = Parallel(n_jobs=n_jobs)(delayed(worker)(i) for i in range(sample_n))

    dev_sum = sum(r[0] for r in results)
    flag_sum = sum(r[1] for r in results)
    deviance_norm = dev_sum / np.maximum(1, flag_sum)

    with h5py.File(refer_h5, "r") as rf:
        prob = rf["prob"][:]

    try:
        os.unlink(refer_h5)
    except OSError:
        pass

    return {"prob": prob, "deviance": deviance_norm}
