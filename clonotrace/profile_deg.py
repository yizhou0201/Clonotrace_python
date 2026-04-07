"""
profile_deg.py - Differential expression analysis

Python port of R/profile_DEG.R from Clonotrace_yuntian.
"""

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

try:
    from sklearn.preprocessing import SplineTransformer
except ImportError:
    SplineTransformer = None


def bin_filter_profile_mass(mass, time, thresh=5, binsize=0.005):
    """Filter cells by profile mass within pseudotime bins.

    Parameters
    ----------
    mass : np.ndarray  (N, 2)  target and control mass
    time : np.ndarray  (N,)  pseudotime values
    thresh : float  minimum total mass per bin
    binsize : float  bin width

    Returns
    -------
    np.ndarray  bool mask (N,)
    """
    mass = np.asarray(mass)
    time = np.asarray(time)
    assert mass.shape[1] == 2, "mass must have exactly 2 columns"
    assert len(mass) == len(time), "mass and time length mismatch"

    df = pd.DataFrame({
        "target": mass[:, 0],
        "control": mass[:, 1],
        "time": time,
    })
    if len(time) == 0:
        return np.zeros(0, dtype=bool)
    breaks = np.arange(time.min(), time.max() + binsize, binsize)
    df["bin"] = pd.cut(df["time"], bins=breaks, include_lowest=True)
    summary = (df.groupby("bin")[["target", "control"]]
               .sum()
               .reset_index())
    keep_bins = summary[(summary["target"] > thresh) & (summary["control"] > thresh)]["bin"]
    return df["bin"].isin(keep_bins).values


def ridge_regression(X, G, lam=1e-4):
    """Ridge regression: beta = (X'X + lambda*I)^{-1} X'G.

    Parameters
    ----------
    X : np.ndarray  (N, d)  design matrix
    G : np.ndarray  (N, genes) or (N,)  response
    lam : float  ridge penalty

    Returns
    -------
    dict with keys 'coef' (d, genes) and 'rss' (genes,)
    """
    X = np.asarray(X, dtype=float)
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G[:, np.newaxis]

    d = X.shape[1]
    XtX = X.T @ X + lam * np.eye(d)
    XtY = X.T @ G
    beta = np.linalg.solve(XtX, XtY)
    G_hat = X @ beta
    rss = np.sum((G - G_hat) ** 2, axis=0)
    return {"coef": beta, "rss": rss}


def _natural_spline_basis(t, df=5, include_intercept=True):
    """Approximate natural spline basis using SplineTransformer.

    Parameters
    ----------
    t : np.ndarray  (N,)
    df : int  degrees of freedom (number of basis functions excluding intercept)
    include_intercept : bool

    Returns
    -------
    np.ndarray  (N, df+1) if include_intercept else (N, df)
    """
    if SplineTransformer is None:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    t = np.asarray(t).reshape(-1, 1)
    n_knots = max(df - 1, 2)
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=3,
        include_bias=False,
        knots="quantile",
    )
    B = transformer.fit_transform(t)
    if include_intercept:
        B = np.hstack([np.ones((len(t), 1)), B])
    return B


def soft_cluster_gam_fit(G, t, P, df=5, lam=1e-4, test="F"):
    """Fit soft cluster-specific GAMs with ridge regularization.

    Parameters
    ----------
    G : np.ndarray  (genes, N)  expression matrix
    t : np.ndarray  (N,)  pseudotime
    P : np.ndarray  (N, m)  soft cluster assignments (rows sum to 1)
    df : int  spline degrees of freedom
    lam : float  ridge penalty
    test : 'F' or 'LRT'

    Returns
    -------
    dict with keys stat, df, coef, design_null, design_full
    """
    G = np.asarray(G, dtype=float)
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)

    n = G.shape[1]
    m = P.shape[1]

    # Spline basis: (N, d)
    B = _natural_spline_basis(t, df=df, include_intercept=True)
    d = B.shape[1]

    # Full model: B * P interaction => (N, m*d)
    X_full = np.column_stack([B * P[:, k:k+1] for k in range(m)])
    X_null = B

    null = ridge_regression(X_null, G.T, lam=lam)
    full = ridge_regression(X_full, G.T, lam=lam)

    df_null = d
    df_full = m * d
    df1 = df_full - df_null
    df2 = n - df_full

    if test == "F":
        F_stat = ((null["rss"] - full["rss"]) / df1) / (full["rss"] / max(df2, 1))
        pval = stats.f.sf(F_stat, df1, max(df2, 1))
        stat_vals = F_stat
    else:  # LRT
        logLik_null = -0.5 * n * np.log(np.maximum(null["rss"], 1e-300))
        logLik_full = -0.5 * n * np.log(np.maximum(full["rss"], 1e-300))
        LRT_stat = 2 * (logLik_full - logLik_null)
        pval = stats.chi2.sf(LRT_stat, df1)
        stat_vals = LRT_stat

    # Effect size (d = actual number of spline basis columns)
    target_pred = X_null @ full["coef"][:d, :]
    control_pred = X_null @ full["coef"][d:2 * d, :]
    sigma = np.sqrt(null["rss"] / max(X_null.shape[0], 1))
    mean_diff = np.mean(target_pred - control_pred, axis=0)
    cohen = mean_diff / np.maximum(sigma, 1e-12)

    stat_df = pd.DataFrame({
        "stat": stat_vals,
        "pval": pval,
        "mean_diff": mean_diff,
        "cohen": cohen,
    })

    return {
        "stat": stat_df,
        "df": [df1, df2],
        "coef": full["coef"],
        "design_null": X_null,
        "design_full": X_full,
    }


def cluster_profile_mass(cell_profile_prob, cluster_label):
    """Aggregate soft profile probabilities by cluster.

    Parameters
    ----------
    cell_profile_prob : np.ndarray or pd.DataFrame  (N, profiles)
    cluster_label : array-like  length N

    Returns
    -------
    np.ndarray  (n_clusters, profiles)
    """
    if not isinstance(cell_profile_prob, pd.DataFrame):
        cell_profile_prob = pd.DataFrame(cell_profile_prob)
    cell_profile_prob = cell_profile_prob.copy()
    cell_profile_prob["cluster"] = cluster_label
    result = cell_profile_prob.groupby("cluster").sum()
    return result.values


def cluster_profile_enrich(cell_profile_prob, cluster_label, permute_n=300):
    """Permutation-based enrichment test of profiles within clusters.

    Returns
    -------
    dict with keys 'prob' and 'pval' (both np.ndarray)
    """
    import numpy as np

    cell_profile_prob = np.asarray(cell_profile_prob)
    cluster_label = np.asarray(cluster_label)
    truth = cluster_profile_mass(cell_profile_prob, cluster_label)

    permute_stack = np.stack([
        cluster_profile_mass(
            cell_profile_prob,
            np.random.permutation(cluster_label)
        )
        for _ in range(permute_n)
    ], axis=2)

    pval = np.zeros_like(truth)
    for i in range(truth.shape[0]):
        for j in range(truth.shape[1]):
            pval[i, j] = 1 - np.sum(truth[i, j] > permute_stack[i, j, :]) / permute_n

    return {"prob": truth, "pval": pval}


def profile_cluster_DEG_permute(P, G, dpt, n=50):
    """Permutation null distribution for profile-based DE.

    Returns
    -------
    list of np.ndarray  p-values for each permutation
    """
    null_p = []
    for _ in range(n):
        P_perm = P[np.random.permutation(len(P))]
        P_perm = np.apply_along_axis(np.random.permutation, 1, P_perm)
        result = soft_cluster_gam_fit(G, dpt, P_perm, df=1, lam=1e-4)
        null_p.append(result["stat"]["pval"].values)
    return null_p


def profile_cluster_DEG(profile, cluster, exprs, cell_meta, cell_profile_prob,
                         cluster_col="cluster", pseudotime_col="cell_t", permute_n=50):
    """DE between profile and background in a single cluster.

    Parameters
    ----------
    profile : str or list  profile column names to test
    cluster : str or int  cluster label
    exprs : np.ndarray or pd.DataFrame  (genes, cells)
    cell_meta : pd.DataFrame  cell metadata
    cell_profile_prob : pd.DataFrame or np.ndarray  (cells, profiles)
    cluster_col : str
    pseudotime_col : str
    permute_n : int

    Returns
    -------
    dict or None  with keys stat, cell, design_null, design_full, coef, df
    """
    if not isinstance(cell_meta, pd.DataFrame):
        cell_meta = pd.DataFrame(cell_meta)
    if not isinstance(cell_profile_prob, pd.DataFrame):
        cell_profile_prob = pd.DataFrame(cell_profile_prob)

    sub_meta = cell_meta.loc[cell_profile_prob.index]
    sub_meta = sub_meta[sub_meta[cluster_col] == cluster]

    P = cell_profile_prob.loc[sub_meta.index]
    if isinstance(profile, str):
        profile = [profile]
    target = P[profile].sum(axis=1).values
    control = (P.sum(axis=1) - P[profile].sum(axis=1)).values
    P_arr = np.column_stack([target, control])

    flag = bin_filter_profile_mass(P_arr, sub_meta[pseudotime_col].values,
                                    thresh=3, binsize=0.005)
    if flag.sum() < 5:
        return None

    sub_meta = sub_meta.iloc[flag]
    P_arr = P_arr[flag]

    if isinstance(exprs, pd.DataFrame):
        G = exprs[sub_meta.index].values
        gene_names = exprs.index
    else:
        G = np.asarray(exprs)[:, flag]
        gene_names = None

    nonzero_mask = (G != 0).sum(axis=1) > 30
    G = G[nonzero_mask]
    if gene_names is not None:
        gene_names = gene_names[nonzero_mask]

    p_null = np.concatenate(profile_cluster_DEG_permute(
        P_arr, G, sub_meta[pseudotime_col].values, n=permute_n
    ))

    result = soft_cluster_gam_fit(G, sub_meta[pseudotime_col].values, P_arr,
                                   df=1, lam=1e-4)

    # Restore gene names as index
    if gene_names is not None:
        result["stat"].index = gene_names

    result["stat"]["pcali"] = result["stat"]["pval"].apply(
        lambda p: float(np.mean(p_null <= p))
    )

    if multipletests is not None:
        _, padj, _, _ = multipletests(result["stat"]["pcali"].values, method="fdr_bh")
    else:
        padj = np.minimum(result["stat"]["pcali"].values * len(result["stat"]), 1.0)

    result["stat"]["padj"] = padj
    result["cell"] = sub_meta.index.tolist()
    return result


def profile_multiclusters_DEG(profile, exprs, cell_meta, cell_profile_prob,
                                clusters=None, cluster_col="cluster",
                                pseudotime_col="cell_t", mass_thresh=100):
    """Profile-specific DE across multiple clusters.

    Returns
    -------
    dict  {cluster: DEG result}
    """
    if not isinstance(cell_meta, pd.DataFrame):
        cell_meta = pd.DataFrame(cell_meta)
    if not isinstance(cell_profile_prob, pd.DataFrame):
        cell_profile_prob = pd.DataFrame(cell_profile_prob)

    if isinstance(profile, (int, str)):
        profile = [str(profile)]
    else:
        profile = [str(p) for p in profile]

    mass = cluster_profile_mass(
        cell_profile_prob[profile].values,
        cell_meta.loc[cell_profile_prob.index, cluster_col].values
    )

    cluster_ids = np.unique(cell_meta.loc[cell_profile_prob.index, cluster_col])
    if clusters is None:
        clusters = cluster_ids.tolist()
    clusters = [str(c) for c in clusters]

    def _run(cl):
        cl_idx = list(cluster_ids).index(cl) if cl in cluster_ids else -1
        if cl_idx < 0:
            return cl, None
        cl_mass = mass[cl_idx, :]
        if cl_mass.sum() > mass_thresh:
            result = profile_cluster_DEG(
                profile, cl, exprs, cell_meta, cell_profile_prob,
                cluster_col, pseudotime_col
            )
        else:
            result = None
        return cl, result

    results = Parallel(n_jobs=1)(delayed(_run)(cl) for cl in clusters)
    return {cl: r for cl, r in results if r is not None}
