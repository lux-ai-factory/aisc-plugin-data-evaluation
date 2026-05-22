"""Multivariate drift detectors: C2ST and MMD on the joint feature vector.

MMD has two implementations selected automatically based on data size:
  - n ≤ 10,000 (per side):  exact O(n²) MMD with RBF kernel
  - n > 10,000:             Random Fourier Features (RFF) approximation, linear-time
                            Rahimi & Recht 2007. Statistically equivalent to exact
                            in the asymptotic regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Above this per-side sample count, switch from exact O(n²) MMD to RFF
MMD_RFF_THRESHOLD = 10_000


@dataclass
class MultivariateResult:
    method: str
    raw: float  # natural-scale stat (AUC for C2ST, MMD² for MMD)
    score: float  # [0,1] severity
    flag: bool
    p_value: float | None = None


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode(ref: pd.DataFrame, ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One-hot categoricals + z-score numeric. Returns ref/eval encoded matrices."""
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = [c for c in ref.columns if not pd.api.types.is_numeric_dtype(ref[c])]
    num_cols = [c for c in ref.columns if pd.api.types.is_numeric_dtype(ref[c])]

    combined = pd.concat([ref, ev], axis=0, ignore_index=True)
    parts: list[np.ndarray] = []

    if num_cols:
        scaler = StandardScaler()
        parts.append(scaler.fit_transform(combined[num_cols].to_numpy()))
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        parts.append(ohe.fit_transform(combined[cat_cols].astype(str).to_numpy()))

    if not parts:
        return np.zeros((len(ref), 0)), np.zeros((len(ev), 0))

    X = np.concatenate(parts, axis=1)
    return X[: len(ref)], X[len(ref) :]


# ---------------------------------------------------------------------------
# C2ST — classifier two-sample test
# ---------------------------------------------------------------------------


def c2st(
    ref: pd.DataFrame,
    ev: pd.DataFrame,
    n_estimators: int = 100,
    cv: int = 5,
    max_n: int = 5000,
    random_state: int = 0,
) -> MultivariateResult:
    """Train a RandomForest to discriminate ref vs eval; report cross-validated AUC.

    Each side capped at `max_n` samples for tractable training time.
    """
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    if len(ref) > max_n:
        ref = ref.sample(n=max_n, random_state=random_state)
    if len(ev) > max_n:
        ev = ev.sample(n=max_n, random_state=random_state)

    X_ref, X_ev = _encode(ref, ev)
    if X_ref.shape[1] == 0:
        return MultivariateResult("c2st", 0.5, 0.0, False)

    X = np.vstack([X_ref, X_ev])
    y = np.concatenate([np.zeros(len(X_ref)), np.ones(len(X_ev))])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        aucs = cross_val_score(clf, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    except Exception:
        return MultivariateResult("c2st", 0.5, 0.0, False)
    auc = float(np.mean(aucs))
    score = max(0.0, 2.0 * (auc - 0.5))  # AUC 0.5 → 0, 1.0 → 1

    # crude p-value via normal approx (used only as a fallback / sanity)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    se = np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))
    z = (auc - 0.5) / max(se, 1e-9)
    p = float(1 - stats.norm.cdf(z))

    return MultivariateResult(
        method="c2st",
        raw=auc,
        score=score,
        flag=(auc > 0.55) and (p < 0.05),  # default fallback; calibration may override
        p_value=p,
    )


# ---------------------------------------------------------------------------
# MMD — RBF kernel + permutation p-value
# ---------------------------------------------------------------------------


def _mmd2_from_K(K_xx: np.ndarray, K_yy: np.ndarray, K_xy: np.ndarray) -> float:
    """Unbiased MMD² estimator (Gretton et al. 2012)."""
    n = K_xx.shape[0]
    m = K_yy.shape[0]
    sum_xx = K_xx.sum() - np.trace(K_xx)
    sum_yy = K_yy.sum() - np.trace(K_yy)
    return sum_xx / (n * (n - 1)) + sum_yy / (m * (m - 1)) - 2.0 * K_xy.mean()


def _mmd_exact(
    X_ref: np.ndarray,
    X_ev: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Exact O(n²) RBF MMD² + permutation p-value. Returns (observed, p, null_vals)."""
    from scipy.spatial.distance import pdist
    from sklearn.metrics.pairwise import rbf_kernel

    combined = np.vstack([X_ref, X_ev])
    sq_dists = pdist(combined, metric="sqeuclidean")
    med = float(np.median(sq_dists))
    gamma = 1.0 / max(med, 1e-9)

    K_xx = rbf_kernel(X_ref, X_ref, gamma=gamma)
    K_yy = rbf_kernel(X_ev, X_ev, gamma=gamma)
    K_xy = rbf_kernel(X_ref, X_ev, gamma=gamma)
    observed = _mmd2_from_K(K_xx, K_yy, K_xy)

    n = len(X_ref)
    K_all = rbf_kernel(combined, combined, gamma=gamma)
    null_vals = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        idx_x, idx_y = perm[:n], perm[n:]
        null_vals.append(
            _mmd2_from_K(
                K_all[np.ix_(idx_x, idx_x)],
                K_all[np.ix_(idx_y, idx_y)],
                K_all[np.ix_(idx_x, idx_y)],
            )
        )
    null_vals = np.array(null_vals)
    p = float((null_vals >= observed).mean()) if len(null_vals) else 1.0
    return float(observed), p, null_vals


def _mmd_rff(
    X_ref: np.ndarray,
    X_ev: np.ndarray,
    n_permutations: int,
    n_features_rff: int,
    gamma_subsample: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Linear-time RBF-MMD² via Random Fourier Features (Rahimi & Recht 2007)."""
    from scipy.spatial.distance import pdist

    d_in = X_ref.shape[1]
    D = n_features_rff

    # Median heuristic on pooled subsample
    sub_a = rng.choice(len(X_ref), min(gamma_subsample, len(X_ref)), replace=False)
    sub_b = rng.choice(len(X_ev), min(gamma_subsample, len(X_ev)), replace=False)
    sub = np.vstack([X_ref[sub_a], X_ev[sub_b]])
    sq = pdist(sub, metric="sqeuclidean")
    med = float(np.median(sq))
    gamma = 1.0 / max(med, 1e-9)

    # RFF weights
    W = rng.standard_normal((D, d_in)) * np.sqrt(2.0 * gamma)
    b = rng.uniform(0.0, 2.0 * np.pi, size=D)

    def _phi(X):
        return np.sqrt(2.0 / D) * np.cos(X @ W.T + b)

    Z_ref = _phi(X_ref)
    Z_ev = _phi(X_ev)

    observed = float(np.sum((Z_ref.mean(0) - Z_ev.mean(0)) ** 2))

    Z_all = np.vstack([Z_ref, Z_ev])
    n = len(Z_ref)
    null_vals = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(Z_all))
        null_vals.append(
            float(np.sum((Z_all[perm[:n]].mean(0) - Z_all[perm[n:]].mean(0)) ** 2))
        )
    null_vals = np.array(null_vals)
    p = float((null_vals >= observed).mean()) if len(null_vals) else 1.0
    return observed, p, null_vals


def mmd_rbf(
    ref: pd.DataFrame,
    ev: pd.DataFrame,
    n_permutations: int = 100,
    max_n: int = 1000,
    rff_max_n: int = 5000,
    n_features_rff: int = 500,
    rff_threshold: int = MMD_RFF_THRESHOLD,
    random_state: int = 0,
) -> MultivariateResult:
    """RBF-kernel MMD with adaptive implementation:

    - If min(len(ref), len(ev)) ≤ rff_threshold:
        exact O(n²) MMD on a sample of up to `max_n` rows per side.
    - Else:
        Random Fourier Features approximation. Subsamples each side to
        `rff_max_n` rows first so the permutation loop stays cheap; this
        does not hurt statistical accuracy at the percentile level we use.

    Both return MMD² with permutation p-value. Same kernel (RBF), same
    median-heuristic bandwidth, statistically equivalent in the asymptotic regime.
    """
    X_ref, X_ev = _encode(ref, ev)
    if X_ref.shape[1] == 0:
        return MultivariateResult("mmd", 0.0, 0.0, False)

    rng = np.random.default_rng(random_state)
    use_rff = min(len(X_ref), len(X_ev)) > rff_threshold

    if use_rff:
        # Subsample BEFORE RFF projection so the permutation loop is cheap.
        # With rff_max_n=5000, Z_all is ~10k × 500 floats = ~40 MB — very fast.
        if len(X_ref) > rff_max_n:
            X_ref = X_ref[rng.choice(len(X_ref), rff_max_n, replace=False)]
        if len(X_ev) > rff_max_n:
            X_ev = X_ev[rng.choice(len(X_ev), rff_max_n, replace=False)]
        observed, p, null_vals = _mmd_rff(
            X_ref,
            X_ev,
            n_permutations=n_permutations,
            n_features_rff=n_features_rff,
            gamma_subsample=1000,
            rng=rng,
        )
        method_name = "mmd_rff"
    else:
        # Cap exact MMD at max_n to keep O(n²) memory tractable
        if len(X_ref) > max_n:
            X_ref = X_ref[rng.choice(len(X_ref), max_n, replace=False)]
        if len(X_ev) > max_n:
            X_ev = X_ev[rng.choice(len(X_ev), max_n, replace=False)]
        observed, p, null_vals = _mmd_exact(X_ref, X_ev, n_permutations, rng)
        method_name = "mmd"

    denom = max(float(np.quantile(null_vals, 0.99)), 1e-12) if len(null_vals) else 1.0
    score = float(min(max(observed / denom, 0.0), 1.0)) if observed > 0 else 0.0

    return MultivariateResult(
        method=method_name,
        raw=float(observed),
        score=score,
        flag=p < 0.05,
        p_value=p,
    )


MULTIVARIATE = {
    "c2st": c2st,
    "mmd": mmd_rbf,
}
