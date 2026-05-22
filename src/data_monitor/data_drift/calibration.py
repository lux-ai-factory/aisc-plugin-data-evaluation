"""Bootstrap-based threshold calibration.

Calibrates per-(metric, feature) univariate thresholds AND multivariate
(C2ST, MMD) thresholds from the reference dataset.

Modes:
  - "iid"   : random row shuffling (tabular data)
  - "block" : contiguous-block shuffling (time-series data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .metrics import METRICS
from .multivariate import c2st, mmd_rbf

Mode = Literal["iid", "block"]


@dataclass
class CalibratedThreshold:
    metric: str
    feature: str
    threshold: float
    null_mean: float
    null_std: float
    n_samples: int


@dataclass
class CalibratedThresholds:
    univariate: dict[tuple[str, str], CalibratedThreshold]
    c2st: float
    mmd: float


# ---------------------------------------------------------------------------
# Bootstrap splits
# ---------------------------------------------------------------------------


def _iid_split(n: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    perm = rng.permutation(n)
    h = n // 2
    return perm[:h].tolist(), perm[h : 2 * h].tolist()


def _block_split(
    n: int, block_size: int, rng: np.random.Generator
) -> tuple[list[int], list[int]]:
    n_blocks = n // block_size
    if n_blocks < 4:
        return _iid_split(n, rng)
    order = rng.permutation(n_blocks)
    half = n_blocks // 2
    a, b = [], []
    for i, blk in enumerate(order):
        idx = list(range(blk * block_size, (blk + 1) * block_size))
        (a if i < half else b).extend(idx)
    return a, b


def _block_size_default(n: int) -> int:
    return max(20, n // 30)


def _feature_types(
    df: pd.DataFrame,
    type_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Determine feature kind, preferring an explicit override over dtype inference."""
    overrides = type_overrides or {}
    out = {}
    for c in df.columns:
        if c in overrides:
            out[c] = overrides[c]
        else:
            out[c] = (
                "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"
            )
    return out


# ---------------------------------------------------------------------------
# Main calibration
# ---------------------------------------------------------------------------


CALIBRATION_MAX_N = 5000
DEFAULT_N_BOOT_MULTIVARIATE = 30  # multivariate calibration is the bottleneck;
# 30 gives a stable 95th percentile estimate
# combined with the safety buffer in the detector


def calibrate(
    ref: pd.DataFrame,
    mode: Mode = "iid",
    n_boot: int = 100,
    alpha: float = 0.05,
    block_size: int | None = None,
    random_state: int = 0,
    include_multivariate: bool = True,
    max_n: int = CALIBRATION_MAX_N,
    n_boot_multivariate: int | None = None,
    n_jobs: int = -1,
    feature_kinds: dict[str, str] | None = None,
) -> CalibratedThresholds:
    """Calibrate thresholds via bootstrap on the reference dataset.

    For each bootstrap iteration, split the reference into two halves
    (random rows in 'iid' mode, contiguous blocks in 'block' mode), compute
    every applicable metric, and accumulate a null distribution. The
    (1-alpha) quantile is the calibrated threshold.

    Args:
        ref: reference dataframe (assumed clean / no drift).
        mode: "iid" (random shuffle) or "block" (contiguous blocks).
        n_boot: number of bootstrap iterations (default 100).
        alpha: significance level (default 0.05 → 95th percentile).
        block_size: only used in "block" mode; default max(20, n//30).
        random_state: seed.
        include_multivariate: if False, only calibrate univariate.

    Returns:
        CalibratedThresholds with .univariate dict, .c2st, .mmd thresholds.
    """
    rng = np.random.default_rng(random_state)

    # Subsample reference if too large (calibration is O(n_boot × multivariate(n))
    # which becomes prohibitive on big datasets; threshold estimation only needs
    # a representative subsample).
    if len(ref) > max_n:
        if mode == "block":
            # Keep order intact (preserve autocorrelation), take a contiguous slice
            start = int(rng.integers(0, len(ref) - max_n))
            ref = ref.iloc[start : start + max_n].reset_index(drop=True)
        else:
            ref = ref.sample(n=max_n, random_state=random_state).reset_index(drop=True)

    n = len(ref)
    types = _feature_types(ref, type_overrides=feature_kinds)
    bs = block_size if block_size is not None else _block_size_default(n)

    # Precompute the split indices upfront (cheap, sequential, uses single rng for determinism)
    splits: list[tuple[list[int], list[int]]] = []
    for _ in range(n_boot):
        if mode == "iid":
            splits.append(_iid_split(n, rng))
        else:
            splits.append(_block_split(n, bs, rng))

    n_mv = min(
        n_boot_multivariate
        if n_boot_multivariate is not None
        else DEFAULT_N_BOOT_MULTIVARIATE,
        n_boot,
    )

    # ----- Univariate calibration (fast; sequential is fine) -----
    null_uni: dict[tuple[str, str], list[float]] = {}
    for metric_name, (_, supports) in METRICS.items():
        for feat, kind in types.items():
            if supports != kind and supports != "both":
                continue
            null_uni[(metric_name, feat)] = []

    for idx_a, idx_b in splits:
        a = ref.iloc[idx_a].reset_index(drop=True)
        b = ref.iloc[idx_b].reset_index(drop=True)
        for (metric_name, feat), bucket in null_uni.items():
            fn, _ = METRICS[metric_name]
            try:
                bucket.append(fn(a[feat], b[feat]).score)
            except Exception:
                continue

    # ----- Multivariate calibration (slow; parallelized, reduced count) -----
    null_c2st: list[float] = []
    null_mmd: list[float] = []

    if include_multivariate and n_mv > 0:
        from joblib import Parallel, delayed

        mv_splits = splits[:n_mv]

        def _one_iter(b_idx_split):
            b_idx, (ia, ib) = b_idx_split
            a_df = ref.iloc[ia].reset_index(drop=True)
            b_df = ref.iloc[ib].reset_index(drop=True)
            try:
                c_raw = c2st(a_df, b_df, random_state=random_state + b_idx).raw
            except Exception:
                c_raw = None
            try:
                m_raw = mmd_rbf(a_df, b_df, random_state=random_state + b_idx).raw
            except Exception:
                m_raw = None
            return c_raw, m_raw

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_iter)(s) for s in enumerate(mv_splits)
        )
        for c_raw, m_raw in results:
            if c_raw is not None:
                null_c2st.append(c_raw)
            if m_raw is not None:
                null_mmd.append(m_raw)

    # ----- Aggregate thresholds -----
    uni_thr: dict[tuple[str, str], CalibratedThreshold] = {}
    for (metric_name, feat), bucket in null_uni.items():
        if not bucket:
            uni_thr[(metric_name, feat)] = CalibratedThreshold(
                metric=metric_name,
                feature=feat,
                threshold=1.0,
                null_mean=0.0,
                null_std=0.0,
                n_samples=0,
            )
            continue
        arr = np.array(bucket)
        uni_thr[(metric_name, feat)] = CalibratedThreshold(
            metric=metric_name,
            feature=feat,
            threshold=float(np.quantile(arr, 1 - alpha)),
            null_mean=float(arr.mean()),
            null_std=float(arr.std()),
            n_samples=len(arr),
        )

    c2st_thr = float(np.quantile(null_c2st, 1 - alpha)) if null_c2st else 0.55
    mmd_thr = float(np.quantile(null_mmd, 1 - alpha)) if null_mmd else 0.0

    return CalibratedThresholds(
        univariate=uni_thr,
        c2st=c2st_thr,
        mmd=mmd_thr,
    )


def detect_autocorrelation(
    df: pd.DataFrame, threshold: float = 0.3
) -> tuple[bool, float]:
    """Lag-1 autocorrelation across numeric columns. Returns (is_ts, max_acf)."""
    max_acf = 0.0
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = df[c].dropna().to_numpy()
        if len(x) < 10:
            continue
        ac = np.corrcoef(x[:-1], x[1:])[0, 1]
        if np.isfinite(ac):
            max_acf = max(max_acf, abs(float(ac)))
    return max_acf > threshold, max_acf
