"""Candidate per-feature drift metrics.

Each metric is a callable
    f(ref: pd.Series, eval: pd.Series) -> MetricResult
returning:
    raw    : float       — the raw statistic on its natural scale
    score  : float[0,1]  — normalized severity, used for decisions / aggregation
    flag   : bool        — drift detected at the metric's natural threshold
    p_value: float | None — only if the metric is a hypothesis test

All metrics drop NaNs in both series before computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

Supports = Literal["numeric", "categorical", "both"]


@dataclass
class MetricResult:
    raw: float
    score: float
    flag: bool
    p_value: float | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clean_numeric(ref: pd.Series, ev: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    r = pd.to_numeric(ref, errors="coerce").dropna().to_numpy()
    e = pd.to_numeric(ev, errors="coerce").dropna().to_numpy()
    return r, e


def _clean_categorical(
    ref: pd.Series, ev: pd.Series
) -> tuple[pd.Series, pd.Series, list[str]]:
    r = ref.dropna().astype(str)
    e = ev.dropna().astype(str)
    levels = sorted(set(r.unique()).union(e.unique()))
    return r, e, levels


def _binned_probs(
    x: np.ndarray, bin_edges: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    counts, _ = np.histogram(x, bins=bin_edges)
    p = counts / max(counts.sum(), 1)
    p = p + eps
    return p / p.sum()


def _clip_score(x: float, denom: float) -> float:
    if not np.isfinite(x):
        return 1.0
    return float(min(max(x, 0.0) / denom, 1.0))


def _safe_p_to_score(p: float | None) -> float:
    if p is None or not np.isfinite(p):
        return 0.0
    return float(1.0 - p)


# ---------------------------------------------------------------------------
# Numeric metrics
# ---------------------------------------------------------------------------


def psi_numeric(ref: pd.Series, ev: pd.Series, n_bins: int = 10) -> MetricResult:
    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    edges = np.histogram(r, bins=n_bins)[1]
    p = _binned_probs(r, edges)
    q = _binned_probs(e, edges)
    raw = float(np.sum((p - q) * np.log(p / q)))
    return MetricResult(
        raw=raw,
        score=_clip_score(raw, denom=0.25),
        flag=raw > 0.1,
    )


def ks_numeric(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    stat, p = stats.ks_2samp(r, e, method="asymp")
    return MetricResult(
        raw=float(stat), score=float(stat), flag=p < 0.05, p_value=float(p)
    )


def wasserstein_numeric(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    raw = float(stats.wasserstein_distance(r, e))
    sigma = float(np.std(r) + 1e-9)
    score = _clip_score(raw / sigma, denom=1.0)
    return MetricResult(raw=raw, score=score, flag=score > 0.25)


def jsd_numeric(ref: pd.Series, ev: pd.Series, n_bins: int = 10) -> MetricResult:
    from scipy.spatial.distance import jensenshannon

    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    edges = np.histogram(r, bins=n_bins)[1]
    p = _binned_probs(r, edges)
    q = _binned_probs(e, edges)
    raw = float(jensenshannon(p, q, base=2) ** 2)  # squared → [0,1]
    return MetricResult(raw=raw, score=raw, flag=raw > 0.1)


def smd_numeric(ref: pd.Series, ev: pd.Series) -> MetricResult:
    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    sigma = float(np.std(r) + 1e-9)
    raw = float(abs(np.mean(r) - np.mean(e)) / sigma)
    return MetricResult(raw=raw, score=_clip_score(raw, 0.25), flag=raw > 0.1)


def welch_t(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    stat, p = stats.ttest_ind(r, e, equal_var=False)
    return MetricResult(
        raw=float(stat),
        score=_safe_p_to_score(p),
        flag=p < 0.05,
        p_value=float(p),
    )


def mannwhitney(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    stat, p = stats.mannwhitneyu(r, e, alternative="two-sided")
    return MetricResult(
        raw=float(stat),
        score=_safe_p_to_score(p),
        flag=p < 0.05,
        p_value=float(p),
    )


def levene(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    stat, p = stats.levene(r, e, center="median")
    return MetricResult(
        raw=float(stat),
        score=_safe_p_to_score(p),
        flag=p < 0.05,
        p_value=float(p),
    )


def anderson_darling(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e = _clean_numeric(ref, ev)
    if len(r) < 2 or len(e) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    try:
        res = stats.anderson_ksamp([r, e], variant="midrank")
        stat = float(res.statistic)
        # p_value attribute is `significance_level` (0-25 percent). Use pvalue if present.
        p = getattr(res, "pvalue", None)
        if p is None:
            p = float(res.significance_level) / 100.0
        else:
            p = float(p)
    except Exception:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    return MetricResult(
        raw=stat,
        score=_safe_p_to_score(p),
        flag=p < 0.05,
        p_value=p,
    )


# ---------------------------------------------------------------------------
# Categorical metrics
# ---------------------------------------------------------------------------


def _category_probs(
    ref: pd.Series, ev: pd.Series, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    r, e, levels = _clean_categorical(ref, ev)
    rc = r.value_counts().reindex(levels, fill_value=0).to_numpy().astype(float)
    ec = e.value_counts().reindex(levels, fill_value=0).to_numpy().astype(float)
    p = (rc + eps) / (rc.sum() + eps * len(levels))
    q = (ec + eps) / (ec.sum() + eps * len(levels))
    return p, q, levels


def chi2_cat(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e, levels = _clean_categorical(ref, ev)
    rc = r.value_counts().reindex(levels, fill_value=0).to_numpy()
    ec = e.value_counts().reindex(levels, fill_value=0).to_numpy()
    if rc.sum() == 0 or ec.sum() == 0 or len(levels) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    contingency = np.array([rc, ec])
    try:
        stat, p, _, _ = stats.chi2_contingency(contingency)
    except Exception:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    return MetricResult(
        raw=float(stat),
        score=_safe_p_to_score(p),
        flag=p < 0.05,
        p_value=float(p),
    )


def jsd_cat(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy.spatial.distance import jensenshannon

    p, q, _ = _category_probs(ref, ev)
    raw = float(jensenshannon(p, q, base=2) ** 2)
    return MetricResult(raw=raw, score=raw, flag=raw > 0.1)


def tvd_cat(ref: pd.Series, ev: pd.Series) -> MetricResult:
    p, q, _ = _category_probs(ref, ev)
    raw = float(0.5 * np.sum(np.abs(p - q)))
    return MetricResult(raw=raw, score=raw, flag=raw > 0.1)


def psi_cat(ref: pd.Series, ev: pd.Series) -> MetricResult:
    p, q, _ = _category_probs(ref, ev)
    raw = float(np.sum((p - q) * np.log(p / q)))
    return MetricResult(raw=raw, score=_clip_score(raw, 0.25), flag=raw > 0.1)


def fisher_cat(ref: pd.Series, ev: pd.Series) -> MetricResult:
    """Fisher exact — only well-defined for 2xN; we fall back to permuted-chi² p otherwise."""
    from scipy import stats

    r, e, levels = _clean_categorical(ref, ev)
    rc = r.value_counts().reindex(levels, fill_value=0).to_numpy()
    ec = e.value_counts().reindex(levels, fill_value=0).to_numpy()
    if rc.sum() == 0 or ec.sum() == 0 or len(levels) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    if len(levels) == 2:
        try:
            _, p = stats.fisher_exact(np.array([rc, ec]))
        except Exception:
            return MetricResult(raw=0.0, score=0.0, flag=False)
    else:
        # For >2 categories use chi² with Monte-Carlo permutation
        try:
            res = stats.chi2_contingency(np.array([rc, ec]))
            p = float(res[1])
        except Exception:
            return MetricResult(raw=0.0, score=0.0, flag=False)
    return MetricResult(
        raw=float(p), score=_safe_p_to_score(p), flag=p < 0.05, p_value=float(p)
    )


def z_test_prop(ref: pd.Series, ev: pd.Series) -> MetricResult:
    """Two-proportion z-test on the most-frequent category (binary or general)."""
    from scipy import stats

    r, e, levels = _clean_categorical(ref, ev)
    if len(levels) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    target = levels[0]
    x1 = (r == target).sum()
    x2 = (e == target).sum()
    n1, n2 = len(r), len(e)
    if n1 == 0 or n2 == 0:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) + 1e-12
    z = (p1 - p2) / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return MetricResult(
        raw=float(z), score=_safe_p_to_score(p), flag=p < 0.05, p_value=p
    )


def cramer_v(ref: pd.Series, ev: pd.Series) -> MetricResult:
    from scipy import stats

    r, e, levels = _clean_categorical(ref, ev)
    rc = r.value_counts().reindex(levels, fill_value=0).to_numpy()
    ec = e.value_counts().reindex(levels, fill_value=0).to_numpy()
    n = rc.sum() + ec.sum()
    if n == 0 or len(levels) < 2:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    contingency = np.array([rc, ec])
    try:
        stat, p, _, _ = stats.chi2_contingency(contingency)
    except Exception:
        return MetricResult(raw=0.0, score=0.0, flag=False)
    v = float(np.sqrt(stat / (n * (min(contingency.shape) - 1))))
    return MetricResult(raw=v, score=min(v, 1.0), flag=v > 0.1, p_value=float(p))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


METRICS: dict[str, tuple[Callable[..., MetricResult], Supports]] = {
    # numeric
    "psi": (psi_numeric, "numeric"),
    "ks": (ks_numeric, "numeric"),
    "wasserstein": (wasserstein_numeric, "numeric"),
    "jsd_num": (jsd_numeric, "numeric"),
    "smd": (smd_numeric, "numeric"),
    "welch_t": (welch_t, "numeric"),
    "mannwhitney": (mannwhitney, "numeric"),
    "levene": (levene, "numeric"),
    "anderson_darling": (anderson_darling, "numeric"),
    # categorical
    "chi2": (chi2_cat, "categorical"),
    "jsd_cat": (jsd_cat, "categorical"),
    "tvd": (tvd_cat, "categorical"),
    "psi_cat": (psi_cat, "categorical"),
    "fisher": (fisher_cat, "categorical"),
    "z_test_prop": (z_test_prop, "categorical"),
    "cramer_v": (cramer_v, "categorical"),
}


def metrics_for(kind: str) -> list[str]:
    return [
        name
        for name, (_, supports) in METRICS.items()
        if supports == kind or supports == "both"
    ]
