"""Drift detector — calibrated univariate diagnostics + multivariate decision.

Pipeline:
  guard → calibrate (IID or block bootstrap) →
  per-feature univariate diagnostic + multivariate decision →
  per-window output.

drift_flag  = c2st.flag OR mmd.flag
drift_score = max(c2st.score, mmd.score)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from ..utils import Feature, FeatureType
from .calibration import (
    CalibratedThresholds,
    calibrate,
    detect_autocorrelation,
)
from .metrics import METRICS, MetricResult
from .multivariate import MultivariateResult, c2st, mmd_rbf


def _empty_mv_result(method: str) -> MultivariateResult:
    """A no-drift placeholder used when multivariate cannot run."""
    return MultivariateResult(
        method=method,
        raw=0.0,
        score=0.0,
        flag=False,
        p_value=1.0,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guard layer
# ---------------------------------------------------------------------------


HIGH_CARD_THRESHOLD = 50
MIN_SAMPLES = 100


@dataclass
class GuardReport:
    schema_aligned_columns: list[str]
    dropped_nan_columns: list[str]
    dropped_constant_columns: list[str]
    high_cardinality_columns: list[str]
    small_sample: bool
    warnings: list[str]


def _guard(
    ref: pd.DataFrame, ev: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, GuardReport]:
    """Validate inputs, drop problematic columns, return cleaned ref/eval."""
    warnings: list[str] = []
    dropped_nan: list[str] = []
    dropped_const: list[str] = []
    high_card: list[str] = []

    # Schema alignment (use intersection)
    common = [c for c in ref.columns if c in ev.columns]
    if len(common) != len(ref.columns):
        missing = set(ref.columns) - set(common)
        warnings.append(f"Columns in reference but not evaluated: {sorted(missing)}")
    ref = ref[common].copy()
    ev = ev[common].copy()

    # All-NaN columns
    for c in list(common):
        if ref[c].isna().all() or ev[c].isna().all():
            dropped_nan.append(c)
            ref = ref.drop(columns=[c])
            ev = ev.drop(columns=[c])
    if dropped_nan:
        warnings.append(f"Dropped all-NaN columns: {dropped_nan}")

    # Constant columns
    for c in list(ref.columns):
        if ref[c].nunique(dropna=True) <= 1:
            dropped_const.append(c)
            ref = ref.drop(columns=[c])
            ev = ev.drop(columns=[c])
    if dropped_const:
        warnings.append(f"Dropped constant columns: {dropped_const}")

    # High-cardinality categoricals
    for c in ref.columns:
        if not pd.api.types.is_numeric_dtype(ref[c]):
            if ref[c].astype(str).nunique() > HIGH_CARD_THRESHOLD:
                high_card.append(c)
    if high_card:
        warnings.append(
            f"High-cardinality categorical (>{HIGH_CARD_THRESHOLD} levels): "
            f"{high_card}. Results may be unreliable."
        )

    small = min(len(ref), len(ev)) < MIN_SAMPLES
    if small:
        warnings.append(
            f"Small sample (n_ref={len(ref)}, n_eval={len(ev)} < {MIN_SAMPLES}). "
            f"Falling back to multivariate-only decision."
        )

    return (
        ref,
        ev,
        GuardReport(
            schema_aligned_columns=list(ref.columns),
            dropped_nan_columns=dropped_nan,
            dropped_constant_columns=dropped_const,
            high_cardinality_columns=high_card,
            small_sample=small,
            warnings=warnings,
        ),
    )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TabularDriftDetector:
    """Drift detector with calibrated thresholds + multivariate decision.

    Workflow:
        detector = TabularDriftDetector()
        detector.fit(features=..., reference=..., is_ts=...)
        result = detector(evaluated_window, date=window_timestamp)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_boot: int = 100,
        safety_buffer: float = 0.03,
        random_state: int = 0,
    ):
        self.alpha = alpha
        self.n_boot = n_boot
        self.safety_buffer = safety_buffer
        self.random_state = random_state

        self.reference: pd.DataFrame | None = None
        self.features: list[Feature] = []
        self.is_ts: bool = False
        self.thresholds: CalibratedThresholds | None = None
        self.guard_report: GuardReport | None = None

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        features: list[Feature],
        reference: pd.DataFrame,
        date_feature: str | None = None,
        is_ts: bool | None = None,
    ) -> None:
        """Calibrate thresholds from the reference dataset.

        is_ts: if None, auto-detect via date_feature OR lag-1 autocorrelation.
        """
        # Restrict reference to the configured features (drop target/date as needed)
        keep = [f.name for f in features if f.name in reference.columns]
        missing = [f.name for f in features if f.name not in reference.columns]
        if missing:
            logger.warning(
                "Features in config but missing from reference (skipped): %s",
                missing,
            )
        ref_used = reference[keep].copy()

        # Auto-detect TS mode if not explicit
        if is_ts is None:
            if date_feature is not None:
                is_ts = True
            else:
                detected, max_acf = detect_autocorrelation(ref_used, threshold=0.3)
                is_ts = detected
                if detected:
                    logger.info(
                        "Auto-detected time-series data (max lag-1 ACF=%.3f); "
                        "switching to block bootstrap.",
                        max_acf,
                    )

        self.is_ts = is_ts
        self.reference = ref_used
        self.features = [f for f in features if f.name in keep]

        mode = "block" if is_ts else "iid"
        logger.info("Calibrating thresholds (mode=%s, n_boot=%d)...", mode, self.n_boot)

        # Build feature_kinds from config so calibration matches eval-time logic.
        feature_kinds = {}
        for f in self.features:
            if f.type in (FeatureType.INTEGER, FeatureType.FLOAT):
                feature_kinds[f.name] = "numeric"
            elif f.type == FeatureType.CATEGORICAL:
                feature_kinds[f.name] = "categorical"

        self.thresholds = calibrate(
            ref_used,
            mode=mode,
            n_boot=self.n_boot,
            alpha=self.alpha,
            random_state=self.random_state,
            feature_kinds=feature_kinds,
        )
        logger.info(
            "Calibrated %d univariate thresholds (c2st=%.3f, mmd=%.5f).",
            len(self.thresholds.univariate),
            self.thresholds.c2st,
            self.thresholds.mmd,
        )

    # -- evaluate (per window) --------------------------------------------

    def __call__(
        self,
        evaluated: pd.DataFrame,
        date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate a window against the calibrated reference.

        Returns the same list-of-dicts format as the previous detector:
            [{metric_name: {score, time, description, feature_pid}}, ...]
        """
        assert self.reference is not None and self.thresholds is not None, (
            "Must call fit() before evaluating."
        )
        if date is None:
            date = datetime.now()

        # Defensive column selection — handle eval missing columns gracefully
        ref_cols = list(self.reference.columns)
        ev_cols = [c for c in ref_cols if c in evaluated.columns]
        if len(ev_cols) < len(ref_cols):
            missing = set(ref_cols) - set(ev_cols)
            logger.warning(
                "Evaluated dataset missing columns present in reference: %s. "
                "These will be dropped from this evaluation.",
                sorted(missing),
            )
        evaluated_aligned = (
            evaluated[ev_cols].copy() if ev_cols else evaluated.iloc[:, :0]
        )

        # Guard at eval time too (drops same-named bad columns if they sneak in)
        ref, ev, guard = _guard(self.reference, evaluated_aligned)
        for w in guard.warnings:
            logger.info("Guard: %s", w)

        # Per-feature univariate (diagnostic)
        per_feature_results: dict[str, dict[str, MetricResult]] = {}
        per_feature_ensemble: dict[str, float] = {}
        feature_pids = {f.name: f.pid for f in self.features}

        # Respect the config's declared feature type (Integer/Float -> numeric;
        # Categorical -> categorical). Fall back to dtype inference only if
        # the feature wasn't in the config.
        feat_type_map = {f.name: f.type for f in self.features}

        def _kind_of(feat: str) -> str:
            ft = feat_type_map.get(feat)
            if ft in (FeatureType.INTEGER, FeatureType.FLOAT):
                return "numeric"
            if ft == FeatureType.CATEGORICAL:
                return "categorical"
            return (
                "numeric" if pd.api.types.is_numeric_dtype(ref[feat]) else "categorical"
            )

        for feat in ref.columns:
            kind = _kind_of(feat)
            per_feature_results[feat] = {}
            flags: list[bool] = []
            applicable_total = (
                0  # count of metrics that should apply (for unbiased fraction)
            )
            for metric_name, (fn, supports) in METRICS.items():
                if supports != kind and supports != "both":
                    continue
                applicable_total += 1
                try:
                    r = fn(ref[feat], ev[feat])
                except Exception as e:
                    logger.warning(
                        "Metric %s on feature %s failed: %s", metric_name, feat, e
                    )
                    # Treat failure as "no signal" (False flag, score=0) so the ensemble
                    # fraction denominator stays correct (don't silently shrink it).
                    per_feature_results[feat][metric_name] = MetricResult(
                        raw=0.0,
                        score=0.0,
                        flag=False,
                        p_value=None,
                    )
                    flags.append(False)
                    continue
                # Calibrated flag overrides default
                t = self.thresholds.univariate.get((metric_name, feat))
                flag = (
                    r.score > t.threshold
                    if (t is not None and t.n_samples > 0)
                    else r.flag
                )
                per_feature_results[feat][metric_name] = MetricResult(
                    raw=r.raw,
                    score=r.score,
                    flag=flag,
                    p_value=r.p_value,
                )
                flags.append(flag)
            per_feature_ensemble[feat] = (
                sum(flags) / applicable_total if applicable_total else 0.0
            )

        # Multivariate (decision)
        if guard.small_sample or len(ref.columns) == 0:
            # Fallback: too little data or no usable columns — emit no-drift placeholders.
            c2st_res = _empty_mv_result("c2st")
            mmd_res = _empty_mv_result("mmd")
        else:
            try:
                c2st_res = c2st(ref, ev, random_state=self.random_state)
            except Exception as e:
                logger.warning("C2ST failed, falling back to no-drift: %s", e)
                c2st_res = _empty_mv_result("c2st")
            try:
                mmd_res = mmd_rbf(ref, ev, random_state=self.random_state)
            except Exception as e:
                logger.warning("MMD failed, falling back to no-drift: %s", e)
                mmd_res = _empty_mv_result("mmd")

            # Apply calibrated thresholds with a small safety buffer.
            # Reason: bootstrap percentile estimates are noisy at finite n_boot;
            # adding a margin reduces FPR inflation without meaningfully hurting TPR
            # (drift signal usually exceeds the threshold by a wide margin).
            c2st_res.flag = c2st_res.raw > (self.thresholds.c2st + self.safety_buffer)
            mmd_res.flag = mmd_res.raw > self.thresholds.mmd * (
                1.0 + self.safety_buffer
            )

        drift_flag = bool(c2st_res.flag or mmd_res.flag)
        drift_score = float(max(c2st_res.score, mmd_res.score))
        drift_raw = float(max(c2st_res.raw, mmd_res.raw))

        # Build output list
        out: list[dict[str, Any]] = []

        out.extend(
            [
                # Top-line
                {
                    "drift_flag": dict(
                        score=float(drift_flag),
                        time=date,
                        description=None,
                    )
                },
                {
                    "drift_score": dict(
                        score=drift_score,
                        time=date,
                        description=None,
                    )
                },
                {
                    "Drift Score": dict(
                        score=drift_raw,
                        time=date,
                        description=None,
                    )
                },
                # Multivariate details
                {
                    "c2st": dict(
                        score=float(c2st_res.score),
                        time=date,
                        description=f"AUC={c2st_res.raw:.3f} thr={self.thresholds.c2st:.3f}",
                    )
                },
                {
                    "mmd": dict(
                        score=float(mmd_res.score),
                        time=date,
                        description=f"MMD²={mmd_res.raw:.5f} thr={self.thresholds.mmd:.5f}",
                    )
                },
            ]
        )

        # Per-feature univariate diagnostics (every metric x every feature).
        # Description packs the raw stat, p-value (if it's a hypothesis test),
        # calibrated score-threshold, and final flag.
        # NOTE: univariate threshold is on the score (normalized [0,1]) scale,
        # not the raw scale; the flag is `score > score_thr`.
        nbr_drifted_features = 0
        for feat, metric_results in per_feature_results.items():
            pid = feature_pids.get(feat)
            for metric_name, r in metric_results.items():
                t = self.thresholds.univariate.get((metric_name, feat))
                parts = [f"raw_stat={r.raw:.4g}"]
                if r.p_value is not None and pd.notna(r.p_value):
                    parts.append(f"p={r.p_value:.4g}")
                if t is not None and t.n_samples > 0:
                    parts.append(f"score_thr={t.threshold:.4f}")
                else:
                    parts.append("score_thr=default")
                parts.append(f"flag={'YES' if r.flag else 'no'}")
                desc = f"{feat} | " + " | ".join(parts)
                out.append(
                    {
                        metric_name: dict(
                            score=float(r.score),
                            time=date,
                            description=desc,
                            feature_pid=str(pid) if pid is not None else None,
                        )
                    }
                )

                if _kind_of(feat) == "numeric":
                    if metric_name == "smd":
                        nbr_drifted_features += int(r.flag)
                else:
                    if metric_name == "psi_cat":
                        nbr_drifted_features += int(r.flag)

            # Ensemble fraction per feature
            out.append(
                {
                    "ensemble_fraction": dict(
                        score=float(per_feature_ensemble[feat]),
                        time=date,
                        description=feat,
                        feature_pid=str(pid) if pid is not None else None,
                    )
                }
            )

        out.append(
            {"Number of Drifted Features": dict(score=nbr_drifted_features, time=date)}
        )

        # Save guard report for the plugin to expose if needed
        self.guard_report = guard

        return out
