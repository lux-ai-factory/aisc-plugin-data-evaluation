import logging
from datetime import datetime
from itertools import chain
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..utils import Feature, FeatureType

logger = logging.getLogger(__name__)


class TabularDriftDetector:
    def __init__(
        self,
        n_bins: int = 10,
        numeric_threshold: float = 0.1,
        categorical_threshold: float = 0.1,
    ):
        self.reference = None
        self.features = None
        self.features_names = None

        self.n_bins = n_bins
        self.bins = {}

        self.numeric_threshold = numeric_threshold
        self.categorical_threshold = categorical_threshold

        self.numeric_features = None
        self.categorical_features = None

    def fit(self, features: list[Feature], reference: pd.DataFrame):
        import numpy as np

        self.reference = reference
        self.features = []
        self.features_names = []

        self.numeric_features = set()
        self.categorical_features = set()

        for feature in features:
            col = feature.name
            if feature.type in (FeatureType.INTEGER, FeatureType.FLOAT):
                self.numeric_features.add(col)
                # store numeric bin edges
                self.bins[col] = np.histogram(
                    reference[col].dropna(), bins=self.n_bins
                )[-1]
                self.features.append(feature)
                self.features_names.append(col)

            elif feature.type == FeatureType.CATEGORICAL:
                self.categorical_features.add(col)
                self.features.append(feature)
                self.features_names.append(col)

    def __call__(self, evaluated: pd.DataFrame, date=None) -> list[dict]:
        if date is None:
            date = datetime.now()

        compute_metric = partial(self._feature_drift, date=date, evaluated=evaluated)

        with ThreadPoolExecutor(max_workers=7) as pool:
            metrics = list(chain.from_iterable(pool.map(compute_metric, self.features)))

        count_features_with_drift = sum(
            m.get("drift_detected", {}).get("score")
            for m in metrics
            if "drift_detected" in m.keys()
        )
        ratio_features_with_drift = count_features_with_drift / len(self.features)

        avg_data_drift = sum(
            m.get("data_drift", {}).get("score")
            for m in metrics
            if "data_drift" in m.keys()
        ) / len(self.features)

        metrics.extend(
            [
                {
                    "Number of Drifted Features": dict(
                        score=count_features_with_drift,
                        time=date,
                        description=None,
                        feature_pid=None,
                    )
                },
                {
                    "ratio_features_with_drift": dict(
                        score=ratio_features_with_drift,
                        time=date,
                        description=None,
                        feature_pid=None,
                    )
                },
                {
                    "Data Drift": dict(
                        score=avg_data_drift,
                        time=date,
                        description=None,
                        feature_pid=None,
                    )
                },
            ]
        )

        return metrics

    def _feature_drift(self, feature, date, evaluated):
        feature_type = feature.type
        logger.debug(f"Processing feature: {feature.name} (type: {feature_type})")
        logger.debug(f"Computing feature drift test for feature type: {feature_type}")

        metrics = {}

        if feature.name in self.numeric_features:
            metrics = self._numeric_drift(
                self.reference[feature.name],
                evaluated[feature.name],
                feature.name,
            )
            for metric_name, score in metrics.items():
                logger.debug(f"Created numerical drift metric: {metric_name} = {score}")

        elif feature.name in self.categorical_features:
            metrics = self._categorical_drift(
                self.reference[feature.name],
                evaluated[feature.name],
                feature.name,
            )
            for metric_name, score in metrics.items():
                logger.debug(
                    f"Created categorical drift metric: {metric_name} = {score}"
                )

        else:
            logger.error(f"Unsupported feature type: {feature_type}")
            raise ValueError(f"Feature type {feature_type} not supported")

        if metrics:
            results = []
            for metric_name, score in metrics.items():
                logger.debug(
                    f"Added metric for feature {feature.name}: {metric_name} = {score}"
                )
                results.append(
                    {
                        metric_name: dict(
                            score=score,
                            time=date,
                            description=feature.name,
                            # feature_pid=feature.pid,
                        )
                    }
                )
            return results

    def _psi(self, ref, eval, bin_edges):
        import numpy as np
        from scipy.stats import chi2

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        eval_counts, _ = np.histogram(eval, bins=bin_edges)

        ref_perc = ref_counts / len(ref)
        eval_perc = eval_counts / len(eval)

        ref_perc = np.where(ref_perc == 0, 1e-6, ref_perc)
        eval_perc = np.where(eval_perc == 0, 1e-6, eval_perc)

        psi = float(np.sum((ref_perc - eval_perc) * np.log(ref_perc / eval_perc)))

        chi_stat = 2 * len(eval) * psi
        # NOTE: drift detected if p_value < 0.05
        psi_pvalue = float(1 - chi2.cdf(chi_stat, len(ref_counts) - 1))

        return psi, psi_pvalue

    def _numeric_drift(self, ref, eval, col):
        import numpy as np
        from scipy.stats import ks_2samp, wasserstein_distance

        ref = ref.dropna()
        eval = eval.dropna()

        psi, psi_p = self._psi(ref, eval, self.bins[col])

        ks_stat, ks_p = ks_2samp(ref, eval)

        wass = wasserstein_distance(ref, eval)

        # Standardized mean difference
        smd = abs(np.mean(ref) - np.mean(eval)) / (np.std(ref) + 1e-6)

        return {
            "psi": psi,
            "psi_chi2_p_value": psi_p,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "wasserstein_distance": wass,
            "standardized_mean_diff": smd,
            "data_drift": psi,
            "drift_detected": psi > self.numeric_threshold,
        }

    def _categorical_drift(self, ref, eval, col):
        import numpy as np
        from scipy.stats import chi2_contingency
        from scipy.spatial.distance import jensenshannon

        ref = ref.astype(str)
        eval = eval.astype(str)

        levels = list(set(ref.unique()).union(set(eval.unique())))

        ref_counts = ref.value_counts().reindex(levels, fill_value=0)
        eval_counts = eval.value_counts().reindex(levels, fill_value=0)

        # Chi-square
        contingency = np.array([ref_counts, eval_counts])
        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)

        # JS divergence
        ref_prob = ref_counts / ref_counts.sum()
        eval_prob = eval_counts / eval_counts.sum()
        js = jensenshannon(ref_prob, eval_prob, base=2) ** 2

        # Total Variation Distance
        tvd = 0.5 * np.sum(np.abs(ref_prob - eval_prob))

        return {
            "chi2_statistic": chi2_stat,
            "chi2_pvalue": chi2_p,
            "jensenshannon_distance": js,
            "total_variance_distance": tvd,
            "data_drift": js,
            "drift_detected": js > self.categorical_threshold,
        }
