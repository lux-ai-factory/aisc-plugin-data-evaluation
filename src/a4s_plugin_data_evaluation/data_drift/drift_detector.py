import logging
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
        global_auc_threshold: float = 0.6,
        random_state: int = 42,
    ):
        self.reference = None
        self.features = None
        self.features_names = None

        self.n_bins = n_bins
        self.bins = {}

        self.numeric_threshold = numeric_threshold
        self.categorical_threshold = categorical_threshold
        self.global_auc_threshold = global_auc_threshold

        self.random_state = random_state

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

    def __call__(self, date, mask, evaluated: pd.DataFrame) -> list[dict]:
        compute_metric = partial(
            self.feature_drift, date=date, mask=mask, evaluated=evaluated
        )

        with ThreadPoolExecutor() as pool:
            metrics = list(chain.from_iterable(pool.map(compute_metric, self.features)))

        ratio_features_with_drift = sum(
            m.get("drift_detected", {}).get("score")
            for m in metrics
            if "drift_detected" in m.keys()
        ) / len(self.features)

        # global_auc = float(self._global_drift(evaluated))
        metrics.extend(
            [
                # {
                #     "global_auc_cls_based": dict(
                #         score=global_auc,
                #         date=date,
                #         description=None,
                #         feature_pid=None,
                #     )
                # },
                # {
                #     "global_drift_detected": dict(
                #         score=global_auc > self.global_auc_threshold,
                #         date=date,
                #         description=None,
                #         feature_pid=None,
                #     )
                # },
                {
                    "ratio_features_with_drift": dict(
                        score=ratio_features_with_drift,
                        date=date,
                        description=None,
                        feature_pid=None,
                    )
                },
            ]
        )

        return metrics

    def feature_drift(self, feature, date, mask, evaluated):
        feature_type = feature.type
        logger.debug(f"Processing feature: {feature.name} (type: {feature_type})")
        logger.debug(f"Computing feature drift test for feature type: {feature_type}")

        masked_evaluated = evaluated.loc[mask] if mask is not None else evaluated

        metrics = {}

        if feature.name in self.numeric_features:
            metrics = self._numeric_drift(
                self.reference[feature.name],
                masked_evaluated[feature.name],
                feature.name,
            )
            for metric_name, score in metrics.items():
                logger.debug(f"Created numerical drift metric: {metric_name} = {score}")

        elif feature.name in self.categorical_features:
            metrics = self._categorical_drift(
                self.reference[feature.name],
                masked_evaluated[feature.name],
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
                            date=date,
                            description=feature.name,
                            feature_pid=feature.pid,
                        )
                    }
                )
            return results

    def _psi(self, ref, new, bin_edges):
        import numpy as np
        from scipy.stats import chi2

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        new_counts, _ = np.histogram(new, bins=bin_edges)

        ref_perc = ref_counts / len(ref)
        new_perc = new_counts / len(new)

        ref_perc = np.where(ref_perc == 0, 1e-6, ref_perc)
        new_perc = np.where(new_perc == 0, 1e-6, new_perc)

        psi = float(np.sum((ref_perc - new_perc) * np.log(ref_perc / new_perc)))

        chi_stat = 2 * len(new) * psi
        # NOTE: drift detected if p_value < 0.05
        psi_pvalue = float(1 - chi2.cdf(chi_stat, len(ref_counts) - 1))

        return psi, psi_pvalue

    def _numeric_drift(self, ref, new, col):
        from scipy.stats import ks_2samp, wasserstein_distance

        ref = ref.dropna()
        new = new.dropna()

        psi, psi_p = self._psi(ref, new, self.bins[col])

        ks_stat, ks_p = ks_2samp(ref, new)

        wass = wasserstein_distance(ref, new)

        return {
            "psi": psi,
            "psi_chi2_p_value": psi_p,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "wasserstein_distance": wass,
            "drift_detected": psi > self.numeric_threshold,
        }

    def _categorical_drift(self, ref, new, col):
        import numpy as np
        from scipy.stats import chi2_contingency
        from scipy.spatial.distance import jensenshannon

        ref = ref.astype(str)
        new = new.astype(str)

        levels = list(set(ref.unique()).union(set(new.unique())))

        ref_counts = ref.value_counts().reindex(levels, fill_value=0)
        new_counts = new.value_counts().reindex(levels, fill_value=0)

        # Chi-square
        contingency = np.array([ref_counts, new_counts])
        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)

        # JS divergence
        ref_prob = ref_counts / ref_counts.sum()
        new_prob = new_counts / new_counts.sum()
        js = jensenshannon(ref_prob, new_prob, base=2) ** 2

        return {
            "chi2_statistic": chi2_stat,
            "chi2_pvalue": chi2_p,
            "jensenshannon_distance": js,
            "drift_detected": js > self.categorical_threshold,
        }

    def _global_drift(self, new_data):
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import roc_auc_score

        ref = self.reference.loc[:, self.features_names].copy()
        new = new_data.loc[:, self.features_names].copy()

        ref["__label__"] = 0
        new["__label__"] = 1

        combined = pd.concat([ref, new], axis=0).reset_index(drop=True)

        y = combined["__label__"]
        X = combined.drop(columns="__label__")

        # encode categorical
        for col in self.categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )

        clf = LogisticRegression(max_iter=5000, solver="lbfgs")

        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        return auc


if __name__ == "__main__":
    import json
    from pathlib import Path

    d = Path.home() / "Documents/projects/data/time_series"

    ref = pd.read_csv(d / "training_data.csv")
    evl = pd.read_csv(d / "testing_data.csv")

    with open(d / "plugin_config.json", "r") as f:
        config = json.load(f)

    features = [Feature(**item) for item in config["features"]]

    detector = TabularDriftDetector()
    detector.fit(features, ref)
    res = detector(None, None, evl)
