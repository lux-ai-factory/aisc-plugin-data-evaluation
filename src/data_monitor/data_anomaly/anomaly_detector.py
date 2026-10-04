from datetime import datetime
from itertools import chain
from typing import Any

import pandas as pd

from ..utils import Feature, FeatureType


class TabularAnomalyDetector:
    reference: pd.DataFrame | None
    features: list[Feature]
    features_names: list[str]
    numeric_features: set[str]
    categorical_features: set[str]

    def __init__(self) -> None:
        self.reference = None
        self.features = []
        self.features_names = []

        self.numeric_features = set()
        self.categorical_features = set()

    def fit(self, features: list[Feature], reference: pd.DataFrame) -> None:
        self.reference = reference
        self.features = []
        self.features_names = []

        self.numeric_features = set()
        self.categorical_features = set()

        for feature in features:
            col = feature.name
            if feature.type in (FeatureType.INTEGER, FeatureType.FLOAT):
                self.numeric_features.add(col)
                self.features.append(feature)
                self.features_names.append(col)

            elif feature.type == FeatureType.CATEGORICAL:
                self.categorical_features.add(col)
                self.features.append(feature)
                self.features_names.append(col)

    def __call__(
        self, evaluated: pd.DataFrame, date: datetime | None = None
    ) -> list[dict[str, Any]]:
        if date is None:
            date = datetime.now()

        metrics: list[dict[str, Any]] = list(
            chain(
                self._analyse_categorical_features(evaluated, date),
                self._analyse_numerical_features(evaluated, date),
            )
        )

        score_pass = 0
        score_low = 0
        score_severe = 0

        for item in metrics:
            metric_name, value = next(iter(item.items()))
            score = value["score"]
            if score == 0:
                score_pass += 1
            else:
                if metric_name in ("Missing Categories", "Distribution Outlier"):
                    score_low += 1
                else:
                    score_severe += 1

        metrics.extend(
            [
                {
                    "Anomaly Pass": dict(
                        score=score_pass,
                        description=None,
                        time=date,
                    )
                },
                {
                    "Anomaly Low": dict(
                        score=score_low,
                        description=None,
                        time=date,
                    )
                },
                {
                    "Anomaly Severe": dict(
                        score=score_severe,
                        description=None,
                        time=date,
                    )
                },
            ]
        )
        return metrics

    def _analyse_categorical_features(
        self, evaluated: pd.DataFrame, date: datetime
    ) -> list[dict[str, Any]]:
        assert self.reference is not None

        metrics: list[dict[str, Any]] = []

        for feat_name in self.categorical_features:
            ref_categories = set(self.reference[feat_name].unique())
            eval_categories = set(evaluated[feat_name].unique())

            new_categories = eval_categories - ref_categories
            missing_categories = ref_categories - eval_categories

            metrics.extend(
                [
                    {
                        "New Categories": dict(
                            score=len(new_categories),
                            description=feat_name,
                            dimensions={"feature": str(feat_name)},
                            time=date,
                        )
                    },
                    {
                        "Missing Categories": dict(
                            score=len(missing_categories),
                            description=feat_name,
                            dimensions={"feature": str(feat_name)},
                            time=date,
                        )
                    },
                ]
            )
        return metrics

    def _analyse_numerical_features(
        self, evaluated: pd.DataFrame, date: datetime
    ) -> list[dict[str, Any]]:
        assert self.reference is not None

        metrics: list[dict[str, Any]] = []

        for feature in self.features:
            if feature.name not in self.numeric_features:
                continue

            ref_series = self.reference[feature.name].dropna()
            eval_series = evaluated[feature.name].dropna()

            if ref_series.empty:
                continue

            ref_mean = ref_series.mean()
            ref_std = ref_series.std()

            if ref_std is None or ref_std == 0 or ref_std != ref_std:
                dist_violations = 0
            else:
                lower = ref_mean - 3 * ref_std
                upper = ref_mean + 3 * ref_std

                violations = (eval_series < lower) | (eval_series > upper)
                dist_violations = int(violations.sum())

            max_violations = int((eval_series > feature.max).sum())
            min_violations = int((eval_series < feature.min).sum())

            metrics.extend(
                [
                    {
                        "Lower Constraint Violations": dict(
                            score=min_violations,
                            description=feature.name,
                            dimensions={"feature": str(feature.name)},
                            time=date,
                        )
                    },
                    {
                        "Upper Constraint Violations": dict(
                            score=max_violations,
                            description=feature.name,
                            dimensions={"feature": str(feature.name)},
                            time=date,
                        )
                    },
                    {
                        "Distribution Outlier": dict(
                            score=dist_violations,
                            description=feature.name,
                            dimensions={"feature": str(feature.name)},
                            time=date,
                        )
                    },
                ]
            )
        return metrics
