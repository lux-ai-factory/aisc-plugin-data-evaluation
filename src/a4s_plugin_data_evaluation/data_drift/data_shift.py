from itertools import chain
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..utils import Feature, FeatureType


class DataShiftMonitor:
    def __init__(self):
        self.reference = None
        self.features = None
        self.features_names = None

        self.numeric_features = None
        self.categorical_features = None

    def fit(self, features: list[Feature], reference: pd.DataFrame):
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

    def __call__(self, date, mask, evaluated: pd.DataFrame) -> list[dict]:
        compute_metric = partial(
            self._data_shift, date=date, mask=mask, evaluated=evaluated
        )

        with ThreadPoolExecutor() as pool:
            metrics = list(chain.from_iterable(pool.map(compute_metric, self.features)))

        return metrics

    def _data_shift(self, feature, date, mask, evaluated):
        feature_type = feature.type

        masked_evaluated = evaluated.loc[mask] if mask is not None else evaluated

        drift = None

        if feature.name in self.numeric_features:
            drift = self._numerical_shift(
                self.reference[feature.name], masked_evaluated[feature.name]
            )
        elif feature.name in self.categorical_features:
            drift = self._categorical_shift(
                self.reference[feature.name], masked_evaluated[feature.name]
            )
        else:
            raise ValueError(f"Feature type {feature_type} not supported")

        return [
            {
                "data_shift": dict(
                    score=drift,
                    date=date,
                    description=feature.name,
                    feature_pid=feature.pid,
                )
            }
        ]

    def _numerical_shift(self, ref, eval):
        import numpy as np

        shift = abs(np.mean(ref) - np.mean(eval)) / (np.std(ref) + 1e-6)
        return shift

    def _categorical_shift(self, ref, eval):
        import numpy as np

        freq_train = ref.value_counts(normalize=True)
        freq_test = eval.value_counts(normalize=True)

        freq_train, freq_test = freq_train.align(freq_test, fill_value=0)
        tvd = 0.5 * np.sum(np.abs(freq_train - freq_test))
        return tvd
