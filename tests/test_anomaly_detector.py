"""Tests for TabularAnomalyDetector."""

import pytest
import pandas as pd
import numpy as np

from a4s_plugin_data_evaluation.utils import Feature, FeatureType
from a4s_plugin_data_evaluation.data_anomaly.anomaly_detector import (
    TabularAnomalyDetector,
)


class TestTabularAnomalyDetector:
    """Tests for TabularAnomalyDetector class."""

    @pytest.fixture
    def numeric_features(self):
        """Create numeric features with defined min/max."""
        return [
            Feature(name="age", min=18.0, max=65.0, type=FeatureType.INTEGER),
            Feature(name="score", min=0.0, max=100.0, type=FeatureType.FLOAT),
        ]

    @pytest.fixture
    def categorical_features(self):
        """Create categorical features."""
        return [
            Feature(name="category", min=0.0, max=0.0, type=FeatureType.CATEGORICAL),
        ]

    @pytest.fixture
    def reference_df(self):
        """Create reference DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 65, 100),
                "score": np.random.uniform(20, 80, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    @pytest.fixture
    def normal_df(self, reference_df):
        """Create DataFrame similar to reference (no anomalies)."""
        np.random.seed(43)
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 65, 50),
                "score": np.random.uniform(20, 80, 50),
                "category": np.random.choice(["A", "B", "C"], 50),
            }
        )

    @pytest.fixture
    def anomalous_df(self):
        """Create DataFrame with anomalies."""
        return pd.DataFrame(
            {
                "age": [10, 80, 25, 30, 100],  # Some outside 18-65 range
                "score": [-10, 50, 150, 60, 70],  # Some outside 0-100 range
                "category": ["A", "D", "E", "B", "F"],  # New categories D, E, F
            }
        )

    def test_fit_numeric_features(self, numeric_features, reference_df):
        detector = TabularAnomalyDetector()
        detector.fit(numeric_features, reference_df)
        assert "age" in detector.numeric_features
        assert "score" in detector.numeric_features

    def test_fit_categorical_features(self, categorical_features, reference_df):
        detector = TabularAnomalyDetector()
        detector.fit(categorical_features, reference_df)
        assert "category" in detector.categorical_features

    def test_call_returns_metrics(self, numeric_features, reference_df, normal_df):
        detector = TabularAnomalyDetector()
        detector.fit(numeric_features, reference_df)
        metrics = detector(normal_df)
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_constraint_violations_detected(
        self, numeric_features, reference_df, anomalous_df
    ):
        detector = TabularAnomalyDetector()
        detector.fit(numeric_features, reference_df)
        metrics = detector(anomalous_df)

        upper_violations = [m for m in metrics if "Upper Constraint Violations" in m]
        lower_violations = [m for m in metrics if "Lower Constraint Violations" in m]

        assert len(upper_violations) > 0
        assert len(lower_violations) > 0

    def test_new_categories_detected(
        self, categorical_features, reference_df, anomalous_df
    ):
        detector = TabularAnomalyDetector()
        detector.fit(categorical_features, reference_df)
        metrics = detector(anomalous_df)

        new_categories = [
            m
            for m in metrics
            if "New Categories" in m and m["New Categories"]["score"] > 0
        ]
        assert len(new_categories) > 0

    def test_missing_categories_detected(self, categorical_features, reference_df):
        detector = TabularAnomalyDetector()
        detector.fit(categorical_features, reference_df)

        # Evaluated df missing category C
        eval_df = pd.DataFrame({"category": ["A", "A", "B", "B"]})
        metrics = detector(eval_df)

        missing_categories = [
            m
            for m in metrics
            if "Missing Categories" in m and m["Missing Categories"]["score"] > 0
        ]
        assert len(missing_categories) > 0

    def test_anomaly_summary_metrics(self, numeric_features, reference_df, normal_df):
        detector = TabularAnomalyDetector()
        detector.fit(numeric_features, reference_df)
        metrics = detector(normal_df)

        metric_names = [list(m.keys())[0] for m in metrics]
        assert "Anomaly Pass" in metric_names
        assert "Anomaly Low" in metric_names
        assert "Anomaly Severe" in metric_names

    def test_distribution_outlier_detected(self, numeric_features, reference_df):
        detector = TabularAnomalyDetector()
        detector.fit(numeric_features, reference_df)

        # Create data with outliers (beyond 3 std from mean)
        eval_df = pd.DataFrame(
            {
                "age": [18, 20, 200, 25, 30],  # 200 is an outlier
                "score": [50, 55, 60, 500, 65],  # 500 is an outlier
            }
        )
        metrics = detector(eval_df)

        dist_outliers = [
            m
            for m in metrics
            if "Distribution Outlier" in m and m["Distribution Outlier"]["score"] > 0
        ]
        assert len(dist_outliers) > 0
