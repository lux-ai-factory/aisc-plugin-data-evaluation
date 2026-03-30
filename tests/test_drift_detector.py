"""Tests for TabularDriftDetector."""

import pytest
import pandas as pd
import numpy as np

from a4s_plugin_data_evaluation.utils import Feature, FeatureType
from a4s_plugin_data_evaluation.data_drift.drift_detector import TabularDriftDetector


class TestTabularDriftDetector:
    """Tests for TabularDriftDetector class."""

    @pytest.fixture
    def numeric_features(self):
        """Create numeric features."""
        return [
            Feature(name="age", min=0.0, max=100.0, type=FeatureType.INTEGER),
            Feature(name="income", min=0.0, max=200000.0, type=FeatureType.FLOAT),
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
                "income": np.random.uniform(30000, 150000, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    @pytest.fixture
    def similar_df(self, reference_df):
        """Create DataFrame similar to reference (low drift)."""
        np.random.seed(43)
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 65, 100),
                "income": np.random.uniform(30000, 150000, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    @pytest.fixture
    def drifted_df(self):
        """Create DataFrame with significant drift."""
        np.random.seed(44)
        return pd.DataFrame(
            {
                "age": np.random.randint(50, 90, 100),  # Shifted age distribution
                "income": np.random.uniform(100000, 300000, 100),  # Shifted income
                "category": np.random.choice(
                    ["C", "D", "E"], 100
                ),  # Different categories
            }
        )

    def test_fit_numeric_features(self, numeric_features, reference_df):
        detector = TabularDriftDetector()
        detector.fit(numeric_features, reference_df)
        assert "age" in detector.numeric_features
        assert "income" in detector.numeric_features
        assert len(detector.bins) == 2

    def test_fit_categorical_features(self, categorical_features, reference_df):
        detector = TabularDriftDetector()
        detector.fit(categorical_features, reference_df)
        assert "category" in detector.categorical_features

    def test_call_returns_metrics(self, numeric_features, reference_df, similar_df):
        detector = TabularDriftDetector()
        detector.fit(numeric_features, reference_df)
        metrics = detector(similar_df)
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_numeric_drift_metrics(self, numeric_features, reference_df, similar_df):
        detector = TabularDriftDetector()
        detector.fit(numeric_features, reference_df)
        metrics = detector(similar_df)

        metric_names = [list(m.keys())[0] for m in metrics]
        assert "psi" in metric_names
        assert "ks_statistic" in metric_names
        assert "wasserstein_distance" in metric_names

    def test_categorical_drift_metrics(
        self, categorical_features, reference_df, similar_df
    ):
        detector = TabularDriftDetector()
        detector.fit(categorical_features, reference_df)
        metrics = detector(similar_df)

        metric_names = [list(m.keys())[0] for m in metrics]
        assert "chi2_statistic" in metric_names
        assert "jensenshannon_distance" in metric_names

    def test_drift_detected_high_drift(
        self, numeric_features, reference_df, drifted_df
    ):
        detector = TabularDriftDetector(numeric_threshold=0.1)
        detector.fit(numeric_features, reference_df)
        metrics = detector(drifted_df)

        drift_detected = [
            m
            for m in metrics
            if "drift_detected" in m and m["drift_detected"]["score"] is True
        ]
        assert len(drift_detected) > 0

    def test_aggregate_metrics_present(
        self, numeric_features, reference_df, similar_df
    ):
        detector = TabularDriftDetector()
        detector.fit(numeric_features, reference_df)
        metrics = detector(similar_df)

        metric_names = [list(m.keys())[0] for m in metrics]
        assert "ratio_features_with_drift" in metric_names
        assert "avg_data_drift" in metric_names
