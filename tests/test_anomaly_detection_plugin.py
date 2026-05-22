"""Integration tests for anomaly detection plugin using real datasets."""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from data_monitor import DataAnomalyPlugin
from data_monitor.utils import Feature, FeatureType


DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "classification"


@pytest.fixture
def data_anomaly_plugin():
    """Create a data anomaly plugin instance."""
    plugin = DataAnomalyPlugin()
    return plugin


class TestDataAnomalyPlugin:
    """Tests for DataAnomalyPlugin class."""

    def test_plugin_name(self, data_anomaly_plugin):
        assert data_anomaly_plugin.plugin_name == "Data Anomaly"

    def test_display_icon(self, data_anomaly_plugin):
        assert data_anomaly_plugin.display_icon == "flag"

    def test_metric_names(self, data_anomaly_plugin):
        metric_names = data_anomaly_plugin.metric_names()
        expected = [
            "New Categories",
            "Missing Categories",
            "Upper Constraint Violations",
            "Lower Constraint Violations",
            "Distribution Outlier",
            "Anomaly Pass",
            "Anomaly Low",
            "Anomaly Severe",
        ]
        assert metric_names == expected

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

    def test_set_input_providers(self, data_anomaly_plugin, reference_df, normal_df):
        import pandas as pd

        reference_df_bytes = reference_df.to_csv(index=False).encode("utf-8")
        evaluated_df_bytes = normal_df.to_csv(index=False).encode("utf-8")

        data_anomaly_plugin.set_input_content("reference-dataset", reference_df_bytes)
        data_anomaly_plugin.set_input_content("evaluated-dataset", evaluated_df_bytes)

        assert (
            data_anomaly_plugin._input_provider_instances.get("reference-dataset")
            is not None
        )
        assert (
            data_anomaly_plugin._input_provider_instances.get("evaluated-dataset")
            is not None
        )
        assert isinstance(
            data_anomaly_plugin.get_input_data("reference-dataset"), pd.DataFrame
        )
        assert isinstance(
            data_anomaly_plugin.get_input_data("evaluated-dataset"), pd.DataFrame
        )

    def test_parse_config_from_dataset(self, data_anomaly_plugin, reference_df):
        reference_df_bytes = reference_df.to_csv(index=False).encode("utf-8")
        config = data_anomaly_plugin.parse_config_from_dataset(reference_df_bytes)

        assert config is not None
        assert "features" in config
        assert len(config["features"]) > 0
        # Check that features have expected structure
        for feature in config["features"]:
            assert "name" in feature
            assert "type" in feature
            assert "min" in feature
            assert "max" in feature

    def test_evaluate(self, data_anomaly_plugin, reference_df, anomalous_df):
        reference_df_bytes = reference_df.to_csv(index=False).encode("utf-8")
        evaluated_df_bytes = anomalous_df.to_csv(index=False).encode("utf-8")

        data_anomaly_plugin.set_input_content("reference-dataset", reference_df_bytes)
        data_anomaly_plugin.set_input_content("evaluated-dataset", evaluated_df_bytes)

        config_data = data_anomaly_plugin.parse_config_from_dataset(reference_df_bytes)

        results = data_anomaly_plugin.evaluate(config_data)

        assert isinstance(results, dict)
        # Check all expected metrics are present
        for metric in data_anomaly_plugin.metric_names():
            assert metric in results
            assert "score" in results[metric][0]
            assert len(results[metric]) > 0
