"""Integration tests for data drift plugin using real datasets."""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from data_monitor import DataDriftPlugin
from data_monitor.utils import Feature, FeatureType


DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "classification"


@pytest.fixture
def data_drift_plugin():
    """Create a data drift plugin instance."""
    plugin = DataDriftPlugin()
    return plugin


class TestDataDriftPlugin:
    """Tests for DataDriftPlugin class."""

    def test_plugin_name(self, data_drift_plugin):
        assert data_drift_plugin.plugin_name == "Data Drift"

    def test_display_icon(self, data_drift_plugin):
        assert data_drift_plugin.display_icon == "alt_route"

    def test_metric_names(self, data_drift_plugin):
        metric_names = data_drift_plugin.metric_names()
        expected = [
            "avg_data_drift",
            "Number of Drifted Features",
            "data_drift",
            "wasserstein_distance",
            "psi",
            "ks_statistic",
            "ks_pvalue",
            "jensenshannon_distance",
            "chi2_statistic",
            "chi2_pvalue",
            "drift_detected",
            "ratio_features_with_drift",
        ]
        assert metric_names == expected

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

    def test_set_input_providers(self, data_drift_plugin, reference_df, similar_df):
        import pandas as pd

        reference_df_bytes = reference_df.to_csv(index=False).encode("utf-8")
        evaluated_df_bytes = similar_df.to_csv(index=False).encode("utf-8")

        data_drift_plugin.set_input_content("reference-dataset", reference_df_bytes)
        data_drift_plugin.set_input_content("evaluated-dataset", evaluated_df_bytes)

        assert (
            data_drift_plugin._input_provider_instances.get("reference-dataset")
            is not None
        )
        assert (
            data_drift_plugin._input_provider_instances.get("evaluated-dataset")
            is not None
        )
        assert isinstance(
            data_drift_plugin.get_input_data("reference-dataset"), pd.DataFrame
        )
        assert isinstance(
            data_drift_plugin.get_input_data("evaluated-dataset"), pd.DataFrame
        )

    def test_parse_config_from_dataset(self, data_drift_plugin, reference_df):
        reference_df_bytes = reference_df.to_csv(index=False).encode("utf-8")
        config = data_drift_plugin.parse_config_from_dataset(reference_df_bytes)

        assert config is not None
        assert "features" in config
        assert len(config["features"]) > 0
        # Check that features have expected structure
        for feature in config["features"]:
            assert "name" in feature
            assert "type" in feature
            assert "min" in feature
            assert "max" in feature

    def test_evaluate(self, data_drift_plugin, reference_df, drifted_df):
        reference_df_bytes = reference_df.to_csv(index=False).encode("utf-8")
        evaluated_df_bytes = drifted_df.to_csv(index=False).encode("utf-8")

        data_drift_plugin.set_input_content("reference-dataset", reference_df_bytes)
        data_drift_plugin.set_input_content("evaluated-dataset", evaluated_df_bytes)

        config_data = data_drift_plugin.parse_config_from_dataset(reference_df_bytes)

        results = data_drift_plugin.evaluate(config_data)

        assert isinstance(results, dict)
        # Check all expected metrics are present
        for metric in data_drift_plugin.metric_names():
            assert metric in results
            assert "score" in results[metric][0]
            assert len(results[metric]) > 0
