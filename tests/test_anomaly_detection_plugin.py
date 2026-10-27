"""Integration tests for DataAnomalyPlugin, driven by generated case fixtures."""

import json

import pytest

from data_monitor import DataAnomalyPlugin
from tests.conftest import load_case


@pytest.fixture
def data_anomaly_plugin():
    """Create a data anomaly plugin instance."""
    return DataAnomalyPlugin()


EXPECTED_METRICS = [
    "New Categories",
    "Missing Categories",
    "Upper Constraint Violations",
    "Lower Constraint Violations",
    "Distribution Outlier",
    "Anomaly Pass",
    "Anomaly Low",
    "Anomaly Severe",
]


class TestDataAnomalyPluginMetadata:
    """Metadata and config contract (no data needed)."""

    def test_plugin_name(self, data_anomaly_plugin):
        assert data_anomaly_plugin.plugin_name == "Data Anomaly"

    def test_display_icon(self, data_anomaly_plugin):
        assert data_anomaly_plugin.display_icon == "flag"

    def test_metric_names(self, data_anomaly_plugin):
        assert data_anomaly_plugin.metric_names() == EXPECTED_METRICS

    def test_input_definitions(self, data_anomaly_plugin):
        assert {input_def.name for input_def in data_anomaly_plugin.input_definitions} == {
            "reference-dataset",
            "evaluated-dataset",
            "datashape",
        }

    def test_config_defaults(self, data_anomaly_plugin):
        config = data_anomaly_plugin.validate_config_form_data({})
        assert config.frequency == ""
        assert config.window_size == ""

    def test_metric_visualizations(self, data_anomaly_plugin):
        charts = data_anomaly_plugin.get_metric_visualizations({"frequency": "", "window_size": ""})
        assert isinstance(charts, list)
        assert charts


class TestDataAnomalyPluginEvaluate:
    """Case-driven evaluate() runs."""

    def test_missing_datashape_raises(self, data_anomaly_plugin):
        case = load_case("anomaly_normal")
        data_anomaly_plugin.set_input_content("reference-dataset", case["reference"])
        data_anomaly_plugin.set_input_content("evaluated-dataset", case["evaluated"])

        with pytest.raises(ValueError, match="Data Shape input is required"):
            data_anomaly_plugin.evaluate(case["config"])

    @pytest.mark.parametrize("case", ["anomaly_normal", "anomaly_outliers"])
    def test_evaluate(self, data_anomaly_plugin, case):
        case_data = load_case(case)
        data_anomaly_plugin.set_input_content("reference-dataset", case_data["reference"])
        data_anomaly_plugin.set_input_content("evaluated-dataset", case_data["evaluated"])
        data_anomaly_plugin.set_input_content(
            "datashape", json.dumps(case_data["datashape"]).encode("utf-8")
        )

        results = data_anomaly_plugin.evaluate(case_data["config"])

        assert isinstance(results, dict)
        assert set(results) == set(EXPECTED_METRICS)
        for metric in EXPECTED_METRICS:
            assert metric in results
            assert results[metric], f"Empty records for metric: {metric}"
            assert "score" in results[metric][0]

        measures = data_anomaly_plugin.export_metrics(results)
        assert len(measures) > 0
        assert {measure.name for measure in measures} == set(EXPECTED_METRICS)

    def test_outliers_raise_severe_count(self, data_anomaly_plugin):
        """The outliers case must detect more anomalies than the normal case."""
        counts = {}
        for case in ["anomaly_normal", "anomaly_outliers"]:
            case_data = load_case(case)
            data_anomaly_plugin.set_input_content("reference-dataset", case_data["reference"])
            data_anomaly_plugin.set_input_content("evaluated-dataset", case_data["evaluated"])
            data_anomaly_plugin.set_input_content(
                "datashape", json.dumps(case_data["datashape"]).encode("utf-8")
            )
            results = data_anomaly_plugin.evaluate(case_data["config"])
            counts[case] = sum(
                item["score"]
                for metric in (
                    "Upper Constraint Violations",
                    "Lower Constraint Violations",
                    "New Categories",
                )
                for item in results[metric]
            )
        assert counts["anomaly_outliers"] > counts["anomaly_normal"]
