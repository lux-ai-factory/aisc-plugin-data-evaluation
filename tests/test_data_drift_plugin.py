"""Integration tests for DataDriftPlugin, driven by generated case fixtures.

Each case folder under ``tests/files/`` (from ``tests/generate_files.py``) has a
reference CSV, an evaluated CSV, a features-only datashape payload and a
webapp-importable ``config.json``.
"""

import json

import pytest

from data_monitor import DataDriftPlugin
from tests.conftest import CASES, load_case


@pytest.fixture
def data_drift_plugin():
    """Create a data drift plugin instance."""
    return DataDriftPlugin()


EXPECTED_METRICS = [
    "drift_flag",
    "drift_score",
    "Drift Score",
    "Number of Drifted Features",
    "c2st",
    "mmd",
    "ensemble_fraction",
    "psi",
    "ks",
    "wasserstein",
    "jsd_num",
    "smd",
    "anderson_darling",
    "levene",
    "chi2",
    "jsd_cat",
    "tvd",
    "psi_cat",
]


class TestDataDriftPluginMetadata:
    """Metadata and config contract (no data needed)."""

    def test_plugin_name(self, data_drift_plugin):
        assert data_drift_plugin.plugin_name == "Data Drift"

    def test_display_icon(self, data_drift_plugin):
        assert data_drift_plugin.display_icon == "alt_route"

    def test_metric_names(self, data_drift_plugin):
        assert data_drift_plugin.metric_names() == EXPECTED_METRICS

    def test_input_definitions(self, data_drift_plugin):
        assert {input_def.name for input_def in data_drift_plugin.input_definitions} == {
            "reference-dataset",
            "evaluated-dataset",
            "datashape",
        }

    def test_config_defaults(self, data_drift_plugin):
        config = data_drift_plugin.validate_config_form_data({})
        assert config.frequency == ""
        assert config.window_size == ""

    def test_metric_visualizations(self, data_drift_plugin):
        charts = data_drift_plugin.get_metric_visualizations({"frequency": "", "window_size": ""})
        assert isinstance(charts, list)
        assert charts


class TestDataDriftPluginEvaluate:
    """Case-driven evaluate() runs."""

    def test_missing_datashape_raises(self, data_drift_plugin):
        case = load_case("no_drift")
        data_drift_plugin.set_input_content("reference-dataset", case["reference"])
        data_drift_plugin.set_input_content("evaluated-dataset", case["evaluated"])

        with pytest.raises(ValueError, match="Data Shape input is required"):
            data_drift_plugin.evaluate(case["config"])

    @pytest.mark.parametrize("case", CASES)
    def test_evaluate(self, data_drift_plugin, case):
        case_data = load_case(case)
        data_drift_plugin.set_input_content("reference-dataset", case_data["reference"])
        data_drift_plugin.set_input_content("evaluated-dataset", case_data["evaluated"])
        data_drift_plugin.set_input_content(
            "datashape", json.dumps(case_data["datashape"]).encode("utf-8")
        )

        results = data_drift_plugin.evaluate(case_data["config"])

        assert isinstance(results, dict)
        # evaluate() emits the registered metrics plus extra per-feature
        # diagnostics; at minimum every registered metric must be present.
        assert set(EXPECTED_METRICS) <= set(results)
        for metric in EXPECTED_METRICS:
            assert metric in results
            assert results[metric], f"Empty records for metric: {metric}"
            assert "score" in results[metric][0]

        measures = data_drift_plugin.export_metrics(results)
        assert len(measures) > 0
        assert {measure.name for measure in measures} == set(EXPECTED_METRICS)

    def test_no_drift_vs_drift_detected(self, data_drift_plugin):
        """The generated no_drift case must not flag, the drift case should."""
        for case in ["no_drift", "drift"]:
            case_data = load_case(case)
            data_drift_plugin.set_input_content("reference-dataset", case_data["reference"])
            data_drift_plugin.set_input_content("evaluated-dataset", case_data["evaluated"])
            data_drift_plugin.set_input_content(
                "datashape", json.dumps(case_data["datashape"]).encode("utf-8")
            )
            results = data_drift_plugin.evaluate(case_data["config"])
            flag = results["drift_flag"][0]["score"]
            if case == "no_drift":
                assert flag == 0
            else:
                assert flag == 1
