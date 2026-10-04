"""What the results dashboard reads from these plugins (aisc docs/superpowers/plugin-dashboards-2026-10-04,
T1.1 to T1.6): each per-feature measure names its feature as data (dimensions), and each plugin's default
charts (get_metric_visualizations) group by what its measures carry.

Dimension values are strings: the engine accepts only str, int or bool in dimensions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aisc_plugin_interface import ChartType
from data_monitor import DataAnomalyPlugin, DataDriftPlugin
from data_monitor.data_anomaly.anomaly_detector import TabularAnomalyDetector
from data_monitor.data_drift.drift_detector import TabularDriftDetector
from data_monitor.utils import Feature, FeatureType

RUN_LEVEL = {"drift_flag", "drift_score", "Drift Score", "Number of Drifted Features", "c2st", "mmd"}
OUTCOMES = {"Anomaly Pass", "Anomaly Low", "Anomaly Severe"}


@pytest.fixture
def features():
    return [Feature(name="age", min=0.0, max=100.0, type=FeatureType.INTEGER),
            Feature(name="income", min=0.0, max=200000.0, type=FeatureType.FLOAT),
            Feature(name="category", min=0.0, max=0.0, type=FeatureType.CATEGORICAL)]


@pytest.fixture
def reference():
    rng = np.random.default_rng(42)
    return pd.DataFrame({"age": rng.integers(18, 65, 120), "income": rng.uniform(30000, 150000, 120),
                         "category": rng.choice(["A", "B", "C"], 120)})


@pytest.fixture
def shifted():
    rng = np.random.default_rng(7)
    return pd.DataFrame({"age": rng.integers(40, 90, 120), "income": rng.uniform(60000, 190000, 120),
                         "category": rng.choice(["A", "B", "C", "D"], 120)})


def rows(output):
    """[(metric name, measure dict)] from a detector's output list of {name: measure}."""
    return [(name, m) for item in output for name, m in item.items()]


def measures(plugin_cls, output):
    """The Measures the plugin would export from a detector output, through its @metric methods."""
    grouped = {}
    for name, m in rows(output):
        grouped.setdefault(name, []).append(m)
    plugin = plugin_cls()
    return [x for name in plugin.get_metrics() for x in plugin.export_metrics(grouped) if x.name == name]


# ── T1.1 to T1.3: dimensions ──────────────────────────────────────────────────

def test_t1_1_per_feature_drift_measures_name_their_feature_statistic_and_flag(features, reference, shifted):
    detector = TabularDriftDetector()
    detector.fit(features, reference)
    out = detector(shifted)
    per_feature = [(n, m) for n, m in rows(out) if n not in RUN_LEVEL and n != "ensemble_fraction"]
    assert per_feature
    for name, m in per_feature:
        dims = m["dimensions"]
        assert dims["feature"] in {"age", "income", "category"}, name
        assert set(dims) <= {"feature", "statistic", "p_value", "flag"}, name
        assert all(isinstance(v, str) for v in dims.values()), (name, dims)
        assert dims["flag"] in {"yes", "no"}
        float(dims["statistic"])
        if "p_value" in dims:
            float(dims["p_value"])
        assert m["description"].startswith(dims["feature"] + " | raw_stat=")   # the text stays as it was


def test_t1_1_the_exported_measures_carry_the_dimensions(features, reference, shifted):
    detector = TabularDriftDetector()
    detector.fit(features, reference)
    exported = measures(DataDriftPlugin, detector(shifted))
    psi = [m for m in exported if m.name == "psi"]
    assert psi and all(m.dimensions and m.dimensions["feature"] in {"age", "income"} for m in psi)
    ens = [m for m in exported if m.name == "ensemble_fraction"]
    assert ens and all(m.dimensions == {"feature": m.description} for m in ens)


def test_t1_2_run_level_drift_measures_name_no_feature(features, reference, shifted):
    detector = TabularDriftDetector()
    detector.fit(features, reference)
    for name, m in rows(detector(shifted)):
        if name in RUN_LEVEL:
            assert "feature" not in (m.get("dimensions") or {}), name


def test_t1_3_per_feature_anomaly_measures_name_their_feature(features, reference, shifted):
    detector = TabularAnomalyDetector()
    detector.fit(features, reference)
    for name, m in rows(detector(shifted)):
        if name in OUTCOMES:
            assert not m.get("dimensions"), name
        else:
            assert m["dimensions"] == {"feature": m["description"]}, name


# ── T1.4 to T1.6: default charts ──────────────────────────────────────────────

@pytest.fixture
def no_windows(monkeypatch):
    """A run with no time windows: the form's date, frequency and window are empty."""
    monkeypatch.setattr(DataDriftPlugin, "validate_config_form_data",
                        lambda self, data: SimpleNamespace(date_feature=None, frequency="", window_size=""))


def test_t1_4_drift_keeps_its_charts_first_then_charts_per_feature(no_windows):
    charts = DataDriftPlugin().get_metric_visualizations({})
    assert [c.title for c in charts] == ["Drift Score", "Number of Drifted Features", "PSI per feature",
                                         "Drift tests per feature"]
    assert [c.chart_type for c in charts] == [ChartType.BARS, ChartType.BARS, ChartType.BARS, ChartType.TABLE]
    assert charts[2].metrics == ["psi"] and charts[2].group_by_dimensions == ["feature"]
    assert charts[3].metrics == ["psi", "smd", "ks", "wasserstein", "levene", "chi2", "psi_cat"]
    assert charts[3].group_by_dimensions == ["feature"]
    exported = set(DataDriftPlugin.metric_names())
    assert all(set(c.metrics) <= exported for c in charts)


def test_t1_4_with_time_windows_drift_score_is_a_line(monkeypatch):
    monkeypatch.setattr(DataDriftPlugin, "validate_config_form_data",
                        lambda self, data: SimpleNamespace(date_feature="date", frequency="1D", window_size="7D"))
    assert DataDriftPlugin().get_metric_visualizations({})[0].chart_type == ChartType.LINE


def test_t1_5_anomaly_keeps_its_pie_first_then_checks_per_feature():
    charts = DataAnomalyPlugin().get_metric_visualizations({})
    assert [c.title for c in charts] == ["Outcome", "Checks per feature"]
    assert charts[0].chart_type == ChartType.PIE and set(charts[0].metrics) == OUTCOMES
    assert charts[1].chart_type == ChartType.TABLE and charts[1].group_by_dimensions == ["feature"]
    assert set(charts[1].metrics) == set(DataAnomalyPlugin.metric_names()) - OUTCOMES


def test_t1_6_every_grouping_dimension_is_carried_by_the_measures_it_groups(features, reference, shifted, no_windows):
    drift = TabularDriftDetector()
    drift.fit(features, reference)
    anomaly = TabularAnomalyDetector()
    anomaly.fit(features, reference)
    carried = {}
    for out in (drift(shifted), anomaly(shifted)):
        for name, m in rows(out):
            carried.setdefault(name, set()).update((m.get("dimensions") or {}).keys())
    for chart in DataDriftPlugin().get_metric_visualizations({}) + DataAnomalyPlugin().get_metric_visualizations({}):
        for dim in chart.group_by_dimensions or []:
            for metric in chart.metrics:
                if metric in carried:                 # a metric this small run produced
                    assert dim in carried[metric], (chart.title, metric, dim)
