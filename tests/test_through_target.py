"""Data drift on what the target answers (AISC's plugin standard, @dataset_through_target): with a target that
has an endpoint (a scorer), every row of both datasets goes through it first, and the answer's columns
(target.score, target.recommendation, ...) drift like any feature. Without an endpoint, drift of the uploads
as before. Run with the stack's plugin interface:

    uv run --with pytest --with-editable <aisc>/shared/plugin-interface python -m pytest -q tests
"""
import json

import numpy as np
import pandas as pd
import pytest

from aisc_plugin_interface.system_under_test import SYSTEM_INPUT, target_access_of
from data_monitor import DataAnomalyPlugin, DataDriftPlugin
from data_monitor.config_form import ConfigForm
from data_monitor.data_drift.plugin import target_features
from data_monitor.utils import FeatureType

PID = "0f7c1e2a-aaaa-bbbb-cccc-1234567890ab"
KEY = "component:0b9c7a1e-0000-4000-8000-000000000002"


def test_drift_goes_through_the_target_when_it_can_and_anomaly_reads_its_inputs():
    assert target_access_of(DataDriftPlugin) == "dataset_through_target"
    assert DataDriftPlugin.dataset_through_target_required is False
    assert set(DataDriftPlugin.dataset_through_target_inputs) == {"reference-dataset", "evaluated-dataset"}
    assert target_access_of(DataAnomalyPlugin) == "inputs"


def test_the_form_has_the_run_settings_of_the_step():
    fields = ConfigForm.model_fields
    assert fields["target_calls_at_once"].default == 1 and fields["target_row_limit"].default == 0


def test_the_targets_numeric_and_few_valued_columns_become_features():
    ref = pd.DataFrame({"amount_eur": [1, 2], "target.score": [500, 700], "target.recommendation": ["Approve", "Review"],
                        "target.explanation": [f"long text {i}" * 20 for i in range(2)], "target.refused": [False, False]})
    ev = ref.assign(**{"target.score": [900, 950], "target.explanation": [f"other {i}" * 20 for i in range(2)]})
    many = pd.DataFrame({"target.request_id": [f"r{i}" for i in range(40)]})
    found = {f.name: f for f in target_features(pd.concat([ref, many]), pd.concat([ev, many]), already={"amount_eur"})}
    assert found["target.score"].type == FeatureType.FLOAT
    assert (found["target.score"].min, found["target.score"].max) == (500.0, 950.0)
    assert found["target.recommendation"].type == FeatureType.CATEGORICAL
    assert "target.explanation" not in found and "target.request_id" not in found     # free text, ids: noise
    assert "amount_eur" not in found                                                   # not the target's


@pytest.fixture
def scorer(stub, monkeypatch):
    def score(request):
        row = json.loads(request["body"])
        s = int(row["income"]) // 1000
        # the reference set gets both answers (a column constant in the reference is not drifted)
        return {"score": s, "recommendation": "Approve" if s > 60 else "Review", "explanation": f"because {s}" * 10}
    stub.route(f"/internal/projects/{PID}/targets/{KEY}/connection", (200, {
        "name": "mcas-score", "label": "MCAS scorer", "kind": "rest", "base_url": stub.base, "method": "POST",
        "path": "/score", "headers": {}, "secret_header": None, "body_template": "{{input}}", "response_path": "$",
        "refusal": None, "model": None, "timeout_s": 5, "secret": None, "updated_at": "2026-10-04T10:00:00Z",
        "allowed_hosts": [stub.host], "denied_addresses": []}))
    stub.route("/score", (200, score))
    monkeypatch.setenv("PLATFORM_URL", stub.base)
    monkeypatch.setenv("PLATFORM_CONNECTIONS_TOKEN", "svc-token")
    return stub


def test_a_run_through_the_scorer_reports_drift_of_its_answers(scorer):
    rng = np.random.default_rng(1)
    reference = pd.DataFrame({"age": rng.integers(20, 60, 40), "income": rng.uniform(30000, 90000, 40)})
    evaluated = pd.DataFrame({"age": rng.integers(20, 60, 40), "income": rng.uniform(120000, 200000, 40)})
    plugin = DataDriftPlugin()
    plugin._set_artifact_callback(lambda name, content: None)
    plugin.set_input_content(SYSTEM_INPUT, json.dumps({"value": f"target:{PID}/{KEY}"}).encode())
    plugin.set_input_content("reference-dataset", reference.to_csv(index=False).encode())
    plugin.set_input_content("evaluated-dataset", evaluated.to_csv(index=False).encode())
    config = plugin.parse_config_from_dataset(reference.to_csv(index=False).encode())
    results = plugin.evaluate(config)
    measured = {m["description"].split(" | ")[0] for ms in results.values() for m in ms if m.get("description")}
    assert {"age", "income", "target.score", "target.recommendation"} <= measured, measured
    assert "target.explanation" not in measured
    assert len([r for r in scorer.seen if r["path"] == "/score"]) == 80
