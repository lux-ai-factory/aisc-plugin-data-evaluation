# aisc-plugin-data-evaluation

[![CI](https://github.com/lux-ai-factory/aisc-plugin-data-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/lux-ai-factory/aisc-plugin-data-evaluation/actions/workflows/ci.yml)

A Python plugin for data evaluation, implementing data drift detection and anomaly detection using statistical methods. Built on the [aisc-plugin-interface](https://github.com/lux-ai-factory/aisc-plugin-interface) framework.

## Features

- **Data Drift Detection** with **multivariate decision** (C2ST + MMD) on the joint feature vector and **per-feature univariate diagnostics** (PSI, KS, Wasserstein, JSD, SMD, Anderson-Darling, Levene, Chi², TVD, PSI-categorical)
- **Data-driven thresholds** — every threshold is calibrated from the reference dataset via bootstrap (no hardcoded `> 0.1` folklore)
- **Tabular and time-series modes** — auto-detected from `date_feature` or lag-1 autocorrelation; TS mode uses block bootstrap to preserve temporal structure
- **Linear-time MMD for big data** — auto-switches to Random Fourier Features approximation above 10k rows; subsamples to 5k for fast permutation tests on multi-million-row data
- **Parallel bootstrap calibration** — multivariate (C2ST + MMD) calibration runs in parallel via joblib, ~3× faster on multi-core machines
- **Guard layer** — schema validation, NaN/constant column handling, high-cardinality warnings, small-sample fallback
- **Anomaly Detection**: Constraint violations, new/missing categories, distribution outliers
- **Time-windowed evaluation**: Compute metrics over sliding windows with configurable frequency
- **Multiple input formats**: CSV and Parquet datasets

## In the AISC stack: drift of what the target answers

Data Drift follows AISC's plugin standard (aisc-plugin-interface, PLUGIN_DEVELOPER_GUIDE.md section 13):

- **With a target that has an endpoint** (a scorer, set under Manage, Targets and endpoints), every row of
  both datasets is first sent through it (`@dataset_through_target`), and the answer's columns drift like
  the uploaded features: numbers (`target.score`) and short values with few levels (`target.recommendation`);
  free text and ids are left out. The answers are saved with the run (`target-answers-<dataset>.csv`).
  Settings: **Calls to the target at once** (default 1) and **Rows sent to the target** (0: all).
- **With a target that has no endpoint** (a dataset component), drift of the uploads, as before.
- Data Anomaly only reads its inputs (`@assesses_inputs`).
- **Results as data (0.4.1):** each per-feature measure also carries `dimensions`: `feature`, plus for Data Drift
  the raw `statistic`, the `p_value` (hypothesis tests only) and the `flag` (`yes`/`no`). These are strings,
  since the engine accepts no floats there. The text `description` is unchanged.
- **Default charts (0.4.1):** both plugins' `get_metric_visualizations` add charts per feature after their own:
  Data Drift adds *PSI per feature* and *Drift tests per feature*, and Data Anomaly adds *Checks per feature*. The
  AISC results dashboard shows them as each plugin's default charts.
- Tests: `uv run --with pytest --with-editable <aisc>/shared/plugin-interface python -m pytest -q tests`
  (the stack's plugin interface; PyPI's `aisc-plugin-interface` has no connector).

## Installation

```bash
uv sync
```

## Quick Start

Both plugins take a reference dataset (training data) and an evaluated dataset (test data) as `*.csv` or `*.parquet` files.

```python
from aisc_plugin_data_evaluation import DataDriftPlugin

plugin = DataDriftPlugin()
plugin.set_input_content("reference-dataset", train_data_bytes)
plugin.set_input_content("evaluated-dataset", test_data_bytes)

results = plugin.evaluate({
    "features": [...],
    "target_feature": "label",
    "date_feature": "timestamp",  # optional
    "frequency": "7D",            # optional
    "window_size": "30D",         # optional
})
```

## Plugins

### Data Drift Plugin

Detects distribution drift between reference and evaluated datasets.
The top-line decision comes from a **multivariate** layer (catches correlation/joint drift), and a rich set of **univariate metrics per feature** is computed for interpretability.

#### Top-line decision
| Output | Description |
|--------|-------------|
| `drift_flag` | True if any multivariate detector fires above its calibrated threshold |
| `drift_score` | `max(c2st.score, mmd.score)` in `[0, 1]` |
| `time` | Window timestamp (or evaluation time if no windowing) |

#### Multivariate (decision layer)
| Method | Description |
|--------|-------------|
| **C2ST** | Classifier two-sample test — RandomForest discriminates reference vs evaluated; CV-AUC is the drift signal |
| **MMD** | Maximum Mean Discrepancy with RBF kernel. Auto-switches to **Random Fourier Features** approximation for n > 10,000 (linear time) |

#### Per-feature univariate (diagnostic layer)
**Numerical features:**
| Metric | Description |
|--------|-------------|
| PSI | Population Stability Index |
| KS | Kolmogorov-Smirnov test statistic |
| Wasserstein | Earth mover's distance |
| JSD | Jensen-Shannon distance (binned) |
| SMD | Standardized mean difference |
| Anderson-Darling | Tail-sensitive 2-sample test |
| Levene | Variance / scale shift test |

**Categorical features:**
| Metric | Description |
|--------|-------------|
| Chi² | Chi-squared contingency test |
| JSD | Jensen-Shannon distance on category proportions |
| TVD | Total variation distance |
| PSI-categorical | Categorical version of PSI |

#### Threshold calibration
Every threshold (per-metric, per-feature, plus the two multivariate ones) is **bootstrapped from the reference dataset**: the reference is repeatedly split in half (random rows in tabular mode, contiguous blocks in TS mode), the metric is computed on each split pair (guaranteed no-drift), and the 95th percentile of that null distribution becomes the threshold. No hardcoded magic numbers.

### Data Anomaly Plugin

Detects anomalies in the evaluated dataset compared to the reference dataset.

**Numerical Features:**
| Metric | Description |
|--------|-------------|
| Upper Constraint Violations | Values exceeding feature max |
| Lower Constraint Violations | Values below feature min |
| Distribution Outlier | Values outside mean +/- 3 std |

**Categorical Features:**
| Metric | Description |
|--------|-------------|
| New Categories | Categories in evaluated but not in reference |
| Missing Categories | Categories in reference but not in evaluated |

**Summary Metrics:**
| Metric | Description |
|--------|-------------|
| Anomaly Pass | Features with no anomalies |
| Anomaly Low | Features with minor anomalies |
| Anomaly Severe | Features with severe anomalies |


## Time-Windowed Evaluation

Metrics can be computed over the entire dataset or over temporal windows:

```python
config = {
    "date_feature": "timestamp",  # Column containing dates
    "frequency": "7D",            # Window hop size (e.g., weekly)
    "window_size": "30D",         # Window duration (e.g., 30 days)
}
```

This produces metrics for each time window, useful for monitoring data quality over time.

## Validation & test coverage

The drift pipeline is validated by **95 drift-specific tests** (plus 38 supporting tests for utilities, iterators, anomaly plugin, etc. — 133 total in the suite) across several layers:

| Layer | Tests | Covers |
|---|---|---|
| `test_drift_pipeline.py` | 40 | Tabular & TS drift scenarios, guard layer, output contract, time-windowing |
| `test_drift_detector.py` | 9 | Core detector API (fit, call, metric names) |
| `test_data_drift_plugin.py` | 13 | Plugin-level integration |
| `test_drift_edge_cases.py` | 17 | Schema mismatches, NaN/Inf, unicode, mixed types, integer dtypes, multi-eval safety, determinism, big-data RFF path |
| `test_drift_production_gotchas.py` | 16 | Memory growth over 50 evals, Parquet I/O, date-column edge cases (timezone, string, NaT), subtle drift sensitivity, pandas nullable dtypes (Int64, pd.NA, Categorical, Boolean) |

Plus a `benchmark/` suite for synthetic drift validation with ground-truth labels (kept as regression artifact).

Real-world validation: pipeline runs end-to-end on the bundled credit-scoring data (`test_data_plugin/files/`) with **100% detection** on injected mean / variance / categorical drift and **0% false alarms** on a clean baseline.

## Development

### Setup

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Commands

```bash
# Run tests
uv run pytest

# Run a single test
uv run pytest tests/test_utils.py::TestGroupMetrics::test_group_single_metric

# Linting
uv run ruff check src/

# Linting with auto-fix
uv run ruff check --fix src/

# Type checking
uv run ty check src/

# Format code
uv run ruff format src/

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to run checks before each commit:

- **Ruff** - Linting and formatting
- **ty** - Type checking

Hooks are installed automatically when you run `uv run pre-commit install`. To skip hooks temporarily:

```bash
git commit --no-verify -m "message"
```

### Project Structure

```
src/aisc_plugin_data_evaluation/
├── __init__.py              # Public exports
├── base_data_plugin.py      # Abstract base class for plugins
├── config_form.py           # Pydantic config + UI schema
├── data_input_provider.py   # CSV/Parquet data reader
├── iterators.py             # Date windowing utilities
├── utils.py                 # Shared utilities and decorators
├── data_anomaly/
│   ├── plugin.py            # DataAnomalyPlugin
│   └── anomaly_detector.py  # Anomaly detection logic
└── data_drift/
    ├── plugin.py            # DataDriftPlugin
    ├── drift_detector.py    # Pipeline orchestration (guard, calibrate, evaluate)
    ├── metrics.py           # 11 univariate metrics
    ├── multivariate.py      # C2ST + MMD (exact / RFF auto-switch)
    └── calibration.py       # IID + block bootstrap calibration, autocorr detect

benchmark/                   # Synthetic-data validation (kept in-repo as regression test)
docs/pipeline.png            # Pipeline diagram
docs/pipeline.pdf            # Pipeline diagram (vector)
PIPELINE.md                  # Pipeline diagram with explanatory text
```

## Tech Stack

- **Python** 3.12+
- **Pydantic** - Configuration and validation
- **NumPy/pandas** - Data processing
- **SciPy** - Statistical tests and metrics
- **scikit-learn** - RandomForest (C2ST), RBF kernel (MMD), one-hot encoding, scaler
- **PyArrow** - Parquet input parsing
- **uv** - Package management
- **Ruff** - Linting and formatting
- **ty** - Type checking

## Plugin Metadata

### Data Drift Plugin

| Field | Value |
|-------|-------|
| Name | Data Drift |
| Description | Detects distribution drift between datasets via a multivariate decision layer (C2ST + MMD with auto RFF for big data) and a univariate diagnostic layer (PSI, KS, Wasserstein, JSD, SMD, Anderson-Darling, Levene, Chi², TVD, PSI-categorical) with data-driven calibration in both tabular and time-series modes. |
| License | - |
| Verification type | Technical test |
| Project | [aisc-plugin-data-evaluation](https://github.com/lux-ai-factory/aisc-plugin-data-evaluation) |
| Branch | main |
| Version | 0.3.1 |
| Project maturity | Deployed |
| Scientific reference | - |
| Verification targets | [Data Drift Detection] [Distribution Analysis] |
| Sector | [AI/ML] [Data Science] [MLOps] |

### Data Anomaly Plugin

| Field | Value |
|-------|-------|
| Name | Data Anomaly |
| Description | Detects anomalies including constraint violations, new/missing categories, and distribution outliers. |
| License | - |
| Verification type | Technical test |
| Project | [aisc-plugin-data-evaluation](https://github.com/lux-ai-factory/aisc-plugin-data-evaluation) |
| Branch | main |
| Version | 0.3.1 |
| Project maturity | Deployed |
| Scientific reference | - |
| Verification targets | [Anomaly Detection] [Data Quality] |
| Sector | [AI/ML] [Data Science] [MLOps] |
