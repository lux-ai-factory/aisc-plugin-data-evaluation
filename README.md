# aisc-plugin-data-evaluation

[![CI](https://github.com/lux-ai-factory/aisc-plugin-data-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/lux-ai-factory/aisc-plugin-data-evaluation/actions/workflows/ci.yml)

A Python plugin for data evaluation, implementing data drift detection and anomaly detection using statistical methods. Built on the [aisc-plugin-interface](https://github.com/lux-ai-factory/aisc-plugin-interface) framework.

## Features

- **Data Drift Detection** with **multivariate decision** (C2ST + MMD) on the joint feature vector and **per-feature univariate diagnostics** (PSI, KS, Wasserstein, JSD, SMD, Anderson-Darling, Levene, Chi², TVD, PSI-categorical)
- **Data-driven thresholds** — every threshold is calibrated from the reference dataset via bootstrap (no hardcoded `> 0.1` folklore)
- **Tabular and time-series modes** — auto-detected from the `date` role in the data shape or lag-1 autocorrelation; TS mode uses block bootstrap to preserve temporal structure
- **Linear-time MMD for big data** — auto-switches to Random Fourier Features approximation above 10k rows; subsamples to 5k for fast permutation tests on multi-million-row data
- **Parallel bootstrap calibration** — multivariate (C2ST + MMD) calibration runs in parallel via joblib, ~3× faster on multi-core machines
- **Guard layer** — schema validation, NaN/constant column handling, high-cardinality warnings, small-sample fallback
- **Anomaly Detection**: Constraint violations, new/missing categories, distribution outliers
- **Time-windowed evaluation**: Compute metrics over sliding windows with configurable frequency
- **Multiple input formats**: CSV and Parquet datasets

## Installation

```bash
uv sync
```

## Quick Start

Both plugins take a reference dataset (training data) and an evaluated dataset (test data) as `*.csv` or `*.parquet` files, plus the project's **Data Shape** AI component (features, target, date):

```python
from data_monitor import DataDriftPlugin

plugin = DataDriftPlugin()
plugin.set_input_content("reference-dataset", train_data_bytes)
plugin.set_input_content("evaluated-dataset", test_data_bytes)
plugin.set_input_content("datashape", datashape_json_bytes)  # project AI component

results = plugin.evaluate({
    "frequency": "7D",   # optional
    "window_size": "30D", # optional
})
```

The **Data Shape** is a first-class AI component of the AISC platform, derived from
the reference dataset on the project Settings page. It carries the feature list and
the `target`/`date`/`feature` column roles — these are intentionally **not** part of
the plugin config (interface `>= 0.3.0`).

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

Metrics can be computed over the entire dataset or over temporal windows.
Set a **Frequency** + **Window Size** in the config to slide windows over time
(the date column itself is resolved from the **Data Shape** input):

```python
config = {
    "frequency": "7D",    # Window hop size (e.g., weekly)
    "window_size": "30D", # Window duration (e.g., 30 days)
}
```

This produces metrics for each time window, useful for monitoring data quality over time.

## Validation & test coverage

The suite is **65 tests** across the smoke, detector, plugin and utility layers,
driven by deterministic generated fixtures:

| Test file | Tests | Covers |
|---|---|---|
| `test_smoke.py` | 2 | Plugin importable + `export_metrics(None)` returns `[]` |
| `test_utils.py` | 11 | Feature models, `group_metrics`, `@add_metrics` |
| `test_iterators.py` | 9 | `get_date_batches`, `DateIterator` |
| `test_drift_detector.py` | 7 | Core detector API (fit, call, metric names) |
| `test_anomaly_detector.py` | 8 | Anomaly detection logic (violations, categories, outliers) |
| `test_data_drift_plugin.py` | 12 | Metadata, config, missing-input, case-driven evaluate |
| `test_anomaly_detection_plugin.py` | 10 | Metadata, config, missing-input, case-driven evaluate |
| `test_config_form.py` | 3 | Config defaults + validation |
| `test_data_input_provider.py` | 3 | CSV/Parquet reading |

Fixtures are **generated**, not committed: `tests/generate_files.py` deterministically
derives each case (reference + evaluated CSVs, features-only datashape payload,
webapp-importable `config.json`) into `tests/files/<case>/` (gitignored), and
`tests/conftest.py` regenerates them automatically at collection time. Cases:
`no_drift`, `drift`, `anomaly_normal`, `anomaly_outliers`.

```bash
uv run python -m pytest -q tests/   # 65 passed
uv run python tests/generate_files.py  # manual regeneration
```

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
src/data_monitor/
├── __init__.py              # Public exports (DataDriftPlugin, DataAnomalyPlugin)
├── base_data_plugin.py      # Abstract base class (config, datashape reading, providers)
├── config_form.py           # Pydantic config (frequency, window_size)
├── data_input_provider.py   # CSV/Parquet data reader
├── iterators.py             # Date windowing utilities
├── utils.py                 # Shared utilities and decorators
├── data_anomaly/
│   ├── plugin.py            # DataAnomalyPlugin
│   └── anomaly_detector.py  # Anomaly detection logic
└── data_drift/
    ├── plugin.py            # DataDriftPlugin
    ├── drift_detector.py    # Pipeline orchestration (guard, calibrate, evaluate)
    ├── metrics.py           # Univariate metrics
    ├── multivariate.py      # C2ST + MMD (exact / RFF auto-switch)
    └── calibration.py       # IID + block bootstrap calibration, autocorr detect

tests/
├── test_utils.py             # Feature models, group_metrics, @add_metrics
├── test_iterators.py         # Date windowing utilities
├── test_drift_detector.py    # Core drift detector API
├── test_anomaly_detector.py  # Anomaly detection logic
├── test_data_drift_plugin.py # Plugin-level drift integration
├── test_anomaly_detection_plugin.py  # Plugin-level anomaly integration
├── test_config_form.py       # Config defaults + validation
└── test_data_input_provider.py      # CSV/Parquet reading
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
