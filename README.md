# a4s-plugin-data-evaluation

[![CI](https://github.com/lux-ai-factory/a4s-plugin-data-evaluation/actions/workflows/ci.yml/badge.svg)](https://github.com/lux-ai-factory/a4s-plugin-data-evaluation/actions/workflows/ci.yml)

A Python plugin for data evaluation, implementing data drift detection and anomaly detection using statistical methods. Built on the [a4s-plugin-interface](https://github.com/lux-ai-factory/a4s-plugin-interface) framework.

## Features

- **Data Drift Detection**: PSI, KS test, Wasserstein distance, Chi-squared test, Jensen-Shannon divergence
- **Anomaly Detection**: Constraint violations, new/missing categories, distribution outliers
- **Time-windowed evaluation**: Compute metrics over sliding windows with configurable frequency
- **Multiple input formats**: CSV and Parquet datasets

## Installation

```bash
uv sync
```

## Quick Start

Both plugins take a reference dataset (training data) and an evaluated dataset (test data) as `*.csv` or `*.parquet` files.

```python
from a4s_plugin_data_evaluation import DataDriftPlugin

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

Detects distribution drift between reference and evaluated datasets at the feature level.

**Numerical Features:**
| Metric | Description |
|--------|-------------|
| PSI | Population Stability Index |
| KS Statistic | Kolmogorov-Smirnov test statistic |
| KS p-value | Kolmogorov-Smirnov test p-value |
| Wasserstein Distance | Earth mover's distance between distributions |
| Standardized Mean Diff | Absolute difference in means normalized by std |

**Categorical Features:**
| Metric | Description |
|--------|-------------|
| Chi-squared Statistic | Chi-squared test statistic |
| Chi-squared p-value | Chi-squared test p-value |
| Jensen-Shannon Distance | JS divergence between category distributions |
| Total Variance Distance | Half the L1 distance between distributions |

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
src/a4s_plugin_data_evaluation/
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
    └── drift_detector.py    # Drift detection logic
```

## Tech Stack

- **Python** 3.12+
- **Pydantic** - Configuration and validation
- **NumPy/pandas** - Data processing
- **SciPy** - Statistical tests and metrics
- **uv** - Package management
- **Ruff** - Linting and formatting
- **ty** - Type checking

## Plugin Metadata

| Field | Value |
|-------|-------|
| Name | A4S Data Evaluation Plugin |
| Description | A Python plugin for data evaluation, implementing data drift detection and anomaly detection using statistical methods. Supports PSI, KS test, Wasserstein distance, Chi-squared test, Jensen-Shannon divergence for drift detection, and constraint violations, new/missing categories, distribution outliers for anomaly detection. |
| License | - |
| Verification type | Technical test |
| Project | [a4s-plugin-data-evaluation](https://github.com/lux-ai-factory/a4s-plugin-data-evaluation) |
| Branch | main |
| Version | 0.1.2 |
| Project maturity | Deployed |
| Scientific reference | - |
| Verification targets | [Data Drift Detection] [Anomaly Detection] [Data Quality] [Distribution Analysis] |
| Sector | [AI/ML] [Data Science] [MLOps] |
