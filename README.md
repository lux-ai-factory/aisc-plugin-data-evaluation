# Data Evaluation Plugins

This repo implements two plugins to evaluate the data quality based on a reference dataset, by reporting the data anomaly and data drift.
These plugins rely on the [a4s-plugin-interface](https://github.com/lux-ai-factory/a4s-plugin-interface).

The reference dataset is the dataset used for training, while the evaluated dataset is the test set.
These dataset can be `*.csv` or `*.parquet` files.

## Data Anomaly Plugin

> Refer to the [plugin file](./src/a4s_plugin_data_evaluation/data_anomaly/plugin.py) for the full implementation.

Anomaly detection is performed on the features level by comparing the range of the numerical features and the categories of the categorical features between the reference and evaluated datasets.
For the former, the values outside and min-max range and mean plus-minus 3 std are reported.
For the latter, we measure the number of the new or missing categories.

## Data Drift Plugin

> Refer to the [plugin file](./src/a4s_plugin_data_evaluation/data_drift/plugin.py) for the full implementation.

Similar to the anomaly detection plugin, the drift of the evaluated dataset compared to the reference dataset is measured on the features level with a distinction between numerical and categorical features.
For the former, we report PSI (Population Stability Index), KS (Kolmogorov-Smirnov) test, Wasserstein distance and the standardized mean difference.
For the latter, the Chi-squared test, the Jensen-Shannon distance and the total variance distance are measured.


# Repo Structure

```bash
.
├── pyproject.toml
├── README.md
├── src
│   └── a4s_plugin_data_evaluation
│       ├── __init__.py
│       ├── base_data_plugin.py
│       ├── config_form.py
│       ├── data_anomaly
│       │   ├── __init__.py
│       │   ├── anomaly_detector.py
│       │   └── plugin.py
│       ├── data_drift
│       │   ├── __init__.py
│       │   ├── drift_detector.py
│       │   └── plugin.py
│       ├── data_input_provider.py
│       ├── iterators.py
│       └── utils.py
└── uv.lock
```
