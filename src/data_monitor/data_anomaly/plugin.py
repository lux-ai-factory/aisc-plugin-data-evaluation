from typing import Any

from aisc_plugin_interface import MetricVisualization, ChartType, PluginFeatureFlags

from ..base_data_plugin import BaseDataPlugin
from ..data_input_provider import dataframe_iter
from ..utils import add_metrics, group_metrics


@add_metrics
class DataAnomalyPlugin(BaseDataPlugin):
    plugin_name = "Data Anomaly"

    description = (
        "Detects anomalies in an evaluated dataset against a reference dataset, "
        "per feature and per time window (new/missing categories, constraint "
        "violations and distribution outliers).\n\n"
        "**Configuration:**\n\n"
        "- **Frequency** — data frequency of batch starts for time-series analysis "
        "(e.g. `30D`, `1M`). Leave empty to evaluate the whole test set at once.\n"
        "- **Window Size** — duration of each analysis window (e.g. `90 days`); "
        "together with Frequency this splits the evaluation into sliding time windows.\n\n"
        "**Try it:** upload reference and evaluated datasets, and attach the "
        "project's **Data Shape** so the target and date columns are resolved automatically."
    )

    ui_icon = "flag"

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(show_dimensions_visualisation=True)

    anomaly_metric_names = [
        "New Categories",
        "Missing Categories",
        "Upper Constraint Violations",
        "Lower Constraint Violations",
        "Distribution Outlier",
        "Anomaly Pass",
        "Anomaly Low",
        "Anomaly Severe",
    ]

    @classmethod
    def metric_names(cls):
        return cls.anomaly_metric_names

    def evaluate(self, config_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        import pandas as pd

        config = self.validate_config_form_data(config_data)
        self.logger.info("Starting anomaly evaluation")

        _, date_feature, features = self._read_datashape()
        frequency = config.frequency.strip()
        window_size = config.window_size.strip()

        if not features:
            self.logger.warning("No input features found after excluding target and date")

        self.logger.debug("Evaluating %d features (excluding target and date)", len(features))

        try:
            reference = self.get_input_data("reference-dataset")
        except Exception:
            self.logger.exception("Failed to load reference dataset")
            raise
        assert isinstance(reference, pd.DataFrame)

        try:
            evaluated = self.get_input_data("evaluated-dataset")
        except Exception:
            self.logger.exception("Failed to load evaluated dataset")
            raise
        assert isinstance(evaluated, pd.DataFrame)

        self.logger.debug(
            "Reference shape: %s, Evaluated shape: %s", reference.shape, evaluated.shape
        )

        from .anomaly_detector import TabularAnomalyDetector

        detector = TabularAnomalyDetector()
        detector.fit(features, reference)

        dates_masks = list(dataframe_iter(evaluated, date_feature, frequency, window_size))
        iterations = len(dates_masks)
        self.logger.info("Processing %d time windows", iterations)

        metrics = []
        for i, (date, mask) in self.progress_bar(
            dates_masks,
            total=iterations,
            start=1,
            with_index=True,
            desc="Data Anomaly metrics",
        ):
            if mask.sum() == 0:
                self.logger.warning(
                    "Window %d/%d (date=%s) has no samples, skipping",
                    i,
                    iterations,
                    date,
                )
                continue
            self.logger.debug(
                "Processing window %d/%d (date=%s, samples=%d)",
                i,
                iterations,
                date,
                mask.sum(),
            )

            try:
                metrics.extend(detector(evaluated.loc[mask], date))
            except Exception:
                self.logger.exception(
                    "Anomaly detection failed for window %d/%d (date=%s)",
                    i,
                    iterations,
                    date,
                )
                raise

        gr_metrics = group_metrics(metrics)

        # save artifact
        df_artifact = pd.DataFrame(
            [
                {"metric": metric, **value}
                for metric, values in gr_metrics.items()
                for value in values
            ]
        )
        self.upload_artifact("results.csv", df_artifact.to_csv(index=False).encode("utf-8"))

        self.logger.info("Anomaly evaluation completed")
        return gr_metrics

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        # config = self.validate_config_form_data(config_data)

        # table = MetricVisualization(
        #     chart_type=ChartType.TABLE, metrics=self.get_metrics()
        # )

        pie_metrics = [
            metric_name for metric_name in self.metric_names() if "Anomaly" in metric_name
        ]

        piechart = MetricVisualization(
            chart_type=ChartType.PIE,
            metrics=pie_metrics,
        )

        per_feature_metrics = [
            "New Categories",
            "Missing Categories",
            "Upper Constraint Violations",
            "Lower Constraint Violations",
            "Distribution Outlier",
        ]

        bars = MetricVisualization(
            chart_type=ChartType.BARS,
            title="Per-Feature Anomalies",
            description="Anomaly counts broken down by feature.",
            metrics=per_feature_metrics,
            metric_label_dimension="feature",
        )

        # # NOTE: add this only if time series ...
        return [piechart, bars]
