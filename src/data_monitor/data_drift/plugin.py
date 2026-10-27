from typing import Any

from aisc_plugin_interface import (
    MetricVisualization,
    ChartType,
    PluginFeatureFlags,
)

from ..base_data_plugin import BaseDataPlugin
from ..data_input_provider import dataframe_iter
from ..utils import add_metrics, group_metrics


@add_metrics
class DataDriftPlugin(BaseDataPlugin):
    plugin_name = "Data Drift"

    description = (
        "Detects statistical drift between a reference dataset and an evaluated "
        "dataset, per feature and per time window.\n\n"
        "**Configuration:**\n\n"
        "- **Frequency** — data frequency of batch starts for time-series analysis "
        "(e.g. `30D`, `1M`). Leave empty to evaluate the whole test set at once.\n"
        "- **Window Size** — duration of each analysis window (e.g. `90 days`); "
        "together with Frequency this splits the evaluation into sliding time windows.\n\n"
        "**Try it:** upload reference and evaluated datasets, and attach the "
        "project's **Data Shape** so the target and date columns are resolved automatically."
    )

    ui_icon = "alt_route"

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(show_dimensions_visualisation=True)

    drift_metric_names = [
        # Top-line decision
        "drift_flag",
        "drift_score",
        "Drift Score",
        "Number of Drifted Features",
        # Multivariate channel
        "c2st",
        "mmd",
        # Per-feature ensemble
        "ensemble_fraction",
        # Per-feature univariate diagnostics (numeric)
        "psi",
        "ks",
        "wasserstein",
        "jsd_num",
        "smd",
        "anderson_darling",
        "levene",
        # Per-feature univariate diagnostics (categorical)
        "chi2",
        "jsd_cat",
        "tvd",
        "psi_cat",
    ]

    @classmethod
    def metric_names(cls):
        return cls.drift_metric_names

    def evaluate(self, config_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        import pandas as pd

        config = self.validate_config_form_data(config_data)
        self.logger.info("Starting drift evaluation")

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

        # TODO: include alpha
        from .drift_detector import TabularDriftDetector

        detector = TabularDriftDetector()
        detector.fit(features, reference, date_feature=date_feature)

        dates_masks = list(dataframe_iter(evaluated, date_feature, frequency, window_size))
        iterations = len(dates_masks)
        self.logger.info("Processing %d time windows", iterations)

        metrics = []
        for i, (date, mask) in self.progress_bar(
            dates_masks,
            total=iterations,
            start=1,
            with_index=True,
            desc="Data Drift metrics",
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
                    "Drift detection failed for window %d/%d (date=%s)",
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

        self.logger.info("Drift evaluation completed")
        return gr_metrics

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        # table = MetricVisualization(
        #     chart_type=ChartType.TABLE, metrics=self.get_metrics()
        # )

        # metrics = self.metric_names()
        is_multivalued = bool(config.frequency.strip() and config.window_size.strip())

        return [
            MetricVisualization(
                chart_type=ChartType.LINE if is_multivalued else ChartType.BARS,
                metrics=["Drift Score"],
            ),
            MetricVisualization(chart_type=ChartType.BARS, metrics=["Number of Drifted Features"]),
            MetricVisualization(
                chart_type=ChartType.BARS,
                title="Per-Feature Drift Diagnostics",
                description="Univariate drift metrics broken down by feature.",
                metrics=[
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
                ],
                metric_label_dimension="feature",
            ),
            MetricVisualization(
                chart_type=ChartType.BARS,
                title="Ensemble Drift Fraction",
                description="Fraction of univariate metrics flagging drift, per feature.",
                metrics=["ensemble_fraction"],
                metric_label_dimension="feature",
            ),
        ]
