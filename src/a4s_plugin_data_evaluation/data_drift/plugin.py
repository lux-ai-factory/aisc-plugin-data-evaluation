from a4s_plugin_interface import TaskProgress
from a4s_plugin_interface.models.measure import MetricVisualization, ChartType

from ..base_data_plugin import BaseDataPlugin
from ..data_input_provider import DataFrameProvider
from ..utils import add_metrics, group_metrics
from .drift_detector import TabularDriftDetector


@add_metrics
class DataDriftPlugin(BaseDataPlugin):
    plugin_name = "Data Drift"

    drift_metric_names = [
        "avg_data_drift",
        "count_features_with_drift",
        "data_drift",
        "wasserstein_distance",
        "psi",
        "psi_chi2_p_value",
        "ks_statistic",
        "ks_pvalue",
        "jensenshannon_distance",
        "chi2_statistic",
        "chi2_pvalue",
        "drift_detected",
        "ratio_features_with_drift",
    ]

    @classmethod
    def metric_names(cls):
        return cls.drift_metric_names

    @property
    def display_icon(self) -> str:
        return "alt_route"

    def evaluate(self, config_data: dict):
        config = self.validate_config_form_data(config_data)
        self.logger.info("Starting drift evaluation")
        self.logger.info("Parsed %d features from config", len(config.features))

        target_col = config.target_feature
        date_feature = config.date_feature
        frequency = config.frequency
        window_size = config.window_size

        features = [
            f for f in config.features if f.name not in (target_col, date_feature)
        ]

        if not features:
            self.logger.warning(
                "No input features found after excluding target and date"
            )

        self.logger.debug(
            "Evaluating %d features (excluding target and date)", len(features)
        )

        datasets = self.get_dataset()

        evaluated = datasets["test"]
        reference = datasets.get("train")

        if reference is None:
            self.logger.critical("Reference dataset is missing")
            raise ValueError("Reference dataset is missing.")

        self.logger.debug(
            "Reference shape: %s, Evaluated shape: %s", reference.shape, evaluated.shape
        )

        detector = TabularDriftDetector()
        detector.fit(features, reference)

        assert isinstance(self.dataset_input_provider, DataFrameProvider)
        dates_masks = list(
            self.dataset_input_provider.iter(date_feature, frequency, window_size)
        )
        iterations = len(dates_masks)
        self.logger.info("Processing %d time windows", iterations)

        metrics = []
        for i, (date, mask) in enumerate(dates_masks, start=1):
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

            self.report_progress(
                TaskProgress(progress=i / iterations, extra={"iteration": i})
            )

        self.logger.info("Drift evaluation completed")
        return group_metrics(metrics)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        metrics = self.metric_names()
        is_multivalued = config.date_feature and config.frequency and config.window_size
        chart_type = ChartType.LINE if is_multivalued else ChartType.BARS

        charts = [
            MetricVisualization(chart_type=chart_type, metrics=[metric])
            for metric in metrics
        ]

        return [table, *charts]
