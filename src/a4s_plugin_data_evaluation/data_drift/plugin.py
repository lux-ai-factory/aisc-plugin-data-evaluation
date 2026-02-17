from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

from ..utils import BaseDataPlugin, add_metrics, merge_dicts
from .drift_detector import TabularDriftDetector
from .data_shift import DataShiftMonitor


@add_metrics
class DataDriftPlugin(BaseDataPlugin):
    metric_names = [
        "data_shift",
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

    @property
    def display_icon(self) -> str:
        return "alt_route"

    def evaluate(self, config_data: dict) -> list[Measure]:
        config = self.validate_config_form_data(config_data)

        target_col = config.target_feature
        date_feature = config.date_feature
        frequency = config.frequency
        window_size = config.window_size

        features = [
            f for f in config.features if f.name not in (target_col, date_feature)
        ]

        datasets = self.get_dataset()

        evaluated = datasets["test"]
        reference = datasets.get("train")
        if reference is None:
            raise ValueError("Reference dataset is missing.")

        df_date_iterator = self.dataset_input_provider.iter(
            date_feature, frequency, window_size
        )

        detector = TabularDriftDetector()
        detector.fit(features, reference)

        data_shift_monitor = DataShiftMonitor()
        data_shift_monitor.fit(features, reference)

        metrics = []

        for date, mask in df_date_iterator:
            metrics.extend(detector(date, mask, evaluated))
            metrics.extend(data_shift_monitor(date, mask, evaluated))

        return merge_dicts(metrics)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        metrics = self.metric_names
        is_multivalued = config.date_feature and config.frequency and config.window_size
        chart_type = ChartType.LINE if is_multivalued else ChartType.BARS

        charts = [
            MetricVisualization(chart_type=chart_type, metrics=[metric])
            for metric in metrics
        ]

        return [table, *charts]
