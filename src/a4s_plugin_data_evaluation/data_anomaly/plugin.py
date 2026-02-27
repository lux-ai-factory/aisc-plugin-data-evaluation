from a4s_plugin_interface import TaskProgress
from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

from ..base_data_plugin import BaseDataPlugin
from ..utils import add_metrics, group_metrics
from .anomaly_detector import TabularAnomalyDetector


@add_metrics
class DataAnomalyPlugin(BaseDataPlugin):
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

    @property
    def display_icon(self) -> str:
        return "flag"

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

        detector = TabularAnomalyDetector()
        detector.fit(features, reference)

        dates_masks = list(
            self.dataset_input_provider.iter(date_feature, frequency, window_size)
        )
        iterations = len(dates_masks)

        metrics = []
        for i, (date, mask) in enumerate(dates_masks, start=1):
            metrics.extend(detector(evaluated.loc[mask], date))
            self.report_progress(
                TaskProgress(progress=i / iterations, extra={"iteration": i})
            )

        return group_metrics(metrics)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        # config = self.validate_config_form_data(config_data)

        table = MetricVisualization(
            chart_type=ChartType.TABLE, metrics=self.get_metrics()
        )

        pie_metrics = [
            metric_name
            for metric_name in self.metric_names()
            if "Anomaly" in metric_name
        ]

        piechart = MetricVisualization(
            chart_type=ChartType.PIE,
            metrics=pie_metrics,
        )

        return [table, piechart]
