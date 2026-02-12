import logging
import pandas as pd
from itertools import chain
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from a4s_plugin_interface.models.measure import Measure, MetricVisualization, ChartType

from ..utils import BaseDataPlugin, Feature, FeatureType, add_metrics, merge_dicts

logger = logging.getLogger(__name__)


def _safe_float(x):
    """Convert to JSON-safe float."""
    import numpy as np

    if np.isnan(x) or np.isinf(x):
        return None
    return float(x)


def _psi_metric(
    x_ref,
    x_eval,
    bins: int = 10,
    min_bin_pct: float = 0.02,
    smoothing: float = 0.5,
) -> dict[str, float | None]:
    import numpy as np
    from scipy.stats import chi2

    # Ensure numpy arrays
    x_ref = np.asarray(x_ref).ravel()
    x_eval = np.asarray(x_eval).ravel()

    # Remove NaNs
    x_ref = x_ref[~np.isnan(x_ref)]
    x_eval = x_eval[~np.isnan(x_eval)]

    if x_ref.size == 0 or x_eval.size == 0:
        raise ValueError("Empty input after removing NaNs.")

    n_ref = x_ref.size
    n_eval = x_eval.size

    # 1️⃣ Freeze binning using reference quantiles
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(x_ref, quantiles))

    if len(bin_edges) <= 2:
        logger.debug("Not enough variation in reference data.")
        return {"psi": None, "psi_chi2_p_value": None}

    # 2️⃣ Compute counts
    ref_counts, _ = np.histogram(x_ref, bins=bin_edges)
    eval_counts, _ = np.histogram(x_eval, bins=bin_edges)

    # 3️⃣ Enforce minimum bin population
    ref_pct_raw = ref_counts / n_ref
    if np.any(ref_pct_raw < min_bin_pct):
        logger.debug(
            "Some bins below minimum population threshold. "
            "Reduce bin count or merge bins."
        )
        return {"psi": None, "psi_chi2_p_value": None}

    # 4️⃣ Apply smoothing (Haldane–Anscombe correction)
    ref_counts = ref_counts + smoothing
    eval_counts = eval_counts + smoothing

    ref_pct = ref_counts / ref_counts.sum()
    eval_pct = eval_counts / eval_counts.sum()

    # 5️⃣ Compute PSI
    psi_values = (ref_pct - eval_pct) * np.log(ref_pct / eval_pct)
    psi_total = float(np.sum(psi_values))

    # 6️⃣ Chi-square test
    chi_stat = 2 * n_eval * psi_total
    # NOTE: drift detected if p_value < 0.05
    p_value = float(1 - chi2.cdf(chi_stat, len(ref_counts) - 1))

    return {"psi": _safe_float(psi_total), "psi_chi2_p_value": _safe_float(p_value)}


def numerical_drift_test(
    x_ref: "pd.Series[float]", x_eval: "pd.Series[float]"
) -> dict[str, float]:
    """Calculate drift between two numerical distributions using Wasserstein distance.

    Args:
        x_ref: Reference distribution as pandas Series
        x_eval: Evaluated distribution to compare against reference

    Returns:
        float: Wasserstein distance between the distributions
    """
    from scipy.stats import wasserstein_distance

    logger.debug(
        f"Computing numerical drift test - Reference shape: {x_ref.shape}, "
        f"Evaluated shape: {x_eval.shape}"
    )

    x_ref = x_ref.to_numpy()
    x_eval = x_eval.to_numpy()

    distance = wasserstein_distance(x_ref, x_eval)
    logger.debug(f"Wasserstein distance computed: {distance}")
    metrics = {"wasserstein_distance": distance, **_psi_metric(x_ref, x_eval)}
    return metrics


def categorical_drift_test(
    x_ref: "pd.Series[int]", x_eval: "pd.Series[int]"
) -> dict[str, float]:
    """Calculate drift between two categorical distributions using Jensen-Shannon distance.

    Args:
        x_ref: Reference distribution as pandas Series
        x_eval: Evaluated distribution to compare against reference

    Returns:
        float: Jensen-Shannon distance between the distributions
    """
    from scipy.spatial.distance import jensenshannon

    logger.debug(
        f"Computing categorical drift test - Reference shape: {x_ref.shape}, "
        f"Evaluated shape: {x_eval.shape}"
    )

    # Get all unique values from both series
    all_categories = pd.Index(x_ref.unique()).union(pd.Index(x_eval.unique()))
    logger.debug(f"Total unique categories: {len(all_categories)}")

    # Compute normalized value counts for both distributions
    ref_counts = x_ref.value_counts(normalize=True)
    eval_counts = x_eval.value_counts(normalize=True)

    # Reindex to ensure both have the same categories (fill missing with 0)
    ref_dist = ref_counts.reindex(all_categories, fill_value=0.0)
    eval_dist = eval_counts.reindex(all_categories, fill_value=0.0)

    distance = jensenshannon(ref_dist.to_numpy(), eval_dist.to_numpy(), base=2) ** 2
    logger.debug(f"Jensen-Shannon distance computed: {distance}")
    return {"jensenshannon": distance}


def feature_drift_test(
    feature: Feature,
    date: pd.Timestamp,
    mask: pd.Series,
    evaluated: pd.DataFrame,
    reference: pd.DataFrame,
) -> Measure:
    """Calculate drift for a specific feature based on its type.

    Args:
        feature: Feature
        date: Timestamp for the metric
        mask: Boolean Series used to filter the DataFrame
        reference: Reference DataFrame
        evaluated: Evaluated DataFrame

    Returns:
        Measure: Drift metric object with computed score

    Raises:
        ValueError: If feature type is not supported
    """
    feature_type = feature.type
    logger.debug(f"Processing feature: {feature.name} (type: {feature_type})")
    logger.debug(f"Computing feature drift test for feature type: {feature_type}")

    masked_evaluated = evaluated.loc[mask]

    metrics = {}

    if feature_type == FeatureType.INTEGER or feature_type == FeatureType.FLOAT:
        metrics = numerical_drift_test(
            reference[feature.name], masked_evaluated[feature.name]
        )
        for metric_name, score in metrics.items():
            logger.debug(f"Created numerical drift metric: {metric_name} = {score}")

    elif feature_type == FeatureType.CATEGORICAL:
        metrics = categorical_drift_test(
            reference[feature.name], masked_evaluated[feature.name]
        )
        for metric_name, score in metrics.items():
            logger.debug(f"Created categorical drift metric: {metric_name} = {score}")

    else:
        logger.error(f"Unsupported feature type: {feature_type}")
        raise ValueError(f"Feature type {feature_type} not supported")

    if metrics:
        results = []
        for metric_name, score in metrics.items():
            logger.debug(
                f"Added metric for feature {feature.name}: {metric_name} = {score}"
            )
            results.append(
                {
                    metric_name: dict(
                        score=score,
                        date=date,
                        description=feature.name,
                        feature_pid=feature.pid,
                    )
                }
            )
        return results


@add_metrics
class DataDriftPlugin(BaseDataPlugin):
    metric_names = [
        "wasserstein_distance",
        "psi",
        "psi_chi2_p_value",
        "jensenshannon",
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

        it = (
            (feature, date, mask)
            for date, mask in df_date_iterator
            for feature in features
        )

        compute_metric = partial(
            feature_drift_test, reference=reference, evaluated=evaluated
        )

        with ThreadPoolExecutor() as pool:
            metrics = list(chain.from_iterable(pool.map(compute_metric, *(zip(*it)))))

        return merge_dicts(metrics)

    def get_metric_visualizations(self, config_data: dict) -> list[MetricVisualization]:
        config = self.validate_config_form_data(config_data)

        metrics = self.get_metrics()

        table = MetricVisualization(chart_type=ChartType.TABLE, metrics=metrics)

        is_multivalued = config.date_feature and config.frequency and config.window_size
        chart_type = ChartType.LINE if is_multivalued else ChartType.BARS

        charts = [
            MetricVisualization(chart_type=chart_type, metrics=[metric])
            for metric in metrics
        ]

        return [table, *charts]
