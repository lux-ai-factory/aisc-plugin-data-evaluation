from abc import abstractmethod
from typing import Any, cast

from aisc_plugin_interface import (
    BaseEvaluationPlugin,
    InputType,
    evaluation_input,
)

from .utils import Feature, feature_from_datashape
from .config_form import ConfigForm
from .data_input_provider import DataFrameProvider


@evaluation_input(
    name="datashape",
    label="Data Shape",
    input_type=InputType.DATASHAPE,
    required=False,
)
@evaluation_input(
    name="evaluated-dataset",
    label="Evaluated Dataset",
    input_provider_class=DataFrameProvider,
    input_type=InputType.DATASET,
)
@evaluation_input(
    name="reference-dataset",
    label="Reference Dataset",
    input_provider_class=DataFrameProvider,
    input_type=InputType.DATASET,
)
class BaseDataPlugin(BaseEvaluationPlugin[ConfigForm]):
    """Base class for data evaluation plugins.

    The data shape (features, target, date) is an AI component owned by the
    project and consumed as a ``datashape`` evaluation input at run time.
    """

    @staticmethod
    def _datashape_features(payload: Any) -> list[dict[str, Any]]:
        """Normalize a datashape payload to a list of feature dicts.

        Accepts a full ``DataShape`` payload and the default features-only
        payload the platform emits (``{"features": [...]}``).
        """
        if not isinstance(payload, dict):
            return []
        raw_features = payload.get("features")
        if not isinstance(raw_features, list):
            return []
        return [cast("dict[str, Any]", f) for f in raw_features if isinstance(f, dict)]

    def _read_datashape(self) -> tuple[str | None, str | None, list[Feature]]:
        """Resolve (target, date, model features) from the datashape input.

        The datashape component's ``json_value`` is the feature schema: a full
        ``DataShape`` payload or the default features-only payload
        (``{"features": [...]}``).  Feature ``role`` designates the target, the
        optional date axis, and the model features.  Unlike model plugins, data
        evaluation does not require a designated target.

        Returns:
            Tuple of (target column or None, date column or None, feature objects).

        Raises:
            ValueError: If the datashape input is missing or has no features.
        """
        payload = self.get_input_data("datashape")
        if payload is None:
            raise ValueError(
                "The Data Shape input is required for evaluation. "
                "Create a data shape from the uploaded dataset on the project "
                "Settings page and attach it as an evaluation input."
            )

        features = self._datashape_features(payload)
        if not features:
            raise ValueError(
                "The Data Shape input carries no features; a data shape must be "
                "derived from a dataset on the project Settings page."
            )

        target_col = next((str(f.get("name")) for f in features if f.get("role") == "target"), None)
        date_feature = next((str(f.get("name")) for f in features if f.get("role") == "date"), None)
        model_features = [feature_from_datashape(f) for f in features if f.get("role") == "feature"]

        return target_col, date_feature, model_features

    @abstractmethod
    def evaluate(self, config_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError
