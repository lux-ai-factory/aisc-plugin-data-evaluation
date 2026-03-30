import copy
from abc import abstractmethod
from typing import Any

from a4s_plugin_interface import InputType
from a4s_plugin_interface.decorators.evaluation_input import evaluation_input
from a4s_plugin_interface.base_evaluation_plugin import (
    BaseEvaluationPlugin,
    PluginFeatureFlags,
)
from a4s_plugin_interface.models.measure import Measure

from .utils import Feature, FeatureType
from .config_form import ConfigForm, FORM_UI_SCHEMA
from .data_input_provider import DataFrameProvider


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
    """Base class for data evaluation plugins."""

    form_ui_schema = FORM_UI_SCHEMA

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=True)

    def parse_config_from_dataset(self, file_content: bytes) -> dict | None:
        import pandas as pd

        self.logger.info("Parsing config from dataset")

        config: ConfigForm = ConfigForm(
            target_feature=None,
            date_feature=None,
            frequency="",
            window_size="",
            features=[],
        )

        try:
            input_provider = DataFrameProvider(file_content)
            input_provider.get_data()
            df: pd.DataFrame = input_provider.get_data()
        except Exception:
            self.logger.exception("Failed to load dataset for config parsing")
            raise

        if df.empty:
            self.logger.warning("Dataset is empty, returning default config")
            return config.model_dump()

        self.logger.debug(
            "Dataset loaded with %d rows and %d columns", len(df), len(df.columns)
        )

        for col_name in df.columns:
            col_data = df[col_name]

            feature_type = FeatureType.CATEGORICAL

            # Check for Date
            if pd.api.types.is_datetime64_any_dtype(col_data):
                feature_type = FeatureType.DATE
            elif pd.api.types.is_object_dtype(col_data):
                temp = pd.to_datetime(col_data, errors="coerce")
                if temp.isna().any():
                    self.logger.warning(
                        "Attempted to parse '%s' as a date, but failed", col_name
                    )
                else:
                    feature_type = FeatureType.DATE

            # Check for Numeric
            if feature_type != FeatureType.DATE:
                if pd.api.types.is_integer_dtype(col_data):
                    feature_type = FeatureType.INTEGER
                elif pd.api.types.is_float_dtype(col_data):
                    feature_type = FeatureType.FLOAT

            # Get Min/Max for Numeric types
            if feature_type in [FeatureType.INTEGER, FeatureType.FLOAT]:
                col_min = float(col_data.min()) if not pd.isna(col_data.min()) else 0.0
                col_max = float(col_data.max()) if not pd.isna(col_data.max()) else 0.0
            else:
                # For Categorical or Date, min/max usually aren't numeric ranges
                col_min = 0.0
                col_max = 0.0

            feature: Feature = Feature(
                name=col_name, min=col_min, max=col_max, type=feature_type
            )
            config.features.append(feature)
            self.logger.debug(
                "Detected feature '%s' as %s (min=%.2f, max=%.2f)",
                col_name,
                feature_type,
                col_min,
                col_max,
            )

        self.logger.info("Parsed %d features from dataset", len(config.features))
        return config.model_dump()

    def on_config_change(
        self, form_data: ConfigForm | None
    ) -> tuple[ConfigForm | None, dict[str, Any], dict[str, Any]]:
        config_schema, ui_schema = self.get_full_schema()
        ui_schema = copy.deepcopy(ui_schema)

        if form_data is None:
            ui_schema["date_feature"] = {"ui:widget": "hidden"}
            ui_schema["target_feature"] = {"ui:widget": "hidden"}
            return None, config_schema, ui_schema

        # Convert to dict for property access if needed
        form_dict = (
            form_data.model_dump() if isinstance(form_data, ConfigForm) else form_data
        )

        if (
            "properties" in config_schema
            and "date_feature" in config_schema["properties"]
        ):
            possible_date_features = [
                f["name"]
                for f in form_dict.get("features", [])
                if f["type"] in (FeatureType.DATE, FeatureType.CATEGORICAL)
            ]
            if possible_date_features:
                # NOTE: adding an empty string will force the user to make a choice
                possible_date_features.insert(0, "")
                config_schema["properties"]["date_feature"]["enum"] = (
                    possible_date_features
                )
                default_date = (
                    possible_date_features[0] if possible_date_features else None
                )
                config_schema["properties"]["date_feature"]["default"] = default_date
            else:
                ui_schema["date_feature"] = {"ui:widget": "hidden"}

        if (
            "properties" in config_schema
            and "target_feature" in config_schema["properties"]
        ):
            possible_target_features = [
                f["name"]
                for f in form_dict.get("features", [])
                if f["type"]
                in (FeatureType.INTEGER, FeatureType.FLOAT, FeatureType.CATEGORICAL)
            ]
            if possible_target_features:
                # NOTE: adding an empty string will force the user to make a choice
                possible_target_features.insert(0, "")
                config_schema["properties"]["target_feature"]["enum"] = (
                    possible_target_features
                )
                default_target = (
                    possible_target_features[-1] if possible_target_features else None
                )
                config_schema["properties"]["target_feature"]["default"] = (
                    default_target
                )
            else:
                ui_schema["target_feature"] = {"ui:widget": "hidden"}

        return form_data, config_schema, ui_schema

    @abstractmethod
    def evaluate(self, config_data: dict) -> list[Measure]:
        raise NotImplementedError
