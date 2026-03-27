import logging
from abc import abstractmethod
from typing import Any

from a4s_plugin_interface.base_evaluation_plugin import (
    BaseEvaluationPlugin,
    PluginFeatureFlags,
)
from a4s_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from a4s_plugin_interface.models.measure import Measure

from .utils import Feature, FeatureType
from .config_form import ConfigForm, FORM_UI_SCHEMA
from .data_input_provider import DataFrameProvider


class BaseDataPlugin(BaseEvaluationPlugin[ConfigForm]):
    _logger: logging.Logger | None = None

    @property
    def logger(self) -> logging.Logger:
        """Return logger from parent class, or create a fallback if not available."""
        if (parent_logger := getattr(super(), "logger", None)) is not None:
            return parent_logger

        if self._logger is None:
            self._logger = logging.getLogger(f"a4s_eval.{self.__class__.__name__}")

        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        """Allow setting the logger (useful for testing)."""
        self._logger = value

    form_ui_schema = FORM_UI_SCHEMA

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=True)

    def parse_config_from_dataset(self) -> dict[str, Any] | None:
        import pandas as pd

        self.logger.info("Parsing config from dataset")

        config: ConfigForm = ConfigForm(
            frequency="",
            window_size="",
            features=[],
            date_feature=None,
            target_feature=None,
        )

        try:
            df: pd.DataFrame = self.get_dataset()["test"]
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

    def set_dataset_input_provider(
        self, file_content: bytes | list[bytes] | None
    ) -> BaseInputProvider:
        self.logger.debug("Setting dataset input provider")

        if file_content is None:
            self.logger.critical("Dataset file content is None")

        try:
            self.dataset_input_provider = DataFrameProvider(
                file_content  # ty: ignore[invalid-argument-type]
            )
        except Exception:
            self.logger.exception("Failed to initialize dataset input provider")
            raise

        return self.dataset_input_provider

    @abstractmethod
    def evaluate(self, config_data: dict) -> list[Measure]:
        raise NotImplementedError
