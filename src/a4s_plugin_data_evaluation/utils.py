import io
import uuid
import logging
from enum import Enum
from typing import Any
from abc import abstractmethod
from collections import defaultdict

import pandas as pd
from pydantic import BaseModel, Field, model_validator, field_serializer

from a4s_plugin_interface import metric
from a4s_plugin_interface.input_providers.base_input_provider import BaseInputProvider
from a4s_plugin_interface.models.measure import Measure
from a4s_plugin_interface.base_evaluation_plugin import (
    BaseEvaluationPlugin,
    PluginFeatureFlags,
)

from .iterators import DateIterator

logger = logging.getLogger(__name__)


def merge_dicts(dicts: list[dict]) -> dict:
    merged = defaultdict(lambda: defaultdict(list))

    for d in dicts:
        for key, values in d.items():
            for k, v in values.items():
                merged[key][k].append(v)

    return {m: dict(vals) for m, vals in merged.items()}


class FeatureType(str, Enum):
    INTEGER = "Integer"
    FLOAT = "Float"
    CATEGORICAL = "Categorical"
    DATE = "Date"


class Feature(BaseModel):
    pid: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(...)
    min: float = Field(...)
    max: float = Field(...)
    type: FeatureType = Field(...)

    @field_serializer("pid")
    def serialize_pid(self, pid: uuid.UUID | None) -> str | None:
        return str(pid) if pid is not None else None


class ConfigForm(BaseModel):
    frequency: str = Field(
        default="",
        title="Frequency",
        description="Data frequency for time-series analysis (e.g., '30D', '1M')",
        examples=["30D", "1M", "7D"],
    )

    window_size: str = Field(
        default="",
        title="Window Size",
        description="Analysis window size (e.g., '90 days', '3 months')",
        examples=["90 days", "3 months"],
    )

    target_feature: str | None = Field(default=None, title="Target Feature")
    date_feature: str | None = Field(default=None, title="Date Feature")

    features: list[Feature] = Field(
        default_factory=list, description="List of features to use for prediction"
    )

    @model_validator(mode="after")
    def validate_special_features(self):
        self.frequency = self.frequency
        self.window_size = self.window_size
        self.target_feature = self.target_feature or None
        self.date_feature = self.date_feature or None

        if self.target_feature is None:
            return self

        target_feature = next(
            (f for f in self.features if f.name == self.target_feature), None
        )
        if target_feature is None:
            raise ValueError("Target feature must be one of the configured features.")

        if self.date_feature is None:
            return self

        date_feature = next(
            (f for f in self.features if f.name == self.date_feature), None
        )
        if date_feature is None:
            raise ValueError("Date feature must be one of the configured features.")

        if date_feature.type != FeatureType.DATE:
            raise ValueError("Date feature must be of type Date.")

        if target_feature == date_feature:
            raise ValueError("Target feature must be different from date feature.")
        return self


class DataFrameProvider(BaseInputProvider):
    def _read_data(self, file_content: bytes | list[bytes]) -> dict[str, Any]:
        if isinstance(file_content, bytes):
            return {"test": self._read_single_file(file_content)}

        return {
            name: self._read_single_file(f)
            for name, f in zip(("train", "test"), file_content)
        }

    def _read_single_file(self, file_content: bytes) -> Any:
        import pandas as pd

        file_stream = io.BytesIO(file_content)
        try:
            return pd.read_parquet(file_stream)
        except Exception:
            file_stream.seek(0)
            try:
                return pd.read_csv(file_stream)
            except Exception as e:
                raise ValueError("File is neither a valid Parquet nor CSV.") from e

    def iter(
        self,
        date_feature: str,
        frequency: str,
        window_size: str,
        date_round: str = "1 D",
    ):
        if date_feature is not None:
            self._data["test"][date_feature] = pd.to_datetime(
                self._data["test"][date_feature]
            )

        yield from DateIterator(
            self._data["test"], date_feature, frequency, window_size, date_round
        )


def add_metrics(cls):
    # this a class decorator that automatically adds the @metric decorator
    for name in cls.metric_names:

        @metric(name)
        def fct(self, evaluation_output: dict, _name=name) -> list[Measure]:
            values: dict = evaluation_output.get(_name, {})
            scores = values.get("score", [])
            dates = values.get("date", [])
            feature_pids = values.get("feature_pid", [])
            descriptions = values.get("description", [])

            if len(scores) == 0:
                return []

            return [
                Measure(
                    name=_name,
                    score=float(score),
                    **({"time": date} if date is not None else {}),
                    **({"description": description} if description is not None else {}),
                    **({"feature_pid": feature_pid} if feature_pid is not None else {}),
                )
                for (score, date, feature_pid, description) in zip(
                    scores, dates, feature_pids, descriptions
                )
                if score is not None
            ]

        setattr(cls, f"export_metric_{name}", fct)

    return cls


class BaseDataPlugin(BaseEvaluationPlugin[ConfigForm]):
    form_ui_schema = {
        "features": {
            "ui:options": {
                "orderable": False,
                "addable": False,
            },
            "items": {
                "ui:field": "LayoutGridField",
                "ui:layoutGrid": {
                    "ui:row": {
                        "className": "row",
                        "children": [
                            {"ui:col": {"className": "col-4", "children": ["name"]}},
                            {"ui:col": {"className": "col-3", "children": ["min"]}},
                            {"ui:col": {"className": "col-3", "children": ["max"]}},
                            {
                                "ui:col": {
                                    "className": "col-2",
                                    "children": ["type"],
                                }
                            },
                        ],
                    }
                },
            },
        },
    }

    @property
    def feature_flags(self) -> PluginFeatureFlags:
        return PluginFeatureFlags(can_parse_config_from_dataset=True)

    def parse_config_from_dataset(self) -> dict | None:
        config: ConfigForm = ConfigForm(
            frequency="",
            window_size="",
            features=[],
            date_feature=None,
            target_feature=None,
        )

        df: pd.DataFrame = self.get_dataset()["test"]

        for col_name in df.columns:
            col_data = df[col_name]

            feature_type = FeatureType.CATEGORICAL

            # Check for Date
            if pd.api.types.is_datetime64_any_dtype(col_data):
                feature_type = FeatureType.DATE
            elif pd.api.types.is_object_dtype(col_data):
                temp = pd.to_datetime(col_data, errors="coerce")
                if temp.isna().any():
                    logger.warning(
                        f"Attempted to parse {col_name} as a date, but failed."
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

        return config.model_dump()

    def on_config_change(
        self, form_data: dict | None
    ) -> tuple[dict | None, dict, dict]:
        config_schema, ui_schema = self.get_full_schema()

        if form_data is None:
            ui_schema["date_feature"] = {"ui:widget": "hidden"}
            ui_schema["target_feature"] = {"ui:widget": "hidden"}
            return None, config_schema, ui_schema

        if (
            "properties" in config_schema
            and "date_feature" in config_schema["properties"]
        ):
            possible_date_features = [
                f["name"]
                for f in form_data.get("features", [])
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
                for f in form_data.get("features", [])
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
        self.dataset_input_provider = DataFrameProvider(file_content)
        return self.dataset_input_provider

    @abstractmethod
    def evaluate(self, config_data: dict) -> list[Measure]:
        raise NotImplementedError
