from typing import Any

from pydantic import BaseModel, Field, model_validator

from .utils import Feature, FeatureType


FORM_UI_SCHEMA: dict[str, dict[str, Any]] = {
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


class ConfigForm(BaseModel):
    target_feature: str | None = Field(default=None, title="Target Feature")
    date_feature: str | None = Field(default=None, title="Date Feature")

    frequency: str = Field(
        default="",
        title="Frequency",
        description="Data frequency of batch starts for time-series analysis (e.g., '30D', '1M')",
        examples=["30D", "1M", "7D"],
    )

    window_size: str = Field(
        default="",
        title="Window Size",
        description="Analysis window size (e.g., '90 days', '3 months')",
        examples=["90 days", "3 months"],
    )

    features: list[Feature] = Field(
        default_factory=list, description="List of features to use for prediction"
    )

    # When the evaluation's target has an endpoint (a scorer), both datasets are first sent through it
    # (AISC's @dataset_through_target), and its answers drift like the other features.
    target_calls_at_once: int = Field(
        default=1, ge=1, le=8, title="Calls to the target at once",
        description="When the target has an endpoint: how many rows are sent to it at the same time.",
    )
    target_row_limit: int = Field(
        default=0, ge=0, title="Rows sent to the target",
        description="When the target has an endpoint: send at most this many rows of each dataset (0: all).",
    )

    @model_validator(mode="after")
    def validate_special_features(self) -> "ConfigForm":
        self.target_feature = self.target_feature or None
        self.date_feature = self.date_feature or None
        self.frequency = self.frequency
        self.window_size = self.window_size

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
