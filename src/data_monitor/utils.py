import uuid
from datetime import datetime
from enum import Enum
from collections import defaultdict
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, field_serializer

from aisc_plugin_interface import Measure, metric


class _HasMetricNames(Protocol):
    @classmethod
    def metric_names(cls) -> list[str]: ...


T = TypeVar("T", bound=_HasMetricNames)


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


def feature_from_datashape(feature: dict[str, Any]) -> Feature:
    """Map a datashape feature dict onto the local :class:`Feature` model.

    The platform's ``semantic_type``/``dtype`` fields are translated into the
    plugin's :class:`FeatureType` so the drift/anomaly detectors keep working
    unchanged against datashape-derived features.
    """
    semantic_type = feature.get("semantic_type")
    dtype = feature.get("dtype", "")

    if semantic_type == "categorical":
        feature_type = FeatureType.CATEGORICAL
    elif semantic_type == "datetime":
        feature_type = FeatureType.DATE
    elif semantic_type in ("numeric", "boolean"):
        feature_type = (
            FeatureType.INTEGER
            if str(dtype).startswith(("int", "uint", "bool"))
            else FeatureType.FLOAT
        )
    else:
        feature_type = FeatureType.CATEGORICAL

    return Feature(
        name=feature["name"],
        min=float(feature.get("min") or 0.0),
        max=float(feature.get("max") or 0.0),
        type=feature_type,
    )


def group_metrics(
    dicts: list[dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Group a list of metric dictionaries by metric name into a single dictionary.

    Args:
        dicts (list[dict]): A list of dictionaries to group.

    Returns:
        dict: A grouped dictionary with consolidated values.

    Example:
        >>> dicts = [{"a": {"x": 1}}, {"a": {"x": 2, "y": 3}, "b": {"z": 4}}]
        >>> result = group_metrics(dicts)
        >>> print(result)
        {'a': [{'x': 1}, {'x': 2, 'y': 3}], 'b': [{'z': 4}]}
    """
    grouped = defaultdict(list)

    for item in dicts:
        for key, value in item.items():
            grouped[key].append(value)

    return dict(grouped)


def add_metrics(cls: type[T]) -> type[T]:
    """Class decorator that automatically adds @metric decorated export methods."""
    for name in cls.metric_names():

        @metric(name)
        def fct(self, evaluation_output: dict | None, _name=name) -> list[Measure]:
            measures: list[dict] = evaluation_output.get(_name, []) if evaluation_output else []

            return [
                Measure(
                    name=_name,
                    score=float(score),
                    time=measure.get("time", datetime.now()),
                    description=measure.get("description"),
                    dimensions=measure.get("dimensions"),
                    direction=measure.get("direction"),
                )
                for measure in measures
                if (score := measure["score"]) is not None
            ]

        setattr(cls, f"export_metric_{name}", fct)

    return cls
