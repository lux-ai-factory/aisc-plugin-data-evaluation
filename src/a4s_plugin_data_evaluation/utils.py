import uuid
from enum import Enum
from collections import defaultdict
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, Field, field_serializer

from a4s_plugin_interface import metric
from a4s_plugin_interface.models.measure import Measure


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


def group_metrics(
    dicts: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, list[Any]]]:
    """Group a list of metric dictionaries by metric name into a single dictionary.

    Args:
        dicts (list[dict]): A list of dictionaries to group.

    Returns:
        dict: A grouped dictionary with consolidated values.

    Example:
        >>> dicts = [{"a": {"x": 1}}, {"a": {"x": 2, "y": 3}, "b": {"z": 4}}]
        >>> result = group_metrics(dicts)
        >>> print(result)
        {'a': {'x': [1, 2], 'y': [3]}, 'b': {'z': [4]}}
    """
    merged = defaultdict(lambda: defaultdict(list))

    for d in dicts:
        for key, values in d.items():
            for k, v in values.items():
                merged[key][k].append(v)

    return {m: dict(vals) for m, vals in merged.items()}


def add_metrics(cls: type[T]) -> type[T]:
    """Class decorator that automatically adds @metric decorated export methods."""
    for name in cls.metric_names():

        @metric(name)
        def fct(
            self, evaluation_output: dict[str, dict[str, list[Any]]], _name: str = name
        ) -> list[Measure]:
            values: dict[str, list[Any]] = evaluation_output.get(_name, {})
            scores = values.get("score", [])
            dates = values.get("date", [])

            if len(scores) == 0:
                return []

            return [
                Measure(
                    name=_name,
                    score=float(score),
                    **({"time": date} if date is not None else {}),
                )
                for (score, date) in zip(scores, dates)
            ]

        setattr(cls, f"export_metric_{name}", fct)

    return cls
