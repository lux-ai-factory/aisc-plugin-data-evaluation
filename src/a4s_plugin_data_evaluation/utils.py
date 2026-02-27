import uuid
from enum import Enum
from typing import Any
from datetime import datetime
from collections import defaultdict

from pydantic import BaseModel, Field, field_serializer

from a4s_plugin_interface import metric
from a4s_plugin_interface.models.measure import Measure


def group_metrics(
    data: list[dict[str, dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group dictionaries by their metric name.

    This function takes a list of dictionaries where each dictionary contains
    a single key representing a metric name, and its value is another dictionary
    containing metric data. It groups all dictionaries sharing the same metric
    name into a list under that metric key.

    Parameters
    ----------
    data : List[Dict[str, Dict[str, Any]]]
        A list of dictionaries, each containing exactly one key (the metric name)
        mapped to a dictionary of metric values.

        Example:
            [
                {"metric1": {"a": 1}},
                {"metric2": {"b": 2}},
                {"metric1": {"c": 3}},
            ]

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        A dictionary mapping each metric name to a list of its corresponding
        metric dictionaries.

        Example:
            {
                "metric1": [{"a": 1}, {"c": 3}],
                "metric2": [{"b": 2}],
            }

    Notes
    -----
    - Assumes each input dictionary contains exactly one key.
    - If a dictionary contains multiple keys, each key-value pair will be grouped independently.
    - The order of values in each list preserves the original input order.
    """
    grouped = defaultdict(list)

    for item in data:
        for key, value in item.items():
            grouped[key].append(value)

    return dict(grouped)


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


def add_metrics(cls):
    # this a class decorator that automatically adds the @metric decorator
    for name in cls.metric_names():

        @metric(name)
        def fct(self, evaluation_output: dict, _name=name) -> list[Measure]:
            measures: list[dict] = evaluation_output.get(_name, [])

            if len(measures) == 0:
                return []
            return [
                Measure(
                    name=_name,
                    score=float(score),
                    time=measure.get("time", datetime.now()),
                    description=measure.get("description"),
                )
                for measure in measures
                if (score := measure["score"]) is not None
            ]

        setattr(cls, f"export_metric_{name}", fct)

    return cls
