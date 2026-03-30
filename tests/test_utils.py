"""Tests for utility functions."""

from a4s_plugin_data_evaluation.utils import (
    Feature,
    FeatureType,
    group_metrics,
)


class TestFeatureType:
    """Tests for FeatureType enum."""

    def test_feature_type_values(self):
        assert FeatureType.INTEGER.value == "Integer"
        assert FeatureType.FLOAT.value == "Float"
        assert FeatureType.CATEGORICAL.value == "Categorical"
        assert FeatureType.DATE.value == "Date"

    def test_feature_type_is_string_enum(self):
        assert isinstance(FeatureType.INTEGER, str)
        assert FeatureType.INTEGER == "Integer"


class TestFeature:
    """Tests for Feature model."""

    def test_feature_creation(self):
        feature = Feature(name="test_col", min=0.0, max=100.0, type=FeatureType.FLOAT)
        assert feature.name == "test_col"
        assert feature.min == 0.0
        assert feature.max == 100.0
        assert feature.type == FeatureType.FLOAT

    def test_feature_has_pid(self):
        feature = Feature(name="age", min=18.0, max=65.0, type=FeatureType.INTEGER)
        assert feature.pid is not None

    def test_feature_pid_serialization(self):
        feature = Feature(name="age", min=18.0, max=65.0, type=FeatureType.INTEGER)
        dump = feature.model_dump()
        assert isinstance(dump["pid"], str)


class TestGroupMetrics:
    """Tests for group_metrics function."""

    def test_group_single_metric(self):
        data = [{"metric1": {"score": 0.9}}]
        result = group_metrics(data)
        assert result == {"metric1": [{"score": 0.9}]}

    def test_group_multiple_same_metrics(self):
        data = [
            {"metric1": {"score": 0.9}},
            {"metric1": {"score": 0.85}},
        ]
        result = group_metrics(data)
        assert result == {"metric1": [{"score": 0.9}, {"score": 0.85}]}

    def test_group_different_metrics(self):
        data = [
            {"metric1": {"score": 0.9}},
            {"metric2": {"score": 0.7}},
        ]
        result = group_metrics(data)
        assert "metric1" in result
        assert "metric2" in result
        assert isinstance(result["metric1"], list) and len(result["metric1"]) == 1
        assert isinstance(result["metric2"], list) and len(result["metric2"]) == 1
        assert result["metric1"][0] == {"score": 0.9}
        assert result["metric2"][0] == {"score": 0.7}

    def test_group_empty_list(self):
        result = group_metrics([])
        assert result == {}

    def test_group_preserves_order(self):
        data = [
            {"metric1": {"a": 1}},
            {"metric1": {"a": 2}},
            {"metric1": {"a": 3}},
        ]
        result = group_metrics(data)
        for item, value in zip(result["metric1"], [1, 2, 3]):
            assert item["a"] == value

    def test_group_multiple_keys(self):
        data = [
            {"a": {"x": 1}},
            {"a": {"x": 2, "y": 3}, "b": {"z": 4}},
        ]
        result = group_metrics(data)
        assert result == {"a": [{"x": 1}, {"x": 2, "y": 3}], "b": [{"z": 4}]}
