"""Tests for ConfigForm and UI schema."""

import pytest

from vera_plugin_data_evaluation.config_form import ConfigForm, FORM_UI_SCHEMA
from vera_plugin_data_evaluation.utils import Feature, FeatureType


class TestConfigForm:
    """Tests for ConfigForm Pydantic model."""

    def test_default_values(self):
        config = ConfigForm()
        assert config.frequency == ""
        assert config.window_size == ""
        assert config.features == []
        assert config.date_feature is None
        assert config.target_feature is None

    def test_with_features(self):
        features = [
            Feature(name="age", min=0.0, max=100.0, type=FeatureType.INTEGER),
            Feature(name="income", min=0.0, max=1000000.0, type=FeatureType.FLOAT),
        ]
        config = ConfigForm(features=features)
        assert len(config.features) == 2
        assert config.features[0].name == "age"

    def test_model_dump(self):
        features = [
            Feature(name="timestamp", min=0.0, max=0.0, type=FeatureType.DATE),
            Feature(name="label", min=0.0, max=1.0, type=FeatureType.INTEGER),
        ]
        config = ConfigForm(
            frequency="1D",
            window_size="7D",
            features=features,
            date_feature="timestamp",
            target_feature="label",
        )
        dump = config.model_dump()
        assert dump["frequency"] == "1D"
        assert dump["window_size"] == "7D"
        assert dump["date_feature"] == "timestamp"
        assert dump["target_feature"] == "label"

    def test_validate_special_features_empty_string_to_none(self):
        """Empty string for target_feature should be normalized to None."""
        config = ConfigForm(target_feature="")
        assert config.target_feature is None

    def test_validate_special_features_valid(self):
        """target_feature must be in features list."""
        features = [
            Feature(name="feat1", min=0.0, max=1.0, type=FeatureType.FLOAT),
            Feature(name="target", min=0.0, max=1.0, type=FeatureType.INTEGER),
        ]
        config = ConfigForm(features=features, target_feature="target")
        assert config.target_feature == "target"

    def test_validate_target_feature_not_in_features_raises(self):
        """target_feature not in features should raise ValueError."""
        features = [
            Feature(name="feat1", min=0.0, max=1.0, type=FeatureType.FLOAT),
        ]
        with pytest.raises(ValueError, match="Target feature must be one of"):
            ConfigForm(features=features, target_feature="nonexistent")

    def test_validate_date_feature_wrong_type_raises(self):
        """date_feature must be of type DATE."""
        features = [
            Feature(name="feat1", min=0.0, max=1.0, type=FeatureType.FLOAT),
            Feature(name="target", min=0.0, max=1.0, type=FeatureType.INTEGER),
        ]
        with pytest.raises(ValueError, match="Date feature must be of type Date"):
            ConfigForm(features=features, target_feature="target", date_feature="feat1")


class TestFormUISchema:
    """Tests for FORM_UI_SCHEMA constant."""

    def test_schema_is_dict(self):
        assert isinstance(FORM_UI_SCHEMA, dict)

    def test_features_has_items_schema(self):
        assert "features" in FORM_UI_SCHEMA
        assert "items" in FORM_UI_SCHEMA["features"]

    def test_features_not_orderable_or_addable(self):
        """Feature list should not be orderable or addable in UI."""
        options = FORM_UI_SCHEMA["features"].get("ui:options", {})
        assert options.get("orderable") is False
        assert options.get("addable") is False
