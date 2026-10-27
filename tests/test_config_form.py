"""Tests for ConfigForm."""

from data_monitor.config_form import ConfigForm


class TestConfigForm:
    """Tests for ConfigForm Pydantic model."""

    def test_default_values(self):
        config = ConfigForm()
        assert config.frequency == ""
        assert config.window_size == ""

    def test_model_dump(self):
        config = ConfigForm(frequency="1D", window_size="7D")
        dump = config.model_dump()
        assert dump["frequency"] == "1D"
        assert dump["window_size"] == "7D"

    def test_custom_time_series_values(self):
        config = ConfigForm(frequency="30D", window_size="90 days")
        assert config.frequency == "30D"
        assert config.window_size == "90 days"
