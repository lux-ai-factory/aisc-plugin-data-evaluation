"""Smoke tests: plugin importable without heavy deps; export_metrics(None) == []."""

from data_monitor import DataAnomalyPlugin, DataDriftPlugin


class TestSmoke:
    """Plugin must be importable and safe before any evaluation run."""

    def test_plugins_importable(self):
        assert DataDriftPlugin() is not None
        assert DataAnomalyPlugin() is not None

    def test_export_metrics_none(self):
        """export_metrics with no prior run returns [] for every metric."""
        for plugin in (DataDriftPlugin(), DataAnomalyPlugin()):
            for result in plugin.export_metrics(None):
                assert result == []
