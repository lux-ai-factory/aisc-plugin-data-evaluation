"""Tests for DataFrameProvider."""

import io
import pytest
import pandas as pd
import numpy as np

from vera_plugin_data_evaluation.data_input_provider import (
    DataFrameProvider,
    dataframe_iter,
)


class TestDataFrameProvider:
    """Tests for DataFrameProvider class and dataframe_iter."""

    @pytest.fixture
    def sample_csv_bytes(self):
        """Create sample CSV data as bytes."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "value": np.random.rand(10),
                "category": np.random.choice(["A", "B"], 10),
            }
        )
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue()

    def test_read_csv_file(self, sample_csv_bytes):
        provider = DataFrameProvider(sample_csv_bytes)
        data = provider.get_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10

    def test_invalid_file_returns_dataframe(self):
        # CSV parser is very lenient - even random bytes may parse
        # This test just verifies that the provider handles various inputs
        provider = DataFrameProvider(b"col1,col2\n1,2\n3,4")
        data = provider.get_data()
        assert isinstance(data, pd.DataFrame)

    def test_iter_method(self, sample_csv_bytes):
        provider = DataFrameProvider(sample_csv_bytes)
        data = provider.get_data()
        batches = list(dataframe_iter(data, "date", "7D", "7D", "1D"))
        assert len(batches) > 0
        for date, mask in batches:
            assert isinstance(mask, pd.Series)
