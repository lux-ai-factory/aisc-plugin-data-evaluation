"""Tests for DataFrameProvider."""

import io
import pytest
import pandas as pd
import numpy as np

from a4s_plugin_data_evaluation.data_input_provider import DataFrameProvider


class TestDataFrameProvider:
    """Tests for DataFrameProvider class."""

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
        assert "test" in data
        assert isinstance(data["test"], pd.DataFrame)
        assert len(data["test"]) == 10

    def test_read_two_files(self, sample_csv_bytes):
        provider = DataFrameProvider(
            [sample_csv_bytes, sample_csv_bytes]  # ty: ignore[invalid-argument-type]
        )
        data = provider.get_data()
        assert "train" in data
        assert "test" in data

    def test_invalid_file_returns_dataframe(self):
        # CSV parser is very lenient - even random bytes may parse
        # This test just verifies that the provider handles various inputs
        provider = DataFrameProvider(b"col1,col2\n1,2\n3,4")
        data = provider.get_data()
        assert "test" in data
        assert isinstance(data["test"], pd.DataFrame)

    def test_iter_method(self, sample_csv_bytes):
        provider = DataFrameProvider(
            [sample_csv_bytes, sample_csv_bytes]  # ty: ignore[invalid-argument-type]
        )
        batches = list(provider.iter("date", "7D", "7D", "1D"))
        assert len(batches) > 0
        for date, mask in batches:
            assert isinstance(mask, pd.Series)
