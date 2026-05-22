"""Tests for date iterators."""

import pytest
import pandas as pd
import numpy as np

from data_monitor.iterators import DateIterator, get_date_batches


class TestGetDateBatches:
    """Tests for get_date_batches function."""

    def test_single_day_range(self):
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-01")
        batches = get_date_batches(start, end, "1D", "1D", "1D")
        assert len(batches) == 1
        assert batches[0][0] == start
        assert batches[0][1] == start + pd.Timedelta(days=1)

    def test_multi_day_range_daily_freq(self):
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-10")
        batches = get_date_batches(start, end, "1D", "1D", "1D")
        assert len(batches) > 0
        # Each batch should have 1-day window
        for batch_start, batch_end in batches:
            assert batch_end - batch_start == pd.Timedelta(days=1)

    def test_weekly_window(self):
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")
        batches = get_date_batches(start, end, "7D", "7D", "1D")
        assert len(batches) > 0
        for batch_start, batch_end in batches:
            assert batch_end - batch_start == pd.Timedelta(days=7)

    def test_batch_dates_do_not_exceed_end_date(self):
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-15")
        batches = get_date_batches(start, end, "1D", "7D", "1D")
        for _, batch_end in batches:
            assert batch_end <= end + pd.Timedelta(days=1)


class TestDateIterator:
    """Tests for DateIterator class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with dates."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "value": np.random.rand(30),
                "category": np.random.choice(["A", "B", "C"], 30),
            }
        )

    def test_iterator_protocol(self, sample_df):
        iterator = DateIterator(sample_df, "date", "7D", "7D", "1D")
        assert iter(iterator) is iterator

    def test_iteration_without_date_feature(self, sample_df):
        """When date_feature is None, should return single batch with all data."""
        iterator = DateIterator(sample_df, None, None, None, None)
        batches = list(iterator)
        assert len(batches) == 1
        date, mask = batches[0]
        assert date is None
        assert mask.sum() == len(sample_df)

    def test_iteration_with_date_feature(self, sample_df):
        iterator = DateIterator(sample_df, "date", "7D", "7D", "1D")
        batches = list(iterator)
        assert len(batches) > 0
        for date, mask in batches:
            assert date is not None
            assert isinstance(mask, pd.Series)
            assert mask.dtype == bool

    def test_masks_cover_data(self, sample_df):
        """Ensure masks properly filter the data."""
        iterator = DateIterator(sample_df, "date", "7D", "7D", "1D")
        for date, mask in iterator:
            # Mask should have same length as DataFrame
            assert len(mask) == len(sample_df)
            # At least one row should match
            assert mask.sum() > 0

    def test_empty_window_skipped(self):
        """Windows with no data should be skipped."""
        # Create sparse data with gaps
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-30"]),
                "value": [1, 2, 3],
            }
        )
        iterator = DateIterator(df, "date", "1D", "1D", "1D")
        batches = list(iterator)
        # Should only have batches where data exists
        for _, mask in batches:
            assert mask.sum() > 0
