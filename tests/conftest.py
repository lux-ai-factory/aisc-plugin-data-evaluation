"""Shared fixtures + auto-regeneration of generated test fixtures.

``tests/generate_files.py`` deterministically derives each case's inputs into
``tests/files/<case>/``. If the fixtures are missing (fresh clone), regenerate
them at import time so no test is skipped.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests import generate_files


def _ensure_fixtures() -> None:
    """Regenerate tests/files/<case>/ fixtures when they are missing."""
    if not generate_files.FILES.exists() or not any(generate_files.FILES.iterdir()):
        generate_files.main()


_ensure_fixtures()

CASES = sorted(c.name for c in generate_files.FILES.iterdir() if c.is_dir())


def load_case(case: str) -> dict[str, Any]:
    """Load a generated case folder (config.json, datashape.json, inputs)."""
    folder = generate_files.FILES / case
    return {
        "config": json.loads((folder / "config.json").read_text()),
        "datashape": json.loads((folder / "datashape.json").read_text()),
        "reference": (folder / "reference.csv").read_bytes(),
        "evaluated": (folder / "evaluated.csv").read_bytes(),
    }


def datashape_payload(df: Any) -> dict[str, Any]:
    """Derive a features-only datashape payload from a DataFrame (test helper)."""
    import pandas as pd

    features = []
    for col_name in df.columns:
        col_data = df[col_name]
        is_categorical = bool(
            pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data)
        )
        entry: dict[str, Any] = {
            "name": col_name,
            "dtype": str(col_data.dtype),
            "semantic_type": "categorical" if is_categorical else "numeric",
            "role": "feature",
            "null_count": int(col_data.isna().sum()),
            "unique_count": int(col_data.nunique()),
        }
        if is_categorical:
            categories = [c for c in col_data.dropna().unique().tolist()]
            entry["categories"] = categories
            entry["category_mapping"] = {str(c): str(c) for c in categories}
        else:
            entry.update(
                {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                }
            )
        features.append(entry)
    return {"features": features}


@pytest.fixture
def datashape_builder() -> Any:
    """Fixture exposing the datashape_payload builder to tests."""
    return datashape_payload
