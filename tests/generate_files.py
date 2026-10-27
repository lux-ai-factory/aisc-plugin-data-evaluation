"""Generate deterministic test fixtures under tests/files/<case>/.

Each case folder is self-contained: the reference + evaluated CSV datasets, the
features-only datashape payload, and a webapp-importable ``config.json`` built
from the real ``ConfigForm`` model.

Regenerate with:

    uv run python tests/generate_files.py

A tests/conftest.py hook regenerates missing fixtures automatically at
collection time, so a fresh clone only needs ``uv sync && pytest``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402  # sys.path patched above for standalone runs
import pandas as pd  # noqa: E402

from data_monitor.config_form import ConfigForm  # noqa: E402

FILES = Path(__file__).resolve().parent / "files"

RNG = np.random.default_rng(42)

CASES: dict[str, dict[str, Any]] = {
    "no_drift": {"rows": 200, "drift": "none"},
    "drift": {"rows": 200, "drift": "shift"},
    "anomaly_normal": {"rows": 200, "anomaly": False},
    "anomaly_outliers": {"rows": 200, "anomaly": True},
}


def _reference_frame(rows: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": RNG.integers(18, 65, rows).astype(float),
            "income": RNG.uniform(30_000, 150_000, rows),
            "category": RNG.choice(["A", "B", "C"], rows),
        }
    )


def _evaluated_frame(reference: pd.DataFrame, drift: str) -> pd.DataFrame:
    df = reference.copy()
    if drift == "shift":
        # Shift age/income distributions and change categories -> drifts.
        df["age"] = df["age"] + 35
        df["income"] = df["income"] * 3.0
        df["category"] = RNG.choice(["C", "D", "E"], len(df))
    else:
        # Same distribution family -> low drift.
        df["age"] = RNG.integers(18, 65, len(df)).astype(float)
        df["income"] = RNG.uniform(30_000, 150_000, len(df))
        df["category"] = RNG.choice(["A", "B", "C"], len(df))
    return df


def _anomaly_frame(reference: pd.DataFrame, anomaly: bool) -> pd.DataFrame:
    if not anomaly:
        return _evaluated_frame(reference, drift="none")
    return pd.DataFrame(
        {
            "age": [10, 80, 25, 30, 100],  # outside 18-65 reference range
            "income": [-10, 50, 150, 60, 70],  # outside reference range
            "category": ["A", "D", "E", "B", "F"],  # new categories D, E, F
        }
    )


def _datashape_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Derive a features-only datashape payload from a DataFrame."""
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


def main() -> None:
    FILES.mkdir(parents=True, exist_ok=True)
    for case, opts in CASES.items():
        out = FILES / case
        out.mkdir(parents=True, exist_ok=True)

        reference = _reference_frame(int(opts["rows"]))
        if case.startswith("anomaly"):
            evaluated = _anomaly_frame(reference, anomaly=bool(opts["anomaly"]))
        else:
            evaluated = _evaluated_frame(reference, drift=opts["drift"])

        reference.to_csv(out / "reference.csv", index=False)
        evaluated.to_csv(out / "evaluated.csv", index=False)
        (out / "datashape.json").write_text(json.dumps(_datashape_payload(reference)) + "\n")
        (out / "config.json").write_text(json.dumps(ConfigForm().model_dump(), indent=2) + "\n")

    print(f"Generated {len(CASES)} cases under {FILES}")


if __name__ == "__main__":
    main()
