"""The package asks for aisc-plugin-interface 0.4.0, and its lock takes it from the wheel under vendor/
(the AISC repo's scripts/vendor-plugin-interface.sh puts it there), never from PyPI, whose latest, 0.3.0,
lacks what the plugin imports.

Run under the lock, so the last case sees the lock's environment:

    uv run --locked --with pytest python -m pytest -q -p no:cacheprovider tests

With AISC_DIR set to an AISC checkout, the vendored wheel is also compared with AISC's.
"""
import hashlib
import importlib
import importlib.metadata
import os
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# required= and strict= first shipped in aisc-plugin-interface 0.4.0
INTERFACE_FLOOR = (0, 4, 0)
FLOOR_TEXT = ".".join(str(p) for p in INTERFACE_FLOOR)
WHEEL = f"aisc_plugin_interface-{FLOOR_TEXT}-py3-none-any.whl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_interface_pin_is_0_4_0():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pins = re.findall(r'"aisc-plugin-interface\s*>=\s*([\d.]+)', text)
    assert pins, "no aisc-plugin-interface>= pin in pyproject.toml"
    for pin in pins:
        assert tuple(int(p) for p in pin.split(".")) == INTERFACE_FLOOR, pin


def test_the_lock_takes_the_interface_from_vendor():
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    entries = [p for p in lock.get("package", []) if p.get("name") == "aisc-plugin-interface"]
    assert len(entries) == 1, entries
    (entry,) = entries
    assert entry.get("version") == FLOOR_TEXT
    assert entry.get("source") == {"path": f"vendor/{WHEEL}"}
    wheel = REPO / "vendor" / WHEEL
    assert wheel.is_file(), f"vendor/{WHEEL} is missing"
    assert (entry.get("wheels") or [{}])[0].get("hash") == "sha256:" + _sha256(wheel)
    assert not [e for e in entries if "registry" in e.get("source", {})]


def test_the_vendored_wheel_is_the_aisc_one():
    aisc_dir = os.environ.get("AISC_DIR")
    if not aisc_dir:
        pytest.skip("AISC_DIR not set: the wheel cannot be compared with AISC's")
    theirs = Path(aisc_dir).expanduser() / "apps/backend/vendor" / WHEEL
    if not theirs.is_file():
        pytest.skip("AISC_DIR not set: the wheel cannot be compared with AISC's")
    ours = REPO / "vendor" / WHEEL
    assert ours.is_file(), f"vendor/{WHEEL} is missing"
    assert _sha256(ours) == _sha256(theirs)


def test_the_installed_interface_is_0_4_0():
    assert importlib.metadata.version("aisc-plugin-interface") == FLOOR_TEXT
    importlib.import_module("aisc_plugin_interface.assesses")
