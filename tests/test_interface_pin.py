"""The package asks for an aisc-plugin-interface that provides what the plugin imports."""
import re
from pathlib import Path

# @assesses and its builders first shipped in aisc-plugin-interface 0.2.9.
VERSION_WITH_ASSESSES = (0, 2, 9)


def test_the_interface_pin_provides_assesses():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    pins = re.findall(r'"aisc-plugin-interface\s*>=\s*([\d.]+)', text)
    assert pins, "no aisc-plugin-interface>= pin in pyproject.toml"
    for pin in pins:
        assert tuple(int(p) for p in pin.split(".")) >= VERSION_WITH_ASSESSES, pin
