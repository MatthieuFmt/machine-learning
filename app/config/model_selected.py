"""FROZEN après pivot v4 A7. NE PAS MODIFIER sans nouveau pivot."""
from __future__ import annotations

MODEL_SELECTED: dict[tuple[str, str], str] = {
    ("US30", "D1"): "rf",
    ("EURUSD", "H4"): "rf",
    ("XAUUSD", "D1"): "stacking",
}
