"""FROZEN après pivot v4 A8. NE PAS MODIFIER sans nouveau pivot."""
from __future__ import annotations

HYPERPARAMS_TUNED: dict[tuple[str, str], dict] = {
    ("US30", "D1"): {
        "model": "rf",
        "params": {'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 100},
        "threshold": 0.55,
        "expected_sharpe_outer": 1.913,
        "expected_wr": 0.575,
    },
    ("EURUSD", "H4"): {
        "model": "rf",
        "params": {'max_depth': 6, 'min_samples_leaf': 10, 'n_estimators': 100},
        "threshold": 0.55,
        "expected_sharpe_outer": 0.592,
        "expected_wr": 0.515,
    },
    ("XAUUSD", "D1"): {
        "model": "stacking",
        "params": {},
        "threshold": 0.5,
        "expected_sharpe_outer": 0.000,
        "expected_wr": 0.000,
    },
}
