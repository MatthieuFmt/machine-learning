"""Vérifie l'extension de HYPERPARAMS_TUNED en C4 sans régression A8."""
from __future__ import annotations

import pytest

from app.config.hyperparams_tuned import HYPERPARAMS_TUNED

_A8_ORIGINAL = {
    ("US30", "D1"): {"model": "rf", "threshold": 0.55},
    ("EURUSD", "H4"): {"model": "rf", "threshold": 0.55},
    ("XAUUSD", "D1"): {"model": "stacking", "threshold": 0.50},
}

_REQUIRED_KEYS = {"model", "params", "threshold", "expected_sharpe_outer", "expected_wr"}
_VALID_THRESHOLDS = {0.50, 0.55, 0.60}


def test_a8_original_entries_preserved() -> None:
    """Les 3 entrées A8 originales ne doivent pas être modifiées."""
    for key, expected in _A8_ORIGINAL.items():
        assert key in HYPERPARAMS_TUNED, f"{key} doit rester dans HYPERPARAMS_TUNED"
        entry = HYPERPARAMS_TUNED[key]
        assert entry["model"] == expected["model"], (
            f"{key} : modèle changé ! attendu {expected['model']}, vu {entry['model']}"
        )
        assert entry["threshold"] == expected["threshold"], (
            f"{key} : threshold changé ! attendu {expected['threshold']}, vu {entry['threshold']}"
        )


def test_a8_entries_count_unchanged() -> None:
    """Le nombre d'entrées total doit être >= 3 (A8 originales)."""
    assert len(HYPERPARAMS_TUNED) >= 3, (
        f"HYPERPARAMS_TUNED doit avoir au moins 3 entrées, a {len(HYPERPARAMS_TUNED)}"
    )


@pytest.mark.parametrize("key", list(HYPERPARAMS_TUNED.keys()))
def test_entry_has_required_keys(key: tuple[str, str]) -> None:
    """Chaque entrée doit avoir toutes les clés requises."""
    entry = HYPERPARAMS_TUNED[key]
    missing = _REQUIRED_KEYS - set(entry.keys())
    assert not missing, f"{key} : clés manquantes {missing}"


@pytest.mark.parametrize("key", list(HYPERPARAMS_TUNED.keys()))
def test_threshold_valid(key: tuple[str, str]) -> None:
    """Chaque threshold doit être dans {0.50, 0.55, 0.60}."""
    threshold = HYPERPARAMS_TUNED[key]["threshold"]
    assert threshold in _VALID_THRESHOLDS, (
        f"{key} : threshold {threshold} hors {_VALID_THRESHOLDS}"
    )
