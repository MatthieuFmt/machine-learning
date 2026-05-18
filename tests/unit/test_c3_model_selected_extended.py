"""Vérifie l'extension de MODEL_SELECTED en C3 sans régression A7."""
from __future__ import annotations

import pytest

from app.config.model_selected import MODEL_SELECTED

_VALID_MODELS = {"rf", "hgbm", "stacking"}
_A7_ORIGINAL = {
    ("US30", "D1"): "rf",
    ("EURUSD", "H4"): "rf",
    ("XAUUSD", "D1"): "stacking",
}

# Couples attendus après C3 (shortlist C2 complète)
_EXPECTED_C3_KEYS = {
    ("US30", "D1"),
    ("EURUSD", "H4"),
    ("XAUUSD", "D1"),
    ("BTCUSD", "D1"),
    ("ETHUSD", "D1"),
    ("ETHUSD", "H4"),
    ("ETHUSD", "H1"),
    ("EURUSD", "D1"),
    ("GBPUSD", "D1"),
    ("GBPUSD", "H4"),
    ("USDCHF", "D1"),
    ("USDCHF", "H4"),
}


def test_a7_original_entries_preserved() -> None:
    """Les 3 entrées A7 originales ne doivent pas être modifiées."""
    for key, expected in _A7_ORIGINAL.items():
        assert key in MODEL_SELECTED, f"{key} doit rester dans MODEL_SELECTED"
        assert (
            MODEL_SELECTED[key] == expected
        ), f"{key} : modèle changé ! attendu {expected}, vu {MODEL_SELECTED[key]}"


def test_a7_entries_count_unchanged() -> None:
    """Le nombre d'entrées total doit être >= 3 (A7 originales)."""
    assert len(MODEL_SELECTED) >= 3, (
        f"MODEL_SELECTED doit avoir au moins 3 entrées, a {len(MODEL_SELECTED)}"
    )


@pytest.mark.parametrize("key", list(MODEL_SELECTED.keys()))
def test_model_in_valid_set(key: tuple[str, str]) -> None:
    """Chaque modèle retenu doit être dans {rf, hgbm, stacking}."""
    assert MODEL_SELECTED[key] in _VALID_MODELS, (
        f"{key} : modèle {MODEL_SELECTED[key]} non reconnu (attendu : {_VALID_MODELS})"
    )


def test_all_c3_shortlist_keys_present() -> None:
    """Tous les couples de la shortlist C2+C3 doivent être présents."""
    missing = _EXPECTED_C3_KEYS - set(MODEL_SELECTED.keys())
    assert not missing, f"Couples manquants dans MODEL_SELECTED : {missing}"
