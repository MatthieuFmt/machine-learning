"""Tests d'intégration pour le pipeline rank_features (Prompt 04)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from app.features.research import rank_features


def _mock_ohlcv(n: int = 300) -> pd.DataFrame:
    """DataFrame OHLCV synthétique avec random walk (train ≤ 2022)."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.01, n))
    high = close + rng.uniform(0.001, 0.005, n)
    low = close - rng.uniform(0.001, 0.005, n)
    open_ = low + rng.uniform(0, high - low)
    volume = rng.uniform(100, 1000, n)
    # Dates jusqu'à mi-2022 pour que train_end="2022-12-31" capte toutes les données
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture(autouse=True)
def _patch_load_asset():
    """Mock global de load_asset pour éviter la dépendance aux CSV réels."""
    mock_df = _mock_ohlcv(300)
    with patch("app.features.research.load_asset", return_value=mock_df):
        yield


# ═══════════════════════════════════════════════════════════════════════════════
# Tests de contrat
# ═══════════════════════════════════════════════════════════════════════════════


def test_rank_features_returns_dataframe():
    """rank_features retourne un DataFrame."""
    result = rank_features("TEST", "D1", target_horizon=5, n_top=10)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_rank_features_sorted_by_composite_rank():
    """Le résultat est trié par composite_rank croissant."""
    result = rank_features("TEST", "D1", target_horizon=5, n_top=10)
    assert result["composite_rank"].is_monotonic_increasing


def test_rank_features_no_duplicate_features():
    """Aucune feature en double."""
    result = rank_features("TEST", "D1", target_horizon=5, n_top=10)
    assert len(result["feature"].unique()) == len(result)


def test_rank_features_required_columns():
    """Toutes les colonnes attendues sont présentes."""
    result = rank_features("TEST", "D1", target_horizon=5, n_top=10)
    expected_cols = [
        "feature", "mutual_info", "abs_corr", "permutation_importance",
        "mutual_info_rank", "abs_corr_rank", "permutation_importance_rank",
        "composite_rank",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Colonne manquante : {col}"


def test_rank_features_scores_non_negative():
    """Mutual info, abs_corr, permutation_importance sont ≥ 0."""
    result = rank_features("TEST", "D1", target_horizon=5, n_top=10)
    assert (result["mutual_info"] >= 0).all()
    assert (result["abs_corr"] >= 0).all()
    assert (result["permutation_importance"] >= 0).all()


def test_rank_features_respects_n_top():
    """Ne retourne pas plus de n_top features."""
    result = rank_features("TEST", "D1", target_horizon=5, n_top=5)
    assert len(result) <= 5


# ═══════════════════════════════════════════════════════════════════════════════
# Tests I/O (JSON)
# ═══════════════════════════════════════════════════════════════════════════════


def test_rank_features_json_output_exists():
    """Le fichier JSON est créé dans predictions/."""
    json_path = Path("predictions/feature_research_TEST_D1.json")
    # Supprimer s'il existe d'un run précédent
    json_path.unlink(missing_ok=True)

    rank_features("TEST", "D1", target_horizon=5, n_top=10)

    assert json_path.exists(), f"Fichier non créé : {json_path}"

    # Nettoyage
    json_path.unlink(missing_ok=True)


def test_rank_features_json_valid():
    """Le JSON contient une liste de dicts avec les clés attendues."""
    json_path = Path("predictions/feature_research_TEST_D1.json")
    json_path.unlink(missing_ok=True)

    rank_features("TEST", "D1", target_horizon=5, n_top=10)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0
    assert "feature" in data[0]
    assert "composite_rank" in data[0]
    assert "mutual_info" in data[0]
    assert "abs_corr" in data[0]
    assert "permutation_importance" in data[0]

    json_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_rank_features_empty_train_range():
    """train_end antérieur aux données → DataValidationError."""
    with (
        patch("app.features.research.load_asset", return_value=_mock_ohlcv(100)),
        pytest.raises(DataValidationError, match="Aucune donnée"),
    ):
        rank_features("TEST", "D1", target_horizon=5, train_end="1990-01-01")


def test_rank_features_horizon_too_large():
    """target_horizon > longueur des données → ValueError."""
    # 100 barres, horizon 200 → tout NaN après shift → dropna vide
    with (
        patch("app.features.research.load_asset", return_value=_mock_ohlcv(100)),
        pytest.raises(ValueError, match="trop grand|dropna"),
    ):
        rank_features("TEST", "D1", target_horizon=200)


def test_rank_features_n_top_greater_than_features():
    """n_top > nombre total de features → retourne tout sans erreur."""
    result = rank_features("TEST", "D1", target_horizon=5, n_top=999)
    assert len(result) >= 1  # au moins 20-22 features
