"""Tests unitaires pour analysis/diagnostics.py."""

import pandas as pd

from learning_machine_learning.analysis.diagnostics import (
    analyze_losses,
    diagnostic_direction,
)


class TestAnalyzeLosses:
    """analyze_losses — analyse des pertes vs gains."""

    def test_empty_trades_returns_zero(self) -> None:
        df = pd.DataFrame({"Pips_Nets": pd.Series([], dtype=float)})
        result = analyze_losses(df)
        assert result["n_total"] == 0
        assert result["n_losses"] == 0
        assert result["n_wins"] == 0
        assert result["win_rate"] == 0.0

    def test_single_win(self, trades_synthetic: pd.DataFrame) -> None:
        wins = trades_synthetic[trades_synthetic["result"] == "win"].head(1)
        result = analyze_losses(wins)
        assert result["n_total"] == 1
        assert result["n_losses"] == 0
        assert result["n_wins"] == 1
        assert result["win_rate"] == 100.0

    def test_single_loss_sl(self, trades_synthetic: pd.DataFrame) -> None:
        losses = trades_synthetic[trades_synthetic["result"] == "loss_sl"].head(1)
        result = analyze_losses(losses)
        assert result["n_total"] == 1
        assert result["n_losses"] == 1
        assert result["n_wins"] == 0
        assert result["win_rate"] == 0.0
        assert result["n_loss_sl"] == 1
        assert result["n_loss_timeout"] == 0

    def test_mixed_trades(self, trades_synthetic: pd.DataFrame) -> None:
        result = analyze_losses(trades_synthetic)
        assert result["n_total"] == 10
        assert result["n_losses"] + result["n_wins"] == 10
        assert 0.0 <= result["win_rate"] <= 100.0
        assert result["n_loss_sl"] > 0
        assert result["n_loss_timeout"] > 0

    def test_top_features_computed(self, trades_synthetic: pd.DataFrame) -> None:
        """Vérifie que top_features est calculé si des colonnes numériques existent."""
        result = analyze_losses(trades_synthetic, top_n=3)
        if result["n_wins"] > 0 and result["n_losses"] > 0:
            assert "top_features" in result
            assert len(result["top_features"]) <= 3
            for feat in result["top_features"]:
                assert "feature" in feat
                assert "mean_wins" in feat
                assert "mean_losses" in feat
                assert "abs_diff" in feat

    def test_all_wins_no_top_features(self) -> None:
        """Si pas de pertes, top_features est vide."""
        df = pd.DataFrame({
            "Pips_Nets": [2.0, 3.0, 1.5],
            "result": ["win", "win", "win"],
        })
        result = analyze_losses(df, top_n=5)
        assert result["n_losses"] == 0
        assert result["top_features"] == []


class TestDiagnosticDirection:
    """diagnostic_direction — ventilation PnL par direction."""

    def test_empty_returns_summary(self) -> None:
        trades = pd.DataFrame({"Pips_Nets": pd.Series([], dtype=float)})
        preds = pd.DataFrame({"Prediction_Modele": pd.Series([], dtype=float)})
        result = diagnostic_direction(trades, preds)
        assert result["total_trades"] == 0
        assert result["total_pnl"] == 0.0

    def test_long_and_short_split(self, trades_synthetic: pd.DataFrame) -> None:
        """Crée des prédictions LONG/SHORT et vérifie la ventilation."""
        preds = pd.DataFrame(
            {"Prediction_Modele": [1.0, -1.0] * 5},
            index=trades_synthetic.index,
        )
        result = diagnostic_direction(trades_synthetic, preds)
        assert result["total_trades"] == 10
        assert "long" in result
        assert "short" in result
        assert result["long"]["n"] + result["short"]["n"] == 10

    def test_all_long(self, trades_synthetic: pd.DataFrame) -> None:
        preds = pd.DataFrame(
            {"Prediction_Modele": [1.0] * len(trades_synthetic)},
            index=trades_synthetic.index,
        )
        result = diagnostic_direction(trades_synthetic, preds)
        assert result["long"]["n"] == len(trades_synthetic)
        assert result["short"]["n"] == 0

    def test_trend_respect_with_ml_data(
        self, trades_synthetic: pd.DataFrame, ml_ready_synthetic: pd.DataFrame,
    ) -> None:
        """Avec Dist_SMA200_D1, vérifie que le % de respect de tendance est calculé."""
        preds = pd.DataFrame(
            {"Prediction_Modele": [1.0, -1.0] * 5},
            index=trades_synthetic.index,
        )
        result = diagnostic_direction(trades_synthetic, preds, ml_ready_synthetic)
        assert "long_trend_respected_pct" in result
        assert "short_trend_respected_pct" in result
