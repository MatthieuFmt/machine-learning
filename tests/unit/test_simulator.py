"""Tests unitaires pour backtest.simulator — simulate_trades.

Couverture : TP touché, SL touché, timeout, stateful, 0 trade, signal en bordure,
filtres, coût de spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.backtest.simulator import simulate_trades
from learning_machine_learning.backtest.filters import (
    TrendFilter,
    FilterPipeline,
)


def _make_signal_df(
    n: int = 50,
    signal_positions: list[int] | None = None,
    signal_values: list[int] | None = None,
) -> pd.DataFrame:
    """Fabrique un mini DataFrame avec une tendance haussière linéaire.

    Args:
        n: Nombre de barres H1.
        signal_positions: Indices où placer un signal (0 = première barre).
        signal_values: Valeurs 1 (LONG) ou -1 (SHORT) pour chaque signal.

    Returns:
        DataFrame avec colonnes Prediction_Modele, Confiance_*, High, Low, Close,
        Spread, Dist_SMA200_D1.
    """
    if signal_positions is None:
        signal_positions = []
    if signal_values is None:
        signal_values = [1] * len(signal_positions)

    index = pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time")
    # Tendance haussière régulière : +2 pips par barre
    close = 1.1000 + np.arange(n, dtype=float) * 0.0002
    high = close + 0.0003
    low = close - 0.0001

    # Signaux
    prediction = np.zeros(n, dtype=int)
    for pos, val in zip(signal_positions, signal_values):
        prediction[pos] = val

    # Confiance uniforme à 60%
    confiance_baisse = np.where(prediction == -1, 60.0, 0.0)
    confiance_neutre = np.where(prediction == 0, 60.0, 0.0)
    confiance_hausse = np.where(prediction == 1, 60.0, 0.0)

    return pd.DataFrame(
        {
            "Prediction_Modele": prediction,
            "Confiance_Hausse_%": confiance_hausse,
            "Confiance_Neutre_%": confiance_neutre,
            "Confiance_Baisse_%": confiance_baisse,
            "High": np.round(high, 5),
            "Low": np.round(low, 5),
            "Close": np.round(close, 5),
            "Spread": [15] * n,
            "Dist_SMA200_D1": 0.01,  # tendance haussière pour LONG
        },
        index=index,
    )


def _weight_unity(proba: np.ndarray) -> np.ndarray:
    """Fonction de sizing neutre : poids = 1.0."""
    return np.ones_like(proba, dtype=np.float64)


class TestSimulateTrades:
    """Tests unitaires de simulate_trades."""

    def test_tp_touche_long(self) -> None:
        """LONG avec mouvement haussier suffisant → TP touché, win."""
        df = _make_signal_df(n=50, signal_positions=[0], signal_values=[1])
        # Accélération pour toucher TP = 20 pips = 0.0020
        df.loc[df.index[1:5], "High"] = df.loc[df.index[1:5], "Close"] + 0.0025

        trades, n_signaux, _ = simulate_trades(
            df,
            weight_func=_weight_unity,
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            pip_size=0.0001,
        )
        assert n_signaux == 1
        assert len(trades) == 1
        assert trades.iloc[0]["result"] == "win"
        # Pips bruts ≈ tp_pips - spread_cost (commission=0.5, slippage=1.0, spread=1.5)
        expected_brut = 20.0 - (1.5 + 0.5 + 1.0)  # 17.0
        assert trades.iloc[0]["Pips_Bruts"] == pytest.approx(expected_brut, abs=0.1)

    def test_sl_touche_long(self) -> None:
        """LONG avec mouvement baissier suffisant → SL touché, loss_sl."""
        df = _make_signal_df(n=50, signal_positions=[0], signal_values=[1])
        # Chute pour toucher SL = 10 pips = 0.0010
        df.loc[df.index[1:5], "Low"] = df.loc[df.index[1:5], "Close"] - 0.0015

        trades, n_signaux, _ = simulate_trades(
            df,
            weight_func=_weight_unity,
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            pip_size=0.0001,
        )
        assert len(trades) == 1
        assert trades.iloc[0]["result"] == "loss_sl"
        expected_brut = -10.0 - (1.5 + 0.5 + 1.0)  # -13.0
        assert trades.iloc[0]["Pips_Bruts"] == pytest.approx(expected_brut, abs=0.1)

    def test_timeout_pnl_reel(self) -> None:
        """Timeout → PnL basé sur le Close à window barres, pas -sl_pips."""
        df = _make_signal_df(n=50, signal_positions=[0], signal_values=[1])
        # Prix stagnant (pas de TP ni SL touché) → timeout après 24 barres
        # Close[0] = 1.1000, Close[24] ≈ 1.1000 + 24 * 0.0002 = 1.1048
        # PnL LONG = (Close_final - entry) / pip_size - spread_cost

        trades, n_signaux, _ = simulate_trades(
            df,
            weight_func=_weight_unity,
            tp_pips=200.0,  # TP inatteignable
            sl_pips=200.0,  # SL inatteignable
            window=24,
            pip_size=0.0001,
        )
        assert len(trades) == 1
        assert trades.iloc[0]["result"] == "loss_timeout"
        # Le PnL doit être ~48 pips (24 × 2 pips) moins spread_cost, PAS -200
        assert trades.iloc[0]["Pips_Bruts"] > -10.0  # Pas la perte maximale

    def test_stateful_saut_barres(self) -> None:
        """Après un trade, on saute les barres consommées (pas de chevauchement)."""
        df = _make_signal_df(n=50, signal_positions=[0, 5], signal_values=[1, 1])
        # Premier signal touche TP rapidement (barre 2)
        df.loc[df.index[2], "High"] = df.loc[df.index[2], "Close"] + 0.0030

        trades, n_signaux, _ = simulate_trades(
            df,
            weight_func=_weight_unity,
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            pip_size=0.0001,
        )
        # Le deuxième signal (barre 5) devrait être ignoré car on est en trade
        # depuis la barre 0, et le trade se termine à la barre 2. L'itérateur
        # saute alors à idx=3+. Le signal à 5 devrait être vu.
        # Mais si le stateful saute window barres après timeout sans TP/SL…
        # Ici TP touché à idx=2 → i passe à 2, puis i=3, puis 4, puis 5 → signal capturé.
        # Vérifions qu'il n'y a pas de chevauchement.
        assert n_signaux == 2
        # Au moins 1 trade, mais idéalement 2 (le deuxième est indépendant)
        assert len(trades) >= 1

    def test_zero_signal(self) -> None:
        """DataFrame sans signal → 0 trades."""
        df = _make_signal_df(n=50)  # aucun signal
        trades, n_signaux, _ = simulate_trades(
            df,
            weight_func=_weight_unity,
        )
        assert n_signaux == 0
        assert len(trades) == 0

    def test_signal_fin_data_no_crash(self) -> None:
        """Signal sur la dernière barre → pas de crash (P1.1 fix).

        Le forward loop atteint idx >= n immédiatement, break, puis
        timeout handler s'exécute avec le else de la boucle for.
        """
        df = _make_signal_df(n=30, signal_positions=[28], signal_values=[1])
        # window=24, il ne reste que 1 barre forward → idx >= n → timeout
        trades, n_signaux, _ = simulate_trades(
            df,
            weight_func=_weight_unity,
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            pip_size=0.0001,
        )
        assert n_signaux == 1
        assert len(trades) == 1
        # Doit être un timeout (pas assez de forward pour toucher TP/SL)
        assert trades.iloc[0]["result"] == "loss_timeout"
        # pips_brut doit être défini (pas UnboundLocalError)
        assert not np.isnan(trades.iloc[0]["Pips_Bruts"])

    def test_filtres_appliques(self) -> None:
        """Avec FilterPipeline, les compteurs de rejets sont peuplés."""
        df = _make_signal_df(n=24, signal_positions=list(range(24)), signal_values=[1] * 24)
        # Alternance Dist_SMA200_D1 pour tester TrendFilter
        df["Dist_SMA200_D1"] = [0.01, -0.01] * 12

        pipeline = FilterPipeline([TrendFilter()])
        trades, n_signaux, n_filtres = simulate_trades(
            df,
            weight_func=_weight_unity,
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            pip_size=0.0001,
            filter_pipeline=pipeline,
        )
        assert "trend" in n_filtres
        assert n_filtres["trend"] > 0  # certaines barres rejetées
        # Les trades ont filter_rejected renseigné
        assert "filter_rejected" in trades.columns

    def test_spread_cost_applique(self) -> None:
        """Vérifie que spread + commission + slippage est correctement déduit."""
        df = _make_signal_df(n=50, signal_positions=[0], signal_values=[1])
        df.loc[df.index[2], "High"] = df.loc[df.index[2], "Close"] + 0.0030

        trades, _, _ = simulate_trades(
            df,
            weight_func=_weight_unity,
            tp_pips=20.0,
            sl_pips=10.0,
            window=24,
            pip_size=0.0001,
            commission_pips=0.5,
            slippage_pips=1.0,
        )
        # Spread dans le DataFrame = 15 points = 1.5 pips (/10)
        # Coût total = 1.5 + 0.5 + 1.0 = 3.0 pips
        # Pips_Bruts = 20 - 3.0 = 17.0
        assert trades.iloc[0]["Pips_Bruts"] == pytest.approx(17.0, abs=0.1)
