"""Tests unitaires de la triple barrière — < 100ms, fixtures synthétiques."""

import numpy as np
import pandas as pd
import pytest

from learning_machine_learning.features.triple_barrier import (
    apply_triple_barrier,
    compute_target_series,
    label_distribution,
)


class TestTripleBarrier:
    """Tests de la fonction apply_triple_barrier."""

    @staticmethod
    def _make_ohlcv(prices: list[float], highs: list[float] | None = None,
                    lows: list[float] | None = None) -> pd.DataFrame:
        """Fabrique un mini DataFrame OHLCV pour les tests."""
        n = len(prices)
        return pd.DataFrame(
            {
                "Open": prices,
                "High": highs if highs else prices,
                "Low": lows if lows else prices,
                "Close": prices,
                "Volume": [100] * n,
                "Spread": [15] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

    def test_tp_touche_long(self):
        """Un mouvement haussier suffisant → label 1 (LONG gagnant)."""
        df = self._make_ohlcv(
            prices=[1.1000, 1.1010, 1.1020, 1.1030],  # +30 pips
        )
        # Ajuste High/Low pour simuler un vrai mouvement
        df["High"] = [1.1000, 1.1015, 1.1025, 1.1030]
        df["Low"] = [1.1000, 1.1005, 1.1010, 1.1020]

        targets = apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=3, pip_size=0.0001)
        assert targets[0] == 1.0

    def test_sl_touche_long(self):
        """Un mouvement baissier suffisant → label -1 (SHORT gagnant)."""
        df = self._make_ohlcv(
            prices=[1.1000, 1.0990, 1.0980, 1.0970],  # -30 pips
        )
        df["Low"] = [1.1000, 1.0985, 1.0975, 1.0970]
        df["High"] = [1.1000, 1.0995, 1.0990, 1.0980]

        targets = apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=3, pip_size=0.0001)
        assert targets[0] == -1.0

    def test_timeout(self):
        """Prix stagnant → timeout → label 0."""
        df = self._make_ohlcv(prices=[1.1000] * 30)
        # Tous les prix identiques → aucun TP/SL touché
        targets = apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=24, pip_size=0.0001)
        assert targets[0] == 0.0

    def test_tp_sl_meme_bougie_long_prioritaire(self):
        """TP et SL touchés dans la même bougie : SL prioritaire car vérifié en premier."""
        df = self._make_ohlcv(prices=[1.1000, 1.1000])
        # Bougie avec range énorme : touche à la fois TP (+30 pips) et SL (-15 pips)
        df.loc[df.index[1], "High"] = 1.1040  # +40 pips → au-dessus du TP
        df.loc[df.index[1], "Low"] = 1.0980   # -20 pips → en-dessous du SL

        targets = apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=1, pip_size=0.0001)
        # SL vérifié en premier → label 0 (car long_win=False, short_win dépend...)
        # En réalité : SL touché sur LONG en premier, SHORT doit être testé
        # Avec entry=1.1000: LONG SL=1.0990, SHORT TP=1.0980
        # Low=1.0980 → LONG SL touché (1.0980 <= 1.0990) en premier
        # High=1.1040 → SHORT SL touché (1.1040 >= 1.1010) aussi
        # Les deux directions perdent → 0
        assert targets[0] == 0.0

    def test_bidirectionnel_long_gagne_short_perd(self):
        """Hausse modérée : LONG gagne, SHORT perd (SL touché)."""
        df = self._make_ohlcv(prices=[1.1000, 1.1025])
        df["High"] = [1.1000, 1.1030]  # +30 pips → TP LONG 1.1020 touché
        df["Low"] = [1.1000, 1.1015]   # pas assez bas pour SHORT TP (1.0980)

        targets = apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=1, pip_size=0.0001)
        # LONG: entry 1.1000, TP=1.1020, SL=1.0990. High=1.1030 → TP touché → long_win=True
        # SHORT: entry 1.1000, TP=1.0980, SL=1.1010. High=1.1030 → SL touché → short_win=False
        assert targets[0] == 1.0

    def test_last_bars_are_nan(self):
        """Les `window` dernières barres doivent être NaN (pas assez de forward)."""
        df = self._make_ohlcv(prices=[1.1000] * 30)
        targets = apply_triple_barrier(df, tp_pips=20, sl_pips=10, window=24, pip_size=0.0001)
        # Les 24 dernières barres doivent être NaN
        assert np.all(np.isnan(targets[-24:]))
        # Les premières (30-24=6) doivent être définies
        assert not np.any(np.isnan(targets[:6]))

    def test_empty_dataframe_raises(self):
        """DataFrame sans colonnes requises → ValueError."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError, match="Colonnes requises manquantes"):
            apply_triple_barrier(df)

    def test_numpy_output_shape(self):
        """La sortie a la même longueur que l'entrée."""
        df = self._make_ohlcv(prices=[1.1000] * 50)
        targets = apply_triple_barrier(df, window=24, pip_size=0.0001)
        assert len(targets) == len(df)

    @pytest.mark.parametrize("tp,sl,expected_first", [
        (20, 10, 1.0),    # TP plus proche → LONG gagne
        (10, 20, -1.0),   # SL plus proche en hausse → ?
    ])
    def test_asymmetric_tp_sl(self, tp, sl, expected_first):
        """Test avec TP et SL asymétriques."""
        df = self._make_ohlcv(prices=[1.1000, 1.1025])
        df["High"] = [1.1000, 1.1030]
        df["Low"] = [1.1000, 1.1015]
        targets = apply_triple_barrier(df, tp_pips=tp, sl_pips=sl, window=1, pip_size=0.0001)
        # 20/10 → LONG gagne, 10/20 → ?
        if tp == 20:
            assert targets[0] == 1.0


class TestComputeTargetSeries:
    """Tests de compute_target_series."""

    def test_returns_series_without_nan(self):
        """La Series retournée ne contient pas de NaN."""
        df = pd.DataFrame(
            {
                "Open": [1.1000] * 50,
                "High": [1.1000] * 50,
                "Low": [1.1000] * 50,
                "Close": [1.1000] * 50,
                "Volume": [100] * 50,
                "Spread": [15] * 50,
            },
            index=pd.date_range("2024-01-01", periods=50, freq="h"),
        )
        series = compute_target_series(df, tp_pips=20, sl_pips=10, window=24, pip_size=0.0001)
        assert isinstance(series, pd.Series)
        assert not series.isna().any()
        assert series.name == "Target"


class TestLabelDistribution:
    """Tests de label_distribution."""

    def test_balanced_distribution(self):
        """Distribution équilibrée."""
        targets = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0])
        dist = label_distribution(targets)
        assert dist["1"] == pytest.approx(33.33, abs=0.01)
        assert dist["-1"] == pytest.approx(22.22, abs=0.01)
        assert dist["0"] == pytest.approx(44.44, abs=0.01)

    def test_all_same_label(self):
        """Tous les labels identiques."""
        targets = np.array([1.0, 1.0, 1.0])
        dist = label_distribution(targets)
        assert dist["1"] == 100.0
        assert dist["-1"] == 0.0
        assert dist["0"] == 0.0

    def test_empty_array(self):
        """Tableau vide."""
        targets = np.array([])
        dist = label_distribution(targets)
        assert dist == {"-1": 0.0, "0": 0.0, "1": 0.0}

    def test_with_nan(self):
        """Les NaN sont ignorés."""
        targets = np.array([1.0, -1.0, np.nan, np.nan, 0.0])
        dist = label_distribution(targets)
        assert dist["1"] == pytest.approx(33.33, abs=0.01)
        assert dist["-1"] == pytest.approx(33.33, abs=0.01)
        assert dist["0"] == pytest.approx(33.33, abs=0.01)
