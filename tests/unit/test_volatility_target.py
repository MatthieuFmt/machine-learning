"""Tests unitaires pour app.backtest.sizing.volatility_target_weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.backtest.sizing import volatility_target_weights


def _series(vals: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(vals), freq="D", tz="UTC")
    return pd.Series(vals, index=idx)


class TestVolatilityTargetWeights:
    def test_rejects_tiny_lookback(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            volatility_target_weights(_series([0.01] * 10), lookback=1)

    def test_warmup_is_nan(self) -> None:
        rng = np.random.default_rng(0)
        r = _series(list(rng.normal(0, 0.01, 100)))
        w = volatility_target_weights(r, lookback=20)
        # shift(1) sur un rolling(20) → 20 premières valeurs NaN.
        assert w.iloc[:20].isna().all()
        assert w.iloc[20:].notna().any()

    def test_no_look_ahead(self) -> None:
        """Le poids en t ne dépend QUE des rendements jusqu'à t-1."""
        rng = np.random.default_rng(1)
        r = _series(list(rng.normal(0, 0.01, 80)))
        w1 = volatility_target_weights(r, lookback=20)
        # On modifie le rendement à l'indice k : les poids ≤ k ne doivent PAS bouger
        # (le poids en k utilise les rendements jusqu'à k-1).
        k = 50
        r2 = r.copy()
        r2.iloc[k] += 0.5  # choc énorme
        w2 = volatility_target_weights(r2, lookback=20)
        pd.testing.assert_series_equal(w1.iloc[: k + 1], w2.iloc[: k + 1])
        # En revanche, le poids en k+1 (qui voit r[k]) doit avoir changé.
        assert not np.isclose(w1.iloc[k + 1], w2.iloc[k + 1])

    def test_inverse_relation_to_volatility(self) -> None:
        """Vol récente haute → poids bas ; vol basse → poids haut."""
        # 60 jours de vol basse puis 60 jours de vol haute.
        rng = np.random.default_rng(2)
        low = list(rng.normal(0, 0.002, 60))
        high = list(rng.normal(0, 0.02, 60))
        r = _series(low + high)
        w = volatility_target_weights(r, lookback=20, max_leverage=100.0)
        w_low_regime = w.iloc[40:60].mean()    # fin de la période basse vol
        w_high_regime = w.iloc[100:120].mean()  # fin de la période haute vol
        assert w_low_regime > w_high_regime

    def test_cap_respected(self) -> None:
        """Vol quasi nulle → poids plafonné à max_leverage (pas d'infini)."""
        r = _series([1e-6] * 40)  # vol ~0
        w = volatility_target_weights(r, lookback=20, max_leverage=3.0)
        valid = w.dropna()
        assert (valid <= 3.0 + 1e-9).all()
        assert (valid >= 0.0).all()

    def test_targets_constant_vol_roughly(self) -> None:
        """Après scaling, la vol réalisée se rapproche de la cible (ordre de grandeur)."""
        rng = np.random.default_rng(3)
        r = _series(list(rng.normal(0, 0.01, 500)))  # ~16 %/an
        w = volatility_target_weights(r, target_vol_annual=0.10, lookback=60)
        scaled = (w * r).dropna()
        realized_ann = scaled.std() * np.sqrt(252)
        # On vise 10 % ; tolérance large (échantillon fini, vol non stationnaire).
        assert 0.05 < realized_ann < 0.20
