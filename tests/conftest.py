"""Fixtures partagées pour tous les tests.

Toutes les données sont synthétiques — seed fixe pour reproductibilité.
Aucun fichier CSV externe pour les tests unitaires → exécution < 100ms.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


SEED = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Générateur numpy déterministe pour toute la session de test."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def ohlcv_h1_synthetic(rng: np.random.Generator) -> pd.DataFrame:
    """1000 barres H1 synthétiques avec tendance + bruit.

    Returns:
        DataFrame indexé par Time avec colonnes Open, High, Low, Close, Volume, Spread.
    """
    n = 1000
    start = datetime(2022, 1, 1, 0, 0)
    index = pd.date_range(start, periods=n, freq="h", name="Time")

    # Prix avec Random Walk + dérive
    drift = 0.00002
    noise = rng.normal(0, 0.0005, n)
    close = 1.1000 + np.cumsum(drift + noise)

    high = close + np.abs(rng.normal(0, 0.0002, n))
    low = close - np.abs(rng.normal(0, 0.0002, n))
    open_price = close - rng.normal(0, 0.0001, n)

    # Ajuster pour cohérence OHLC
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    volume = rng.integers(100, 500, n)
    spread = rng.integers(10, 18, n)  # en points (10-18 = 1.0-1.8 pips)

    return pd.DataFrame(
        {
            "Open": np.round(open_price, 5),
            "High": np.round(high, 5),
            "Low": np.round(low, 5),
            "Close": np.round(close, 5),
            "Volume": volume,
            "Spread": spread,
        },
        index=index,
    )


@pytest.fixture(scope="session")
def ohlcv_h4_synthetic(rng: np.random.Generator) -> pd.DataFrame:
    """250 barres H4 synthétiques."""
    n = 250
    start = datetime(2022, 1, 1, 0, 0)
    index = pd.date_range(start, periods=n, freq="4h", name="Time")

    drift = 0.00008
    noise = rng.normal(0, 0.0008, n)
    close = 1.1000 + np.cumsum(drift + noise)
    high = close + np.abs(rng.normal(0, 0.0003, n))
    low = close - np.abs(rng.normal(0, 0.0003, n))
    open_price = close - rng.normal(0, 0.0002, n)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    return pd.DataFrame(
        {
            "Open": np.round(open_price, 5),
            "High": np.round(high, 5),
            "Low": np.round(low, 5),
            "Close": np.round(close, 5),
            "Volume": rng.integers(400, 2000, n),
            "Spread": rng.integers(12, 20, n),
        },
        index=index,
    )


@pytest.fixture(scope="session")
def ohlcv_d1_synthetic(rng: np.random.Generator) -> pd.DataFrame:
    """100 barres D1 synthétiques."""
    n = 100
    start = datetime(2022, 1, 1)
    index = pd.date_range(start, periods=n, freq="D", name="Time")

    drift = 0.0003
    noise = rng.normal(0, 0.003, n)
    close = 1.1000 + np.cumsum(drift + noise)
    high = close + np.abs(rng.normal(0, 0.001, n))
    low = close - np.abs(rng.normal(0, 0.001, n))
    open_price = close - rng.normal(0, 0.0005, n)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    return pd.DataFrame(
        {
            "Open": np.round(open_price, 5),
            "High": np.round(high, 5),
            "Low": np.round(low, 5),
            "Close": np.round(close, 5),
            "Volume": rng.integers(1000, 10000, n),
            "Spread": rng.integers(12, 20, n),
        },
        index=index,
    )


@pytest.fixture(scope="session")
def trades_synthetic() -> pd.DataFrame:
    """10 trades synthétiques pour les tests de métriques."""
    times = pd.date_range("2024-01-01", periods=10, freq="3h")
    return pd.DataFrame(
        {
            "Pips_Nets": [2.5, -1.0, 3.0, -1.5, 2.0, -0.5, 4.0, -1.0, 2.5, -1.5],
            "Pips_Bruts": [3.0, -0.5, 3.5, -1.0, 2.5, 0.0, 4.5, -0.5, 3.0, -1.0],
            "Weight": [0.9, 1.0, 0.95, 1.0, 0.85, 1.0, 1.1, 1.0, 1.0, 1.0],
            "result": [
                "win",
                "loss_sl",
                "win",
                "loss_sl",
                "win",
                "loss_timeout",
                "win",
                "loss_sl",
                "win",
                "loss_timeout",
            ],
            "filter_rejected": [""] * 10,
        },
        index=times,
    )


@pytest.fixture(scope="session")
def ml_ready_synthetic(
    ohlcv_h1_synthetic: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """DataFrame ML-ready synthétique pour tests d'intégration modèle."""
    n = len(ohlcv_h1_synthetic)
    df = pd.DataFrame(index=ohlcv_h1_synthetic.index)

    # Target : -1, 0, 1 équilibré
    df["Target"] = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25])

    # Features techniques simulées
    df["Spread"] = ohlcv_h1_synthetic["Spread"]
    df["Log_Return"] = rng.normal(0, 0.001, n)
    df["Dist_EMA_9"] = rng.normal(0, 0.002, n)
    df["Dist_EMA_21"] = rng.normal(0, 0.003, n)
    df["Dist_EMA_50"] = rng.normal(0, 0.004, n)
    df["RSI_14"] = rng.uniform(30, 70, n)
    df["ADX_14"] = rng.uniform(15, 40, n)
    df["ATR_Norm"] = rng.uniform(0.001, 0.008, n)
    df["BB_Width"] = rng.uniform(0.005, 0.020, n)
    df["Hour_Sin"] = np.sin(ohlcv_h1_synthetic.index.hour * 2 * np.pi / 24)
    df["Hour_Cos"] = np.cos(ohlcv_h1_synthetic.index.hour * 2 * np.pi / 24)

    # Features H4/D1
    df["RSI_14_H4"] = rng.uniform(35, 65, n)
    df["Dist_EMA_20_H4"] = rng.normal(0, 0.003, n)
    df["Dist_EMA_50_H4"] = rng.normal(0, 0.004, n)
    df["RSI_14_D1"] = rng.uniform(35, 65, n)
    df["Dist_EMA_20_D1"] = rng.normal(0, 0.005, n)
    df["Dist_EMA_50_D1"] = rng.normal(0, 0.006, n)

    # Macro
    df["XAU_Return"] = rng.normal(0, 0.005, n)
    df["CHF_Return"] = rng.normal(0, 0.003, n)

    # Régime
    df["Volatilite_Realisee_24h"] = rng.uniform(0.001, 0.010, n)
    df["Range_ATR_ratio"] = rng.uniform(0.5, 1.5, n)
    df["RSI_D1_delta"] = rng.normal(0, 3, n)
    df["Dist_SMA200_D1"] = rng.normal(0.001, 0.010, n)

    return df
