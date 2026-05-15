# Pivot v4 — A5 : Génération du superset de features (~70)

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md) — vue d'ensemble pivot v4 (phase A étendue) — **lire la section "Ordre d'exécution strict — RÉVISÉ"**
2. [A1_audit_simulator.md](A1_audit_simulator.md) — **A1 ✅ Terminé obligatoire** (sizing + DD bornés)

> 📌 **Note ordre révisé** : A5 vient maintenant immédiatement après A1, AVANT A2/A3/A4. Le simulateur n'est pas requis pour A5 (pure feature engineering). A2/A3 viendront après A9.
3. [app/features/indicators.py](../../app/features/indicators.py) — 18 indicateurs existants
4. [app/features/economic.py](../../app/features/economic.py) — 9 features économiques
5. [app/features/regime.py](../../app/features/regime.py) — features régime
6. [../00_constitution.md](../00_constitution.md) — règles 6, 7, 9, 13, 14

## Objectif
Construire un **superset de ~70 features** rigoureusement vectorisées et anti-look-ahead, organisées en 7 catégories sémantiques. Toutes calculées sur **train ≤ 2022 uniquement** dans ce prompt. **Pas de sélection ici** — le ranking se fait en A6.

> **Principe** : un superset large permet à A6 de trouver les **vraies features prédictives** sans biais d'a priori. Le RF avec `max_depth=4` gère naturellement le bruit, donc fournir 70 features ne fait pas exploser l'overfit.

## Type d'opération
🔧 **Infrastructure ML** — **0 n_trial consommé**. Aucune lecture du test set 2024+.

## Definition of Done (testable)

- [ ] `app/features/superset.py` (NOUVEAU) contient `build_superset(df, asset=None) -> pd.DataFrame` retournant ≥ 60 colonnes.
- [ ] Toutes les fonctions de features sont décorées `@look_ahead_safe`.
- [ ] **Aucune feature ne contient de NaN** au-delà du warmup (max 200 barres pour SMA200).
- [ ] Indicateurs manquants ajoutés dans [app/features/indicators.py](../../app/features/indicators.py) : MACD, Stoch K/D, Williams %R, CCI, MFI, BB width, Keltner width (s'ils ne sont pas déjà présents).
- [ ] `tests/unit/test_superset_features.py` (NOUVEAU) avec ≥ 11 tests :
  - 1 test par catégorie (présence des features)
  - 1 test anti-look-ahead sur agrégat
  - 1 test "pas de NaN après warmup"
  - 1 test "≥ 60 colonnes"
  - 1 test "noms uniques"
  - 1 test "dtypes float64 partout"
- [ ] `rtk make verify` → 0 erreur
- [ ] `JOURNAL.md` mis à jour avec liste des 7 catégories et nombre de features par catégorie

## NE PAS FAIRE

- ❌ Ne PAS sélectionner les features ici. La sélection = A6.
- ❌ Ne PAS toucher au test set ≥ 2024.
- ❌ Ne PAS ajouter une feature sans test anti-look-ahead correspondant.
- ❌ Ne PAS faire de boucle Python `for row in df.iterrows()`. Tout vectorisé pandas.
- ❌ Ne PAS dépendre de `pandas_ta` ou autre lib externe. Tout réimplémenté.
- ❌ Ne PAS hardcoder de cross-asset si l'utilisateur n'a fourni qu'1 actif.
- ❌ Ne PAS incrémenter `n_trials`.

## Étapes détaillées

### Étape 1 — Inventaire des 7 catégories de features

**Catégorie 1 — Indicateurs techniques classiques** (~25 features)
- Trend : `sma_20, sma_50, sma_200, ema_12, ema_26`
- Distance prix-MA : `dist_sma_20, dist_sma_50, dist_sma_200, dist_ema_12, dist_ema_26` (normalisées par ATR)
- Slope : `slope_sma_20, slope_sma_50` (pente sur 5 barres / ATR)
- Momentum : `rsi_7, rsi_14, rsi_21, macd, macd_signal, macd_hist`
- Oscillateurs : `stoch_k_14, stoch_d_14, williams_r_14, cci_20, mfi_14`
- Volatilité : `atr_14, atr_pct_14, bb_width_20, kc_width_20`

**Catégorie 2 — Price action** (~10 features)
- `body_to_range_ratio` : `|close - open| / (high - low)`
- `upper_shadow_ratio`, `lower_shadow_ratio`
- `gap_overnight` : `(open - close.shift(1)) / close.shift(1)`
- `consecutive_up`, `consecutive_down` : compteurs vectorisés
- `range_atr_ratio` : `(high - low) / atr_14`
- `inside_bar`, `outside_bar`, `doji` : booléens

**Catégorie 3 — Statistiques rolling** (~10 features)
- Z-scores : `close_zscore_20, close_zscore_60, atr_zscore_60, volume_zscore_20`
- Percentiles : `return_percentile_20, return_percentile_60, vol_percentile_60`
- Moments : `skew_returns_20, kurt_returns_20, autocorr_returns_lag1_20`

**Catégorie 4 — Régime de marché** (~7 features)
- `efficiency_ratio_20` (Kaufman)
- `adx_14` (déjà cat 1)
- `trend_strength` : 1 si `adx_14 > 25`
- `dist_sma_200_abs_atr`
- `regime_trending_binary` : combo ADX > 25 AND |dist_sma_200| > 2
- `vol_regime_low`, `vol_regime_mid`, `vol_regime_high` (terciles ATR 60j, one-hot)

**Catégorie 5 — Économiques** (9 features, déjà construites en Prompt 05)
- 6 booléennes `event_high_within_{1,4,24}h_{USD,EUR}`
- 3 numériques `hours_since_last_{nfp,fomc}, hours_to_next_event_high`

**Catégorie 6 — Session de marché** (~8 features)
- `session_tokyo, session_london, session_ny, session_overlap_london_ny`
- Encodage cyclique : `day_sin, day_cos, month_sin, month_cos`

**Catégorie 7 — Cross-asset / macro** (~3-5 features, optionnel selon dispo)
- `usdchf_return_5` (proxy USD strength)
- `xauusd_return_5` (gold = risk-off)
- `btcusd_return_5` (crypto risk-on)

**Total cible : 65-71 features** selon dispo Volume et cross-asset.

### Étape 2 — Implémenter `app/features/superset.py`

```python
"""Superset de features pour méta-labeling ML (pivot v4 A5).

Toutes les fonctions sont vectorisées, anti-look-ahead via @look_ahead_safe.
Ce module GÉNÈRE le superset, il ne SÉLECTIONNE PAS (cf. A6).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from app.features.indicators import (
    rsi, atr, adx, macd, stoch_k, stoch_d, williams_r,
    cci, mfi, bb_width, keltner_width,
)
from app.features.regime import efficiency_ratio
from app.testing.look_ahead_validator import look_ahead_safe


# ─────────────────────────────────────────────────────────────
# Catégorie 1 — Indicateurs techniques
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    atr14 = atr(df, 14).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    for period in [20, 50, 200]:
        sma = close.rolling(period, min_periods=period // 2).mean()
        out[f"sma_{period}"] = sma
        out[f"dist_sma_{period}"] = (close - sma) / atr14
    for period in [12, 26]:
        ema = close.ewm(span=period, adjust=False, min_periods=period // 2).mean()
        out[f"ema_{period}"] = ema
        out[f"dist_ema_{period}"] = (close - ema) / atr14
    out["slope_sma_20"] = close.rolling(20).mean().diff(5) / (5 * atr14)
    out["slope_sma_50"] = close.rolling(50).mean().diff(5) / (5 * atr14)
    return out


@look_ahead_safe
def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    for period in [7, 14, 21]:
        out[f"rsi_{period}"] = rsi(close, period)
    macd_line, signal_line, hist = macd(close, fast=12, slow=26, signal=9)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    return out


@look_ahead_safe
def oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["stoch_k_14"] = stoch_k(df, 14)
    out["stoch_d_14"] = stoch_d(df, 14, 3)
    out["williams_r_14"] = williams_r(df, 14)
    out["cci_20"] = cci(df, 20)
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        out["mfi_14"] = mfi(df, 14)
    return out


@look_ahead_safe
def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    atr14 = atr(df, 14)
    out = pd.DataFrame(index=df.index)
    out["atr_14"] = atr14
    out["atr_pct_14"] = atr14 / close
    out["bb_width_20"] = bb_width(close, 20, 2.0)
    out["kc_width_20"] = keltner_width(df, 20, 2.0)
    return out


# ─────────────────────────────────────────────────────────────
# Catégorie 2 — Price action
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, low_, c = df["Open"], df["High"], df["Low"], df["Close"]
    range_ = (h - low_).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["body_to_range_ratio"] = (c - o).abs() / range_
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - low_
    out["upper_shadow_ratio"] = upper_shadow / range_
    out["lower_shadow_ratio"] = lower_shadow / range_
    out["gap_overnight"] = (o - c.shift(1)) / c.shift(1)

    up = (c.diff() > 0).astype(int)
    out["consecutive_up"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    down = (c.diff() < 0).astype(int)
    out["consecutive_down"] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)

    out["range_atr_ratio"] = range_ / atr(df, 14).replace(0, np.nan)
    out["inside_bar"] = ((h < h.shift(1)) & (low_ > low_.shift(1))).astype(int)
    out["outside_bar"] = ((h > h.shift(1)) & (low_ < low_.shift(1))).astype(int)
    out["doji"] = ((c - o).abs() < 0.1 * range_).astype(int)
    return out


# ─────────────────────────────────────────────────────────────
# Catégorie 3 — Statistiques rolling
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))
    atr14 = atr(df, 14)
    out = pd.DataFrame(index=df.index)
    for window in [20, 60]:
        mean = close.rolling(window).mean()
        std = close.rolling(window).std().replace(0, np.nan)
        out[f"close_zscore_{window}"] = (close - mean) / std
    atr_mean = atr14.rolling(60).mean()
    atr_std = atr14.rolling(60).std().replace(0, np.nan)
    out["atr_zscore_60"] = (atr14 - atr_mean) / atr_std
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        vol_mean = df["Volume"].rolling(20).mean()
        vol_std = df["Volume"].rolling(20).std().replace(0, np.nan)
        out["volume_zscore_20"] = (df["Volume"] - vol_mean) / vol_std
    for window in [20, 60]:
        out[f"return_percentile_{window}"] = log_ret.rolling(window).rank(pct=True)
    vol_realized = log_ret.rolling(20).std()
    out["vol_percentile_60"] = vol_realized.rolling(60).rank(pct=True)
    out["skew_returns_20"] = log_ret.rolling(20).skew()
    out["kurt_returns_20"] = log_ret.rolling(20).kurt()
    # Autocorr lag 1 sur fenêtre 20j
    out["autocorr_returns_lag1_20"] = (
        log_ret.rolling(20).apply(
            lambda x: x.autocorr(1) if x.std() > 0 else 0.0,
            raw=False,
        )
    )
    return out


# ─────────────────────────────────────────────────────────────
# Catégorie 4 — Régime de marché
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    adx14 = adx(df, 14)
    atr14 = atr(df, 14).replace(0, np.nan)
    sma200 = close.rolling(200, min_periods=50).mean()
    out = pd.DataFrame(index=df.index)
    out["efficiency_ratio_20"] = efficiency_ratio(close, 20)
    out["trend_strength"] = (adx14 > 25).astype(int)
    out["dist_sma_200_abs_atr"] = ((close - sma200) / atr14).abs()
    out["regime_trending_binary"] = (
        (adx14 > 25) & (((close - sma200) / atr14).abs() > 2.0)
    ).astype(int)
    # Vol regime : terciles ATR sur 60j → one-hot
    atr_rolling = atr(df, 14).rolling(60, min_periods=20)
    p33 = atr_rolling.quantile(0.33)
    p66 = atr_rolling.quantile(0.66)
    atr14_raw = atr(df, 14)
    out["vol_regime_low"] = (atr14_raw <= p33).astype(int)
    out["vol_regime_mid"] = ((atr14_raw > p33) & (atr14_raw <= p66)).astype(int)
    out["vol_regime_high"] = (atr14_raw > p66).astype(int)
    return out


# ─────────────────────────────────────────────────────────────
# Catégorie 5 — Économiques
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def economic_features_for_index(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Délègue à app/features/economic.py si calendrier dispo. Sinon fallback −1."""
    from app.features.economic import load_calendar, compute_event_features
    try:
        years = sorted({ts.year for ts in price_index})
        cal = load_calendar(years)
        return compute_event_features(price_index, cal)
    except Exception:
        return pd.DataFrame(
            -1.0,
            index=price_index,
            columns=[
                "event_high_within_1h_USD", "event_high_within_4h_USD", "event_high_within_24h_USD",
                "event_high_within_1h_EUR", "event_high_within_4h_EUR", "event_high_within_24h_EUR",
                "hours_since_last_nfp", "hours_since_last_fomc", "hours_to_next_event_high",
            ],
        )


# ─────────────────────────────────────────────────────────────
# Catégorie 6 — Sessions
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def session_features(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sessions UTC + encodage cyclique du temps."""
    idx = pd.DatetimeIndex(price_index)
    hour = idx.hour
    weekday = idx.weekday
    month = idx.month
    out = pd.DataFrame(index=idx)
    out["session_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)
    out["session_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    out["session_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    out["session_overlap_london_ny"] = ((hour >= 13) & (hour < 16)).astype(int)
    out["day_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["day_cos"] = np.cos(2 * np.pi * weekday / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    return out


# ─────────────────────────────────────────────────────────────
# Catégorie 7 — Cross-asset (optionnel)
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def cross_asset_features(price_index: pd.DatetimeIndex, asset: str | None = None) -> pd.DataFrame:
    """Features cross-asset si données disponibles. Sinon colonnes NaN (droppées en sortie)."""
    from app.data.loader import load_asset
    from app.data.registry import discover_assets

    out = pd.DataFrame(index=price_index)
    try:
        available = discover_assets()
    except Exception:
        return out
    tf_target = "D1"  # par défaut
    macro_map = {
        "usdchf_return_5": ("USDCHF", tf_target),
        "xauusd_return_5": ("XAUUSD", tf_target),
        "btcusd_return_5": ("BTCUSD", tf_target),
    }
    for name, (sym, tf) in macro_map.items():
        if sym in available and tf in available.get(sym, []) and sym != asset:
            try:
                df_macro = load_asset(sym, tf)
                ret = np.log(df_macro["Close"] / df_macro["Close"].shift(5))
                out[name] = ret.reindex(price_index, method="ffill")
            except Exception:
                out[name] = np.nan
        else:
            out[name] = np.nan
    return out


# ─────────────────────────────────────────────────────────────
# Agrégateur principal
# ─────────────────────────────────────────────────────────────

@look_ahead_safe
def build_superset(df: pd.DataFrame, asset: Optional[str] = None) -> pd.DataFrame:
    """Construit le superset de ~70 features pour méta-labeling.

    Args:
        df: DataFrame OHLCV (PascalCase : Open, High, Low, Close, Volume).
        asset: Nom de l'actif (pour exclure du cross-asset).

    Returns:
        DataFrame indexé comme df, 60-75 colonnes float64, NaN possibles
        sur les premiers ~200 lignes (warmup SMA200).
    """
    if df.empty:
        return pd.DataFrame()
    parts = [
        trend_features(df),
        momentum_features(df),
        oscillator_features(df),
        volatility_features(df),
        price_action_features(df),
        statistical_features(df),
        regime_features(df),
        economic_features_for_index(df.index),
        session_features(df.index),
        cross_asset_features(df.index, asset),
    ]
    out = pd.concat(parts, axis=1)
    out = out.dropna(axis=1, how="all")
    out = out.astype(np.float64)
    out = out.loc[:, ~out.columns.duplicated()]
    return out
```

### Étape 3 — Ajouter les indicateurs manquants dans `app/features/indicators.py`

Vérifier d'abord :
```bash
rtk grep -n "^def (macd|stoch_k|stoch_d|williams_r|cci|mfi|bb_width|keltner_width)" app/features/indicators.py
```

S'ils manquent, ajouter :

```python
@look_ahead_safe
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD = EMA(fast) - EMA(slow), Signal = EMA(MACD, signal), Hist = MACD - Signal."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


@look_ahead_safe
def stoch_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic %K."""
    lowest = df["Low"].rolling(period).min()
    highest = df["High"].rolling(period).max()
    rng = (highest - lowest).replace(0, np.nan)
    return 100 * (df["Close"] - lowest) / rng


@look_ahead_safe
def stoch_d(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> pd.Series:
    """Stochastic %D = SMA(K, smooth)."""
    return stoch_k(df, period).rolling(smooth).mean()


@look_ahead_safe
def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest = df["High"].rolling(period).max()
    lowest = df["Low"].rolling(period).min()
    rng = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - df["Close"]) / rng


@look_ahead_safe
def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * md.replace(0, np.nan))


@look_ahead_safe
def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf = tp * df["Volume"]
    pos = raw_mf.where(tp > tp.shift(1), 0.0)
    neg = raw_mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum().replace(0, np.nan)
    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


@look_ahead_safe
def bb_width(close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Largeur des bandes de Bollinger normalisée par la SMA."""
    sma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return (2 * mult * sd) / sma.replace(0, np.nan)


@look_ahead_safe
def keltner_width(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Largeur Keltner = 2 * mult * ATR / EMA(close)."""
    ema = df["Close"].ewm(span=period, adjust=False).mean()
    atr_p = atr(df, period)
    return (2 * mult * atr_p) / ema.replace(0, np.nan)
```

### Étape 4 — Tests `tests/unit/test_superset_features.py`

```python
"""Tests du superset de features pivot v4 A5."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.superset import build_superset
from app.testing.look_ahead_validator import assert_no_look_ahead


@pytest.fixture
def synthetic_df():
    """500 barres OHLCV synthétiques reproductibles."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    close = 40_000 + rng.normal(0, 100, n).cumsum()
    open_ = close + rng.normal(0, 20, n)
    high = np.maximum(close, open_) + rng.uniform(10, 50, n)
    low = np.minimum(close, open_) - rng.uniform(10, 50, n)
    volume = rng.uniform(1000, 10_000, n)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=idx)


def test_superset_size(synthetic_df):
    feat = build_superset(synthetic_df)
    assert feat.shape[1] >= 60, f"Attendu ≥ 60, obtenu {feat.shape[1]}"


def test_superset_no_nan_after_warmup(synthetic_df):
    feat = build_superset(synthetic_df)
    after_warmup = feat.iloc[250:]
    nan_cols = after_warmup.columns[after_warmup.isna().any()].tolist()
    allowed_prefixes = ("usdchf_", "xauusd_", "btcusd_")
    forbidden = [c for c in nan_cols if not c.startswith(allowed_prefixes)]
    assert not forbidden, f"NaN après warmup: {forbidden}"


def test_superset_unique_columns(synthetic_df):
    feat = build_superset(synthetic_df)
    assert feat.columns.is_unique


def test_superset_dtypes_float64(synthetic_df):
    feat = build_superset(synthetic_df)
    non_float = feat.dtypes[feat.dtypes != np.float64]
    assert non_float.empty, f"Dtypes non-float64: {non_float.to_dict()}"


def test_superset_anti_look_ahead(synthetic_df):
    def feat_close_zscore(df):
        return build_superset(df)["close_zscore_20"]
    assert_no_look_ahead(feat_close_zscore, synthetic_df, n_samples=20)


def test_category_trend(synthetic_df):
    feat = build_superset(synthetic_df)
    expected = ["sma_20", "sma_50", "sma_200", "dist_sma_20", "dist_sma_50",
                "dist_sma_200", "dist_ema_12", "dist_ema_26", "slope_sma_20", "slope_sma_50"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_momentum(synthetic_df):
    feat = build_superset(synthetic_df)
    expected = ["rsi_7", "rsi_14", "rsi_21", "macd", "macd_signal", "macd_hist"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_price_action(synthetic_df):
    feat = build_superset(synthetic_df)
    expected = ["body_to_range_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
                "gap_overnight", "consecutive_up", "consecutive_down", "range_atr_ratio",
                "inside_bar", "outside_bar", "doji"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_stats(synthetic_df):
    feat = build_superset(synthetic_df)
    expected = ["close_zscore_20", "close_zscore_60", "atr_zscore_60",
                "return_percentile_20", "return_percentile_60",
                "skew_returns_20", "kurt_returns_20"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_regime(synthetic_df):
    feat = build_superset(synthetic_df)
    expected = ["efficiency_ratio_20", "trend_strength",
                "vol_regime_low", "vol_regime_mid", "vol_regime_high"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"


def test_category_sessions(synthetic_df):
    feat = build_superset(synthetic_df)
    expected = ["session_tokyo", "session_london", "session_ny",
                "session_overlap_london_ny", "day_sin", "day_cos", "month_sin", "month_cos"]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Manquant: {missing}"
```

### Étape 5 — Vérification manuelle (sur demande utilisateur)

```bash
rtk python -c "
from app.data.loader import load_asset
from app.features.superset import build_superset
df = load_asset('US30', 'D1')
df_train = df.loc[:'2022-12-31']
feat = build_superset(df_train, asset='US30')
print(f'Shape: {feat.shape}')
print(f'NaN après row 250: {feat.iloc[250:].isna().sum().sum()}')
print('Catégories par préfixe:')
for prefix in ['sma_', 'dist_', 'rsi_', 'macd', 'stoch_', 'williams_', 'cci', 'atr_', 'bb_', 'kc_', 'body_', 'consecutive_', 'range_', 'inside_', 'outside_', 'doji', 'close_zscore', 'return_percentile', 'skew_', 'kurt_', 'autocorr_', 'efficiency_', 'trend_', 'regime_', 'vol_regime', 'event_', 'hours_', 'session_', 'day_', 'month_', 'usdchf_', 'xauusd_', 'btcusd_']:
    n = sum(c.startswith(prefix) for c in feat.columns)
    if n > 0:
        print(f'  {prefix}*: {n}')
"
```

## Tests unitaires associés

11 tests dans `tests/unit/test_superset_features.py` (cf. Étape 4).

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A5 : Feature superset (~70 features)

- **Statut** : ✅ Terminé
- **Type** : Infrastructure ML (0 n_trial)
- **Fichiers créés** : `app/features/superset.py`, `tests/unit/test_superset_features.py`
- **Fichiers modifiés** : `app/features/indicators.py` (ajout MACD/Stoch/CCI/MFI/Williams/BB_width/KC_width si manquants)
- **Catégories** (sur US30 D1) :
  - Trend : 12 features (5 SMA + 5 dist + 2 slope)
  - Momentum : 6 features (3 RSI + 3 MACD)
  - Oscillator : 5 features (Stoch K/D + Williams + CCI + MFI)
  - Volatilité : 4 features
  - Price action : 10 features
  - Statistical : 11 features
  - Régime : 5 features (+ 3 one-hot vol)
  - Économique : 9 features
  - Session : 8 features
  - Cross-asset : 3 features (USDCHF, XAUUSD, BTCUSD)
- **Total** : ~71 features US30 D1 (avec Volume), ~70 EURUSD H4
- **Tests** : 11/11 passed
- **Anti-look-ahead** : OK
- **make verify** : ✅ passé
- **Prochaine étape** : A6 — Feature ranking train + bootstrap stabilisation
```

## Critères go/no-go

- **GO Phase A6** si :
  - `build_superset(df_train)` retourne ≥ 60 colonnes pour US30 D1
  - Tous les tests A5 passent
  - Anti-look-ahead OK
  - Aucun NaN après warmup 200 barres (hors cross-asset optionnel)
- **NO-GO, revenir à** : cette phase si NaN persistants ou look-ahead détecté. Fixer la feature défaillante avant de continuer.

## Annexes

### A1 — Pourquoi 70 features et pas 200

- > 100 features : RF max_depth=4 ne peut pas toutes les exploiter, dilution.
- < 30 features : risque de manquer des signaux non-évidents.
- 60-80 = sweet spot empirique pour méta-labeling sur ~500-1000 samples train (Donchian D1).

### A2 — Pourquoi pas de pandas-ta

- Dépendance externe = risque de break sur upgrade pandas.
- Implémentations parfois buggées (cf. issues GitHub).
- Réimplémenter en pandas pur = contrôle total + tests anti-look-ahead garantis.

### A3 — Cross-asset : pourquoi optionnel

Si l'utilisateur n'a que US30 dans `data/raw/`, les features cross-asset retournent NaN qui sont dropées. Aucun crash. Si USDCHF, XAUUSD, BTCUSD sont disponibles → +3 features informatives.

### A4 — Pourquoi normaliser par ATR pour les distances prix-MA

`dist_sma_50` brut dépend de l'échelle du prix (US30 ≈ 40k, EURUSD ≈ 1.1). Normaliser par ATR rend la feature **comparable cross-asset** → le même RF entraîné peut servir pour H_new4 portfolio.

### A5 — Vol regime one-hot vs ordinal

Encoder le régime de volatilité en 3 colonnes one-hot (low/mid/high) au lieu d'1 ordinal car :
- RF gère mieux les catégoriels one-hot
- Évite l'hypothèse linéaire low < mid < high
- Coût : +2 colonnes (négligeable)

### A6 — Cyclic encoding (sin/cos) pour day/month

Si on encode `day_of_week` en 0-4, le RF traite lundi (0) et vendredi (4) comme "loin", ce qui est faux (la semaine est cyclique). `sin/cos` capture la cyclicité naturellement, plus efficace pour le RF.

### A7 — Économique en fallback −1

Si `data/raw/economic_calendar/` est absent ou erreur de chargement, on retourne −1 partout au lieu de NaN. Le RF gère ces valeurs comme une catégorie "no calendar info" sans casser le pipeline.

### A8 — Pourquoi vectoriser à 100 %

Une boucle Python sur 500 000 barres = ~10 minutes. La version vectorisée = ~2 secondes. Sur 70 features × N actifs × N retrains walk-forward, c'est la différence entre 1 minute et 10 heures.

## Fin du prompt A5.
**Suivant** : [A6_feature_ranking.md](A6_feature_ranking.md)
