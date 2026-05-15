"""Run H2 — Méta-labeling RF sur US30 D1 (Donchian, 2 configs).

Hypothèse : Le méta-labeling RF appliqué au Donchian US30 D1
reproduit le Sharpe +8.84 walk-forward de v2 H05 avec les coûts v3.

Teste DEUX configurations :
  - Config A : Donchian(N=20, M=20) — paramètres exacts de v2 H05
  - Config B : Donchian(N=100, M=10) — paramètres v3 H06

TP/SL adaptatifs = 2×/1× ATR(20) à l'entrée, timeout 5 barres D1.
Coûts v3 US30 : spread 3.0 + slippage 5.0 = 8.0 points total par trade.

Règles critiques :
- Split temporel figé : train ≤ 2022-12-31, val = 2023, test ≥ 2024-01-01
- Coûts v3 US30 : 8.0 points/trade
- Sweep seuil méta sur TRAIN UNIQUEMENT : [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
- 8 features SANS contexte de marché
- RF : 200 arbres, max_depth=4, class_weight='balanced', random_state=42
- Directions LONG + SHORT
- validate_edge() avec n_trials=25
- Pas de look-ahead
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.analysis.edge_validation import EdgeReport, validate_edge
from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.strategies.donchian import DonchianBreakout
from app.testing.snooping_guard import check_unlocked, read_oos

logger = logging.getLogger("h2_us30_meta")

# ═══════════════════════════════════════════════════════════════════════════════
# Constantes figées ex ante
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")

CONFIGS: dict[str, dict[str, int]] = {
    "A": {"donchian_N": 20, "donchian_M": 20},   # v2 H05 exact
    "B": {"donchian_N": 100, "donchian_M": 10},   # v3 H06
}

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

META_THRESHOLDS: list[float] = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
N_TRIALS: int = 25  # 24 précédents + H2

# TP/SL adaptatifs
TP_ATR_MULT: float = 2.0
SL_ATR_MULT: float = 1.0
ATR_PERIOD: int = 20
TIMEOUT_BARS: int = 5

COST_TOTAL_POINTS: float = 8.0  # spread 3.0 + slippage 5.0
PIP_SIZE: float = 1.0  # US30

US30_CONFIG: AssetConfig = ASSET_CONFIGS["US30"]

OUTPUT_PATH = Path("predictions/h2_us30_meta.json")
LOG_PATH = Path("logs/h2_us30_meta.log")


# ═══════════════════════════════════════════════════════════════════════════════
# Indicateurs vectorisés
# ═══════════════════════════════════════════════════════════════════════════════

def _atr_wilder(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20,
) -> np.ndarray:
    """ATR Wilder smoothing vectorisé — 1 boucle warmup uniquement."""
    n = len(close)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    atr_arr = np.full(n, np.nan)
    atr_arr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, n):
        atr_arr[i] = (1.0 - alpha) * atr_arr[i - 1] + alpha * tr[i]
    return atr_arr


def _rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI Wilder smoothing vectorisé."""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = gain[1:period + 1].mean()
    avg_loss[period] = loss[1:period + 1].mean()
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain[i] = (1.0 - alpha) * avg_gain[i - 1] + alpha * gain[i]
        avg_loss[i] = (1.0 - alpha) * avg_loss[i - 1] + alpha * loss[i]
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0.0)
    return 100.0 - (100.0 / (1.0 + rs))


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation stateful avec TP/SL adaptatifs ATR
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_adaptive_trades(
    df: pd.DataFrame,
    signals: pd.Series,
    atr_vals: np.ndarray,
    tp_atr_mult: float = TP_ATR_MULT,
    sl_atr_mult: float = SL_ATR_MULT,
    timeout_bars: int = TIMEOUT_BARS,
    cost_total_points: float = COST_TOTAL_POINTS,
    pip_size: float = PIP_SIZE,
) -> tuple[list[dict[str, Any]], pd.Series]:
    """Simulation stateful bar-by-bar avec TP/SL basés sur ATR.

    Pour chaque barre où signal ≠ 0 :
    - TP = entry ± tp_atr_mult * ATR(entry)
    - SL = entry ∓ sl_atr_mult * ATR(entry)
    - Timeout = timeout_bars barres
    - TP prime sur SL si même barre (conservateur)
    - Coût total déduit du PnL

    Args:
        df: DataFrame OHLC, index=DatetimeIndex trié.
        signals: pd.Series −1/0/1, même index que df.
        atr_vals: np.ndarray ATR(20), même longueur que df.
        tp_atr_mult: Multiplicateur ATR pour TP.
        sl_atr_mult: Multiplicateur ATR pour SL.
        timeout_bars: Durée max du trade en barres.
        cost_total_points: Coût total (spread + slippage) en points.
        pip_size: Taille du pip (1.0 pour US30).

    Returns:
        (trades, meta_labels):
            trades: liste de dicts avec pips_net, entry_time, exit_time, etc.
            meta_labels: pd.Series 1/0/NaN, même index que df.
    """
    n = len(df)
    meta_labels = pd.Series(np.nan, index=df.index, dtype="float64")

    if n == 0:
        return [], meta_labels

    closes = df["Close"].values.astype(np.float64)
    highs = df["High"].values.astype(np.float64)
    lows = df["Low"].values.astype(np.float64)
    times = df.index

    trades: list[dict[str, Any]] = []
    i = 0

    while i < n:
        sig_val = int(signals.iloc[i])

        if sig_val == 0:
            i += 1
            continue

        # Vérification ATR valide
        entry_atr = float(atr_vals[i])
        if np.isnan(entry_atr) or entry_atr <= 0:
            i += 1
            continue

        signal = sig_val
        entry_bar = i
        entry_price = float(closes[i])

        tp_dist = tp_atr_mult * entry_atr
        sl_dist = sl_atr_mult * entry_atr

        if signal == 1:  # LONG
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:  # SHORT
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist

        result: int = 0
        pips_net = -cost_total_points
        exit_time = times[entry_bar]
        exit_price = entry_price

        for j in range(1, timeout_bars + 1):
            idx = i + j
            if idx >= n:
                exit_idx = n - 1
                exit_time = times[exit_idx]
                exit_price = float(closes[exit_idx])
                if signal == 1:
                    pips_net = (exit_price - entry_price) / pip_size - cost_total_points
                else:
                    pips_net = (entry_price - exit_price) / pip_size - cost_total_points
                result = 0
                i = n
                break

            curr_high = float(highs[idx])
            curr_low = float(lows[idx])

            if signal == 1:
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low <= sl_price
            else:
                tp_hit = curr_low <= tp_price
                sl_hit = curr_high >= sl_price

            if tp_hit and sl_hit:
                result = 1
                exit_price = tp_price
                pips_net = tp_dist / pip_size - cost_total_points
                exit_time = times[idx]
                i = idx
                break
            elif sl_hit:
                result = 0
                exit_price = sl_price
                pips_net = -sl_dist / pip_size - cost_total_points
                exit_time = times[idx]
                i = idx
                break
            elif tp_hit:
                result = 1
                exit_price = tp_price
                pips_net = tp_dist / pip_size - cost_total_points
                exit_time = times[idx]
                i = idx
                break
        else:
            # Timeout
            exit_idx = min(i + timeout_bars, n - 1)
            exit_time = times[exit_idx]
            exit_price = float(closes[exit_idx])
            if signal == 1:
                pips_net = (exit_price - entry_price) / pip_size - cost_total_points
            else:
                pips_net = (entry_price - exit_price) / pip_size - cost_total_points
            result = 0
            i += timeout_bars

        trades.append({
            "entry_time": str(times[entry_bar]),
            "exit_time": str(exit_time),
            "signal": signal,
            "entry_price": entry_price,
            "exit_price": float(exit_price),
            "pips_net": float(pips_net),
            "result": "win" if result == 1 else "loss",
        })
        meta_labels.iloc[entry_bar] = float(result)

    return trades, meta_labels


def _compute_bt_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcule métriques backtest à partir d'une liste de trades.

    Sharpe via resample quotidien (méthode canonique).
    """
    if not trades:
        return {
            "sharpe": 0.0, "wr": 0.0, "total_trades": 0,
            "total_pnl_pips": 0.0, "max_drawdown_pips": 0.0,
            "profit_factor": 0.0, "mean_pnl_per_trade": 0.0,
            "trades": [],
        }

    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    total_trades = len(trades)
    wins = pnls[pnls > 0]

    wr = float(len(wins) / total_trades) if total_trades > 0 else 0.0
    total_pnl = float(pnls.sum())

    from app.backtest.metrics import sharpe_daily_from_trades
    sharpe = sharpe_daily_from_trades(trades)

    cumsum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumsum)
    max_dd = float((cumsum - peak).min())

    wins_arr = pnls[pnls > 0]
    losses_arr = pnls[pnls < 0]
    if len(losses_arr) > 0:
        pf = float(wins_arr.sum() / abs(losses_arr.sum())) if wins_arr.sum() > 0 else 0.0
    elif len(wins_arr) > 0:
        pf = float("inf")
    else:
        pf = 0.0

    return {
        "sharpe": sharpe,
        "wr": wr,
        "total_trades": total_trades,
        "total_pnl_pips": total_pnl,
        "max_drawdown_pips": max_dd,
        "profit_factor": pf,
        "mean_pnl_per_trade": float(pnls.mean()),
        "trades": trades,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Construction des 8 features du méta-modèle
# ═══════════════════════════════════════════════════════════════════════════════

def build_meta_features(
    df: pd.DataFrame,
    donchian_signals: pd.Series,
    donchian_N: int,
) -> pd.DataFrame:
    """Construit les 8 features du méta-modèle pour TOUTES les barres.

    Features (sans look-ahead) :
    1. dist_breakout_pct  : Distance Close au canal Donchian (% ATR)
    2. force_breakout_pct : (Close − Open) / Open × 100
    3. atr_ratio          : ATR(14) / ATR médian(20)
    4. rsi_14             : RSI(14)
    5. retracement_pct    : Retracement depuis le dernier high N (%)
    6. days_since_signal  : Jours depuis le dernier signal Donchian
    7. vol_ratio          : Volume / Volume médian(20)
    8. day_of_week        : Jour de la semaine (0=Lundi → 4=Vendredi)
    """
    idx = df.index
    feats = pd.DataFrame(index=idx)

    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    open_ = df["Open"].values.astype(np.float64)

    # ATR(14) partagé
    atr14 = _atr_wilder(high, low, close, 14)

    # ── Feature 1 : Distance au breakout (% ATR) ──
    hh_n = df["High"].rolling(donchian_N).max().shift(1).values
    feats["dist_breakout_pct"] = np.where(
        (atr14 > 0) & (~np.isnan(hh_n)),
        (close - hh_n) / atr14 * 100.0,
        0.0,
    )

    # ── Feature 2 : Force du breakout ──
    feats["force_breakout_pct"] = np.where(
        open_ > 0,
        (close - open_) / open_ * 100.0,
        0.0,
    )

    # ── Feature 3 : ATR ratio ──
    atr_series = pd.Series(atr14, index=idx)
    atr_median_20 = atr_series.rolling(20, min_periods=5).median().values
    feats["atr_ratio"] = np.where(
        (atr_median_20 > 0) & (~np.isnan(atr14)),
        atr14 / atr_median_20,
        1.0,
    )

    # ── Feature 4 : RSI(14) ──
    feats["rsi_14"] = _rsi_wilder(close, 14)

    # ── Feature 5 : Retracement depuis le dernier high N ──
    rolling_high_n = df["High"].rolling(donchian_N).max().values
    feats["retracement_pct"] = np.where(
        rolling_high_n > 0,
        (close - rolling_high_n) / rolling_high_n * 100.0,
        0.0,
    )

    # ── Feature 6 : Jours depuis le dernier signal ──
    sig_arr = donchian_signals.values.astype(np.int8)
    days_since = np.full(len(df), np.nan, dtype=np.float64)
    last_sig = -1
    for k in range(len(df)):
        if sig_arr[k] != 0:
            if last_sig >= 0:
                days_since[k] = float(k - last_sig)
            last_sig = k
        elif last_sig >= 0:
            days_since[k] = float(k - last_sig)
    feats["days_since_signal"] = days_since

    # ── Feature 7 : Volume ratio ──
    if "Volume" in df.columns and not df["Volume"].isna().all():
        vol = df["Volume"].values.astype(np.float64)
        vol_series = pd.Series(vol, index=idx)
        vol_median_20 = vol_series.rolling(20, min_periods=5).median().values
        feats["vol_ratio"] = np.where(
            vol_median_20 > 0,
            vol / vol_median_20,
            1.0,
        )
    else:
        feats["vol_ratio"] = 1.0

    # ── Feature 8 : Jour de la semaine ──
    feats["day_of_week"] = idx.dayofweek.astype(np.float64)

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _split_periods(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel strict."""
    train = df[df.index <= TRAIN_END]
    val = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test = df[df.index >= TEST_START]
    return train, val, test


def _build_equity_from_trades(
    trades: list[dict[str, Any]],
    capital: float = 10_000.0,
    pip_value_eur: float = 0.92,
) -> tuple[pd.Series, pd.DataFrame]:
    """Construit equity curve monétaire et trades DataFrame pour validate_edge()."""
    if not trades:
        empty_idx = pd.DatetimeIndex([])
        return pd.Series(dtype=np.float64, index=empty_idx), pd.DataFrame({"pnl": []})
    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    exit_times = pd.to_datetime([t["exit_time"] for t in trades])
    equity_vals = capital + np.cumsum(pnls * pip_value_eur)
    equity = pd.Series(equity_vals, index=exit_times).sort_index()
    trades_df = pd.DataFrame({"pnl": pnls}, index=exit_times)
    return equity, trades_df


def _filter_signals_by_proba(
    signals: pd.Series,
    proba_win: np.ndarray,
    threshold: float,
    signal_idx: np.ndarray,
) -> pd.Series:
    """Filtre les signaux : ne garde que ceux où proba_win > threshold."""
    filtered = signals.copy()
    for k, pos in enumerate(signal_idx):
        if proba_win[k] <= threshold:
            filtered.iloc[pos] = 0
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline complet pour une config
# ═══════════════════════════════════════════════════════════════════════════════

def run_config(
    df: pd.DataFrame,
    config_name: str,
    donchian_N: int,
    donchian_M: int,
) -> dict[str, Any] | None:
    """Exécute le pipeline complet méta-labeling pour une config Donchian.

    Returns:
        dict résultats ou None si abandon (mono-classe train, 0 signaux, etc.).
    """
    logger.info("=" * 60)
    logger.info("Configuration %s — Donchian(N=%d, M=%d)", config_name, donchian_N, donchian_M)

    # ── Split ──────────────────────────────────────────────────────────
    df_train, df_val, df_test = _split_periods(df)
    logger.info(
        "Split : train=%d (≤%s), val=%d (%s→%s), test=%d (≥%s)",
        len(df_train), TRAIN_END.date(),
        len(df_val), VAL_START.date(), VAL_END.date(),
        len(df_test), TEST_START.date(),
    )

    # ── Signaux Donchian ────────────────────────────────────────────────
    strat = DonchianBreakout(N=donchian_N, M=donchian_M)
    signals_train = strat.generate_signals(df_train)
    signals_val = strat.generate_signals(df_val)
    signals_test = strat.generate_signals(df_test)

    for period_name, sig in [("train", signals_train), ("val", signals_val), ("test", signals_test)]:
        n_long = int((sig == 1).sum())
        n_short = int((sig == -1).sum())
        logger.info("Signaux %s : %d total (LONG=%d, SHORT=%d)", period_name, n_long + n_short, n_long, n_short)

    n_sig = {
        "train": int((signals_train != 0).sum()),
        "val": int((signals_val != 0).sum()),
        "test": int((signals_test != 0).sum()),
    }

    if n_sig["train"] == 0:
        logger.error("Config %s : 0 signal train — abandon.", config_name)
        return None

    # ── ATR(20) par période ─────────────────────────────────────────────
    atr_train = _atr_wilder(
        df_train["High"].values, df_train["Low"].values, df_train["Close"].values, ATR_PERIOD,
    )
    atr_val = _atr_wilder(
        df_val["High"].values, df_val["Low"].values, df_val["Close"].values, ATR_PERIOD,
    )
    atr_test = _atr_wilder(
        df_test["High"].values, df_test["Low"].values, df_test["Close"].values, ATR_PERIOD,
    )

    # ── Baseline backtest (sans méta-labeling) ──────────────────────────
    logger.info("── Baseline (Donchian pur, TP/SL ATR-adaptatifs) ──")

    bt_train_base_trades, _ = _simulate_adaptive_trades(df_train, signals_train, atr_train)
    bt_train_base = _compute_bt_metrics(bt_train_base_trades)
    logger.info(
        "TRAIN base : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts, dd=%.1f pts",
        bt_train_base["sharpe"], bt_train_base["wr"] * 100,
        bt_train_base["total_trades"], bt_train_base["total_pnl_pips"],
        bt_train_base["max_drawdown_pips"],
    )

    bt_val_base_trades, _ = _simulate_adaptive_trades(df_val, signals_val, atr_val)
    bt_val_base = _compute_bt_metrics(bt_val_base_trades)
    logger.info(
        "VAL  base : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts, dd=%.1f pts",
        bt_val_base["sharpe"], bt_val_base["wr"] * 100,
        bt_val_base["total_trades"], bt_val_base["total_pnl_pips"],
        bt_val_base["max_drawdown_pips"],
    )

    bt_test_base_trades, _ = _simulate_adaptive_trades(df_test, signals_test, atr_test)
    bt_test_base = _compute_bt_metrics(bt_test_base_trades)
    logger.info(
        "TEST base : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts, dd=%.1f pts",
        bt_test_base["sharpe"], bt_test_base["wr"] * 100,
        bt_test_base["total_trades"], bt_test_base["total_pnl_pips"],
        bt_test_base["max_drawdown_pips"],
    )

    # ── Méta-labels ─────────────────────────────────────────────────────
    logger.info("Génération méta-labels (TP/SL ATR-adaptatifs)...")

    _, meta_train = _simulate_adaptive_trades(df_train, signals_train, atr_train)
    _, meta_val = _simulate_adaptive_trades(df_val, signals_val, atr_val)
    _, meta_test_labels = _simulate_adaptive_trades(df_test, signals_test, atr_test)

    n_meta = {
        "train": int(meta_train.notna().sum()),
        "val": int(meta_val.notna().sum()),
        "test": int(meta_test_labels.notna().sum()),
    }
    for period, ml in [("train", meta_train), ("val", meta_val), ("test", meta_test_labels)]:
        n_w = int((ml == 1).sum())
        n_l = int((ml == 0).sum())
        logger.info(
            "Méta-labels %s : %d labels, %d wins (%.1f%%), %d losses",
            period, n_meta[period], n_w,
            n_w / max(n_meta[period], 1) * 100, n_l,
        )

    # Vérification mono-classe
    n_wins_train = int((meta_train == 1).sum())
    n_losses_train = int((meta_train == 0).sum())
    if n_wins_train == 0 or n_losses_train == 0:
        logger.error(
            "Config %s : méta-labels train MONO-CLASSE — %d wins, %d losses. "
            "Le RF ne peut pas apprendre.",
            config_name, n_wins_train, n_losses_train,
        )
        return {
            "config_name": config_name,
            "donchian_N": donchian_N, "donchian_M": donchian_M,
            "abandoned": True,
            "reason": f"meta_labels_mono_classe_train ({n_wins_train}W/{n_losses_train}L)",
            "train": {"base": bt_train_base},
            "val": {"base": bt_val_base},
            "test": {"base": bt_test_base},
        }

    # ── Construction des 8 features ──────────────────────────────────────
    logger.info("Construction des 8 features du méta-modèle...")
    X_train_all = build_meta_features(df_train, signals_train, donchian_N)
    X_val_all = build_meta_features(df_val, signals_val, donchian_N)
    X_test_all = build_meta_features(df_test, signals_test, donchian_N)
    logger.info(
        "Features shape : train=%s, val=%s, test=%s",
        X_train_all.shape, X_val_all.shape, X_test_all.shape,
    )

    # ── Entraînement RF ──────────────────────────────────────────────────
    train_mask = meta_train.notna()
    X_train_ml = X_train_all.loc[train_mask]
    y_train_ml = meta_train.loc[train_mask].astype(int)

    if X_train_ml.empty:
        logger.error("Config %s : 0 échantillon méta-labellisé train.", config_name)
        return None

    X_train_ml = X_train_ml.fillna(0.0)
    logger.info("Entraînement RF sur %d échantillons...", len(X_train_ml))

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train_ml, y_train_ml)

    logger.info("Classes RF : %s", rf.classes_.tolist())
    if 1 in rf.classes_:
        class1_pos = int(list(rf.classes_).index(1))
    else:
        class1_pos = 0
        logger.warning(
            "Config %s : RF mono-classe %s → proba[:,0] utilisée comme proxy P(win)",
            config_name, rf.classes_.tolist(),
        )

    # Feature importance
    feat_names = list(X_train_all.columns)
    importances = sorted(
        zip(feat_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    logger.info("Feature importance :")
    for name, imp in importances:
        logger.info("  %-22s %.4f", name, imp)

    # ── Sweep seuil sur TRAIN UNIQUEMENT ─────────────────────────────────
    logger.info("Sweep seuil méta sur TRAIN (%d valeurs)...", len(META_THRESHOLDS))

    X_train_pred = X_train_all.fillna(0.0)
    proba_train_all = rf.predict_proba(X_train_pred)
    proba_win_train = proba_train_all[:, class1_pos]

    train_sig_idx = np.where(signals_train.values != 0)[0]
    proba_win_train_sig = proba_win_train[train_sig_idx]

    sweep_results: dict[float, dict[str, Any]] = {}
    best_sharpe = -np.inf
    best_threshold = META_THRESHOLDS[0]

    for thresh in META_THRESHOLDS:
        filtered = _filter_signals_by_proba(
            signals_train, proba_win_train_sig, thresh, train_sig_idx,
        )
        trades_f, _ = _simulate_adaptive_trades(df_train, filtered, atr_train)
        bt_f = _compute_bt_metrics(trades_f)
        sr = float(bt_f["sharpe"])
        sweep_results[thresh] = {
            "sharpe": sr,
            "n_trades": int(bt_f["total_trades"]),
            "wr": float(bt_f["wr"]),
            "total_pnl_pips": float(bt_f["total_pnl_pips"]),
        }
        if sr > best_sharpe:
            best_sharpe = sr
            best_threshold = thresh

    logger.info("Résultats sweep train :")
    for thresh in META_THRESHOLDS:
        r = sweep_results[thresh]
        marker = " ★" if thresh == best_threshold else ""
        logger.info(
            "  seuil=%.2f  sharpe=%+.4f  wr=%.1f%%  trades=%d%s",
            thresh, r["sharpe"], r["wr"] * 100, r["n_trades"], marker,
        )
    logger.info(
        "Meilleur seuil TRAIN : %.2f (Sharpe=%.4f)", best_threshold, best_sharpe,
    )

    # ── Évaluation VAL ───────────────────────────────────────────────────
    logger.info("── Évaluation VAL 2023 (seuil=%.2f) ──", best_threshold)

    X_val_pred = X_val_all.fillna(0.0)
    proba_val_all = rf.predict_proba(X_val_pred)
    proba_win_val = proba_val_all[:, class1_pos]
    val_sig_idx = np.where(signals_val.values != 0)[0]
    proba_win_val_sig = proba_win_val[val_sig_idx]

    signals_val_filt = _filter_signals_by_proba(
        signals_val, proba_win_val_sig, best_threshold, val_sig_idx,
    )
    n_val_after = int((signals_val_filt != 0).sum())
    logger.info("VAL après méta-filtre : %d → %d signaux", n_sig["val"], n_val_after)

    trades_val_meta, _ = _simulate_adaptive_trades(df_val, signals_val_filt, atr_val)
    bt_val_meta = _compute_bt_metrics(trades_val_meta)
    logger.info(
        "VAL méta : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts",
        bt_val_meta["sharpe"], bt_val_meta["wr"] * 100,
        bt_val_meta["total_trades"], bt_val_meta["total_pnl_pips"],
    )

    # ── Évaluation TEST ──────────────────────────────────────────────────
    logger.info("── Évaluation TEST ≥ 2024 (seuil=%.2f) ──", best_threshold)

    X_test_pred = X_test_all.fillna(0.0)
    proba_test_all = rf.predict_proba(X_test_pred)
    proba_win_test = proba_test_all[:, class1_pos]
    test_sig_idx = np.where(signals_test.values != 0)[0]
    proba_win_test_sig = proba_win_test[test_sig_idx]

    signals_test_filt = _filter_signals_by_proba(
        signals_test, proba_win_test_sig, best_threshold, test_sig_idx,
    )
    n_test_after = int((signals_test_filt != 0).sum())
    logger.info(
        "TEST après méta-filtre : %d → %d signaux", n_sig["test"], n_test_after,
    )

    trades_test_meta, _ = _simulate_adaptive_trades(df_test, signals_test_filt, atr_test)
    bt_test_meta = _compute_bt_metrics(trades_test_meta)

    sr_test = float(bt_test_meta["sharpe"])
    wr_test = float(bt_test_meta["wr"])
    n_trades_test = int(bt_test_meta["total_trades"])
    pnl_test = float(bt_test_meta["total_pnl_pips"])
    dd_test = float(bt_test_meta["max_drawdown_pips"])

    logger.info(
        "TEST méta : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts, dd=%.1f",
        sr_test, wr_test * 100, n_trades_test, pnl_test, dd_test,
    )

    # ── Assemblage résultat ──────────────────────────────────────────────
    return {
        "config_name": config_name,
        "donchian_N": donchian_N,
        "donchian_M": donchian_M,
        "abandoned": False,
        "rf_classes": rf.classes_.tolist(),
        "feature_importance": {name: round(imp, 6) for name, imp in importances},
        "best_threshold_train": best_threshold,
        "sweep_results": {str(k): v for k, v in sweep_results.items()},
        "train": {
            "n_signaux": n_sig["train"],
            "n_meta_labels": n_meta["train"],
            "n_wins_meta": n_wins_train,
            "n_losses_meta": n_losses_train,
            "base": {
                "sharpe": float(bt_train_base["sharpe"]),
                "wr": float(bt_train_base["wr"]),
                "n_trades": int(bt_train_base["total_trades"]),
                "total_pnl_pips": float(bt_train_base["total_pnl_pips"]),
                "max_dd_pips": float(bt_train_base["max_drawdown_pips"]),
            },
        },
        "val": {
            "n_signaux": n_sig["val"],
            "n_signaux_after_meta": n_val_after,
            "base": {
                "sharpe": float(bt_val_base["sharpe"]),
                "wr": float(bt_val_base["wr"]),
                "n_trades": int(bt_val_base["total_trades"]),
                "total_pnl_pips": float(bt_val_base["total_pnl_pips"]),
                "max_dd_pips": float(bt_val_base["max_drawdown_pips"]),
            },
            "meta": {
                "sharpe": float(bt_val_meta["sharpe"]),
                "wr": float(bt_val_meta["wr"]),
                "n_trades": int(bt_val_meta["total_trades"]),
                "total_pnl_pips": float(bt_val_meta["total_pnl_pips"]),
                "max_dd_pips": float(bt_val_meta["max_drawdown_pips"]),
            },
        },
        "test": {
            "n_signaux": n_sig["test"],
            "n_signaux_after_meta": n_test_after,
            "base": {
                "sharpe": float(bt_test_base["sharpe"]),
                "wr": float(bt_test_base["wr"]),
                "n_trades": int(bt_test_base["total_trades"]),
                "total_pnl_pips": float(bt_test_base["total_pnl_pips"]),
                "max_dd_pips": float(bt_test_base["max_drawdown_pips"]),
            },
            "meta": {
                "sharpe": sr_test,
                "wr": wr_test,
                "n_trades": n_trades_test,
                "total_pnl_pips": pnl_test,
                "max_dd_pips": dd_test,
            },
        },
        # Pour validate_edge (utilise les trades du test méta)
        "_test_meta_trades": trades_test_meta,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrateur principal
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """H2 : Méta-labeling RF sur US30 D1 (2 configs Donchian)."""
    set_global_seeds()
    check_unlocked()

    logger.info("=" * 60)
    logger.info("H2 — Méta-labeling RF sur US30 D1 (Donchian)")
    logger.info("Coûts v3 US30 : %.1f points/trade (spread 3.0 + slippage 5.0)", COST_TOTAL_POINTS)
    logger.info(
        "TP/SL adaptatifs : %.0f×/%.0f× ATR(%d), timeout %d barres D1",
        TP_ATR_MULT, SL_ATR_MULT, ATR_PERIOD, TIMEOUT_BARS,
    )
    logger.info("RF : %s", RF_PARAMS)
    logger.info("Configs : A (N=20, M=20), B (N=100, M=10)")
    logger.info("n_trials cumulé : %d", N_TRIALS)

    # ── 1. Chargement données ─────────────────────────────────────────────
    logger.info("Chargement US30 D1 depuis data/raw/US30/...")
    try:
        df = load_asset("US30", "D1")
    except Exception as e:
        logger.warning("load_asset a échoué (%s), fallback chargement direct...", e)
        path = Path("data/raw/US30/USA30IDXUSD_D1.csv")
        # Détection nb colonnes
        import csv as _csv
        with open(path, encoding="utf-8-sig") as f:
            reader = _csv.reader(f, delimiter="\t")
            header_raw = next(reader)
            first_row = next(reader)
        n_headers = len(header_raw)
        n_data = len(first_row)
        if n_data > n_headers:
            col_names = (
                ["Open", "High", "Low", "Close", "Volume", "Spread"]
                if n_data == 7 and n_headers == 6
                else [f"Col_{i}" for i in range(n_data - 1)]
            )
            df = pd.read_csv(path, sep="\t", index_col=0, names=col_names, skiprows=1)
        else:
            df = pd.read_csv(path, sep="\t")
            if "Time" in df.columns:
                df["Time"] = pd.to_datetime(df["Time"], utc=True)
                df = df.set_index("Time")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        # Normalisation colonnes
        rename = {}
        for col in df.columns:
            low = col.strip().lower()
            if low == "open":
                rename[col] = "Open"
            elif low == "high":
                rename[col] = "High"
            elif low == "low":
                rename[col] = "Low"
            elif low == "close":
                rename[col] = "Close"
            elif low == "volume":
                rename[col] = "Volume"
        df = df.rename(columns=rename)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        required = ["Open", "High", "Low", "Close"]
        for c in required:
            if c not in df.columns:
                logger.error("Colonne %s manquante après normalisation. Colonnes: %s", c, list(df.columns))
                return

    logger.info("US30 D1 : %d barres, %s → %s", len(df), df.index[0], df.index[-1])
    logger.info("Colonnes : %s", sorted(df.columns))

    # ── 2. Exécution Config A ─────────────────────────────────────────────
    logger.info("")
    logger.info("▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    logger.info("CONFIG A : Donchian(N=20, M=20) — paramètres v2 H05")
    logger.info("▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    result_a = run_config(df, "A", CONFIGS["A"]["donchian_N"], CONFIGS["A"]["donchian_M"])

    # ── 3. Exécution Config B ─────────────────────────────────────────────
    logger.info("")
    logger.info("▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")
    logger.info("CONFIG B : Donchian(N=100, M=10) — paramètres v3 H06")
    logger.info("▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    result_b = run_config(df, "B", CONFIGS["B"]["donchian_N"], CONFIGS["B"]["donchian_M"])

    # ── 4. Sélection de la meilleure config pour validate_edge ────────────
    # Critère : Sharpe test méta le plus élevé parmi les configs non-abandonnées
    best_result = None
    best_sharpe_test = -np.inf
    for res in [result_a, result_b]:
        if res is None:
            continue
        if res.get("abandoned", False):
            continue
        sr = res["test"]["meta"]["sharpe"]
        if sr > best_sharpe_test:
            best_sharpe_test = sr
            best_result = res

    if best_result is not None:
        test_trades = best_result.get("_test_meta_trades", [])
    else:
        test_trades = []

    # ── 5. read_oos + validate_edge ──────────────────────────────────────
    sr_for_edge = best_result["test"]["meta"]["sharpe"] if best_result else 0.0
    n_for_edge = best_result["test"]["meta"]["n_trades"] if best_result else 0
    read_oos(prompt="12", hypothesis="H2", sharpe=sr_for_edge, n_trades=n_for_edge)

    equity_test, trades_df_test = _build_equity_from_trades(test_trades)
    if not trades_df_test.empty and len(equity_test) >= 2:
        edge_report = validate_edge(equity_test, trades_df_test, N_TRIALS)
    else:
        edge_report = EdgeReport(
            go=False,
            reasons=["Aucun trade sur test (toutes configs)"],
            metrics={"sharpe": 0.0, "wr": 0.0, "n_trades": 0.0, "trades_per_year": 0.0},
        )

    logger.info("Edge : go=%s, reasons=%s", edge_report.go, edge_report.reasons)
    logger.info("Edge metrics : %s", edge_report.metrics)

    # ── 6. Rapport comparatif console ─────────────────────────────────────
    n_years_test = max((df[df.index >= TEST_START].index[-1] - df[df.index >= TEST_START].index[0]).days / 365.25, 0.01)
    df_test_period = df[df.index >= TEST_START]

    print("\n" + "=" * 80)
    print("  H2 — Méta-labeling RF sur US30 D1 (Donchian, 2 configs)")
    print("=" * 80)
    print(f"  Split         : train ≤ 2022-12-31 | val = 2023 | test ≥ 2024-01-01")
    print(f"  Coûts         : {COST_TOTAL_POINTS:.0f} points/trade (spread 3.0 + slippage 5.0)")
    print(f"  TP/SL         : {TP_ATR_MULT:.0f}x/{SL_ATR_MULT:.0f}x ATR({ATR_PERIOD}), timeout {TIMEOUT_BARS} barres D1")
    print(f"  RF            : {RF_PARAMS['n_estimators']} arbres, max_depth={RF_PARAMS['max_depth']}, "
          f"class_weight={RF_PARAMS['class_weight']}")
    print(f"  Seuils sweep  : {META_THRESHOLDS}")
    print(f"  Barres total   : {len(df):,}")
    print("-" * 80)
    print(f"  {'Période':<20} {'Sharpe':>8} {'WR':>8} {'Trades':>7} {'PnL (pts)':>12} {'Max DD':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*12} {'-'*8}")

    for label, res in [("Config A (N=20,M=20)", result_a), ("Config B (N=100,M=10)", result_b)]:
        if res is None:
            print(f"  {label:<20} {'ABANDON':>8} (0 signal)")
            continue
        if res.get("abandoned"):
            print(f"  {label:<20} {'ABANDON':>8} ({res.get('reason', '?')})")
            continue
        print(f"\n  ▸ {label}")
        print(f"  {'  Train (base)':<20} {res['train']['base']['sharpe']:>8.4f} "
              f"{res['train']['base']['wr']*100:>7.1f}% {res['train']['base']['n_trades']:>7} "
              f"{res['train']['base']['total_pnl_pips']:>+11.1f} {res['train']['base']['max_dd_pips']:>8.1f}")
        print(f"  {'  Val (base)':<20} {res['val']['base']['sharpe']:>8.4f} "
              f"{res['val']['base']['wr']*100:>7.1f}% {res['val']['base']['n_trades']:>7} "
              f"{res['val']['base']['total_pnl_pips']:>+11.1f} {res['val']['base']['max_dd_pips']:>8.1f}")
        print(f"  {'  Val (méta)':<20} {res['val']['meta']['sharpe']:>8.4f} "
              f"{res['val']['meta']['wr']*100:>7.1f}% {res['val']['meta']['n_trades']:>7} "
              f"{res['val']['meta']['total_pnl_pips']:>+11.1f} {res['val']['meta']['max_dd_pips']:>8.1f}")
        print(f"  {'  Test (base)':<20} {res['test']['base']['sharpe']:>8.4f} "
              f"{res['test']['base']['wr']*100:>7.1f}% {res['test']['base']['n_trades']:>7} "
              f"{res['test']['base']['total_pnl_pips']:>+11.1f} {res['test']['base']['max_dd_pips']:>8.1f}")
        print(f"  {'  Test (méta) ★':<20} {res['test']['meta']['sharpe']:>8.4f} "
              f"{res['test']['meta']['wr']*100:>7.1f}% {res['test']['meta']['n_trades']:>7} "
              f"{res['test']['meta']['total_pnl_pips']:>+11.1f} {res['test']['meta']['max_dd_pips']:>8.1f}")
        print(f"  {'  Seuil optimal':<20} {res['best_threshold_train']:.2f}")

    print("-" * 80)

    # Métriques edge sur la meilleure config
    if best_result is not None:
        sr_final = best_result["test"]["meta"]["sharpe"]
        wr_final = best_result["test"]["meta"]["wr"]
        n_final = best_result["test"]["meta"]["n_trades"]
        tpa_final = n_final / n_years_test if n_years_test > 0 else 0.0
        dd_final = best_result["test"]["meta"]["max_dd_pips"]
        best_cfg_name = best_result["config_name"]
    else:
        sr_final = 0.0
        wr_final = 0.0
        n_final = 0
        tpa_final = 0.0
        dd_final = 0.0
        best_cfg_name = "N/A"

    print(f"  Meilleure config   : {best_cfg_name}")
    print(f"  Sharpe test (méta) : {sr_final:.4f}")
    print(f"  WR test (méta)     : {wr_final*100:.1f}%")
    print(f"  Trades test (méta) : {n_final}")
    print(f"  Trades/an test     : {tpa_final:.1f}")
    print(f"  Max DD test (méta) : {dd_final:.1f} pts")
    dsr_val = edge_report.metrics.get("dsr", float("nan"))
    p_val = edge_report.metrics.get("p_value", float("nan"))
    print(f"  DSR                : {dsr_val:.2f} (p={p_val:.3f})"
          if not np.isnan(dsr_val) else "  DSR                : NaN")
    print(f"  n_trials cumul     : {N_TRIALS}")
    print(f"  GO                 : {edge_report.go}")
    if edge_report.reasons:
        for r in edge_report.reasons:
            print(f"    → {r}")
    print("=" * 80)

    # ── 7. GO/NO-GO ──────────────────────────────────────────────────────
    sharpe_go = sr_final >= 1.0
    wr_go = wr_final >= 0.35
    tpa_go = tpa_final >= 30
    dsr_ok = not np.isnan(dsr_val) and dsr_val > 0

    go_final = sharpe_go and wr_go and tpa_go and dsr_ok
    if go_final:
        logger.info("★★★ GO — Tous les critères H2 satisfaits ★★★")
    else:
        logger.info("✗ NO-GO — Critères non satisfaits :")
        if not sharpe_go:
            logger.info("  → Sharpe test %.4f < 1.0", sr_final)
        if not wr_go:
            logger.info("  → WR test %.1f%% < 35%%", wr_final * 100)
        if not tpa_go:
            logger.info("  → Trades/an %.1f < 30", tpa_final)
        if not dsr_ok:
            logger.info("  → DSR non significatif (DSR=%.2f, p=%.3f)", dsr_val, p_val)

    # ── 8. Sauvegarde JSON ───────────────────────────────────────────────
    def _clean_result(res: dict[str, Any] | None) -> dict[str, Any] | None:
        if res is None:
            return None
        # Retirer les données non-sérialisables
        clean = {k: v for k, v in res.items() if k != "_test_meta_trades"}
        return clean

    summary: dict[str, Any] = {
        "hypothesis": "H2",
        "prompt": "12",
        "title": "Méta-labeling RF sur US30 D1 (Donchian, 2 configs)",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_trials_cumul": N_TRIALS,
        "split": {
            "train": "≤ 2022-12-31",
            "val": "2023-01-01 → 2023-12-31",
            "test": "≥ 2024-01-01",
        },
        "config": {
            "tp_atr_mult": TP_ATR_MULT,
            "sl_atr_mult": SL_ATR_MULT,
            "atr_period": ATR_PERIOD,
            "timeout_bars": TIMEOUT_BARS,
            "cost_total_points": COST_TOTAL_POINTS,
            "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
            "meta_thresholds_swept": META_THRESHOLDS,
        },
        "features": [
            "dist_breakout_pct", "force_breakout_pct", "atr_ratio", "rsi_14",
            "retracement_pct", "days_since_signal", "vol_ratio", "day_of_week",
        ],
        "results": {
            "A": _clean_result(result_a),
            "B": _clean_result(result_b),
        },
        "best_config": best_cfg_name,
        "edge_report": {
            "go": edge_report.go,
            "reasons": edge_report.reasons,
            "metrics": edge_report.metrics,
        },
        "go_nogo": {
            "sharpe_test": sr_final,
            "sharpe_threshold": 1.0,
            "sharpe_go": sharpe_go,
            "wr_test": wr_final,
            "wr_threshold": 0.35,
            "wr_go": wr_go,
            "trades_per_year": round(tpa_final, 2),
            "trades_per_year_threshold": 30,
            "trades_per_year_go": tpa_go,
            "dsr_go": dsr_ok,
            "go_final": go_final,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Résultats sauvegardés : %s", OUTPUT_PATH)
    logger.info("Log : %s", LOG_PATH)
    logger.info("Terminé — %s", "GO" if go_final else "NO-GO")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setStream(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w"),
            stream_handler,
        ],
    )
    main()
