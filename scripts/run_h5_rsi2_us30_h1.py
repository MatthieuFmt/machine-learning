"""Run H5 — Mean-reversion RSI(2) extrême sur US30 H1.

Stratégie :
- RSI(2) < 10 → LONG (survente extrême)
- RSI(2) > 90 → SHORT (surachat extrême)
- Filtre ATR(14) > 2× médiane ATR(14) 20j → NO TRADE
- Sortie : RSI(2) retraverse 50 OU SL = 1.5× ATR(14) entrée
- Pas de TP

Règles critiques :
- Split temporel figé : train ≤ 2022-12-31 23:00, val = 2023, test ≥ 2024-01-01
- Coûts v3 US30 : spread 3.0 + slippage 5.0 = 8.0 points total
- Capital 10k€, risque 2%/trade
- validate_edge() avec n_trials=24
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

from app.analysis.edge_validation import EdgeReport, validate_edge
from app.core.seeds import set_global_seeds
from app.features.indicators import atr, rsi
from app.testing.snooping_guard import check_unlocked, read_oos

logger = logging.getLogger("h5_rsi2_us30_h1")

# ═══════════════════════════════════════════════════════════════════════════════
# Constantes figées ex ante
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")

RSI_PERIOD: int = 2
RSI_OVERSOLD: int = 10
RSI_OVERBOUGHT: int = 90
RSI_EXIT: int = 50
ATR_PERIOD: int = 14
ATR_FILTER_MULT: float = 2.0
ATR_MEDIAN_DAYS: int = 20  # fenêtre glissante pour médiane ATR
SL_ATR_MULT: float = 1.5

COST_POINTS: float = 8.0  # spread 3 + slippage 5
CAPITAL: float = 10_000.0

N_TRIALS: int = 24  # 23 précédents + H5

OUTPUT_PATH = Path("predictions/h5_rsi2_us30_h1.json")
LOG_PATH = Path("logs/h5_rsi2_us30_h1.log")

# ═══════════════════════════════════════════════════════════════════════════════
# Chargement données (direct pandas — le loader échoue sur gaps H1)
# ═══════════════════════════════════════════════════════════════════════════════


def load_us30_h1() -> pd.DataFrame:
    """Charge US30 H1 directement, sans la validation stricte des gaps.

    Le CSV a 7 colonnes de donnees pour 6 headers (timestamps servent d'index implicite).
    On utilise index_col=0 + noms explicites.
    """
    path = Path("data/raw/US30/USA30IDXUSD_H1.csv")
    df = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
        names=["Open", "High", "Low", "Close", "Volume", "Spread"],
        skiprows=1,
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    # Volume en float
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).astype(np.float64)
    # Garder uniquement les colonnes utiles
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ═══════════════════════════════════════════════════════════════════════════════
# Indicateurs
# ═══════════════════════════════════════════════════════════════════════════════


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule RSI(2), ATR(14), médiane ATR 20j sur le DataFrame.

    Returns:
        DataFrame avec colonnes ajoutées : rsi_2, atr_14, atr_median_20d, atr_filter.
        Même index que df.
    """
    result = df.copy()

    # RSI(2) — Wilder smoothing standard (alpha = 1/period)
    result["rsi_2"] = rsi(df["Close"], period=RSI_PERIOD)

    # ATR(14)
    result["atr_14"] = atr(df["High"], df["Low"], df["Close"], period=ATR_PERIOD)

    # Médiane ATR sur ~20 jours (H1 → 480 barres)
    median_window = ATR_MEDIAN_DAYS * 24  # 480 barres H1
    result["atr_median_20d"] = (
        result["atr_14"].rolling(window=median_window, min_periods=median_window // 2).median()
    )

    # Filtre ATR : True si volatilité excessive
    result["atr_filter"] = result["atr_14"] > (ATR_FILTER_MULT * result["atr_median_20d"])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Génération des signaux
# ═══════════════════════════════════════════════════════════════════════════════


def generate_signals(df: pd.DataFrame) -> pd.Series:
    """Génère les signaux RSI(2) extrême avec filtre ATR.

    Returns:
        pd.Series : 1=LONG, -1=SHORT, 0=FLAT. Même index que df.
    """
    signals = pd.Series(0, index=df.index, dtype=int)

    rsi_vals = df["rsi_2"].values
    atr_filter = df["atr_filter"].values

    long_condition = (rsi_vals < RSI_OVERSOLD) & (~atr_filter)
    short_condition = (rsi_vals > RSI_OVERBOUGHT) & (~atr_filter)

    signals[long_condition] = 1
    signals[short_condition] = -1

    return signals


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest stateful RSI(2) — moteur custom
# ═══════════════════════════════════════════════════════════════════════════════


def run_rsi2_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
) -> dict[str, Any]:
    """Backtest stateful bar-by-bar pour stratégie RSI(2) mean-reversion.

    Règles :
    - 1 trade à la fois (stateful).
    - Entrée au Close de la barre où signal ≠ 0.
    - Sortie au premier de :
        a) RSI(2) retraverse 50 (LONG: RSI passe > 50, SHORT: RSI passe < 50)
        b) SL = 1.5 × ATR(14) touché (sur High/Low intra-barre)
    - Coûts déduits : COST_POINTS points par trade (entrée + sortie).
    - Pas de TP, pas de timeout.

    Args:
        df: DataFrame avec colonnes Open, High, Low, Close, rsi_2, atr_14.
        signals: pd.Series 1=LONG, -1=SHORT, 0=FLAT.

    Returns:
        dict avec sharpe, wr, total_trades, total_pnl_points, max_dd_points,
        trades, equity, trades_df.
    """
    # Alignement
    common_idx = df.index.intersection(signals.index)
    df_aligned = df.loc[common_idx]
    sig_aligned = signals.loc[common_idx]

    n = len(df_aligned)
    if n == 0:
        return _empty_bt_result()

    closes = df_aligned["Close"].values.astype(np.float64)
    highs = df_aligned["High"].values.astype(np.float64)
    lows = df_aligned["Low"].values.astype(np.float64)
    rsi_vals = df_aligned["rsi_2"].values.astype(np.float64)
    atr_vals = df_aligned["atr_14"].values.astype(np.float64)
    sig_arr = sig_aligned.values.astype(np.int8)
    times = df_aligned.index

    trades: list[dict[str, Any]] = []
    in_position: bool = False
    position_type: int = 0
    entry_idx: int = -1
    entry_price: float = 0.0
    sl_price: float = 0.0
    sl_atr_entry: float = 0.0

    i = 0
    while i < n:
        if not in_position:
            sig = int(sig_arr[i])
            if sig != 0 and not np.isnan(rsi_vals[i]) and not np.isnan(atr_vals[i]):
                if atr_vals[i] <= 0:
                    i += 1
                    continue
                in_position = True
                position_type = sig
                entry_idx = i
                entry_price = closes[i]
                sl_atr_entry = atr_vals[i]
                sl_dist = SL_ATR_MULT * sl_atr_entry
                if position_type == 1:  # LONG
                    sl_price = entry_price - sl_dist
                else:  # SHORT
                    sl_price = entry_price + sl_dist
            i += 1
            continue

        # ── En position : vérifier sortie ──────────────────────────────
        exit_now = False
        exit_type = ""
        exit_price = closes[i]

        prev_rsi = rsi_vals[i - 1] if i > 0 else np.nan
        curr_rsi = rsi_vals[i]

        if position_type == 1:  # LONG
            # RSI traverse 50 vers le haut
            if not np.isnan(curr_rsi) and not np.isnan(prev_rsi):
                if curr_rsi > RSI_EXIT and prev_rsi <= RSI_EXIT:
                    exit_now = True
                    exit_type = "rsi_cross"
                    exit_price = closes[i]
            # SL touché (intra-barre)
            if not exit_now and lows[i] <= sl_price:
                exit_now = True
                exit_type = "sl"
                exit_price = sl_price
        else:  # SHORT
            # RSI traverse 50 vers le bas
            if not np.isnan(curr_rsi) and not np.isnan(prev_rsi):
                if curr_rsi < RSI_EXIT and prev_rsi >= RSI_EXIT:
                    exit_now = True
                    exit_type = "rsi_cross"
                    exit_price = closes[i]
            # SL touché (intra-barre)
            if not exit_now and highs[i] >= sl_price:
                exit_now = True
                exit_type = "sl"
                exit_price = sl_price

        if exit_now:
            if position_type == 1:
                pnl_points = (exit_price - entry_price) - COST_POINTS
            else:
                pnl_points = (entry_price - exit_price) - COST_POINTS

            trades.append({
                "entry_time": str(times[entry_idx]),
                "exit_time": str(times[i]),
                "signal": position_type,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pips_net": float(pnl_points),
                "result": "win" if pnl_points > 0 else "loss",
                "exit_type": exit_type,
                "sl_atr_entry": float(sl_atr_entry),
            })

            in_position = False
            position_type = 0
            # Ne pas ré-entrer sur la même barre
            i += 1
            continue

        i += 1

    # ── Clôture forcée en fin de données ───────────────────────────────
    if in_position:
        exit_idx = n - 1
        if position_type == 1:
            pnl_points = (closes[exit_idx] - entry_price) - COST_POINTS
        else:
            pnl_points = (entry_price - closes[exit_idx]) - COST_POINTS
        trades.append({
            "entry_time": str(times[entry_idx]),
            "exit_time": str(times[exit_idx]),
            "signal": position_type,
            "entry_price": float(entry_price),
            "exit_price": float(closes[exit_idx]),
            "pips_net": float(pnl_points),
            "result": "win" if pnl_points > 0 else "loss",
            "exit_type": "end_of_data",
            "sl_atr_entry": float(sl_atr_entry),
        })

    return _compute_bt_metrics(trades)


def _compute_bt_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcule les métriques à partir de la liste de trades.

    Sharpe calcule via sharpe_daily_from_trades() — resample quotidien,
    forward fill, pct_change, annualisation sqrt(252). Methode canonique.
    """
    if not trades:
        return _empty_bt_result()

    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    total_trades = len(trades)
    wins = pnls[pnls > 0]

    wr = float(len(wins) / total_trades) if total_trades > 0 else 0.0
    total_pnl = float(pnls.sum())

    # Sharpe canonique : daily resample
    from app.backtest.metrics import sharpe_daily_from_trades
    sharpe = sharpe_daily_from_trades(trades)

    # Equity curve: capital + cumsum(pnl * pip_value_eur)
    pip_value = 0.92
    equity_vals = CAPITAL + np.cumsum(pnls * pip_value)
    exit_times = pd.to_datetime([t["exit_time"] for t in trades])
    equity = pd.Series(equity_vals, index=exit_times).sort_index()

    # Max drawdown en points (PNL brut)
    cumsum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumsum)
    dd = cumsum - peak
    max_dd_points = float(dd.min())

    # Max DD % sur equity monetaire
    equity_cummax = equity.cummax()
    dd_pct = (equity_cummax - equity) / equity_cummax.replace(0.0, np.nan)
    max_dd_pct = float(dd_pct.max()) if not np.isnan(dd_pct.max()) else 0.0

    # Profit factor
    wins_arr = pnls[pnls > 0]
    losses_arr = pnls[pnls < 0]
    if len(losses_arr) > 0:
        pf = float(wins_arr.sum() / abs(losses_arr.sum())) if wins_arr.sum() > 0 else 0.0
    elif len(wins_arr) > 0:
        pf = float("inf")
    else:
        pf = 0.0

    # Trades par type de sortie
    exit_types: dict[str, int] = {}
    for t in trades:
        et = t.get("exit_type", "unknown")
        exit_types[et] = exit_types.get(et, 0) + 1

    trades_df = pd.DataFrame({
        "pnl": pnls,
    }, index=pd.to_datetime([t["exit_time"] for t in trades]))

    return {
        "sharpe": sharpe,
        "wr": wr,
        "total_trades": total_trades,
        "total_pnl_points": total_pnl,
        "max_dd_points": max_dd_points,
        "max_dd_pct": max_dd_pct,
        "profit_factor": pf,
        "mean_pnl_per_trade": float(pnls.mean()),
        "exit_types": exit_types,
        "trades": trades,
        "equity": equity,
        "trades_df": trades_df,
    }


def _empty_bt_result() -> dict[str, Any]:
    """Résultat vide quand aucun trade n'est généré."""
    return {
        "sharpe": 0.0,
        "wr": 0.0,
        "total_trades": 0,
        "total_pnl_points": 0.0,
        "max_dd_points": 0.0,
        "max_dd_pct": 0.0,
        "profit_factor": 0.0,
        "mean_pnl_per_trade": 0.0,
        "exit_types": {},
        "trades": [],
        "equity": pd.Series(dtype=np.float64),
        "trades_df": pd.DataFrame({"pnl": []}),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _split_periods(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel strict H1."""
    train = df[df.index <= TRAIN_END]
    val = df[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test = df[df.index >= TEST_START]
    return train, val, test


def _build_equity_from_trades(
    trades: list[dict[str, Any]],
) -> tuple[pd.Series, pd.DataFrame]:
    """Construit equity curve monetaire (capital + cumsum PnL) et trades DataFrame.

    Equity en EUR pour que validate_edge() calcule un DD% correct.
    """
    if not trades:
        empty_idx = pd.DatetimeIndex([])
        return pd.Series(dtype=np.float64, index=empty_idx), pd.DataFrame({"pnl": []})
    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    exit_times = pd.to_datetime([t["exit_time"] for t in trades])
    pip_value = 0.92
    equity_vals = CAPITAL + np.cumsum(pnls * pip_value)
    equity = pd.Series(equity_vals, index=exit_times).sort_index()
    trades_df = pd.DataFrame({"pnl": pnls}, index=exit_times)
    return equity, trades_df


def _compute_donchian_correlation(
    rsi2_equity: pd.Series,
) -> float:
    """Calcule la corrélation rolling 60j avec Donchian D1 si dispo.

    Returns:
        Corrélation moyenne, ou NaN si Donchian indisponible.
    """
    donchian_path = Path("predictions/h06_donchian_multi_asset.json")
    if not donchian_path.exists():
        logger.info("Donchian D1 non trouve (%s) - correlation ignoree.", donchian_path)
        return float("nan")

    try:
        with open(donchian_path, encoding="utf-8") as f:
            donchian_data = json.load(f)
    except Exception:
        logger.warning("Lecture Donchian JSON echouee - correlation ignoree.")
        return float("nan")

    # Extraire trades Donchian US30
    results = donchian_data.get("results", {})
    us30_result = results.get("US30", {})
    donchian_trades = us30_result.get("test", {}).get("trades", [])
    if not donchian_trades:
        logger.info("Aucun trade Donchian US30 dans le JSON.")
        return float("nan")

    donchian_pnls = np.array([t["pips_net"] for t in donchian_trades], dtype=np.float64)
    donchian_times = pd.to_datetime([t["exit_time"] for t in donchian_trades])
    donchian_equity = pd.Series(np.cumsum(donchian_pnls), index=donchian_times).sort_index()

    # Rééchantillonner en daily pour aligner
    rsi2_daily = rsi2_equity.resample("D").last().ffill().pct_change().dropna()
    donchian_daily = donchian_equity.resample("D").last().ffill().pct_change().dropna()

    common_idx = rsi2_daily.index.intersection(donchian_daily.index)
    if len(common_idx) < 60:
        return float("nan")

    ra = rsi2_daily.loc[common_idx]
    rb = donchian_daily.loc[common_idx]
    rolling_corr = ra.rolling(window=60).corr(rb)
    return float(rolling_corr.dropna().mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrateur principal
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """H5 : Mean-reversion RSI(2) extrême sur US30 H1."""
    set_global_seeds()
    check_unlocked()

    logger.info("=== H5 - Mean-reversion RSI(2) extreme sur US30 H1 ===")
    logger.info(
        "Parametres : RSI(%d) < %d LONG, RSI > %d SHORT, sortie RSI traverse %d",
        RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT, RSI_EXIT,
    )
    logger.info(
        "Filtre ATR : ATR(%d) > %.1fx mediane ATR %dj -> NO TRADE",
        ATR_PERIOD, ATR_FILTER_MULT, ATR_MEDIAN_DAYS,
    )
    logger.info("SL = %.1f x ATR(14) entree, pas de TP", SL_ATR_MULT)
    logger.info("Couts : %.1f points/trade (spread 3 + slippage 5)", COST_POINTS)
    logger.info("n_trials cumul : %d", N_TRIALS)

    # ── 1. Chargement données ─────────────────────────────────────────────
    logger.info("Chargement US30 H1 depuis data/raw/US30/USA30IDXUSD_H1.csv...")
    df = load_us30_h1()
    logger.info("US30 H1 : %d barres, %s -> %s", len(df), df.index[0], df.index[-1])
    logger.info("Colonnes : %s", sorted(df.columns))

    # ── 2. Calcul des indicateurs ─────────────────────────────────────────
    logger.info("Calcul RSI(%d) + ATR(%d) + mediane ATR...", RSI_PERIOD, ATR_PERIOD)
    df = compute_indicators(df)

    # ── 3. Split temporel strict ──────────────────────────────────────────
    df_train, df_val, df_test = _split_periods(df)
    logger.info(
        "Split : train=%d (<=%s), val=%d (%s->%s), test=%d (>=%s)",
        len(df_train), TRAIN_END,
        len(df_val), VAL_START.date(), VAL_END.date(),
        len(df_test), TEST_START.date(),
    )

    # ── 4. Génération signaux par période ─────────────────────────────────
    signals_train = generate_signals(df_train)
    signals_val = generate_signals(df_val)
    signals_test = generate_signals(df_test)

    n_sig = {
        "train": int((signals_train != 0).sum()),
        "val": int((signals_val != 0).sum()),
        "test": int((signals_test != 0).sum()),
    }
    n_long = {
        "train": int((signals_train == 1).sum()),
        "val": int((signals_val == 1).sum()),
        "test": int((signals_test == 1).sum()),
    }
    n_short = {
        "train": int((signals_train == -1).sum()),
        "val": int((signals_val == -1).sum()),
        "test": int((signals_test == -1).sum()),
    }
    logger.info(
        "Signaux : train=%d (LONG=%d, SHORT=%d), val=%d (L=%d, S=%d), test=%d (L=%d, S=%d)",
        n_sig["train"], n_long["train"], n_short["train"],
        n_sig["val"], n_long["val"], n_short["val"],
        n_sig["test"], n_long["test"], n_short["test"],
    )

    # ── 5. Backtest train ─────────────────────────────────────────────────
    logger.info("-- Backtest TRAIN (<= 2022-12-31) --")
    bt_train = run_rsi2_backtest(df_train, signals_train)
    logger.info(
        "TRAIN : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts, dd=%.1f pts",
        bt_train["sharpe"], bt_train["wr"] * 100,
        bt_train["total_trades"], bt_train["total_pnl_points"],
        bt_train["max_dd_points"],
    )
    logger.info("Types de sortie train : %s", bt_train.get("exit_types", {}))

    # ── 6. Backtest val ───────────────────────────────────────────────────
    logger.info("-- Backtest VAL (2023) --")
    bt_val = run_rsi2_backtest(df_val, signals_val)
    logger.info(
        "VAL : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts, dd=%.1f pts",
        bt_val["sharpe"], bt_val["wr"] * 100,
        bt_val["total_trades"], bt_val["total_pnl_points"],
        bt_val["max_dd_points"],
    )
    logger.info("Types de sortie val : %s", bt_val.get("exit_types", {}))

    # ── 7. Backtest test (UNE SEULE FOIS) ─────────────────────────────────
    logger.info("-- Backtest TEST (>= 2024) --")
    bt_test = run_rsi2_backtest(df_test, signals_test)

    sr_test = float(bt_test["sharpe"])
    wr_test = float(bt_test["wr"])
    n_trades_test = bt_test["total_trades"]
    pnl_test = float(bt_test["total_pnl_points"])
    dd_test = float(bt_test["max_dd_points"])
    mean_pnl = float(bt_test["mean_pnl_per_trade"])
    exit_types_test = bt_test.get("exit_types", {})

    logger.info(
        "TEST : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pts, dd=%.1f pts, mean=%.1f pts",
        sr_test, wr_test * 100, n_trades_test, pnl_test, dd_test, mean_pnl,
    )
    logger.info("Types de sortie test : %s", exit_types_test)

    # ── 8. read_oos + validate_edge ───────────────────────────────────────
    read_oos(prompt="12", hypothesis="H5", sharpe=sr_test, n_trades=n_trades_test)

    equity_test, trades_df_test = _build_equity_from_trades(bt_test["trades"])
    if not trades_df_test.empty and len(equity_test) >= 2:
        edge_report = validate_edge(equity_test, trades_df_test, N_TRIALS)
    else:
        edge_report = EdgeReport(
            go=False,
            reasons=["Aucun trade sur test"],
            metrics={"sharpe": 0.0, "wr": 0.0, "n_trades": 0.0, "trades_per_year": 0.0},
        )

    logger.info("Edge : go=%s, reasons=%s", edge_report.go, edge_report.reasons)
    logger.info("Edge metrics : %s", edge_report.metrics)

    # ── 9. Corrélation Donchian D1 ────────────────────────────────────────
    corr_donchian = _compute_donchian_correlation(equity_test)
    if not np.isnan(corr_donchian):
        logger.info("Correlation 60j vs Donchian D1 : %.4f", corr_donchian)

    # ── 10. Rapport console ───────────────────────────────────────────────
    n_years_test = max((df_test.index[-1] - df_test.index[0]).days / 365.25, 0.01)
    trades_per_year = n_trades_test / n_years_test

    n_years_train = max((df_train.index[-1] - df_train.index[0]).days / 365.25, 0.01)
    tpa_train = bt_train["total_trades"] / n_years_train

    n_years_val = max((df_val.index[-1] - df_val.index[0]).days / 365.25, 0.01)
    tpa_val = bt_val["total_trades"] / n_years_val

    print("\n" + "=" * 72)
    print("  H5 - Mean-reversion RSI(2) extreme sur US30 H1")
    print("=" * 72)
    print(f"  Split         : train <= 2022-12-31 | val = 2023 | test >= 2024-01-01")
    print(f"  RSI(2)        : LONG < 10, SHORT > 90, exit traverse 50")
    print(f"  Filtre ATR    : ATR(14) > 2x mediane ATR 20j -> NO TRADE")
    print(f"  SL            : {SL_ATR_MULT}x ATR(14) entree")
    print(f"  Couts         : {COST_POINTS:.0f} points/trade (spread 3 + slippage 5)")
    print(f"  Barres H1     : {len(df):,} total")
    print("-" * 72)
    print(f"  {'Periode':<16} {'Sharpe':>8} {'WR':>8} {'Trades':>7} {'PnL (pts)':>10} {'T/an':>7} {'Max DD':>8}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*7} {'-'*8}")
    print(f"  {'Train':<16} {bt_train['sharpe']:>8.4f} {bt_train['wr']*100:>7.1f}% "
          f"{bt_train['total_trades']:>7} {bt_train['total_pnl_points']:>+10.1f} "
          f"{tpa_train:>7.1f} {bt_train['max_dd_points']:>8.1f}")
    print(f"  {'Val':<16} {bt_val['sharpe']:>8.4f} {bt_val['wr']*100:>7.1f}% "
          f"{bt_val['total_trades']:>7} {bt_val['total_pnl_points']:>+10.1f} "
          f"{tpa_val:>7.1f} {bt_val['max_dd_points']:>8.1f}")
    print(f"  {'Test *':<16} {sr_test:>8.4f} {wr_test*100:>7.1f}% "
          f"{n_trades_test:>7} {pnl_test:>+10.1f} "
          f"{trades_per_year:>7.1f} {dd_test:>8.1f}")
    print("-" * 72)
    print(f"  Mean PnL/test : {mean_pnl:+.1f} pts")
    dsr_val = edge_report.metrics.get("dsr", float("nan"))
    p_val = edge_report.metrics.get("p_value", float("nan"))
    print(f"  DSR           : {dsr_val:.2f} (p={p_val:.3f})"
          if not np.isnan(dsr_val) else "  DSR           : NaN")
    if not np.isnan(corr_donchian):
        print(f"  r Donchian D1 : {corr_donchian:.4f}")
    print(f"  n_trials cumul: {N_TRIALS}")
    print(f"  GO            : {edge_report.go}")
    if edge_report.reasons:
        for r in edge_report.reasons:
            print(f"    -> {r}")
    print(f"  Sorties test  : {exit_types_test}")
    print("=" * 72)

    # ── 11. GO/NO-GO ──────────────────────────────────────────────────────
    sharpe_go = sr_test >= 1.0
    tpa_go = trades_per_year >= 100
    dd_go = abs(dd_test) / max(abs(pnl_test), 1.0) < 0.20 if dd_test < 0 else True

    go_final = sharpe_go and tpa_go and dd_go
    if go_final:
        logger.info("*** GO - Tous les criteres H5 satisfaits ***")
    else:
        logger.info("x NO-GO - Criteres non satisfaits :")
        if not sharpe_go:
            logger.info("  -> Sharpe test %.4f < 1.0", sr_test)
        if not tpa_go:
            logger.info("  -> Trades/an %.1f < 100", trades_per_year)
        if not dd_go:
            logger.info("  -> Max DD excessif")

    # ── 12. Sauvegarde JSON ───────────────────────────────────────────────
    summary: dict[str, Any] = {
        "hypothesis": "H5",
        "prompt": "12",
        "title": "Mean-reversion RSI(2) extrême sur US30 H1",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_trials_cumul": N_TRIALS,
        "split": {
            "train": f"≤ {TRAIN_END}",
            "val": f"{VAL_START} → {VAL_END}",
            "test": f"≥ {TEST_START}",
        },
        "config": {
            "rsi_period": RSI_PERIOD,
            "rsi_oversold": RSI_OVERSOLD,
            "rsi_overbought": RSI_OVERBOUGHT,
            "rsi_exit": RSI_EXIT,
            "atr_period": ATR_PERIOD,
            "atr_filter_mult": ATR_FILTER_MULT,
            "atr_median_days": ATR_MEDIAN_DAYS,
            "sl_atr_mult": SL_ATR_MULT,
            "cost_points": COST_POINTS,
            "capital": CAPITAL,
        },
        "data": {
            "n_barres_total": len(df),
            "n_barres_train": len(df_train),
            "n_barres_val": len(df_val),
            "n_barres_test": len(df_test),
            "signaux_train": n_sig["train"],
            "signaux_val": n_sig["val"],
            "signaux_test": n_sig["test"],
        },
        "train": {
            "sharpe": float(bt_train["sharpe"]),
            "wr": float(bt_train["wr"]),
            "n_trades": bt_train["total_trades"],
            "total_pnl_points": float(bt_train["total_pnl_points"]),
            "max_dd_points": float(bt_train["max_dd_points"]),
            "profit_factor": float(bt_train["profit_factor"]) if bt_train["profit_factor"] != float("inf") else None,
            "mean_pnl_per_trade": float(bt_train["mean_pnl_per_trade"]),
            "trades_per_year": round(tpa_train, 1),
            "exit_types": bt_train.get("exit_types", {}),
        },
        "val": {
            "sharpe": float(bt_val["sharpe"]),
            "wr": float(bt_val["wr"]),
            "n_trades": bt_val["total_trades"],
            "total_pnl_points": float(bt_val["total_pnl_points"]),
            "max_dd_points": float(bt_val["max_dd_points"]),
            "profit_factor": float(bt_val["profit_factor"]) if bt_val["profit_factor"] != float("inf") else None,
            "mean_pnl_per_trade": float(bt_val["mean_pnl_per_trade"]),
            "trades_per_year": round(tpa_val, 1),
            "exit_types": bt_val.get("exit_types", {}),
        },
        "test": {
            "sharpe": sr_test,
            "wr": wr_test,
            "n_trades": n_trades_test,
            "total_pnl_points": pnl_test,
            "max_dd_points": dd_test,
            "profit_factor": float(bt_test["profit_factor"]) if bt_test["profit_factor"] != float("inf") else None,
            "mean_pnl_per_trade": mean_pnl,
            "trades_per_year": round(trades_per_year, 1),
            "exit_types": exit_types_test,
        },
        "correlation_donchian_60d": None if np.isnan(corr_donchian) else round(corr_donchian, 4),
        "edge_report": {
            "go": edge_report.go,
            "reasons": edge_report.reasons,
            "metrics": edge_report.metrics,
        },
        "go_nogo": {
            "sharpe_test": sr_test,
            "sharpe_threshold": 1.0,
            "sharpe_go": sharpe_go,
            "trades_per_year": round(trades_per_year, 1),
            "trades_per_year_threshold": 100,
            "trades_per_year_go": tpa_go,
            "max_dd_go": dd_go,
            "go_final": go_final,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Resultats sauvegardes : %s", OUTPUT_PATH)
    logger.info("Log : %s", LOG_PATH)
    logger.info("Termine - %s", "GO" if go_final else "NO-GO")


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
