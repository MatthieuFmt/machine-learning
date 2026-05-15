"""Run H1 — Méta-labeling RF sur XAUUSD D1 (Donchian).

Hypothèse : Le méta-labeling RF appliqué au Donchian XAUUSD D1
peut faire passer le WR de 22.5% → ≥30% et atteindre Sharpe ≥ 1.0.

Règles critiques :
- Split temporel figé : train ≤ 2022-12-31, val = 2023, test ≥ 2024-01-01
- Coûts v3 XAUUSD : spread 25 pips + slippage 10 pips
- Sweep seuil méta sur TRAIN UNIQUEMENT : [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
- 8 features SANS contexte de marché
- RF : 200 arbres, max_depth=4, class_weight='balanced', random_state=42
- Direction LONG uniquement (XAUUSD biais long)
- validate_edge() avec n_trials=23
- read_oos() obligatoire avant analyse test
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ajouter la racine du projet au PYTHONPATH pour les imports `app.*`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.analysis.edge_validation import EdgeReport, validate_edge
from app.backtest.deterministic import run_deterministic_backtest
from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.strategies.donchian import DonchianBreakout
from app.testing.snooping_guard import check_unlocked, read_oos

logger = logging.getLogger("h1_xauusd_meta")

# ═══════════════════════════════════════════════════════════════════════════════
# Constantes figées ex ante
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_END = pd.Timestamp("2022-12-31", tz="UTC")
VAL_START = pd.Timestamp("2023-01-01", tz="UTC")
VAL_END = pd.Timestamp("2023-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")

DONCHIAN_N: int = 100
DONCHIAN_M: int = 20

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

META_THRESHOLDS: list[float] = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
N_TRIALS: int = 23  # 22 précédents + H1

XAUUSD_CONFIG: AssetConfig = ASSET_CONFIGS["XAUUSD"]

OUTPUT_PATH = Path("predictions/h1_xauusd_meta.json")
LOG_PATH = Path("logs/h1_xauusd_meta.log")


# ═══════════════════════════════════════════════════════════════════════════════
# Indicateurs techniques vectorisés
# ═══════════════════════════════════════════════════════════════════════════════

def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR vectorisé — Wilder's smoothing, zéro boucle sauf warmup."""
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


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI vectorisé — Wilder's smoothing."""
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
# Construction des 8 features du méta-modèle (SANS contexte de marché)
# ═══════════════════════════════════════════════════════════════════════════════

def build_meta_features(
    df: pd.DataFrame,
    donchian_signals: pd.Series,
) -> pd.DataFrame:
    """Construit les 8 features du méta-modèle pour TOUTES les barres.

    Features (toutes sans look-ahead, info dispo à la barre t) :
    1. dist_breakout_pct  : Distance Close au canal Donchian (% ATR)
    2. force_breakout_pct : (Close − Open) / Open × 100
    3. atr_ratio          : ATR(14) / ATR médian(20)
    4. rsi_14             : RSI(14) au breakout
    5. retracement_pct    : Retracement depuis le dernier high N (%)
    6. days_since_signal  : Jours depuis le dernier signal Donchian
    7. vol_ratio          : Volume / Volume médian(20), NaN si pas de Volume
    8. day_of_week        : Jour de la semaine (0=Lundi → 4=Vendredi)

    Args:
        df: DataFrame OHLC(V), index=DatetimeIndex trié.
        donchian_signals: pd.Series −1/0/1, même index que df.

    Returns:
        DataFrame de features, même index que df.
    """
    n = len(df)
    idx = df.index
    feats = pd.DataFrame(index=idx)

    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    open_ = df["Open"].values.astype(np.float64)

    # ── ATR(14) partagé ──
    atr14 = _atr(high, low, close, 14)

    # ── Feature 1 : Distance au breakout (% ATR) ──
    hh_n = df["High"].rolling(DONCHIAN_N).max().shift(1).values
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
        1.0,  # fallback neutre
    )

    # ── Feature 4 : RSI(14) ──
    feats["rsi_14"] = _rsi(close, 14)

    # ── Feature 5 : Retracement depuis le dernier high N ──
    rolling_high_n = df["High"].rolling(DONCHIAN_N).max().values
    feats["retracement_pct"] = np.where(
        rolling_high_n > 0,
        (close - rolling_high_n) / rolling_high_n * 100.0,
        0.0,
    )

    # ── Feature 6 : Jours depuis le dernier signal Donchian ──
    sig_arr = donchian_signals.values.astype(np.int8)
    days_since = np.full(n, np.nan, dtype=np.float64)
    last_sig = -1
    for i in range(n):
        if sig_arr[i] != 0:
            if last_sig >= 0:
                days_since[i] = float(i - last_sig)
            last_sig = i
        elif last_sig >= 0:
            days_since[i] = float(i - last_sig)
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
        feats["vol_ratio"] = 1.0  # neutre si pas de volume

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
) -> tuple[pd.Series, pd.DataFrame]:
    """Construit equity curve et trades DataFrame pour validate_edge()."""
    if not trades:
        empty_idx = pd.DatetimeIndex([])
        return pd.Series(dtype=np.float64, index=empty_idx), pd.DataFrame({"pnl": []})
    pnls = np.array([t["pips_net"] for t in trades], dtype=np.float64)
    exit_times = pd.to_datetime([t["exit_time"] for t in trades])
    equity = pd.Series(np.cumsum(pnls), index=exit_times).sort_index()
    trades_df = pd.DataFrame({"pnl": pnls}, index=exit_times)
    return equity, trades_df


def _run_backtest(
    df_period: pd.DataFrame,
    signals: pd.Series,
    config: AssetConfig,
) -> dict[str, Any]:
    """Wrapper stateful backtest avec coûts v3."""
    return run_deterministic_backtest(
        df=df_period,
        signals=signals,
        tp_pips=config.tp_points,
        sl_pips=config.sl_points,
        window_hours=config.window_hours,
        commission_pips=config.spread_pips,
        slippage_pips=config.slippage_pips,
        pip_size=config.pip_size,
    )


def _filter_signals_by_proba(
    signals: pd.Series,
    proba_win: np.ndarray,
    threshold: float,
    signal_idx: np.ndarray,
) -> pd.Series:
    """Filtre les signaux : ne garde que ceux où proba_win > threshold.

    Args:
        signals: Série complète des signaux (même index que df).
        proba_win: Array de P(win) aligné sur signal_idx.
        threshold: Seuil de probabilité.
        signal_idx: Indices (positions) des barres où signal ≠ 0.

    Returns:
        Copie de signals avec les signaux filtrés mis à 0.
    """
    filtered = signals.copy()
    for k, pos in enumerate(signal_idx):
        if proba_win[k] <= threshold:
            filtered.iloc[pos] = 0
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrateur principal
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """H1 : Méta-labeling RF sur XAUUSD D1 (Donchian)."""
    set_global_seeds()
    check_unlocked()

    cfg = XAUUSD_CONFIG
    logger.info("=== H1 — Méta-labeling RF XAUUSD D1 (Donchian N=%d, M=%d) ===",
                DONCHIAN_N, DONCHIAN_M)
    logger.info("Coûts v3 : spread=%.0f, slippage=%.0f, TP=%.0f, SL=%.0f, window=%dh",
                cfg.spread_pips, cfg.slippage_pips, cfg.tp_points, cfg.sl_points,
                cfg.window_hours)
    logger.info("RF : %s", RF_PARAMS)
    logger.info("n_trials cumulé : %d", N_TRIALS)

    # ── 1. Chargement données ─────────────────────────────────────────────
    logger.info("Chargement XAUUSD D1 depuis data/raw/...")
    df = load_asset("XAUUSD", "D1")
    logger.info("XAUUSD D1 : %d barres, %s -> %s", len(df), df.index[0], df.index[-1])
    logger.info("Colonnes : %s", sorted(df.columns))

    # ── 2. Split temporel strict ──────────────────────────────────────────
    df_train, df_val, df_test = _split_periods(df)
    logger.info("Split : train=%d (≤%s), val=%d (%s→%s), test=%d (≥%s)",
                len(df_train), TRAIN_END.date(),
                len(df_val), VAL_START.date(), VAL_END.date(),
                len(df_test), TEST_START.date())

    # ── 3. Signaux Donchian (LONG uniquement) ─────────────────────────────
    strat = DonchianBreakout(N=DONCHIAN_N, M=DONCHIAN_M)
    signals_train = strat.generate_signals(df_train)
    signals_val = strat.generate_signals(df_val)
    signals_test = strat.generate_signals(df_test)

    # Forcer LONG uniquement
    for sig in (signals_train, signals_val, signals_test):
        sig[sig == -1] = 0

    n_sig = {
        "train": int((signals_train != 0).sum()),
        "val": int((signals_val != 0).sum()),
        "test": int((signals_test != 0).sum()),
    }
    logger.info("Signaux LONG : train=%d, val=%d, test=%d",
                n_sig["train"], n_sig["val"], n_sig["test"])

    if n_sig["train"] == 0:
        logger.error("Aucun signal LONG sur train -- abandon.")
        return

    # ── 4. Méta-labels via backtest stateful (PnL réel, pas TP-hit only) ──
    # compute_meta_labels ne marque win=1 que si TP explicitement touché.
    # Avec XAUUSD TP=600 pips (40% du prix), le TP n'est quasi jamais atteint
    # en 5 barres D1 → 0 win → RF mono-classe.
    # Solution : utiliser le PnL réel du backtest stateful (inclut timeout wins).
    logger.info("Génération méta-labels via backtest stateful (basé PnL réel)...")

    def _generate_meta_labels_from_bt(
        df_p: pd.DataFrame, sig_p: pd.Series,
    ) -> pd.Series:
        """Génère des méta-labels 1/0 basés sur le PnL réel du backtest."""
        meta_series = pd.Series(np.nan, index=df_p.index, dtype="float64")
        sig_idx = np.where(sig_p.values != 0)[0]
        if len(sig_idx) == 0 or len(df_p) == 0:
            return meta_series

        bt = _run_backtest(df_p, sig_p, cfg)
        trades = bt.get("trades", [])

        # Mapping: entry_time -> pips_net
        entry_to_pnl: dict[str, float] = {}
        for t in trades:
            # Parse l'entry_time pour matcher l'index du signal
            entry_ts = pd.Timestamp(t["entry_time"])
            # Cherche la barre la plus proche dans df_p
            if entry_ts in df_p.index:
                entry_to_pnl[str(entry_ts)] = float(t["pips_net"])
            else:
                # Tolérance 1 jour
                closest = df_p.index[df_p.index.get_indexer([entry_ts], method="nearest")[0]]
                entry_to_pnl[str(closest)] = float(t["pips_net"])

        # Pour chaque barre avec signal, assigner le label
        for pos in sig_idx:
            ts_str = str(df_p.index[pos])
            if ts_str in entry_to_pnl:
                pnl = entry_to_pnl[ts_str]
                # Méta-label : 1 si PnL > 0 après coûts, 0 sinon
                meta_series.iloc[pos] = 1.0 if pnl > 0 else 0.0

        return meta_series

    meta_train = _generate_meta_labels_from_bt(df_train, signals_train)
    meta_val = _generate_meta_labels_from_bt(df_val, signals_val)
    meta_test_labels = _generate_meta_labels_from_bt(df_test, signals_test)

    n_meta = {
        "train": int(meta_train.notna().sum()),
        "val": int(meta_val.notna().sum()),
        "test": int(meta_test_labels.notna().sum()),
    }
    for period, ml in (("train", meta_train), ("val", meta_val), ("test", meta_test_labels)):
        n_w = int((ml == 1).sum()) if len(ml) > 0 else 0
        logger.info("Méta-labels %s : %d labels, %d wins (%.1f%%)",
                    period, n_meta[period], n_w,
                    n_w / max(n_meta[period], 1) * 100)

    # Vérification : il faut au moins 1 échantillon de chaque classe
    n_wins_train = int((meta_train == 1).sum())
    n_losses_train = int((meta_train == 0).sum())
    if n_wins_train == 0 or n_losses_train == 0:
        logger.error(
            "Méta-labels train mono-classe : %d wins, %d losses. "
            "Le RF ne peut pas apprendre. Abandon.",
            n_wins_train, n_losses_train,
        )
        return

    # ── 5. Construction des 8 features ────────────────────────────────────
    logger.info("Construction des 8 features du méta-modèle...")
    X_train_all = build_meta_features(df_train, signals_train)
    X_val_all = build_meta_features(df_val, signals_val)
    X_test_all = build_meta_features(df_test, signals_test)

    logger.info("Features shape : train=%s, val=%s, test=%s",
                X_train_all.shape, X_val_all.shape, X_test_all.shape)

    # ── 6. Entraînement RF sur train (échantillons méta-labellisés) ──────
    # On n'entraîne QUE sur les barres où un méta-label existe (signal ≠ 0)
    train_mask = meta_train.notna()
    X_train_ml = X_train_all.loc[train_mask]
    y_train_ml = meta_train.loc[train_mask].astype(int)

    if X_train_ml.empty:
        logger.error("Aucun échantillon méta-labellisé sur train — abandon.")
        return

    logger.info("Entraînement RF sur %d échantillons...", len(X_train_ml))

    # Fill NaN restants dans les features d'entraînement
    X_train_ml = X_train_ml.fillna(0.0)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train_ml, y_train_ml)

    logger.info("Classes RF : %s", rf.classes_.tolist())
    if 1 in rf.classes_:
        class1_pos = int(list(rf.classes_).index(1))
    else:
        # Mono-classe 0 → fallback : utiliser la seule colonne dispo
        class1_pos = 0
        logger.warning(
            "RF mono-classe %s → proba[:,0] utilisée comme proxy P(win)",
            rf.classes_.tolist(),
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

    # ── 7. Sweep seuil sur TRAIN UNIQUEMENT ───────────────────────────────
    logger.info("Sweep seuil méta sur TRAIN (%d valeurs)...", len(META_THRESHOLDS))

    # Prédiction sur TOUTES les barres train (fillna pour les barres sans signal)
    X_train_pred = X_train_all.fillna(0.0)
    proba_train_all = rf.predict_proba(X_train_pred)
    proba_win_train = proba_train_all[:, class1_pos]

    # Indices des barres avec signal ≠ 0
    train_sig_idx = np.where(signals_train.values != 0)[0]
    proba_win_train_sig = proba_win_train[train_sig_idx]

    sweep_results: dict[float, dict[str, Any]] = {}
    best_sharpe = -np.inf
    best_threshold = META_THRESHOLDS[0]

    for thresh in META_THRESHOLDS:
        filtered = _filter_signals_by_proba(
            signals_train, proba_win_train_sig, thresh, train_sig_idx,
        )
        bt = _run_backtest(df_train, filtered, cfg)
        sr = float(bt.get("sharpe", 0.0))
        sweep_results[thresh] = {
            "sharpe": sr,
            "n_trades": int(bt.get("total_trades", 0)),
            "wr": float(bt.get("wr", 0.0)),
            "total_pnl_pips": float(bt.get("total_pnl_pips", 0.0)),
        }
        if sr > best_sharpe:
            best_sharpe = sr
            best_threshold = thresh

    logger.info("Résultats sweep train :")
    for thresh in META_THRESHOLDS:
        r = sweep_results[thresh]
        marker = " ★" if thresh == best_threshold else ""
        logger.info("  seuil=%.2f  sharpe=%+.4f  wr=%.1f%%  trades=%d%s",
                    thresh, r["sharpe"], r["wr"] * 100, r["n_trades"], marker)
    logger.info("Meilleur seuil TRAIN : %.2f (Sharpe=%.4f)", best_threshold, best_sharpe)

    # ── 8. Évaluation VAL avec seuil fixe ─────────────────────────────────
    logger.info("── Évaluation VAL 2023 (seuil=%.2f) ──", best_threshold)

    # Baseline Donchian pur
    bt_val_base = _run_backtest(df_val, signals_val, cfg)
    logger.info("VAL baseline : sharpe=%.4f, wr=%.1f%%, trades=%d",
                bt_val_base["sharpe"], bt_val_base["wr"] * 100,
                bt_val_base["total_trades"])

    # Méta-labeling
    X_val_pred = X_val_all.fillna(0.0)
    proba_val_all = rf.predict_proba(X_val_pred)
    proba_win_val = proba_val_all[:, class1_pos]
    val_sig_idx = np.where(signals_val.values != 0)[0]
    proba_win_val_sig = proba_win_val[val_sig_idx]

    signals_val_filtered = _filter_signals_by_proba(
        signals_val, proba_win_val_sig, best_threshold, val_sig_idx,
    )
    n_val_after = int((signals_val_filtered != 0).sum())
    logger.info("VAL après méta-filtre : %d → %d signaux", n_sig["val"], n_val_after)

    bt_val = _run_backtest(df_val, signals_val_filtered, cfg)
    logger.info("VAL méta : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pips",
                bt_val["sharpe"], bt_val["wr"] * 100,
                bt_val["total_trades"], bt_val["total_pnl_pips"])

    # ── 9. Évaluation TEST (UNE SEULE FOIS) ──────────────────────────────
    logger.info("── Évaluation TEST ≥ 2024 (seuil=%.2f) ──", best_threshold)

    # Baseline Donchian pur
    bt_test_base = _run_backtest(df_test, signals_test, cfg)
    logger.info("TEST baseline : sharpe=%.4f, wr=%.1f%%, trades=%d",
                bt_test_base["sharpe"], bt_test_base["wr"] * 100,
                bt_test_base["total_trades"])

    # Méta-labeling
    X_test_pred = X_test_all.fillna(0.0)
    proba_test_all = rf.predict_proba(X_test_pred)
    proba_win_test = proba_test_all[:, class1_pos]
    test_sig_idx = np.where(signals_test.values != 0)[0]
    proba_win_test_sig = proba_win_test[test_sig_idx]

    signals_test_filtered = _filter_signals_by_proba(
        signals_test, proba_win_test_sig, best_threshold, test_sig_idx,
    )
    n_test_after = int((signals_test_filtered != 0).sum())
    logger.info("TEST après méta-filtre : %d → %d signaux", n_sig["test"], n_test_after)

    bt_test = _run_backtest(df_test, signals_test_filtered, cfg)
    sr_test = float(bt_test["sharpe"])
    wr_test = float(bt_test["wr"])
    n_trades_test = int(bt_test["total_trades"])
    pnl_test = float(bt_test["total_pnl_pips"])
    dd_test = float(bt_test.get("max_drawdown_pips", 0.0))

    logger.info("TEST méta : sharpe=%.4f, wr=%.1f%%, trades=%d, pnl=%+.1f pips, dd=%.1f",
                sr_test, wr_test * 100, n_trades_test, pnl_test, dd_test)

    # ── 10. read_oos + validate_edge ──────────────────────────────────────
    read_oos(prompt="11", hypothesis="H1", sharpe=sr_test, n_trades=n_trades_test)

    equity_test, trades_df_test = _build_equity_from_trades(bt_test["trades"])
    if not trades_df_test.empty:
        edge_report = validate_edge(equity_test, trades_df_test, N_TRIALS)
    else:
        edge_report = EdgeReport(
            go=False,
            reasons=["Aucun trade sur test"],
            metrics={"sharpe": 0.0, "wr": 0.0, "n_trades": 0.0, "trades_per_year": 0.0},
        )

    logger.info("Edge : go=%s, reasons=%s", edge_report.go, edge_report.reasons)
    logger.info("Edge metrics : %s", edge_report.metrics)

    # ── 11. Rapport console ───────────────────────────────────────────────
    n_years_test = max((df_test.index[-1] - df_test.index[0]).days / 365.25, 0.01)
    trades_per_year = n_trades_test / n_years_test

    print("\n" + "=" * 72)
    print("  H1 — Méta-labeling RF sur XAUUSD D1 (Donchian N=100, M=20)")
    print("=" * 72)
    print(f"  Split         : train ≤ 2022-12-31 | val = 2023 | test ≥ 2024-01-01")
    print(f"  Coûts         : spread=25 pips + slippage=10 pips (total 35 pips)")
    print(f"  TP/SL         : 600/300 pips, window=120h (D1)")
    print(f"  Direction     : LONG uniquement")
    print(f"  RF            : {RF_PARAMS['n_estimators']} arbres, max_depth={RF_PARAMS['max_depth']}, "
          f"class_weight={RF_PARAMS['class_weight']}")
    print(f"  Seuil optimal : {best_threshold:.2f} (sweepé sur train uniquement, "
          f"{len(META_THRESHOLDS)} valeurs)")
    print("-" * 72)
    print(f"  {'Période':<16} {'Sharpe':>8} {'WR':>8} {'Trades':>7} {'PnL (pips)':>12}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*7} {'─'*12}")

    bt_train_base = _run_backtest(df_train, signals_train, cfg)
    print(f"  {'Train (base)':<16} {bt_train_base['sharpe']:>8.4f} "
          f"{bt_train_base['wr']*100:>7.1f}% {bt_train_base['total_trades']:>7}")

    print(f"  {'Val (base)':<16} {bt_val_base['sharpe']:>8.4f} "
          f"{bt_val_base['wr']*100:>7.1f}% {bt_val_base['total_trades']:>7}")

    print(f"  {'Val (méta)':<16} {bt_val['sharpe']:>8.4f} "
          f"{bt_val['wr']*100:>7.1f}% {bt_val['total_trades']:>7} "
          f"{bt_val['total_pnl_pips']:>+11.1f}")

    print(f"  {'Test (base)':<16} {bt_test_base['sharpe']:>8.4f} "
          f"{bt_test_base['wr']*100:>7.1f}% {bt_test_base['total_trades']:>7}")

    print(f"  {'Test (méta) ★':<16} {sr_test:>8.4f} "
          f"{wr_test*100:>7.1f}% {n_trades_test:>7} "
          f"{pnl_test:>+11.1f}")

    print("-" * 72)
    print(f"  Max DD test    : {dd_test:.1f} pips")
    print(f"  Trades/an test : {trades_per_year:.1f}")
    dsr_val = edge_report.metrics.get("dsr", float("nan"))
    p_val = edge_report.metrics.get("p_value", float("nan"))
    print(f"  DSR            : {dsr_val:.2f} (p={p_val:.3f})"
          if not np.isnan(dsr_val) else "  DSR            : NaN")
    print(f"  n_trials cumul : {N_TRIALS}")
    print(f"  GO             : {edge_report.go}")
    if edge_report.reasons:
        for r in edge_report.reasons:
            print(f"    → {r}")
    print("=" * 72)

    # ── 12. GO/NO-GO ──────────────────────────────────────────────────────
    sharpe_go = sr_test >= 1.0
    wr_go = wr_test >= 0.30
    tpa_go = trades_per_year >= 30
    dsr_ok = not np.isnan(dsr_val) and dsr_val > 0

    go_final = sharpe_go and wr_go and tpa_go and dsr_ok
    if go_final:
        logger.info("★★★ GO — Tous les critères H1 satisfaits ★★★")
    else:
        logger.info("✗ NO-GO — Critères non satisfaits :")
        if not sharpe_go:
            logger.info("  → Sharpe test %.4f < 1.0", sr_test)
        if not wr_go:
            logger.info("  → WR test %.1f%% < 30%%", wr_test * 100)
        if not tpa_go:
            logger.info("  → Trades/an %.1f < 30", trades_per_year)
        if not dsr_ok:
            logger.info("  → DSR non significatif (DSR=%.2f, p=%.3f)", dsr_val, p_val)

    # ── 13. Sauvegarde JSON ───────────────────────────────────────────────
    summary: dict[str, Any] = {
        "hypothesis": "H1",
        "prompt": "11",
        "title": "Méta-labeling RF sur XAUUSD D1 (Donchian Breakout)",
        "timestamp": datetime.now(UTC).isoformat(),
        "n_trials_cumul": N_TRIALS,
        "split": {
            "train": "≤ 2022-12-31",
            "val": "2023-01-01 → 2023-12-31",
            "test": "≥ 2024-01-01",
        },
        "config": {
            "donchian_N": DONCHIAN_N,
            "donchian_M": DONCHIAN_M,
            "direction": "LONG uniquement",
            "tp_points": cfg.tp_points,
            "sl_points": cfg.sl_points,
            "window_hours": cfg.window_hours,
            "spread_pips": cfg.spread_pips,
            "slippage_pips": cfg.slippage_pips,
            "total_cost_pips": cfg.total_cost_pips,
            "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
            "meta_thresholds_swept": META_THRESHOLDS,
            "best_threshold_train": best_threshold,
        },
        "features": [
            "dist_breakout_pct",
            "force_breakout_pct",
            "atr_ratio",
            "rsi_14",
            "retracement_pct",
            "days_since_signal",
            "vol_ratio",
            "day_of_week",
        ],
        "feature_importance": {name: round(imp, 6) for name, imp in importances},
        "train": {
            "n_barres": len(df_train),
            "n_signaux_long": n_sig["train"],
            "n_meta_labels": n_meta["train"],
            "n_wins": int((meta_train == 1).sum()),
            "sweep_results": {str(k): v for k, v in sweep_results.items()},
            "best_threshold": best_threshold,
        },
        "val": {
            "n_barres": len(df_val),
            "n_signaux_long": n_sig["val"],
            "n_signaux_after_meta": n_val_after,
            "baseline": {
                "sharpe": float(bt_val_base["sharpe"]),
                "wr": float(bt_val_base["wr"]),
                "n_trades": int(bt_val_base["total_trades"]),
            },
            "meta": {
                "sharpe": float(bt_val["sharpe"]),
                "wr": float(bt_val["wr"]),
                "n_trades": int(bt_val["total_trades"]),
                "total_pnl_pips": float(bt_val["total_pnl_pips"]),
            },
        },
        "test": {
            "n_barres": len(df_test),
            "n_signaux_long": n_sig["test"],
            "n_signaux_after_meta": n_test_after,
            "baseline": {
                "sharpe": float(bt_test_base["sharpe"]),
                "wr": float(bt_test_base["wr"]),
                "n_trades": int(bt_test_base["total_trades"]),
            },
            "meta": {
                "sharpe": sr_test,
                "wr": wr_test,
                "n_trades": n_trades_test,
                "total_pnl_pips": pnl_test,
                "max_dd_pips": dd_test,
                "trades_per_year": round(trades_per_year, 2),
            },
        },
        "edge_report": {
            "go": edge_report.go,
            "reasons": edge_report.reasons,
            "metrics": edge_report.metrics,
        },
        "go_nogo": {
            "sharpe_test": sr_test,
            "sharpe_threshold": 1.0,
            "sharpe_go": sharpe_go,
            "wr_test": wr_test,
            "wr_threshold": 0.30,
            "wr_go": wr_go,
            "trades_per_year": round(trades_per_year, 2),
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
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # StreamHandler avec encoding UTF-8 pour éviter les UnicodeEncodeError sur Windows
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
