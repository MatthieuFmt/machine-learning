"""CPCV simplifié pour H04 — comparaison méta-labeling vs baseline.

Implémente run_cpcv_meta_vs_baseline : CPCV purged time-series
cross-validation comparant Donchian pur vs Donchian+RF méta-labeling.

Algorithme :
1. Divise le dataset en n_splits groupes consécutifs sans overlap
2. Pour chaque fold k :
   a. Train = groupes[0:k+1] (cumulatif)
   b. Purge_hours entre train et test
   c. Test = groupes[k+n_test_splits] (saut)
   d. Baseline : backtest Donchian pur sur test → Sharpe
   e. Méta : entraîne RF sur train, filtre signaux sur test, backtest → Sharpe
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.backtest.deterministic import run_deterministic_backtest
from app.models.meta_rf import train_meta_rf


def run_cpcv_meta_vs_baseline(
    df: pd.DataFrame,
    donchian_signals: pd.Series,
    features: pd.DataFrame,
    meta_labels: pd.Series,
    rf_params: dict[str, Any],
    tp_pips: float = 200.0,
    sl_pips: float = 100.0,
    window_hours: int = 120,
    commission_pips: float = 3.0,
    slippage_pips: float = 5.0,
    pip_size: float = 1.0,
    threshold: float = 0.50,
    n_splits: int = 5,
    n_test_splits: int = 2,
    purge_hours: int = 120,
) -> dict[str, Any]:
    """CPCV purged time-series cross-validation.

    Compare Donchian pur (baseline) vs Donchian+RF méta-labeling
    sur des splits temporels avec purge.

    Args:
        df: DataFrame OHLC complet.
        donchian_signals: Signaux Donchian (−1, 0, 1), même index que df.
        features: DataFrame de features (7-8 colonnes), même index.
        meta_labels: Méta-labels (1/0/NaN), même index.
        rf_params: Paramètres pour RandomForestClassifier.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window_hours: Durée max d'un trade en heures.
        commission_pips: Commission en pips.
        slippage_pips: Slippage en pips.
        pip_size: Taille d'un pip.
        threshold: Seuil de probabilité RF calibré.
        n_splits: Nombre de groupes de backtesting.
        n_test_splits: Nombre de groupes en test OOS.
        purge_hours: Heures de purge entre train et test.

    Returns:
        Dict avec :
            baseline_sharpe_mean, baseline_sharpe_std,
            meta_sharpe_mean, meta_sharpe_std,
            p_value_paired_ttest,
            split_details: list[dict],
    """
    # ── Alignement ──────────────────────────────────────────────────────
    common_idx = df.index.intersection(donchian_signals.index)
    common_idx = common_idx.intersection(features.index)
    common_idx = common_idx.intersection(meta_labels.index)

    df = df.loc[common_idx]
    signals = donchian_signals.loc[common_idx]
    X = features.loc[common_idx]
    y_meta = meta_labels.loc[common_idx]

    n = len(df)
    split_details: list[dict[str, Any]] = []
    baseline_sharpes: list[float] = []
    meta_sharpes: list[float] = []

    if n < n_splits:
        return {
            "baseline_sharpe_mean": 0.0,
            "baseline_sharpe_std": 0.0,
            "meta_sharpe_mean": 0.0,
            "meta_sharpe_std": 0.0,
            "p_value_paired_ttest": 1.0,
            "split_details": [],
        }

    # ── Découpage en n_splits groupes contigus ──────────────────────────
    group_size = n // n_splits
    group_boundaries: list[tuple[int, int]] = []
    for g in range(n_splits):
        start = g * group_size
        end = start + group_size if g < n_splits - 1 else n
        group_boundaries.append((start, end))

    # ── Calcul de la purge en nombre de barres ──────────────────────────
    times = df.index
    if n >= 2:
        typical_td = (times[-1] - times[0]) / n
        typical_hours = typical_td.total_seconds() / 3600.0
        purge_bars = max(1, int(purge_hours / typical_hours)) if typical_hours > 0 else 0
    else:
        purge_bars = 0

    # ── Boucle CPCV ────────────────────────────────────────────────────
    max_k = n_splits - n_test_splits  # dernier fold où on peut avoir test après train

    for k in range(max_k):
        # Train : groupes 0 à k (cumulatif)
        train_end_group = k + 1
        train_start_idx = group_boundaries[0][0]
        train_end_idx = group_boundaries[train_end_group - 1][1]

        # Test : groupe k + n_test_splits (saut de n_test_splits après train)
        test_group = k + n_test_splits
        if test_group >= n_splits:
            continue

        test_start_idx = group_boundaries[test_group][0]
        test_end_idx = group_boundaries[test_group][1]

        # Purge : on exclut purge_bars barres entre train et test
        purge_start = train_end_idx
        purge_end = min(purge_start + purge_bars, test_start_idx)

        train_indices = np.arange(train_start_idx, purge_start)
        test_indices = np.arange(max(test_start_idx, purge_end), test_end_idx)

        if len(train_indices) < 30 or len(test_indices) < 10:
            continue

        # ── Baseline : Donchian pur sur test ───────────────────────────
        df_test = df.iloc[test_indices]
        signals_test = signals.iloc[test_indices]

        bt_baseline = run_deterministic_backtest(
            df=df_test,
            signals=signals_test,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            window_hours=window_hours,
            commission_pips=commission_pips,
            slippage_pips=slippage_pips,
            pip_size=pip_size,
        )

        # ── Méta-labeling : entraîne RF sur train, filtre sur test ─────
        X_train_fold = X.iloc[train_indices]
        y_train_fold = y_meta.iloc[train_indices]
        X_test_fold = X.iloc[test_indices]
        signals_test_fold = signals.iloc[test_indices]

        try:
            rf_model, _, _ = train_meta_rf(
                X=X_train_fold,
                y=y_train_fold,
                params=rf_params,
            )

            # predict_proba sur test
            proba = rf_model.predict_proba(X_test_fold)
            class1_idx = list(rf_model.classes_).index(1) if 1 in rf_model.classes_ else 1
            proba_win = proba[:, class1_idx]

            filtered_signals = signals_test_fold.copy()
            mask_reject = proba_win <= threshold
            filtered_signals.loc[filtered_signals.index[mask_reject]] = 0

            bt_meta = run_deterministic_backtest(
                df=df_test,
                signals=filtered_signals,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                window_hours=window_hours,
                commission_pips=commission_pips,
                slippage_pips=slippage_pips,
                pip_size=pip_size,
            )
        except (ValueError, RuntimeError):
            bt_meta = {"sharpe": 0.0, "total_trades": 0, "wr": 0.0, "total_pnl_pips": 0.0}

        baseline_sharpes.append(bt_baseline["sharpe"])
        meta_sharpes.append(bt_meta["sharpe"])

        split_details.append({
            "fold": int(k),
            "train_start": str(times[train_indices[0]].date()),
            "train_end": str(times[train_indices[-1]].date()),
            "test_start": str(times[test_indices[0]].date()),
            "test_end": str(times[test_indices[-1]].date()),
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "baseline_sharpe": float(bt_baseline["sharpe"]),
            "baseline_trades": int(bt_baseline["total_trades"]),
            "meta_sharpe": float(bt_meta["sharpe"]),
            "meta_trades": int(bt_meta["total_trades"]),
        })

    # ── Agrégation statistique ──────────────────────────────────────────
    n_valid = len(baseline_sharpes)

    if n_valid < 2:
        return {
            "baseline_sharpe_mean": float(np.mean(baseline_sharpes)) if n_valid > 0 else 0.0,
            "baseline_sharpe_std": 0.0,
            "meta_sharpe_mean": float(np.mean(meta_sharpes)) if n_valid > 0 else 0.0,
            "meta_sharpe_std": 0.0,
            "p_value_paired_ttest": 1.0,
            "split_details": split_details,
        }

    baseline_arr = np.array(baseline_sharpes, dtype=np.float64)
    meta_arr = np.array(meta_sharpes, dtype=np.float64)

    # t-test apparié
    t_stat, p_value = stats.ttest_rel(meta_arr, baseline_arr)

    return {
        "baseline_sharpe_mean": float(baseline_arr.mean()),
        "baseline_sharpe_std": float(baseline_arr.std(ddof=1)),
        "meta_sharpe_mean": float(meta_arr.mean()),
        "meta_sharpe_std": float(meta_arr.std(ddof=1)),
        "p_value_paired_ttest": float(p_value),
        "split_details": split_details,
    }
