"""H04 -- Orchestrateur meta-labeling RF + CPCV.

Pipeline complet :
1. Charge USA30IDXUSD D1
2. Split train<=2022 / val=2023 / test>=2024
3. Genere signaux Donchian(20,20)
4. Construit features H01 + Donchian_Position
5. Calcule meta-labels (gagnant/perdant par trade simule)
6. Sur train : entraine RF -> calibre seuil -> best_threshold
7. Sur val : backtest baseline vs meta-labeling
8. Sur test : backtest baseline vs meta-labeling
9. CPCV sur train+val+test combine
10. Sauvegarde rapport JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning_v2.backtest.cpcv import run_cpcv_meta_vs_baseline
from learning_machine_learning_v2.backtest.deterministic import run_deterministic_backtest
from learning_machine_learning_v2.backtest.meta_labeling import compute_meta_labels
from learning_machine_learning_v2.models.meta_rf import calibrate_threshold, train_meta_rf
from learning_machine_learning_v2.pipelines.us30 import Us30Pipeline
from learning_machine_learning_v2.strategies.donchian import DonchianBreakout

# -- Configuration figee ex ante -------------------------------------------
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")

TP_PIPS = 200.0
SL_PIPS = 100.0
WINDOW_HOURS = 120
COMMISSION_PIPS = 3.0
SLIPPAGE_PIPS = 5.0
PIP_SIZE = 1.0  # US30

RF_META_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}

CPCV_CONFIG: dict[str, Any] = {
    "n_splits": 5,
    "n_test_splits": 2,
    "purge_hours": 120,
}

THRESHOLD_GRID: list[float] = [0.45, 0.50, 0.55, 0.60, 0.65]


def _backtest_sharpe_only(
    df: pd.DataFrame,
    signals: pd.Series,
) -> float:
    """Wrapper : execute le backtest deterministe et retourne uniquement le Sharpe."""
    result = run_deterministic_backtest(
        df=df,
        signals=signals,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        window_hours=WINDOW_HOURS,
        commission_pips=COMMISSION_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        pip_size=PIP_SIZE,
    )
    return float(result["sharpe"])


def _filter_signals_by_proba(
    signals: pd.Series,
    proba_win: pd.Series,
    threshold: float,
) -> pd.Series:
    """Filtre les signaux Donchian : ne garde que ceux ou proba > threshold."""
    filtered = signals.copy()
    mask_reject = proba_win <= threshold
    filtered.loc[filtered.index[mask_reject]] = 0
    return filtered


def run() -> None:
    """Execute le pipeline H04 complet."""
    pipeline = Us30Pipeline()

    # -- 1. Chargement --------------------------------------------------
    data = pipeline.load_data()
    d1 = data["us30_d1"].sort_index()
    print(f"Donnees chargees : {len(d1)} barres, {d1.index.min().date()} -> {d1.index.max().date()}")

    # -- 2. Split temporel ----------------------------------------------
    train_mask = d1.index <= TRAIN_END
    val_mask = (d1.index >= VAL_START) & (d1.index <= VAL_END)
    test_mask = d1.index >= TEST_START

    df_train = d1.loc[train_mask]
    df_val = d1.loc[val_mask]
    df_test = d1.loc[test_mask]

    print(f"Split : train={len(df_train)} (<={TRAIN_END.date()}), "
          f"val={len(df_val)} ({VAL_START.date()}->{VAL_END.date()}), "
          f"test={len(df_test)} (>={TEST_START.date()})")

    # -- 3. Generation signaux Donchian(20,20) --------------------------
    donchian = DonchianBreakout(N=20, M=20)
    all_signals = donchian.generate_signals(d1)
    signals_train = all_signals.loc[df_train.index]
    signals_val = all_signals.loc[df_val.index]
    signals_test = all_signals.loc[df_test.index]

    n_long = int((all_signals == 1).sum())
    n_short = int((all_signals == -1).sum())
    print(f"Signaux Donchian(20,20) : {n_long} LONG, {n_short} SHORT, "
          f"{(n_long + n_short)} total")

    # -- 4. Construction features ---------------------------------------
    ml_data = pipeline.build_features(data)
    # ml_data contient : Open, High, Low, Close, Target, RSI_14, ADX_14,
    #   Dist_SMA50, Dist_SMA200, ATR_Norm, Log_Return_5d, (+-Volume_Ratio)

    # Ajouter Donchian_Position comme feature
    ml_data["Donchian_Position"] = all_signals.reindex(ml_data.index).fillna(0).astype(int)

    # Colonnes de features (exclut OHLC et Target)
    feature_cols = [
        "RSI_14", "ADX_14", "Dist_SMA50", "Dist_SMA200",
        "ATR_Norm", "Log_Return_5d", "Donchian_Position",
    ]
    if "Volume_Ratio" in ml_data.columns:
        feature_cols.append("Volume_Ratio")

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"ml_data : {len(ml_data)} lignes apres dropna (build_features)")

    # -- 5. Meta-labels sur tout le dataset -----------------------------
    meta_labels_all = compute_meta_labels(
        df=d1,
        donchian_signals=all_signals,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        window_hours=WINDOW_HOURS,
        pip_size=PIP_SIZE,
    )

    # Aligner sur ml_data (qui a perdu des lignes avec dropna)
    common_idx = ml_data.index.intersection(meta_labels_all.index)
    ml_data = ml_data.loc[common_idx]
    meta_labels_all = meta_labels_all.loc[common_idx]
    all_signals = all_signals.loc[common_idx]

    n_win = int((meta_labels_all == 1).sum())
    n_loss = int((meta_labels_all == 0).sum())
    print(f"Meta-labels : {n_win} gagnants, {n_loss} perdants "
          f"(sur {len(common_idx)} barres alignees)")

    # -- 6. Entrainement RF + calibration seuil sur train ---------------
    train_common = ml_data.index.intersection(df_train.index)
    X_train = ml_data.loc[train_common, feature_cols]
    y_train = meta_labels_all.loc[train_common]
    df_train_aligned = d1.loc[train_common]
    signals_train_aligned = all_signals.loc[train_common]

    print(f"\nTrain meta-labeling : {len(X_train)} lignes, "
          f"dont {int(y_train.notna().sum())} avec signal")

    rf_model, X_train_f, y_train_f = train_meta_rf(
        X=X_train,
        y=y_train,
        params=RF_META_PARAMS,
    )
    print(f"RF entraine sur {len(X_train_f)} echantillons "
          f"(classe 1: {int(y_train_f.sum())}, "
          f"classe 0: {int(len(y_train_f) - y_train_f.sum())})")

    best_threshold, calibration_results = calibrate_threshold(
        rf=rf_model,
        X_train=X_train,
        y_train=y_train,
        df_train=df_train_aligned,
        donchian_signals_train=signals_train_aligned,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        window_hours=WINDOW_HOURS,
        commission_pips=COMMISSION_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        pip_size=PIP_SIZE,
        thresholds=THRESHOLD_GRID,
    )

    print(f"\nCalibration seuil sur train:")
    for t in sorted(calibration_results.keys()):
        r = calibration_results[t]
        marker = " <-- BEST" if t == best_threshold else ""
        print(f"  threshold={t:.2f} -> Sharpe={r['sharpe']:.4f}, "
              f"trades={r['trades']}{marker}")
    print(f"Best threshold = {best_threshold}")

    # -- 7. Evaluation sur Val (2023) -----------------------------------
    val_common = ml_data.index.intersection(df_val.index)
    X_val = ml_data.loc[val_common, feature_cols]
    df_val_aligned = d1.loc[val_common]
    signals_val_aligned = all_signals.loc[val_common]

    # Baseline val
    sharpe_val_baseline = _backtest_sharpe_only(df_val_aligned, signals_val_aligned)

    # Meta-labeling val
    proba_val = rf_model.predict_proba(X_val)
    class1_idx = list(rf_model.classes_).index(1) if 1 in rf_model.classes_ else 1
    proba_win_val = pd.Series(proba_val[:, class1_idx], index=X_val.index)
    filtered_val = _filter_signals_by_proba(signals_val_aligned, proba_win_val, best_threshold)
    sharpe_val_meta = _backtest_sharpe_only(df_val_aligned, filtered_val)

    print(f"\n=== RESULTATS VAL (2023) ===")
    print(f"Baseline Donchian pur  - Sharpe Val: {sharpe_val_baseline:.4f}")
    print(f"Meta-labeling RF       - Sharpe Val: {sharpe_val_meta:.4f}")

    # -- 8. Evaluation sur Test (2024-2025) -----------------------------
    test_common = ml_data.index.intersection(df_test.index)
    X_test = ml_data.loc[test_common, feature_cols]
    df_test_aligned = d1.loc[test_common]
    signals_test_aligned = all_signals.loc[test_common]

    # Baseline test
    sharpe_test_baseline = _backtest_sharpe_only(df_test_aligned, signals_test_aligned)

    # Meta-labeling test
    proba_test = rf_model.predict_proba(X_test)
    proba_win_test = pd.Series(proba_test[:, class1_idx], index=X_test.index)
    filtered_test = _filter_signals_by_proba(signals_test_aligned, proba_win_test, best_threshold)
    sharpe_test_meta = _backtest_sharpe_only(df_test_aligned, filtered_test)

    print(f"\n=== RESULTATS TEST (2024-2025) ===")
    print(f"Baseline Donchian pur  - Sharpe Test: {sharpe_test_baseline:.4f}")
    print(f"Meta-labeling RF       - Sharpe Test: {sharpe_test_meta:.4f}")

    # -- 9. CPCV sur train+val+test combine -----------------------------
    print(f"\n=== CPCV ({CPCV_CONFIG['n_splits']} splits, "
          f"test={CPCV_CONFIG['n_test_splits']}, "
          f"purge={CPCV_CONFIG['purge_hours']}h) ===")

    df_all = d1.loc[common_idx]
    signals_all = all_signals.loc[common_idx]
    X_all = ml_data.loc[common_idx, feature_cols]
    y_all = meta_labels_all.loc[common_idx]

    cpcv_result = run_cpcv_meta_vs_baseline(
        df=df_all,
        donchian_signals=signals_all,
        features=X_all,
        meta_labels=y_all,
        rf_params=RF_META_PARAMS,
        tp_pips=TP_PIPS,
        sl_pips=SL_PIPS,
        window_hours=WINDOW_HOURS,
        commission_pips=COMMISSION_PIPS,
        slippage_pips=SLIPPAGE_PIPS,
        pip_size=PIP_SIZE,
        threshold=best_threshold,
        n_splits=CPCV_CONFIG["n_splits"],
        n_test_splits=CPCV_CONFIG["n_test_splits"],
        purge_hours=CPCV_CONFIG["purge_hours"],
    )

    print(f"CPCV Baseline          - Sharpe mean: {cpcv_result['baseline_sharpe_mean']:.4f} "
          f"+/- {cpcv_result['baseline_sharpe_std']:.4f}")
    print(f"CPCV Meta-labeling     - Sharpe mean: {cpcv_result['meta_sharpe_mean']:.4f} "
          f"+/- {cpcv_result['meta_sharpe_std']:.4f}")
    print(f"p-value (t-test)       - {cpcv_result['p_value_paired_ttest']:.4f}")

    # -- 10. Decision GO/NO-GO ------------------------------------------
    meta_beats_baseline_test = sharpe_test_meta > sharpe_test_baseline
    meta_beats_baseline_cpcv = (
        cpcv_result["meta_sharpe_mean"] > cpcv_result["baseline_sharpe_mean"]
    )
    go = meta_beats_baseline_test and meta_beats_baseline_cpcv
    verdict = "GO - Le ML ameliore la baseline. Passage en paper trading." if go else \
              "NO-GO - Le ML n'apporte rien. On conserve la baseline Donchian pur."

    print(f"\n>>> GO/NO-GO: {verdict}")

    # -- 11. Sauvegarde rapport -----------------------------------------
    predictions_dir = Path("predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "hypothesis": "v2_04",
        "instrument": "USA30IDXUSD",
        "primary_tf": "D1",
        "strategy": "Donchian(20,20) + RF meta-labeling",
        "rf_params": {k: v for k, v in RF_META_PARAMS.items() if k != "n_jobs"},
        "threshold_grid": THRESHOLD_GRID,
        "best_threshold": best_threshold,
        "calibration_results": {
            f"{t:.2f}": calibration_results[t] for t in sorted(calibration_results.keys())
        },
        "backtest_config": {
            "tp_pips": TP_PIPS,
            "sl_pips": SL_PIPS,
            "window_hours": WINDOW_HOURS,
            "commission_pips": COMMISSION_PIPS,
            "slippage_pips": SLIPPAGE_PIPS,
        },
        "split": {
            "train_end": str(TRAIN_END.date()),
            "val": f"{VAL_START.date()} -> {VAL_END.date()}",
            "test_start": str(TEST_START.date()),
        },
        "results": {
            "val": {
                "baseline_sharpe": sharpe_val_baseline,
                "meta_sharpe": sharpe_val_meta,
            },
            "test": {
                "baseline_sharpe": sharpe_test_baseline,
                "meta_sharpe": sharpe_test_meta,
            },
            "cpcv": {
                "baseline_sharpe_mean": cpcv_result["baseline_sharpe_mean"],
                "baseline_sharpe_std": cpcv_result["baseline_sharpe_std"],
                "meta_sharpe_mean": cpcv_result["meta_sharpe_mean"],
                "meta_sharpe_std": cpcv_result["meta_sharpe_std"],
                "p_value_paired_ttest": cpcv_result["p_value_paired_ttest"],
                "n_splits": CPCV_CONFIG["n_splits"],
                "n_test_splits": CPCV_CONFIG["n_test_splits"],
                "purge_hours": CPCV_CONFIG["purge_hours"],
                "split_details": cpcv_result["split_details"],
            },
        },
        "go_no_go": {
            "verdict": "GO" if go else "NO-GO",
            "meta_beats_baseline_test": meta_beats_baseline_test,
            "meta_beats_baseline_cpcv": meta_beats_baseline_cpcv,
            "p_value": cpcv_result["p_value_paired_ttest"],
        },
        "n_trials_cumulatif_v2": 4,
    }

    output_path = predictions_dir / "meta_labeling_cpcv_results.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nRapport sauvegarde : {output_path}")


if __name__ == "__main__":
    run()
