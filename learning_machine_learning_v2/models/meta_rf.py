"""RandomForest pour méta-labeling — filtre les signaux Donchian.

H04 : le RF n'apprend pas une direction (Hausse/Baisse) mais estime
P(trade gagnant | features + signal Donchian). Seulement sur les barres
où Donchian émet un signal.

Fonctions :
- train_meta_rf : entraîne le RF sur les échantillons méta-labellisés
- calibrate_threshold : grid search du seuil de probabilité optimal
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning_v2.backtest.deterministic import run_deterministic_backtest


def train_meta_rf(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """Entraîne un RandomForest sur les échantillons méta-labellisés.

    Filtre les NaN de y (barres sans signal Donchian) avant entraînement.

    Args:
        X: DataFrame de features (incluant Donchian_Position), index=Time.
        y: pd.Series méta-labels (1=gagnant, 0=perdant, NaN=pas de signal).
        params: Dict de paramètres pour RandomForestClassifier.

    Returns:
        Tuple (model entraîné, X_filtered, y_filtered).
        X_filtered et y_filtered sont les échantillons utilisés pour l'entraînement
        (sans les NaN).
    """
    valid_mask = y.notna()
    X_filtered = X.loc[valid_mask]
    y_filtered = y.loc[valid_mask]

    if X_filtered.empty:
        raise ValueError(
            "Aucun échantillon méta-labellisé disponible. "
            "Vérifier que des signaux Donchian sont présents dans la période."
        )

    model = RandomForestClassifier(**params)
    model.fit(X_filtered, y_filtered)

    return model, X_filtered, y_filtered


def calibrate_threshold(
    rf: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_train: pd.DataFrame,
    donchian_signals_train: pd.Series,
    tp_pips: float = 200.0,
    sl_pips: float = 100.0,
    window_hours: int = 120,
    commission_pips: float = 3.0,
    slippage_pips: float = 5.0,
    pip_size: float = 1.0,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Calibre le seuil de probabilité RF qui maximise le Sharpe sur train.

    Pour chaque seuil :
    1. predict_proba sur X_train → probabilité classe 1
    2. Filtre les signaux Donchian : ne garde que ceux où proba > seuil
    3. Backtest déterministe avec signaux filtrés → Sharpe train

    Args:
        rf: RandomForestClassifier déjà entraîné.
        X_train: Features d'entraînement (même index que df_train).
        y_train: Méta-labels d'entraînement (non utilisé directement ici).
        df_train: DataFrame OHLC d'entraînement.
        donchian_signals_train: Signaux Donchian sur train (−1, 0, 1).
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window_hours: Durée max d'un trade en heures.
        commission_pips: Commission en pips par trade.
        slippage_pips: Slippage estimé en pips.
        pip_size: Taille d'un pip (1.0 pour US30).
        thresholds: Liste des seuils à tester (défaut: [0.45, 0.50, 0.55, 0.60, 0.65]).

    Returns:
        Tuple (best_threshold, results_dict) où results_dict contient
        {threshold: {"sharpe": float, "trades": int}}.
    """
    if thresholds is None:
        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65]

    # Alignement
    common_idx = X_train.index.intersection(donchian_signals_train.index)
    common_idx = common_idx.intersection(df_train.index)
    X_aligned = X_train.loc[common_idx]
    signals_aligned = donchian_signals_train.loc[common_idx]
    df_aligned = df_train.loc[common_idx]

    # Prédiction de proba classe 1 sur tout X_train
    proba_class1 = rf.predict_proba(X_aligned)
    # La classe 1 peut être à l'index 0 ou 1 selon les classes
    class1_idx = list(rf.classes_).index(1) if 1 in rf.classes_ else 1
    proba_win = proba_class1[:, class1_idx]

    results: dict[str, Any] = {}

    for threshold in thresholds:
        # Filtre : signal Donchian pris seulement si proba > threshold
        filtered_signals = signals_aligned.copy()
        mask_filter = proba_win <= threshold
        filtered_signals.loc[filtered_signals.index[mask_filter]] = 0

        # Backtest déterministe
        bt_result = run_deterministic_backtest(
            df=df_aligned,
            signals=filtered_signals,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            window_hours=window_hours,
            commission_pips=commission_pips,
            slippage_pips=slippage_pips,
            pip_size=pip_size,
        )

        results[threshold] = {
            "sharpe": bt_result["sharpe"],
            "trades": bt_result["total_trades"],
            "wr": bt_result["wr"],
            "total_pnl_pips": bt_result["total_pnl_pips"],
        }

    # Meilleur seuil = celui qui maximise le Sharpe
    if results:
        best_threshold = max(results, key=lambda t: results[t]["sharpe"])
    else:
        best_threshold = 0.50

    return best_threshold, results
