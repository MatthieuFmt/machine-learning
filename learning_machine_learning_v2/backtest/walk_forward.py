"""Moteur de walk-forward avec réentraînement périodique — H05.

Simule le trading en conditions réalistes :
- Réentraînement tous les 6 mois
- Slippage aléatoire reproductible (seed=42)
- Zéro look-ahead : les features à la barre t utilisent info ≤ t
- Application du spread + slippage aléatoire à chaque trade

Réutilise run_deterministic_backtest pour la simulation bar-by-bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from learning_machine_learning_v2.backtest.deterministic import run_deterministic_backtest
from learning_machine_learning_v2.models.meta_rf import calibrate_threshold, train_meta_rf


def apply_slippage(
    entry_price: float,
    direction: int,
    slippage_fixed: float,
    slippage_random: float,
    spread: float,
    pip_size: float,
    rng: np.random.Generator,
) -> float:
    """Calcule le prix d'exécution avec slippage réaliste.

    Args:
        entry_price: Prix d'entrée théorique (Close de la barre de signal).
        direction: 1=LONG, -1=SHORT.
        slippage_fixed: Slippage fixe toujours défavorable (pips).
        slippage_random: Amplitude du slippage aléatoire uniforme ± (pips).
        spread: Spread bid/ask total (pips).
        pip_size: Taille d'un pip.
        rng: Générateur aléatoire avec seed fixe.

    Returns:
        Prix d'exécution réel après slippage.
    """
    random_component = rng.uniform(-slippage_random, slippage_random)
    total_slippage = (slippage_fixed + spread / 2.0 + random_component) * pip_size
    if direction == 1:  # LONG -> achat plus cher
        return entry_price + total_slippage
    else:  # SHORT -> vente moins chere
        return entry_price - total_slippage


def _adjust_trades_slippage(
    trades: list[dict[str, Any]],
    rng: np.random.Generator,
    spread_pips: float,
    slippage_random: float,
) -> list[dict[str, Any]]:
    """Applique le coût additionnel (spread + slippage aléatoire) à chaque trade.

    Le slippage fixe et la commission sont déjà intégrés par run_deterministic_backtest.
    On ajoute uniquement le spread bid/ask et la composante aléatoire.

    Args:
        trades: Liste de trades issus de run_deterministic_backtest.
        rng: Générateur aléatoire reproductible.
        spread_pips: Spread bid/ask total (pips), toujours défavorable.
        slippage_random: Amplitude du slippage aléatoire ± (pips).

    Returns:
        Liste de trades avec pips_net ajusté et extra_cost_pips ajouté.
    """
    for trade in trades:
        # random_component > 0 -> execution favorable -> cout reduit
        random_component = rng.uniform(-slippage_random, slippage_random)
        extra_cost = spread_pips - random_component
        trade["pips_net_raw"] = trade["pips_net"]
        trade["pips_net"] = trade["pips_net"] - extra_cost
        trade["extra_cost_pips"] = float(extra_cost)
    return trades


def _compute_sharpe_from_trades(trades: list[dict[str, Any]]) -> float:
    """Sharpe annualisé à partir d'une liste de trades — calculé sur les returns
    quotidiens de la courbe d'equity.

    Remplace l'ancien _compute_sharpe qui calculait (mean/std)*sqrt(252) sur
    les PnL par trade — formule incorrecte qui gonflait le Sharpe d'un facteur
    sqrt(252 / trades_par_an).
    """
    from learning_machine_learning_v2.backtest.metrics import sharpe_daily_from_trades
    return sharpe_daily_from_trades(trades)


def run_walk_forward(
    df: pd.DataFrame,
    donchian_signals: pd.Series,
    features: pd.DataFrame,
    meta_labels: pd.Series,
    retrain_dates: list[str],
    initial_train_end: str,
    use_rf: bool,
    rf_params: dict[str, Any] | None = None,
    thresholds: list[float] | None = None,
    tp_pips: float = 200.0,
    sl_pips: float = 100.0,
    window_hours: int = 120,
    commission_pips: float = 3.0,
    slippage_pips: float = 5.0,
    slippage_random: float = 2.0,
    spread_pips: float = 2.0,
    pip_size: float = 1.0,
) -> dict[str, Any]:
    """Walk-forward avec réentraînement périodique tous les 6 mois.

    Algorithme :
    1. Segment initial: [debut_donnees, initial_train_end] -> entrainement RF si use_rf.
    2. Découpe les dates retrain_dates en segments temporels.
    3. Pour chaque segment :
       a. Extrait df_segment (après la date de réentraînement précédente).
       b. Si use_rf: RF.predict_proba -> filtre signaux avec seuil calibre.
       c. Backtest déterministe sur le segment avec signaux (filtrés ou non).
       d. Applique slippage réaliste (spread + aléatoire) à chaque trade.
       e. Cumule les résultats (equity curve, trades).
       f. Réentraîne RF sur df[:fin_segment] (si use_rf et pas dernier segment).
    4. Retourne métriques agrégées.

    Args:
        df: DataFrame OHLC, index=Time (datetime).
        donchian_signals: pd.Series −1/0/1, même index que df.
        features: DataFrame de features, index=Time.
        meta_labels: pd.Series 1/0/NaN, index=Time.
        retrain_dates: Liste des dates de réentraînement (str "YYYY-MM-DD").
        initial_train_end: Date de fin d'entraînement initial (str "YYYY-MM-DD").
        use_rf: Si True, filtre les signaux avec RF méta-labeling.
        rf_params: Paramètres RandomForestClassifier.
        thresholds: Liste des seuils à tester pour la calibration.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window_hours: Durée max d'un trade en heures.
        commission_pips: Commission en pips (round-trip).
        slippage_pips: Slippage fixe en pips (round-trip).
        slippage_random: Amplitude du slippage aléatoire ± (pips).
        spread_pips: Spread bid/ask total (pips).
        pip_size: Taille d'un pip (1.0 pour US30).

    Returns:
        dict avec clés:
            sharpe: float — Sharpe ratio annualisé.
            wr: float — Win rate (0.0 à 1.0).
            trades: int — Nombre total de trades.
            pnl_pips: float — PnL total en pips.
            equity_curve: pd.Series — Courbe d'equity (index=exit_time).
            segment_details: list[dict] — Détail par segment.
    """
    # ── Nettoyage index ──
    if "Time" in df.columns:
        df = df.set_index("Time")

    # Alignement global sur l'intersection des index
    common_idx = df.index.intersection(donchian_signals.index)
    common_idx = common_idx.intersection(features.index)
    common_idx = common_idx.intersection(meta_labels.index)

    df = df.loc[common_idx].sort_index()
    donchian_signals = donchian_signals.loc[common_idx].sort_index()
    features = features.loc[common_idx].sort_index()
    meta_labels = meta_labels.loc[common_idx].sort_index()

    feature_cols = features.columns.tolist()

    # RNG reproductible (seed=42)
    rng = np.random.default_rng(42)

    # Conversion des dates
    train_end_ts = pd.Timestamp(initial_train_end)
    retrain_ts = [pd.Timestamp(d) for d in retrain_dates]

    # ── Construction des segments ──
    # Segment 0: (initial_train_end, retrain_dates[0]]
    # Segment i: (retrain_dates[i-1], retrain_dates[i]] pour i=1..len-1
    # Dernier: (retrain_dates[-1], fin_données]
    segment_boundaries = [train_end_ts] + retrain_ts + [df.index[-1] + pd.Timedelta(days=1)]
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(len(segment_boundaries) - 1):
        segments.append((segment_boundaries[i], segment_boundaries[i + 1]))

    # ── État mutable ──
    all_trades: list[dict[str, Any]] = []
    segment_details: list[dict[str, Any]] = []
    rf_model: RandomForestClassifier | None = None
    best_threshold: float = 0.50

    # ── Entraînement initial (sur données ≤ initial_train_end) ──
    if use_rf:
        train_mask = df.index <= train_end_ts
        if train_mask.any():
            X_train = features.loc[train_mask, feature_cols]
            y_train = meta_labels.loc[train_mask]
            signals_train = donchian_signals.loc[train_mask]
            df_train = df.loc[train_mask]

            rf_model, _, _ = train_meta_rf(X_train, y_train, rf_params or {})

            best_threshold, _ = calibrate_threshold(
                rf=rf_model,
                X_train=X_train,
                y_train=y_train,
                df_train=df_train,
                donchian_signals_train=signals_train,
                tp_pips=tp_pips,
                sl_pips=sl_pips,
                window_hours=window_hours,
                commission_pips=commission_pips,
                slippage_pips=slippage_pips,
                pip_size=pip_size,
                thresholds=thresholds,
            )

    # ── Walk-forward sur chaque segment ──
    for seg_idx, (seg_start_excl, seg_end_incl) in enumerate(segments):
        # Le segment trade les barres STRICTEMENT après la date de fin d'entraînement
        seg_mask = (df.index > seg_start_excl) & (df.index <= seg_end_incl)
        if not seg_mask.any():
            continue

        df_seg = df.loc[seg_mask]
        signals_seg = donchian_signals.loc[seg_mask]

        # Filtrer signaux avec RF si use_rf
        if use_rf and rf_model is not None:
            X_seg = features.loc[seg_mask, feature_cols]
            proba = rf_model.predict_proba(X_seg)
            class1_idx = list(rf_model.classes_).index(1) if 1 in rf_model.classes_ else 1
            proba_win = pd.Series(proba[:, class1_idx], index=X_seg.index)

            filtered_signals = signals_seg.copy()
            mask_reject = proba_win <= best_threshold
            filtered_signals.loc[filtered_signals.index[mask_reject]] = 0
            signals_seg = filtered_signals

        # Backtest déterministe sur le segment
        bt_result = run_deterministic_backtest(
            df=df_seg,
            signals=signals_seg,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            window_hours=window_hours,
            commission_pips=commission_pips,
            slippage_pips=slippage_pips,
            pip_size=pip_size,
        )

        # Appliquer slippage réaliste (spread + aléatoire) à chaque trade
        seg_trades = _adjust_trades_slippage(
            bt_result.get("trades", []),
            rng,
            spread_pips,
            slippage_random,
        )
        all_trades.extend(seg_trades)

        # Métriques segment
        seg_pnls = np.array([t["pips_net"] for t in seg_trades], dtype=np.float64)
        n_trades = len(seg_pnls)
        n_wins = int((seg_pnls > 0).sum())
        seg_wr = n_wins / n_trades if n_trades > 0 else 0.0
        seg_pnl = float(seg_pnls.sum()) if n_trades > 0 else 0.0
        seg_sharpe = _compute_sharpe_from_trades(seg_trades)

        seg_label = (
            f"{df_seg.index[0].date()}"
            f" -> {seg_end_incl.date()}"
        )
        segment_details.append({
            "segment": seg_label,
            "sharpe": seg_sharpe,
            "wr": seg_wr,
            "trades": n_trades,
            "pnl_pips": seg_pnl,
        })

        # ── Réentraînement cumulatif après le segment (sauf dernier) ──
        if use_rf and seg_idx < len(segments) - 1:
            retrain_end = seg_end_incl
            retrain_mask = df.index <= retrain_end
            if retrain_mask.any():
                X_retrain = features.loc[retrain_mask, feature_cols]
                y_retrain = meta_labels.loc[retrain_mask]
                signals_retrain = donchian_signals.loc[retrain_mask]
                df_retrain = df.loc[retrain_mask]

                rf_model, _, _ = train_meta_rf(X_retrain, y_retrain, rf_params or {})

                best_threshold, _ = calibrate_threshold(
                    rf=rf_model,
                    X_train=X_retrain,
                    y_train=y_retrain,
                    df_train=df_retrain,
                    donchian_signals_train=signals_retrain,
                    tp_pips=tp_pips,
                    sl_pips=sl_pips,
                    window_hours=window_hours,
                    commission_pips=commission_pips,
                    slippage_pips=slippage_pips,
                    pip_size=pip_size,
                    thresholds=thresholds,
                )

    # ── Métriques agrégées ──
    all_pnls = np.array([t["pips_net"] for t in all_trades], dtype=np.float64)
    total_trades = len(all_pnls)
    total_wins = int((all_pnls > 0).sum())
    wr = total_wins / total_trades if total_trades > 0 else 0.0
    total_pnl = float(all_pnls.sum()) if total_trades > 0 else 0.0
    sharpe = _compute_sharpe_from_trades(all_trades)

    # Courbe d'equity (index = exit_time de chaque trade)
    if all_trades:
        exit_times = [pd.Timestamp(t["exit_time"]) for t in all_trades]
        equity = pd.Series(np.cumsum([t["pips_net"] for t in all_trades]), index=exit_times)
    else:
        equity = pd.Series(dtype=np.float64)

    return {
        "sharpe": sharpe,
        "wr": wr,
        "trades": total_trades,
        "pnl_pips": total_pnl,
        "equity_curve": equity,
        "segment_details": segment_details,
    }
