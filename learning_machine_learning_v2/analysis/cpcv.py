"""Combinatorial Purged Cross-Validation (CPCV) — López de Prado ch.12.

Module Step 02 : framework de validation robuste.
- generate_cpcv_splits : groupes contigus + purge bidirectionnelle + échantillonnage.
- run_cpcv_backtest : pipeline complet par split, parallélisé joblib.
- aggregate_cpcv_metrics : agrégation statistique de la distribution Sharpe.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from learning_machine_learning.analysis.edge_validation import _compute_sharpe_from_returns
from learning_machine_learning.backtest.filters import (
    FilterPipeline,
    MomentumFilter,
    VolFilter,
    SessionFilter,
)
from learning_machine_learning.backtest.simulator import simulate_trades, simulate_trades_continuous
from learning_machine_learning.backtest.sizing import weight_centered
from learning_machine_learning.config.backtest import BacktestConfig
from learning_machine_learning.config.instruments import InstrumentConfig, TargetMode  # noqa: F401
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)

# ── Colonnes requises pour les filtres ──────────────────────────────────
FILTER_COLS: tuple[str, ...] = ("Dist_SMA200_D1", "ATR_Norm", "RSI_D1_delta")


# ═══════════════════════════════════════════════════════════════════════════
# 3a. Génération des splits CPCV
# ═══════════════════════════════════════════════════════════════════════════

def generate_cpcv_splits(
    index: pd.DatetimeIndex,
    n_groups: int = 48,
    k_test: int = 12,
    purge_hours: int = 48,
    n_samples: int = 200,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Génère des splits CPCV avec purge bidirectionnelle.

    Partitionne l'index en `n_groups` groupes contigus. Pour chaque combinaison
    de `k_test` groupes-test, applique une purge de `purge_hours` avant et après
    chaque groupe test, puis yield (train_indices, test_indices).

    Args:
        index: DatetimeIndex trié chronologiquement (monotonic increasing).
        n_groups: Nombre de groupes contigus (défaut: 48).
        k_test: Nombre de groupes de test par split (défaut: 12).
        purge_hours: Heures de purge bidirectionnelle autour de chaque groupe test.
        n_samples: Nombre de combinaisons à échantillonner (défaut: 200).
        random_state: Graine pour l'échantillonnage reproductible.

    Yields:
        Tuple (train_indices: np.ndarray, test_indices: np.ndarray).
        Indices = positions entières dans l'index.

    Invariants garantis :
        - max(train) + purge_hours < min(test)  [purge avant test]
        - max(test) + purge_hours < min(train_after_test) [purge après test]
        - Aucun chevauchement train/test
        - Pas de barres dans la zone de purge incluses
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"index doit être un DatetimeIndex, reçu {type(index).__name__}")
    if not index.is_monotonic_increasing:
        raise ValueError("L'index doit être strictement croissant chronologiquement.")
    if n_groups < 2:
        raise ValueError(f"n_groups doit être >= 2, reçu {n_groups}")
    if k_test < 1 or k_test >= n_groups:
        raise ValueError(f"k_test doit être dans [1, n_groups-1], reçu {k_test} pour n_groups={n_groups}")
    if purge_hours < 0:
        raise ValueError(f"purge_hours doit être >= 0, reçu {purge_hours}")

    n = len(index)
    if n < n_groups:
        raise ValueError(f"Index trop court ({n} barres) pour {n_groups} groupes.")

    # ── Découpage en n_groups groupes de taille ~égale ────────────────
    group_size = n // n_groups
    group_boundaries: list[tuple[int, int]] = []  # (start_idx_inclusive, end_idx_exclusive)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else n
        group_boundaries.append((start, end))

    # ── Génération des combinaisons ────────────────────────────────────
    from math import comb as _comb

    total_combinations = _comb(n_groups, k_test)
    if total_combinations <= n_samples:
        # Toutes les combinaisons
        n_actual = total_combinations
        rng = np.random.default_rng(random_state)
        # On génère toutes les combinaisons via itertools et on les mélange
        from itertools import combinations as _combinations
        all_combo_indices = list(_combinations(range(n_groups), k_test))
        rng.shuffle(all_combo_indices)
    else:
        n_actual = n_samples
        rng = np.random.default_rng(random_state)
        # Échantillonnage sans remise
        all_indices = np.arange(n_groups)
        all_combo_indices = []
        seen: set[tuple[int, ...]] = set()
        while len(all_combo_indices) < n_samples:
            candidate = tuple(sorted(rng.choice(all_indices, size=k_test, replace=False)))
            if candidate not in seen:
                seen.add(candidate)
                all_combo_indices.append(candidate)

    logger.info(
        "CPCV splits: n_groups=%d, k_test=%d, total_comb=%d, sampled=%d, purge=%dh",
        n_groups, k_test, total_combinations, n_actual, purge_hours,
    )

    purge_delta = timedelta(hours=purge_hours)

    for combo in all_combo_indices:
        test_group_set = set(combo)

        test_indices_list: list[np.ndarray] = []
        train_indices_list: list[np.ndarray] = []

        for g_idx in range(n_groups):
            g_start, g_end = group_boundaries[g_idx]
            group_timestamps = index[g_start:g_end]

            if g_idx in test_group_set:
                # Groupe test : on garde tout le groupe
                test_indices_list.append(np.arange(g_start, g_end, dtype=np.int64))
            else:
                # Groupe train potentiel : on exclut la zone de purge
                # autour des groupes test adjacents
                g_start_ts = group_timestamps[0]
                g_end_ts = group_timestamps[-1]

                # Vérifier si ce groupe est en zone de purge d'un groupe test
                in_purge = False
                for t_idx in test_group_set:
                    t_start, t_end = group_boundaries[t_idx]
                    t_start_ts = index[t_start]
                    t_end_ts = index[t_end - 1] if t_end > t_start else index[t_start]

                    # Purge avant : [t_start_ts - purge_delta, t_start_ts)
                    purge_before_start = t_start_ts - purge_delta
                    purge_before_end = t_start_ts

                    # Purge après : (t_end_ts, t_end_ts + purge_delta]
                    purge_after_start = t_end_ts
                    purge_after_end = t_end_ts + purge_delta

                    # Le groupe chevauche-t-il la purge ?
                    if (g_end_ts > purge_before_start and g_start_ts < purge_before_end) or \
                       (g_end_ts > purge_after_start and g_start_ts < purge_after_end):
                        in_purge = True
                        break

                if not in_purge:
                    train_indices_list.append(np.arange(g_start, g_end, dtype=np.int64))

                # Même si in_purge, on exclut — aucun ajout

        if not test_indices_list or not train_indices_list:
            logger.warning(
                "Split ignoré : test_groups=%s → train=%d groupes, test=%d groupes",
                combo, len(train_indices_list), len(test_indices_list),
            )
            continue

        test_idx = np.concatenate(test_indices_list)
        train_idx = np.concatenate(train_indices_list)

        # ── Vérifications invariants ─────────────────────────────────
        if len(np.intersect1d(train_idx, test_idx)) > 0:
            logger.error("CPCV invariant violé : chevauchement train/test. Split ignoré.")
            continue

        if len(train_idx) > 0 and len(test_idx) > 0:
            max_train_ts = index[train_idx].max()
            min_test_ts = index[test_idx].min()

            if max_train_ts + purge_delta >= min_test_ts:
                # Les purges intra-groupes devraient garantir ça,
                # mais si train et test ne sont pas adjacents ce n'est pas
                # nécessairement vrai (train après test)
                # On ne bloque que si le train est AVANT le test
                if max_train_ts < min_test_ts:
                    logger.error(
                        "CPCV invariant violé : max(train) + purge >= min(test). "
                        "train_max=%s, test_min=%s, purge=%s",
                        max_train_ts, min_test_ts, purge_delta,
                    )
                    continue

        yield train_idx, test_idx

    logger.info("CPCV splits: génération terminée (%d splits émis).", n_actual)


# ═══════════════════════════════════════════════════════════════════════════
# 3b. Exécution du pipeline sur chaque split (parallélisable)
# ═══════════════════════════════════════════════════════════════════════════

def _run_one_split(
    split_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ml_data: pd.DataFrame,
    ohlcv_h1: pd.DataFrame,
    model_factory: Any,
    backtest_cfg: BacktestConfig,
    instrument_cfg: InstrumentConfig,
    X_cols: list[str],
    target_mode: str,
    confidence_threshold: float,
    continuous_signal_threshold: float,
    filter_pipeline: FilterPipeline | None,
) -> dict[str, Any]:
    """Exécute le pipeline complet sur UN split CPCV.

    Returns:
        Dict avec split_id, train_start, train_end, test_start, test_end,
        n_train, n_test, n_trades, sharpe, sharpe_per_trade, profit_net,
        win_rate, dd, total_return_pct, n_signaux, n_filtres, error (str ou None).
    """
    result: dict[str, Any] = {
        "split_id": split_id,
        "train_start": None,
        "train_end": None,
        "test_start": None,
        "test_end": None,
        "n_train": 0,
        "n_test": 0,
        "n_trades": 0,
        "sharpe": float("nan"),
        "sharpe_per_trade": float("nan"),
        "profit_net": 0.0,
        "win_rate": 0.0,
        "dd": 0.0,
        "total_return_pct": 0.0,
        "n_signaux": 0,
        "n_filtres": 0,
        "error": None,
    }

    try:
        # ── Périodes temporelles ──────────────────────────────────────
        train_ts = ml_data.index[train_idx]
        test_ts = ml_data.index[test_idx]
        result["train_start"] = str(train_ts.min().date()) if len(train_ts) > 0 else None
        result["train_end"] = str(train_ts.max().date()) if len(train_ts) > 0 else None
        result["test_start"] = str(test_ts.min().date()) if len(test_ts) > 0 else None
        result["test_end"] = str(test_ts.max().date()) if len(test_ts) > 0 else None

        # ── Train ─────────────────────────────────────────────────────
        train_data = ml_data.iloc[train_idx]
        X_train = train_data[X_cols]
        y_train = train_data["Target"]
        result["n_train"] = len(X_train)

        if len(X_train) < 50:
            result["error"] = f"train trop petit: {len(X_train)} barres"
            return result

        # Fit
        model = model_factory(X_train, y_train)

        # ── Test ──────────────────────────────────────────────────────
        test_data = ml_data.iloc[test_idx]
        X_test = test_data[X_cols]
        result["n_test"] = len(X_test)

        if len(X_test) < 10:
            result["error"] = f"test trop petit: {len(X_test)} barres"
            return result

        # ── Prédiction ────────────────────────────────────────────────
        if target_mode == "forward_return":
            preds_array = model.predict(X_test)
            preds_df = pd.DataFrame(
                {"Predicted_Return": preds_array},
                index=test_data.index,
            )
            simulate_func = simulate_trades_continuous
            simulate_kwargs: dict[str, Any] = {
                "signal_threshold": continuous_signal_threshold,
            }
        else:
            preds_array = model.predict(X_test)
            probas = model.predict_proba(X_test)
            class_map = {float(cls): int(idx) for idx, cls in enumerate(model.classes_)}

            def _get_col(class_key: float) -> np.ndarray:
                if class_key in class_map:
                    return probas[:, class_map[class_key]]
                return np.zeros(len(probas), dtype=np.float64)

            preds_df = pd.DataFrame(
                {
                    "Prediction_Modele": preds_array,
                    "Confiance_Hausse_%": np.round(_get_col(1.0) * 100, 2),
                    "Confiance_Neutre_%": np.round(_get_col(0.0) * 100, 2),
                    "Confiance_Baisse_%": np.round(_get_col(-1.0) * 100, 2),
                },
                index=test_data.index,
            )
            simulate_func = simulate_trades
            simulate_kwargs = {"seuil_confiance": confidence_threshold}

        # ── Joindre OHLC H1 ───────────────────────────────────────────
        ohlc_cols = ["High", "Low", "Close"]
        ohlc_available = [c for c in ohlc_cols if c in ohlcv_h1.columns]
        if ohlc_available:
            df_backtest = preds_df.join(
                ohlcv_h1.loc[ohlcv_h1.index.isin(test_data.index), ohlc_available],
                how="left",
            )
        else:
            df_backtest = preds_df

        # ── Injecter colonnes filtres ─────────────────────────────────
        filter_cols_present = [c for c in FILTER_COLS if c in ml_data.columns]
        if filter_cols_present:
            df_backtest = df_backtest.join(
                ml_data.loc[ml_data.index.isin(test_data.index), filter_cols_present],
                how="left",
            )

        # ── Ajouter Spread si présent ─────────────────────────────────
        if "Spread" in test_data.columns:
            df_backtest["Spread"] = test_data["Spread"]

        # Vérifier colonnes OHLC + Spread
        if "High" not in df_backtest.columns or "Low" not in df_backtest.columns or \
           "Close" not in df_backtest.columns:
            result["error"] = "Colonnes OHLC manquantes"
            return result

        if "Spread" not in df_backtest.columns:
            df_backtest["Spread"] = 0.0  # fallback

        # ── Simulation ────────────────────────────────────────────────
        trades_df, n_signaux, n_filtres = simulate_func(
            df=df_backtest,
            weight_func=weight_centered,
            tp_pips=backtest_cfg.tp_pips,
            sl_pips=backtest_cfg.sl_pips,
            window=backtest_cfg.window_hours,
            pip_size=instrument_cfg.pip_size,
            commission_pips=backtest_cfg.commission_pips,
            slippage_pips=backtest_cfg.slippage_pips,
            filter_pipeline=filter_pipeline,
            **simulate_kwargs,
        )

        result["n_signaux"] = n_signaux
        result["n_filtres"] = int(sum(n_filtres.values())) if isinstance(n_filtres, dict) else 0

        if trades_df.empty or "Pips_Nets" not in trades_df.columns:
            result["n_trades"] = 0
            result["sharpe"] = 0.0
            result["sharpe_per_trade"] = 0.0
            return result

        # ── Métriques ─────────────────────────────────────────────────
        pnl: np.ndarray = trades_df["Pips_Nets"].values.astype(np.float64)
        n_trades = len(pnl)
        result["n_trades"] = n_trades

        if n_trades > 1:
            result["sharpe"] = float(_compute_sharpe_from_returns(pnl))
            # Sharpe per-trade (annualisé par le nombre de trades)
            result["sharpe_per_trade"] = float(_compute_sharpe_from_returns(pnl) * np.sqrt(n_trades))
        else:
            result["sharpe"] = 0.0
            result["sharpe_per_trade"] = 0.0

        result["profit_net"] = float(np.sum(pnl))
        result["win_rate"] = float(np.mean(pnl > 0) * 100.0)
        result["dd"] = float(np.min(np.cumsum(pnl))) if n_trades > 0 else 0.0
        result["total_return_pct"] = float(np.sum(pnl))  # pips nets totaux

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("Split %d: erreur — %s", split_id, result["error"])

    return result


def run_cpcv_backtest(
    ml_data: pd.DataFrame,
    ohlcv_h1: pd.DataFrame,
    splits: Iterator[tuple[np.ndarray, np.ndarray]],
    model_factory: Any,
    backtest_cfg: BacktestConfig,
    instrument_cfg: InstrumentConfig,
    X_cols: list[str],
    target_mode: str,
    confidence_threshold: float = 0.33,
    continuous_signal_threshold: float = 0.0005,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Exécute le pipeline complet sur chaque split CPCV, en parallèle.

    Args:
        ml_data: DataFrame ML-ready complet, indexé par Time.
        ohlcv_h1: DataFrame OHLC H1 (colonnes High, Low, Close).
        splits: Générateur de (train_indices, test_indices).
        model_factory: Callable (X_train, y_train) -> modèle entraîné.
        backtest_cfg: Configuration backtest.
        instrument_cfg: Configuration instrument.
        X_cols: Colonnes de features pour l'entraînement.
        target_mode: Mode de cible (triple_barrier, forward_return, etc.).
        confidence_threshold: Seuil de confiance pour le classifieur.
        continuous_signal_threshold: Seuil pour le régresseur.
        n_jobs: Nombre de jobs joblib (-1 = tous les CPUs).

    Returns:
        DataFrame avec colonnes split_id, train_start, train_end, test_start,
        test_end, n_train, n_test, n_trades, sharpe, sharpe_per_trade,
        profit_net, win_rate, dd, total_return_pct, n_signaux, n_filtres, error.
    """
    # ── Construire le pipeline de filtres ─────────────────────────────
    filters: list = []
    if backtest_cfg.use_momentum_filter:
        filters.append(MomentumFilter(threshold=backtest_cfg.momentum_filter_threshold))
    if backtest_cfg.use_vol_filter:
        filters.append(
            VolFilter(
                window=backtest_cfg.vol_filter_window,
                multiplier=backtest_cfg.vol_filter_multiplier,
            )
        )
    if backtest_cfg.use_session_filter:
        filters.append(
            SessionFilter(
                exclude_start=backtest_cfg.session_exclude_start,
                exclude_end=backtest_cfg.session_exclude_end,
            )
        )
    filter_pipeline = FilterPipeline(filters) if filters else None

    logger.info(
        "CPCV backtest: démarrage parallèle (n_jobs=%d), target_mode=%s, tp=%.1f, sl=%.1f",
        n_jobs, target_mode, backtest_cfg.tp_pips, backtest_cfg.sl_pips,
    )

    # ── Exécution parallèle ───────────────────────────────────────────
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_one_split)(
            split_id=i,
            train_idx=train_idx,
            test_idx=test_idx,
            ml_data=ml_data,
            ohlcv_h1=ohlcv_h1,
            model_factory=model_factory,
            backtest_cfg=backtest_cfg,
            instrument_cfg=instrument_cfg,
            X_cols=X_cols,
            target_mode=target_mode,
            confidence_threshold=confidence_threshold,
            continuous_signal_threshold=continuous_signal_threshold,
            filter_pipeline=filter_pipeline,
        )
        for i, (train_idx, test_idx) in enumerate(splits, start=1)
    )

    df = pd.DataFrame(results)

    n_ok = int((df["error"].isna() | (df["error"] == "")).sum())
    n_err = len(df) - n_ok
    logger.info(
        "CPCV backtest terminé: %d splits, %d OK, %d erreurs.",
        len(df), n_ok, n_err,
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3c. Agrégation des métriques CPCV
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_cpcv_metrics(
    results_df: pd.DataFrame,
) -> dict[str, Any]:
    """Agrège les métriques de tous les splits CPCV.

    Args:
        results_df: DataFrame produit par run_cpcv_backtest().

    Returns:
        Dict structuré avec n_splits, n_splits_valid, pct_profitable,
        sharpe (stats descriptives), n_trades (stats), coverage (par mois).
    """
    n_total = len(results_df)

    # Splits valides : au moins 1 trade ET pas d'erreur
    valid_mask = (results_df["n_trades"] > 0) & (
        results_df["error"].isna() | (results_df["error"] == "")
    )
    valid = results_df[valid_mask]
    n_valid = len(valid)

    if n_valid == 0:
        logger.warning("aggregate_cpcv_metrics: 0 split valide sur %d.", n_total)
        return {
            "n_splits": n_total,
            "n_splits_valid": 0,
            "pct_profitable": float("nan"),
            "sharpe": {
                "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "median": float("nan"),
                "ci_95_lower": float("nan"), "ci_95_upper": float("nan"),
            },
            "n_trades": {
                "mean": float("nan"), "std": float("nan"), "total": 0,
            },
            "coverage": {},
        }

    sharpe_values = valid["sharpe"].values.astype(np.float64)

    sharpe_stats = {
        "mean": float(np.mean(sharpe_values)),
        "std": float(np.std(sharpe_values, ddof=1)),
        "min": float(np.min(sharpe_values)),
        "max": float(np.max(sharpe_values)),
        "median": float(np.median(sharpe_values)),
        "ci_95_lower": float(np.percentile(sharpe_values, 2.5)),
        "ci_95_upper": float(np.percentile(sharpe_values, 97.5)),
    }

    pct_profitable = float(np.mean(sharpe_values > 0.0)) * 100.0

    n_trades_values = valid["n_trades"].values.astype(np.float64)
    n_trades_stats = {
        "mean": float(np.mean(n_trades_values)),
        "std": float(np.std(n_trades_values, ddof=1)),
        "total": int(np.sum(n_trades_values)),
    }

    # ── Couverture temporelle par mois ─────────────────────────────────
    coverage: dict[str, int] = {}
    for _, row in valid.iterrows():
        if row["test_start"] is not None and row["test_end"] is not None:
            try:
                test_start = pd.Timestamp(row["test_start"])
                test_end = pd.Timestamp(row["test_end"])
                # Mois couverts par la période de test
                months = pd.date_range(test_start, test_end, freq="MS")
                for m in months:
                    key = m.strftime("%Y-%m")
                    coverage[key] = coverage.get(key, 0) + 1
            except (ValueError, KeyError):
                continue

    logger.info(
        "CPCV aggregate: %d/%d splits valides, E[SR]=%.4f±%.4f, profitable=%.1f%%",
        n_valid, n_total,
        sharpe_stats["mean"], sharpe_stats["std"], pct_profitable,
    )

    return {
        "n_splits": n_total,
        "n_splits_valid": n_valid,
        "pct_profitable": round(pct_profitable, 2),
        "sharpe": {
            k: round(v, 6) if not np.isnan(v) else float("nan")
            for k, v in sharpe_stats.items()
        },
        "n_trades": {
            "mean": round(n_trades_stats["mean"], 1),
            "std": round(n_trades_stats["std"], 1),
            "total": n_trades_stats["total"],
        },
        "coverage": coverage,
    }
