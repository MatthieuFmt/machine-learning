"""NR7 + ML méta-labeling (Étape 2 cascade, Phase H3).

Sur les signaux NR7 baseline (Étape 1), un modèle ML décide take/skip
pour chaque trade candidate, sur base de features connues au close du
jour J (NR setup), avant l'entry à Open J+1 (pas de fuite future).

Architecture :
    1. Pour chaque trade NR7, calculer un vecteur de features à `setup_date`
       (close J : indicateurs US500 + macro shiftés +1 jour).
    2. Label = 1 si pips_net > 0 dans le trade Étape 1.
    3. Entraîner HistGradientBoostingClassifier (sklearn natif).
    4. Threshold de probabilité tuné sur CV pour maximiser Sharpe annualisé.
    5. Prédiction OOS : trade pris seulement si P(win) ≥ threshold.

Features (11) :
    - vix_level, vix_zscore_60, dxy_zscore_60, yield_slope_10y_3m  (macro)
    - us500_return_5d, us500_return_20d, us500_dist_sma_200, us500_rsi_14  (context)
    - range_NR_atr20_ratio (compression relative — clé NR7)
    - signal_direction (interaction long/short)
    - day_of_week (Monday/Friday effects connus)

Régularisation modeste (n_train ≈ 307 vs 81 Pre-FOMC) : on autorise
plus de capacité (max_iter=100, max_leaf_nodes=15, min_samples_leaf=20).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from app.core.logging import get_logger
from app.features.indicators import atr, dist_sma, rsi

logger = get_logger(__name__)


# Hyperparams figés — pas de tuning OOS
HGB_PARAMS: dict = {
    "max_iter": 100,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}

FEATURE_NAMES: list[str] = [
    "vix_level",
    "vix_zscore_60",
    "dxy_zscore_60",
    "yield_slope_10y_3m",
    "us500_return_5d",
    "us500_return_20d",
    "us500_dist_sma_200",
    "us500_rsi_14",
    "range_NR_atr20_ratio",
    "signal_direction",
    "day_of_week",
]


@dataclass(frozen=True)
class MetaLabelResult:
    """Résultat d'une lecture train ou OOS du méta-modèle NR7."""
    threshold: float
    n_trades_pre_filter: int
    n_trades_post_filter: int
    sharpe_pre: float
    sharpe_post: float
    wr_pre: float
    wr_post: float
    mean_pnl_pre: float
    mean_pnl_post: float


def build_features_at_entry(
    df_d1_us500: pd.DataFrame,
    df_macro: pd.DataFrame,
    trades: list[dict],
    atr_period: int = 20,
    sma_period: int = 200,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Construit la matrice de features X pour chaque trade NR7.

    Une ligne par trade. Features = valeurs au close du jour J (NR setup),
    AVANT l'entry à Open J+1 (anti-look-ahead).

    Args:
        df_d1_us500: OHLCV US500 D1 indexé tz-aware UTC.
        df_macro: DataFrame macro indexé daily UTC (sortie de build_macro_dataframe).
        trades: Liste de trades dicts (sortie de simulate_nr_breakout_trades).
        atr_period, sma_period, rsi_period: Périodes des indicateurs.

    Returns:
        DataFrame indexé par setup_date (UTC normalized), colonnes = FEATURE_NAMES.
        Lignes avec features warmup peuvent contenir NaN.
    """
    df = df_d1_us500.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ── Indicateurs US500 sur l'ensemble de la série ───────────────
    atr_n = atr(high, low, close, period=atr_period)
    dist_sma_n = dist_sma(close, sma_period).rename("dist_sma_n")
    rsi_n = rsi(close, period=rsi_period).rename("rsi_n")
    ret_5d = close.pct_change(periods=5).rename("ret_5d")
    ret_20d = close.pct_change(periods=20).rename("ret_20d")

    # ── Macro shifté +1 jour (anti-look-ahead conservateur) ────────
    macro_shifted = df_macro.copy()
    macro_shifted.index = macro_shifted.index + pd.Timedelta(days=1)
    macro_shifted = macro_shifted.sort_index()

    rows: list[dict] = []
    for trade in trades:
        setup_date_str = trade["setup_date"]
        setup_ts = pd.Timestamp(setup_date_str).tz_localize("UTC")

        # Localiser la barre D1 du jour J (setup)
        j_mask = df.index.normalize() == setup_ts.normalize()
        j_bars = df[j_mask]
        if j_bars.empty:
            continue
        j_idx = j_bars.index[-1]

        atr_val = float(atr_n.get(j_idx, np.nan))
        range_nr = float(trade.get("range_J", np.nan))
        range_atr_ratio = (
            range_nr / atr_val if (atr_val and not np.isnan(atr_val) and atr_val > 0)
            else np.nan
        )

        entry_dow = float(pd.Timestamp(trade["entry_time"]).dayofweek)

        row = {
            "setup_date": setup_ts.normalize(),
            "us500_return_5d": float(ret_5d.get(j_idx, np.nan)),
            "us500_return_20d": float(ret_20d.get(j_idx, np.nan)),
            "us500_dist_sma_200": float(dist_sma_n.get(j_idx, np.nan)),
            "us500_rsi_14": float(rsi_n.get(j_idx, np.nan)),
            "range_NR_atr20_ratio": float(range_atr_ratio) if not np.isnan(range_atr_ratio) else np.nan,
            "signal_direction": float(trade["signal"]),
            "day_of_week": entry_dow,
        }

        # Macro à j_idx (merge_asof backward avec macro_shifted)
        macro_eligible = macro_shifted[macro_shifted.index <= j_idx]
        if not macro_eligible.empty:
            last_macro = macro_eligible.iloc[-1]
            row["vix_level"] = float(last_macro.get("vix_level", np.nan))
            row["vix_zscore_60"] = float(last_macro.get("vix_zscore_60", np.nan))
            row["dxy_zscore_60"] = float(last_macro.get("dxy_zscore_60", np.nan))
            row["yield_slope_10y_3m"] = float(last_macro.get("yield_slope_10y_3m", np.nan))
        else:
            row["vix_level"] = np.nan
            row["vix_zscore_60"] = np.nan
            row["dxy_zscore_60"] = np.nan
            row["yield_slope_10y_3m"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("setup_date")
        # Réordonner les colonnes selon FEATURE_NAMES
        out = out[FEATURE_NAMES]
    return out


def train_meta_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str] | None = None,
    hgb_params: dict | None = None,
) -> HistGradientBoostingClassifier:
    """Entraîne le HistGradientBoostingClassifier sur les features train.

    Args:
        X_train: DataFrame features (lignes = trades train).
        y_train: 1 si winner (pips_net > 0), 0 sinon. Même index que X_train.
        feature_cols: Sous-ensemble de colonnes. Défaut : toutes FEATURE_NAMES.
        hgb_params: Override des HGB_PARAMS par défaut (pour test variantes).

    Returns:
        Modèle entraîné.
    """
    cols = feature_cols if feature_cols is not None else FEATURE_NAMES
    params = hgb_params if hgb_params is not None else HGB_PARAMS
    X = X_train[cols].values
    y = y_train.values.astype(int)

    model = HistGradientBoostingClassifier(**params)
    model.fit(X, y)
    return model


def cv_select_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pnls_train: pd.Series,
    feature_cols: list[str] | None = None,
    n_splits: int = 5,
    threshold_grid: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    trades_per_year_annualisation: float = 25.0,
    hgb_params: dict | None = None,
) -> tuple[float, dict]:
    """Choisit le threshold de probabilité qui maximise le Sharpe CV moyen.

    Args:
        X_train: Features train.
        y_train: Labels binaires.
        pnls_train: pips_net réels par trade.
        feature_cols: Colonnes à utiliser.
        n_splits: Nombre de folds StratifiedKFold.
        threshold_grid: Seuils à tester.
        trades_per_year_annualisation: Facteur d'annualisation Sharpe per-trade
            (NR7 ≈ 25 trades/an).

    Returns:
        (best_threshold, per_threshold_stats dict)
    """
    cols = feature_cols if feature_cols is not None else FEATURE_NAMES
    params = hgb_params if hgb_params is not None else HGB_PARAMS
    X = X_train[cols].values
    y = y_train.values.astype(int)
    pnls = pnls_train.values.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    per_thresh: dict[float, list[float]] = {t: [] for t in threshold_grid}
    for tr_idx, val_idx in skf.split(X, y):
        model = HistGradientBoostingClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx])
        proba_val = model.predict_proba(X[val_idx])[:, 1]
        pnls_val = pnls[val_idx]

        for thresh in threshold_grid:
            mask = proba_val >= thresh
            if mask.sum() < 3:
                per_thresh[thresh].append(0.0)
                continue
            kept = pnls_val[mask]
            std = kept.std() if kept.std() > 0 else 1e-9
            sharpe = (kept.mean() / std) * np.sqrt(trades_per_year_annualisation)
            per_thresh[thresh].append(float(sharpe))

    means = {t: float(np.mean(v)) for t, v in per_thresh.items()}
    best = max(means, key=means.get)
    return best, {"per_threshold": means, "details": per_thresh}


def filter_trades(
    trades: list[dict],
    X: pd.DataFrame,
    model: HistGradientBoostingClassifier,
    threshold: float,
    feature_cols: list[str] | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Applique le filtre méta sur une liste de trades NR7.

    Args:
        trades: Liste de trades dict (sortie de simulate_nr_breakout_trades).
        X: Features alignées aux trades (index = setup_date normalized UTC).
        model: Modèle entraîné.
        threshold: Seuil de probabilité pour take.
        feature_cols: Colonnes à utiliser.

    Returns:
        (trades filtrés, probabilités par trade dans l'ordre de X.index ∩ trades).
    """
    cols = feature_cols if feature_cols is not None else FEATURE_NAMES
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return [], np.array([])

    trades_df["setup_date_ts"] = pd.to_datetime(trades_df["setup_date"]).dt.tz_localize("UTC").dt.normalize()
    trades_df = trades_df.set_index("setup_date_ts")

    common = X.index.intersection(trades_df.index)
    X_aligned = X.loc[common, cols]
    trades_aligned = trades_df.loc[common]

    mask_valid = X_aligned.notna().all(axis=1)
    probas = np.zeros(len(X_aligned))
    if mask_valid.sum() > 0:
        probas_valid = model.predict_proba(X_aligned[mask_valid].values)[:, 1]
        probas[mask_valid.values] = probas_valid

    take = probas >= threshold
    kept_trades = trades_aligned[take].reset_index(drop=True)
    return kept_trades.to_dict(orient="records"), probas
