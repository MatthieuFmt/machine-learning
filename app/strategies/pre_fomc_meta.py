"""Pre-FOMC Drift + ML méta-labeling (Étape 2 cascade).

Sur les signaux Pre-FOMC drift (Étape 1), un modèle ML décide
take/skip pour chaque trade candidate, sur base de features connues
à FOMC - 24h (pas de fuite future).

Architecture :
    1. Pour chaque event FOMC, calculer un vecteur de features à T_entry
    2. Label = 1 si pips_net > 0 dans le trade Étape 1
    3. Entraîner HistGradientBoostingClassifier (sklearn natif)
    4. Threshold de probabilité tuné sur CV pour maximiser Sharpe
    5. Prédiction OOS : trade pris seulement si P(win) ≥ threshold

Features (10) :
    - VIX level, VIX zscore 60j, DXY zscore 60j, yield slope 10y-3m  (macro)
    - US500 return 5d, return 20d, ATR% 14, dist SMA200, RSI 14   (US500 context)
    - days_since_last_fomc                                         (calendar)

Régularisation forte (n_train ≈ 88, donc max_iter=50, max_leaf_nodes=8).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from app.core.logging import get_logger
from app.features.indicators import atr, atr_pct, dist_sma, rsi

logger = get_logger(__name__)


# Hyperparams figés — pas de tuning OOS (n_trial = 0)
HGB_PARAMS: dict = {
    "max_iter": 50,
    "learning_rate": 0.08,
    "max_leaf_nodes": 8,
    "min_samples_leaf": 8,
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
    "us500_atr_pct_14",
    "us500_dist_sma_200",
    "us500_rsi_14",
    "days_since_last_fomc",
]


@dataclass(frozen=True)
class MetaLabelResult:
    """Résultat d'une lecture train ou OOS du méta-modèle."""
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
    df_us500_h1: pd.DataFrame,
    df_macro: pd.DataFrame,
    fomc_times: pd.DatetimeIndex,
    hours_before_entry: int = 24,
) -> pd.DataFrame:
    """Construit la matrice de features X pour chaque FOMC event.

    Une ligne par event FOMC. Les features sont les valeurs des indicateurs
    à la barre H1 qui sert d'entrée (FOMC - hours_before_entry).

    Args:
        df_us500_h1: OHLCV US500 H1 indexé UTC.
        df_macro: DataFrame macro indexé daily UTC (sortie de build_macro_dataframe).
        fomc_times: DatetimeIndex UTC des FOMC events à considérer.
        hours_before_entry: Décalage de l'entrée (défaut 24h).

    Returns:
        DataFrame indexé par fomc_time (UTC), colonnes = FEATURE_NAMES.
        Lignes avec features incomplètes (warmup) sont conservées avec NaN.
    """
    df = df_us500_h1.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Indicateurs US500 calculés sur l'ensemble de la série (pas de leak)
    atr_14 = atr(high, low, close, period=14)
    atr_pct_14 = atr_pct(close, atr_14).rename("atr_pct_14")
    dist_sma200 = dist_sma(close, 200).rename("dist_sma_200")
    rsi_14 = rsi(close, period=14).rename("rsi_14")
    # 5 jours H1 = 5*24 bars, 20 jours = 20*24
    ret_5d = close.pct_change(periods=5 * 24).rename("ret_5d")
    ret_20d = close.pct_change(periods=20 * 24).rename("ret_20d")

    # Décale +1 jour le macro (anti-look-ahead, cohérent avec F5)
    macro_shifted = df_macro.copy()
    macro_shifted.index = macro_shifted.index + pd.Timedelta(days=1)
    macro_shifted = macro_shifted.sort_index()

    sorted_times = pd.DatetimeIndex(sorted(set(fomc_times)))
    rows: list[dict] = []
    last_fomc: pd.Timestamp | None = None

    for fomc_ts in sorted_times:
        entry_ts = fomc_ts - pd.Timedelta(hours=hours_before_entry)

        # Trouve la 1ère barre H1 ≥ entry_ts
        entry_bars = df.index[df.index >= entry_ts]
        if len(entry_bars) == 0:
            continue
        entry_bar = entry_bars[0]

        # Features US500 à entry_bar
        row = {
            "fomc_time": fomc_ts,
            "entry_time": entry_bar,
            "us500_return_5d": float(ret_5d.get(entry_bar, np.nan)),
            "us500_return_20d": float(ret_20d.get(entry_bar, np.nan)),
            "us500_atr_pct_14": float(atr_pct_14.get(entry_bar, np.nan)),
            "us500_dist_sma_200": float(dist_sma200.get(entry_bar, np.nan)),
            "us500_rsi_14": float(rsi_14.get(entry_bar, np.nan)),
        }

        # Macro à entry_bar (merge_asof backward avec macro_shifted)
        macro_eligible = macro_shifted[macro_shifted.index <= entry_bar]
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

        # Days since last FOMC
        if last_fomc is None:
            row["days_since_last_fomc"] = np.nan
        else:
            row["days_since_last_fomc"] = float((fomc_ts - last_fomc).days)
        last_fomc = fomc_ts

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("fomc_time")
    return out


def train_meta_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str] | None = None,
) -> HistGradientBoostingClassifier:
    """Entraîne le HistGradientBoostingClassifier sur les features train.

    Args:
        X_train: DataFrame features (lignes = events FOMC train).
        y_train: 1 si winner, 0 sinon, même index que X_train.
        feature_cols: Sous-ensemble de colonnes. Défaut : toutes les FEATURE_NAMES.

    Returns:
        Modèle entraîné.
    """
    cols = feature_cols if feature_cols is not None else FEATURE_NAMES
    X = X_train[cols].values
    y = y_train.values.astype(int)

    model = HistGradientBoostingClassifier(**HGB_PARAMS)
    model.fit(X, y)
    return model


def cv_select_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pnls_train: pd.Series,
    feature_cols: list[str] | None = None,
    n_splits: int = 3,
    threshold_grid: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
) -> tuple[float, dict]:
    """Choisit le threshold de probabilité qui maximise le Sharpe CV moyen.

    Args:
        X_train: Features train.
        y_train: Labels binaires.
        pnls_train: pips_net réels par trade (pour calcul Sharpe in-CV).
        feature_cols: Colonnes à utiliser.
        n_splits: Nombre de folds StratifiedKFold.
        threshold_grid: Seuils à tester.

    Returns:
        (best_threshold, per_threshold_stats dict)
    """
    cols = feature_cols if feature_cols is not None else FEATURE_NAMES
    X = X_train[cols].values
    y = y_train.values.astype(int)
    pnls = pnls_train.values.astype(float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    per_thresh: dict[float, list[float]] = {t: [] for t in threshold_grid}
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        model = HistGradientBoostingClassifier(**HGB_PARAMS)
        model.fit(X[tr_idx], y[tr_idx])
        proba_val = model.predict_proba(X[val_idx])[:, 1]
        pnls_val = pnls[val_idx]

        for thresh in threshold_grid:
            mask = proba_val >= thresh
            if mask.sum() < 3:
                per_thresh[thresh].append(0.0)
                continue
            kept = pnls_val[mask]
            # Sharpe simple : mean / std × sqrt(n_per_year approx)
            std = kept.std() if kept.std() > 0 else 1e-9
            # 8 FOMC par an → annualisation × sqrt(8)
            sharpe = (kept.mean() / std) * np.sqrt(8)
            per_thresh[thresh].append(float(sharpe))

    # Best threshold = celui dont la moyenne Sharpe CV est max
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
    """Applique le filtre méta sur une liste de trades.

    Args:
        trades: Liste de trades dict (sortie de simulate_pre_fomc_trades).
        X: Features alignées aux trades (index = fomc_time).
        model: Modèle entraîné.
        threshold: Seuil de probabilité pour take.
        feature_cols: Colonnes à utiliser.

    Returns:
        (trades filtrés, probabilités par trade)
    """
    cols = feature_cols if feature_cols is not None else FEATURE_NAMES
    # Aligner trades sur X (par fomc_time)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return [], np.array([])
    trades_df["fomc_time"] = pd.to_datetime(trades_df["fomc_time"], utc=True)
    trades_df = trades_df.set_index("fomc_time")
    common = X.index.intersection(trades_df.index)
    X_aligned = X.loc[common, cols]
    trades_aligned = trades_df.loc[common]

    # Drop lignes avec NaN dans features → on garde le trade mais on marque proba=0
    mask_valid = X_aligned.notna().all(axis=1)
    probas = np.zeros(len(X_aligned))
    if mask_valid.sum() > 0:
        probas_valid = model.predict_proba(X_aligned[mask_valid].values)[:, 1]
        probas[mask_valid.values] = probas_valid

    take = probas >= threshold
    kept_trades = trades_aligned[take].reset_index()
    kept_trades_list = kept_trades.to_dict(orient="records")
    return kept_trades_list, probas
