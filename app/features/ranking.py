"""Ranking robuste des features avec bootstrap stability (pivot v4 A6).

Méthode :
  1. Pour chaque bootstrap (n=5) :
     - Resample (avec remplacement) train ≤ 2022
     - Calculer 3 scores : mutual info, permutation importance RF, |corr|
     - Score composite = moyenne du rank de chacune des 3 métriques
  2. Stability score d'une feature = % de bootstraps où elle est dans le top K
  3. Top final = features avec stability >= 0.6 ET composite_rank ≤ top_k
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance


@dataclass
class RankingResult:
    """Résultat du ranking bootstrap avec scores de stabilité."""

    top_features: tuple[str, ...]
    stability_score: dict[str, float]
    metrics_per_feature: pd.DataFrame
    n_bootstrap: int = 5
    top_k: int = 15
    seed: int = 42


def _score_one_bootstrap(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> pd.DataFrame:
    """Pour un seul resample (avec remplacement), calcule les 3 métriques."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx_resampled = rng.choice(n, size=n, replace=True)
    X_b = X.iloc[idx_resampled].reset_index(drop=True)
    y_b = y.iloc[idx_resampled].reset_index(drop=True)

    # Drop NaN rows (safety — should already be clean)
    mask = X_b.notna().all(axis=1) & y_b.notna()
    X_b = X_b.loc[mask]
    y_b = y_b.loc[mask].astype(int)
    if len(X_b) < 50:
        return pd.DataFrame({"feature": X.columns, "score": 0.0})

    # 1. Mutual info classif
    mi = mutual_info_classif(X_b.values, y_b.values, random_state=seed)

    # 2. Permutation importance avec RF
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_b.values, y_b.values)
    perm = permutation_importance(
        rf,
        X_b.values,
        y_b.values,
        n_repeats=5,
        random_state=seed,
        n_jobs=-1,
    )

    # 3. Absolute Spearman correlation (robuste aux non-linéarités modestes)
    corr_abs = X_b.corrwith(y_b, method="spearman").abs().fillna(0.0).values

    df = pd.DataFrame({
        "feature": X.columns,
        "mutual_info": mi,
        "perm_importance": perm.importances_mean,
        "abs_corr": corr_abs,
    })
    # Rangs (1 = meilleur)
    for col in ["mutual_info", "perm_importance", "abs_corr"]:
        df[f"{col}_rank"] = df[col].rank(ascending=False, method="min")
    df["composite_rank"] = df[
        ["mutual_info_rank", "perm_importance_rank", "abs_corr_rank"]
    ].mean(axis=1)
    return df


def rank_features_bootstrap(
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 5,
    top_k: int = 15,
    seed: int = 42,
    stability_threshold: float = 0.6,
) -> RankingResult:
    """Ranking robuste : 5 bootstraps, garde les features stables.

    Args:
        X: DataFrame de features train uniquement (NaN tolérés, droppés par bootstrap).
        y: Series binaire (0/1) train uniquement.
        n_bootstrap: nombre de resamples avec remplacement.
        top_k: taille du top final.
        seed: graine reproductible.
        stability_threshold: fraction minimum de bootstraps où la feature doit
                              apparaître dans le top K pour être retenue.

    Returns:
        RankingResult avec top_features stable et scores détaillés.
    """
    if X.empty or len(X.columns) < top_k:
        raise ValueError(f"Trop peu de features : {len(X.columns)} < top_k={top_k}")

    all_dfs: list[pd.DataFrame] = []
    appearance: dict[str, int] = {f: 0 for f in X.columns}

    for i in range(n_bootstrap):
        df_i = _score_one_bootstrap(X, y, seed + i)
        all_dfs.append(df_i)
        top_i = df_i.nsmallest(top_k, "composite_rank")["feature"].tolist()
        for f in top_i:
            appearance[f] += 1

    stability = {f: appearance[f] / n_bootstrap for f in X.columns}

    # Score moyen composite sur tous les bootstraps
    avg = pd.concat(all_dfs).groupby("feature").mean(numeric_only=True)
    avg["stability"] = avg.index.map(stability)

    # Top final : features stables (stability >= threshold) classées par composite rank
    stable = avg[avg["stability"] >= stability_threshold].copy()
    if len(stable) < top_k:
        # Fallback : si pas assez de features stables, prendre les top_k par stability
        top = avg.sort_values(
            ["stability", "composite_rank"], ascending=[False, True]
        ).head(top_k)
    else:
        top = stable.sort_values("composite_rank").head(top_k)

    return RankingResult(
        top_features=tuple(top.index.tolist()),
        stability_score=stability,
        metrics_per_feature=avg.reset_index(),
        n_bootstrap=n_bootstrap,
        top_k=top_k,
        seed=seed,
    )
