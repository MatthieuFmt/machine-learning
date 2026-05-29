# Pivot v4 — A6 : Ranking & sélection robuste des features

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. [A5_feature_generation.md](A5_feature_generation.md) — **✅ Terminé**
3. [app/features/superset.py](../../app/features/superset.py) — créé en A5
4. [app/features/research.py](../../app/features/research.py) — harness existant (Prompt 04)
5. [../00_constitution.md](../00_constitution.md) — règles 9, 14
6. Bailey & López de Prado, "The Probability of Backtest Overfitting" (2014) — référence anti-snooping

## Objectif
Ranker rigoureusement les ~70 features sur **train ≤ 2022 uniquement** par 3 métriques complémentaires (mutual info, permutation importance RF, corrélation absolue), stabiliser le ranking via **bootstrap 5×**, et **figer le top 15 par actif** dans `app/config/features_selected.py`. Ce top sera utilisé par A7/A8/B1/B2 sans modification ultérieure.

> **Principe critique** : ranker des features sur train ne consomme **pas de n_trial** car ce n'est pas une décision GO/NO-GO sur le test set. Mais le ranking doit être **figé** avant la moindre lecture OOS. Toute modification post-A6 = data snooping.

## Type d'opération
🔧 **Sélection de features train-only** — **0 n_trial consommé**. Aucune lecture du test set 2024+.

## Definition of Done (testable)

- [ ] `app/features/ranking.py` (NOUVEAU) contient :
  - `rank_features_bootstrap(X, y, n_bootstrap=5, top_k=15, seed=42) -> RankingResult`
  - `RankingResult` dataclass avec : `top_features: list[str]`, `stability_score: dict[str, float]`, `metrics_per_feature: pd.DataFrame`
- [ ] `scripts/run_a6_feature_ranking.py` :
  - Pour chaque actif/TF dans `RANKING_CONFIGS` (US30 D1, EURUSD H4 prioritaires, autres secondaires)
  - Charge train ≤ 2022, calcule le superset, génère la cible binaire (winner Donchian)
  - Lance le bootstrap ranking
  - Sauvegarde dans `predictions/feature_ranking_<asset>_<tf>.json`
- [ ] `app/config/features_selected.py` (NOUVEAU, FROZEN) contient le top 15 par (asset, tf), figé après run :
  ```python
  FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {
      ("US30", "D1"): ("rsi_14", "adx_14", "dist_sma_200", ...),
      ("EURUSD", "H4"): ("...",),
  }
  ```
- [ ] `docs/feature_ranking_v4.md` (NOUVEAU) : rapport détaillé par actif (top 15 + scores + interprétation)
- [ ] `tests/unit/test_feature_ranking.py` (NOUVEAU) : ≥ 7 tests :
  - Bootstrap reproductible (même seed → même résultat)
  - Top K est trié
  - Stability score ∈ [0, 1]
  - Pas de NaN dans les rankings
  - Test sur série synthétique : feature corrélée à y doit être dans le top
  - Test sur features pures bruit : pas dans le top
  - Test sur train_end : aucune donnée > 2022-12-31 dans le ranking
- [ ] `n_trials` inchangé dans `JOURNAL.md`
- [ ] `rtk make verify` → 0 erreur

## NE PAS FAIRE

- ❌ Ne PAS utiliser le test set ≥ 2024. **Hard filter dans le script**.
- ❌ Ne PAS modifier `FEATURES_SELECTED` après ce prompt. Frozen dataclass.
- ❌ Ne PAS ranker sur des features dérivées du test set (ex : "mean global", "stat sur tout le DataFrame").
- ❌ Ne PAS faire de feature engineering "manuel" post-ranking (= snooping caché).
- ❌ Ne PAS incrémenter `n_trials`.
- ❌ Ne PAS sélectionner top K > 20 (overfit) ni < 10 (perte d'information).

## Étapes détaillées

### Étape 1 — `app/features/ranking.py`

```python
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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance


@dataclass
class RankingResult:
    top_features: tuple[str, ...]
    stability_score: dict[str, float]
    metrics_per_feature: pd.DataFrame
    n_bootstrap: int
    top_k: int
    seed: int


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

    # Drop NaN rows
    mask = X_b.notna().all(axis=1) & y_b.notna()
    X_b = X_b.loc[mask]
    y_b = y_b.loc[mask].astype(int)
    if len(X_b) < 50:
        return pd.DataFrame({"feature": X.columns, "score": 0.0})

    # 1. Mutual info classif
    mi = mutual_info_classif(X_b.values, y_b.values, random_state=seed)

    # 2. Permutation importance avec RF
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=10,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    rf.fit(X_b.values, y_b.values)
    perm = permutation_importance(
        rf, X_b.values, y_b.values,
        n_repeats=5, random_state=seed, n_jobs=-1,
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

    all_dfs = []
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
        top = avg.sort_values(["stability", "composite_rank"],
                              ascending=[False, True]).head(top_k)
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
```

### Étape 2 — `scripts/run_a6_feature_ranking.py`

```python
"""Pivot v4 A6 — Ranking robuste des features train uniquement.

⚠️ Aucune lecture du test set ≥ 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.config.instruments import ASSET_CONFIGS
from app.features.superset import build_superset
from app.features.ranking import rank_features_bootstrap
from app.strategies.donchian import DonchianBreakout
from app.strategies.mean_reversion import MeanReversionRSIBB  # créé en B2
from app.backtest.deterministic import run_deterministic_backtest


CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")


RANKING_CONFIGS = [
    {"asset": "US30",   "tf": "D1", "strat_cls": DonchianBreakout, "strat_kwargs": {"N": 20, "M": 20}},
    {"asset": "EURUSD", "tf": "H4", "strat_cls": MeanReversionRSIBB, "strat_kwargs": {}},
    {"asset": "XAUUSD", "tf": "D1", "strat_cls": DonchianBreakout, "strat_kwargs": {"N": 100, "M": 20}},
]


def build_target_from_strat(df_train, strat, cfg):
    """Génère la cible binaire (winner) à partir des trades sur train."""
    trades = run_deterministic_backtest(df_train, strat, cfg)
    if trades.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    y = (trades["Pips_Bruts"] > 0).astype(int)
    return trades, y


def rank_one_config(cfg_entry: dict) -> dict:
    asset = cfg_entry["asset"]
    tf = cfg_entry["tf"]
    cfg = ASSET_CONFIGS[asset]

    df = load_asset(asset, tf)
    df_train = df.loc[df.index <= CUTOFF]
    if df_train.empty:
        return {"asset": asset, "tf": tf, "error": "no train data"}

    strat = cfg_entry["strat_cls"](**cfg_entry["strat_kwargs"])
    trades, y = build_target_from_strat(df_train, strat, cfg)
    if trades.empty or len(y) < 50:
        return {"asset": asset, "tf": tf, "error": f"too few trades ({len(y)})"}

    X_full = build_superset(df_train, asset=asset)
    # Align X to trades entry timestamps
    X = X_full.loc[trades.index].dropna(axis=1, how="all")
    # Drop NaN rows
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    y_aligned = y.loc[X.index]

    if len(X) < 80:
        return {"asset": asset, "tf": tf, "error": f"too few train samples after align ({len(X)})"}

    result = rank_features_bootstrap(X, y_aligned, n_bootstrap=5, top_k=15, seed=42)
    return {
        "asset": asset,
        "tf": tf,
        "n_train_trades": len(X),
        "winner_rate": float(y_aligned.mean()),
        "top_features": list(result.top_features),
        "stability_score": result.stability_score,
        "metrics_per_feature": result.metrics_per_feature.to_dict(orient="records"),
    }


def main() -> int:
    set_global_seeds()
    out: dict = {}
    for entry in RANKING_CONFIGS:
        key = f"{entry['asset']}_{entry['tf']}"
        print(f"Ranking {key}...")
        out[key] = rank_one_config(entry)

    out_path = Path("predictions/feature_ranking_v4.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    # Génère features_selected.py
    selected_lines = []
    for entry in RANKING_CONFIGS:
        key = f"{entry['asset']}_{entry['tf']}"
        if "top_features" in out[key]:
            tup = tuple(out[key]["top_features"])
            selected_lines.append(
                f'    ("{entry["asset"]}", "{entry["tf"]}"): {tup!r},'
            )

    config_content = '"""FROZEN après pivot v4 A6. NE PAS MODIFIER sans nouveau pivot."""\n' \
                    'from __future__ import annotations\n\n' \
                    'FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {\n'
    config_content += "\n".join(selected_lines)
    config_content += '\n}\n'
    Path("app/config/features_selected.py").write_text(config_content, encoding="utf-8")

    print(f"Top features sauvegardés dans {out_path} et app/config/features_selected.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — `app/config/features_selected.py` (généré par le script)

Le script Étape 2 crée automatiquement ce fichier. Exemple de contenu attendu :

```python
"""FROZEN après pivot v4 A6. NE PAS MODIFIER sans nouveau pivot."""
from __future__ import annotations

FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {
    ("US30", "D1"): (
        "rsi_14", "adx_14", "dist_sma_200", "atr_pct_14", "macd_hist",
        "close_zscore_60", "dist_sma_50", "vol_regime_high", "regime_trending_binary",
        "return_percentile_60", "stoch_k_14", "efficiency_ratio_20",
        "event_high_within_24h_USD", "consecutive_up", "session_overlap_london_ny",
    ),
    ("EURUSD", "H4"): (
        "rsi_14", "bb_width_20", "session_overlap_london_ny", "close_zscore_20",
        "atr_zscore_60", "macd_hist", "return_percentile_20", "stoch_d_14",
        "williams_r_14", "body_to_range_ratio", "day_sin", "hours_to_next_event_high",
        "vol_regime_high", "efficiency_ratio_20", "kc_width_20",
    ),
    ("XAUUSD", "D1"): (
        "rsi_14", "dist_sma_200", "adx_14", "atr_pct_14", "macd_signal",
        "xauusd_return_5", "close_zscore_60", "regime_trending_binary",
        "return_percentile_60", "skew_returns_20", "stoch_k_14",
        "event_high_within_24h_USD", "vol_regime_high", "trend_strength", "doji",
    ),
}
```

> ⚠️ Les listes exactes dépendront du run. Le script remplit automatiquement.

### Étape 4 — Tests `tests/unit/test_feature_ranking.py`

```python
"""Tests du ranking robuste pivot v4 A6."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features.ranking import rank_features_bootstrap, RankingResult


def _make_synthetic_xy(n: int = 500, n_features: int = 30, n_relevant: int = 5, seed: int = 0):
    """Génère X avec n_relevant features prédictives + bruit."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    # Cible : combinaison linéaire des n_relevant premières features + seuillage
    logit = X.iloc[:, :n_relevant].sum(axis=1) + rng.normal(0, 0.3, n)
    y = pd.Series((logit > 0).astype(int))
    return X, y


def test_ranking_reproducible():
    X, y = _make_synthetic_xy(seed=0)
    r1 = rank_features_bootstrap(X, y, n_bootstrap=3, top_k=10, seed=42)
    r2 = rank_features_bootstrap(X, y, n_bootstrap=3, top_k=10, seed=42)
    assert r1.top_features == r2.top_features


def test_ranking_returns_correct_size():
    X, y = _make_synthetic_xy()
    r = rank_features_bootstrap(X, y, n_bootstrap=3, top_k=10, seed=42)
    assert len(r.top_features) <= 10


def test_relevant_features_in_top():
    """Les features prédictives doivent être dans le top."""
    X, y = _make_synthetic_xy(n=800, n_relevant=5, seed=0)
    r = rank_features_bootstrap(X, y, n_bootstrap=5, top_k=10, seed=42)
    relevant = {"f0", "f1", "f2", "f3", "f4"}
    overlap = relevant & set(r.top_features)
    assert len(overlap) >= 3, f"Seulement {len(overlap)}/5 features pertinentes dans le top 10"


def test_pure_noise_features_excluded():
    """Sur 25 features pures bruit, le top 10 n'inclut PAS toutes les 5 pertinentes
    de l'autre seed (séparation possible)."""
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(0, 1, (n, 25)), columns=[f"f{i}" for i in range(25)])
    y = pd.Series(rng.integers(0, 2, n))
    r = rank_features_bootstrap(X, y, n_bootstrap=5, top_k=10, seed=42)
    # Sur bruit pur, la stabilité doit être faible
    avg_stab = np.mean(list(r.stability_score.values()))
    assert avg_stab < 0.6, f"Stabilité moyenne trop élevée sur bruit : {avg_stab}"


def test_stability_in_range():
    X, y = _make_synthetic_xy()
    r = rank_features_bootstrap(X, y, n_bootstrap=5, top_k=10, seed=42)
    for f, s in r.stability_score.items():
        assert 0.0 <= s <= 1.0, f"Stabilité hors [0, 1] pour {f}: {s}"


def test_no_nan_in_metrics():
    X, y = _make_synthetic_xy()
    r = rank_features_bootstrap(X, y, n_bootstrap=3, top_k=10, seed=42)
    assert not r.metrics_per_feature.isna().any().any()


def test_raises_if_too_few_features():
    X, y = _make_synthetic_xy(n_features=5)
    with pytest.raises(ValueError, match="Trop peu de features"):
        rank_features_bootstrap(X, y, n_bootstrap=3, top_k=10)
```

### Étape 5 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_a6_feature_ranking.py
```

Sortie attendue :
```
Ranking US30_D1...
Ranking EURUSD_H4...
Ranking XAUUSD_D1...
Top features sauvegardés dans predictions/feature_ranking_v4.json et app/config/features_selected.py
```

### Étape 6 — Rapport `docs/feature_ranking_v4.md`

```markdown
# Feature Ranking v4 (pivot v4 A6)

**Date** : YYYY-MM-DD
**Méthode** : Bootstrap stability 5× × 3 métriques (MI + perm imp + Spearman |corr|)
**Périmètre** : train ≤ 2022-12-31 UNIQUEMENT
**n_trials** : inchangé (analyse train pure)

## Top 15 par actif

### US30 D1 (Donchian winner target)

| Rank | Feature | MI | Perm | |corr| | Stability |
|---|---|---|---|---|---|
| 1 | rsi_14 | 0.045 | 0.012 | 0.18 | 1.0 |
| 2 | ... | ... | ... | ... | ... |
...

### EURUSD H4 (MeanReversionRSIBB winner target)

| Rank | Feature | MI | Perm | |corr| | Stability |
|---|---|---|---|---|---|
| 1 | rsi_14 | ... | ... | ... | ... |
...

### XAUUSD D1 (Donchian winner target)
...

## Interprétation

### Patterns dominants
- **US30** : RSI + ADX + distance SMA200 dominent → confirmation que c'est un actif trend-following dont la qualité de signal dépend du régime.
- **EURUSD H4** : RSI + BB width + session overlap → mean-reversion sensible à la liquidité.
- **XAUUSD** : RSI + dist_sma_200 + xauusd_return_5 → trend + auto-régression.

### Features absentes (intéressant)
- ...

### Stabilité
- US30 : N features stables (stability ≥ 0.8) sur 71 → ratio %
- EURUSD : ...
- XAUUSD : ...

## Décision de gel

Les top 15 par actif sont FIGÉS dans `app/config/features_selected.py`.
**Aucune modification autorisée** jusqu'à la fin de Phase B.

## Limites
- n_bootstrap=5 = compromis vitesse/stabilité. 20 serait mieux mais lent.
- Si on rentraîne sur train+val au moment du walk-forward (B1+), le ranking est figé sur train ≤ 2022. C'est conservateur (pas d'info de 2023 dans le ranking).
```

## Tests unitaires associés

7 tests dans `tests/unit/test_feature_ranking.py` (cf. Étape 4).

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A6 : Feature ranking + bootstrap stability

- **Statut** : ✅ Terminé
- **Type** : Sélection de features train (0 n_trial)
- **Fichiers créés** : `app/features/ranking.py`, `scripts/run_a6_feature_ranking.py`, `app/config/features_selected.py`, `tests/unit/test_feature_ranking.py`, `docs/feature_ranking_v4.md`, `predictions/feature_ranking_v4.json`
- **Top 15 par actif** :
  - US30 D1 : [liste]
  - EURUSD H4 : [liste]
  - XAUUSD D1 : [liste]
- **Tests** : 7/7 passed
- **make verify** : ✅ passé
- **Notes** : Test set ≥ 2024 NON LU. Top features FIGÉS pour Phase B.
- **Prochaine étape** : A7 — Model selection RF vs HistGBM vs Stacking
```

## Critères go/no-go

- **GO Phase A7** si :
  - `app/config/features_selected.py` contient ≥ 1 actif avec top 15
  - Tests A6 passent
  - Stability moyenne des top 15 ≥ 0.6 (sinon ranking instable, refaire avec n_bootstrap=10)
- **NO-GO, revenir à** : si stability < 0.4 sur le top 5 → revoir n_bootstrap ou la cible binaire (peut-être prédire returns continus serait mieux).

## Annexes

### A1 — Pourquoi 3 métriques et pas 1

- **Mutual info** : capture relations non-linéaires mais sensible à la discrétisation
- **Permutation importance RF** : robuste, mais lent et dépend du modèle (RF specific)
- **Spearman |corr|** : monotone non-linéaire, rapide, indépendant du modèle

Un score composite (moyenne des rangs) est plus robuste que n'importe lequel pris isolément. C'est l'approche de López de Prado dans "Advances in Financial Machine Learning" ch. 8.

### A2 — Pourquoi bootstrap stability

Sur 500 samples train, le ranking d'une feature change si on resample. Une feature **vraiment prédictive** apparaît dans le top de **TOUS** les bootstraps. Une feature qui n'apparaît que dans 1/5 bootstraps est probablement un artefact.

Seuil 0.6 = présente dans 3/5 bootstraps. Compromis entre conservatisme et inclusion.

### A3 — Pourquoi top 15

- > 20 features : RF max_depth=4 sature, surcharge non utile
- < 10 features : risque de manquer des signaux complémentaires
- 15 = sweet spot empirique pour méta-labeling sur 500-1000 samples train

### A4 — Pourquoi ranker sur target binaire winner

La cible du méta-labeling = "ce trade Donchian sera-t-il gagnant?" → binaire 0/1.
Ranker les features par leur capacité à prédire cette binaire = exactement ce qu'on veut.

Alternative : ranker par retours continus (régression). Mais le méta-labeling est binaire, donc le ranking doit l'être aussi.

### A5 — Pourquoi pas LASSO / Elastic Net pour la sélection

LASSO sélectionne implicitement mais :
- Sensible aux features corrélées (élimine arbitrairement l'une des deux)
- Linéaire → manque les interactions non-linéaires
- Dépend lourdement de l'hyperparam alpha

Bootstrap + 3 métriques agnostiques au modèle = plus robuste pour un superset hétérogène.

### A6 — Que faire si stability < 0.4 sur le top 5

Probables causes :
1. Trop peu de samples (< 200) → besoin de plus de données train
2. Cible bruitée → la WR Donchian est inhérentement aléatoire
3. Features colinéaires → l'algorithme alterne entre des features quasi-équivalentes

Mitigations :
- Augmenter n_bootstrap à 10 ou 20
- Ajouter une étape de **clustering features** avant ranking (cf. López de Prado "Distance Matrix") — hors scope v4
- Élargir le superset (A5)

### A7 — Pourquoi figer top 15 dans un fichier Python (et pas JSON)

Un fichier `.py` avec frozen tuples = immutable au sens Python + visible dans le code review + utilisable directement (`from app.config.features_selected import FEATURES_SELECTED`). JSON nécessiterait du parsing à chaque import.

Le seul risque : modification post-A6. Mitigation : faire une vérification dans le test A9 que le hash du fichier n'a pas changé entre A6 et B1.

## Fin du prompt A6.
**Suivant** : [A7_model_selection.md](A7_model_selection.md)
