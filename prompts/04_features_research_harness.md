# Prompt 04 — Harness de recherche d'indicateurs

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/03_data_layer.md`

## Objectif
Construire un harness systématique qui teste **tous les indicateurs techniques classiques** sur un actif/TF donné et les classe par pouvoir prédictif (mutual information avec le retour forward + permutation importance d'un RF). Ce harness sert pour H11/H12 (méta-labeling) — il ne s'agit PAS de trouver un edge ici, mais de RANGER les features par utilité.

## Definition of Done (testable)
- [ ] `app/features/indicators.py` contient ≥ 15 indicateurs purement vectorisés (zéro boucle) :
  - Trend : `sma(n)`, `ema(n)`, `dist_sma(n)`, `slope_sma(n)`
  - Momentum : `rsi(14)`, `macd(12,26,9)`, `stoch(14,3)`, `williams_r(14)`, `cci(20)`
  - Volatilité : `atr(14)`, `atr_pct(14)`, `bbands_width(20,2)`, `keltner_width(20,2)`
  - Volume : `obv()`, `mfi(14)` (si volume > 0)
  - Régime : `adx(14)`, `efficiency_ratio(20)`, `realized_vol(20)`
- [ ] `app/features/research.py` contient `rank_features(asset, tf, target_horizon, n_top=20) -> pd.DataFrame` qui :
  - Calcule tous les indicateurs sur la série (train ≤ 2022 uniquement).
  - Crée une cible `forward_return = close.shift(-target_horizon) / close - 1`.
  - Calcule pour chaque feature : mutual information, Pearson corr abs, permutation importance (RF 100 arbres).
  - Retourne un DataFrame trié par score composite (moyenne rang sur les 3 métriques).
- [ ] Sortie sauvegardée dans `predictions/feature_research_<ASSET>_<TF>.json`.
- [ ] Script CLI : `scripts/run_feature_research.py --asset US30 --tf D1 --horizon 5`.
- [ ] `tests/unit/test_indicators.py` : pour chaque indicateur, un test de **non-look-ahead** (cf. Règle 7 de la constitution) — la valeur à `t` ne change pas si on tronque la série à `t+k`.
- [ ] `tests/unit/test_feature_research.py` : test que `rank_features` retourne un DataFrame trié, que les scores sont entre 0 et 1, que aucune feature n'apparaît en doublon.
- [ ] `rtk pytest tests/unit/test_indicators.py tests/unit/test_feature_research.py -v` passe.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS implémenter pandas-ta. Tout doit être vectorisé pandas/numpy pur (lisible, sans dépendance lourde).
- Ne PAS utiliser le test set (≥ 2024) pour le ranking. Train ≤ 2022 uniquement.
- Ne PAS chercher d'edge à ce stade — c'est juste un ranking.
- Ne PAS supprimer les colonnes du DataFrame source.
- Ne PAS commit.

## Étapes

### Étape 1 — Indicateurs vectorisés
Exemple `rsi` vectorisé :
```python
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))
```

Implémenter de la même manière tous les indicateurs listés. Chaque indicateur a un test de non-look-ahead.

### Étape 2 — Module `research.py`
```python
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from app.data.loader import load_asset
from app.features.indicators import compute_all_indicators


def rank_features(
    asset: str,
    tf: str,
    target_horizon: int,
    n_top: int = 20,
    train_end: str = "2022-12-31",
) -> pd.DataFrame:
    df = load_asset(asset, tf)
    df = df.loc[:train_end]

    features = compute_all_indicators(df)
    target = (df["close"].shift(-target_horizon) / df["close"] - 1).rename("y")

    aligned = pd.concat([features, target], axis=1).dropna()
    X = aligned.drop(columns=["y"])
    y = aligned["y"]

    mi = mutual_info_regression(X, y, random_state=42)
    corr = X.corrwith(y).abs()

    rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    result = pd.DataFrame(
        {
            "feature": X.columns,
            "mutual_info": mi,
            "abs_corr": corr.values,
            "permutation_importance": perm.importances_mean,
        }
    )

    for col in ["mutual_info", "abs_corr", "permutation_importance"]:
        result[f"{col}_rank"] = result[col].rank(ascending=False)
    result["composite_rank"] = result[
        ["mutual_info_rank", "abs_corr_rank", "permutation_importance_rank"]
    ].mean(axis=1)
    result = result.sort_values("composite_rank").head(n_top)

    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"feature_research_{asset}_{tf}.json"
    out_path.write_text(json.dumps(result.to_dict(orient="records"), indent=2))

    return result
```

### Étape 3 — Script CLI
`scripts/run_feature_research.py` qui parse `--asset`, `--tf`, `--horizon` et appelle `rank_features`.

### Étape 4 — Tests
- Pour chaque indicateur : test `non_look_ahead(indicator_fn, series)` qui vérifie `indicator_fn(series[:n])[n-1] == indicator_fn(series)[n-1]`.
- Pour `rank_features` : test que les 3 scores sont valides, que le tri est correct.

### Étape 5 — Documentation
Ajouter une section dans `INVENTORY.md` listant les 15+ indicateurs et leur module.

## Logging
```markdown
## 2026-MM-DD — Prompt 04 : Feature research harness
- **Statut** : ✅ Terminé
- **Fichiers créés** : app/features/indicators.py (X indicateurs), app/features/research.py, scripts/run_feature_research.py, tests/unit/test_indicators.py, tests/unit/test_feature_research.py
- **Tests pytest** : ✅ Y tests, 0 failures
- **Résultats indicatifs** (à compléter après exécution manuelle utilisateur) :
  - Top 5 features US30 D1 horizon=5 : ...
  - Top 5 features XAUUSD D1 horizon=5 : ...
```

## Critères go/no-go
- **GO prompt 05** si : tous les tests anti-look-ahead passent, le script CLI fonctionne sur US30 D1.
- **NO-GO, revenir à** : ce prompt si un seul indicateur a un look-ahead leak.
