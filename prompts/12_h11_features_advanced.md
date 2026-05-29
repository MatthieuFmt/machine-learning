# Prompt 12 — H11 bis : Features avancées (régime, statistiques)

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/11_h10_h11_h12_meta_labeling.md`
4. `docs/v3_roadmap.md`

## Objectif
Étendre le set de features du méta-labeling avec des transformations statistiques avancées (z-scores, percentiles, vol regime) et re-runner H11 avec ces features pour vérifier amélioration. Si gain marginal, intégrer définitivement.

## Definition of Done (testable)
- [ ] `app/features/advanced.py` contient :
  - `atr_zscore(df, period=14, lookback=200)` : (ATR_t - mean) / std sur fenêtre roulante
  - `return_percentile(close, period=20)` : percentile du log-return des 20 derniers jours
  - `vol_regime(close, lookback=60)` : terciles low/med/high de la vol réalisée (catégoriel encodé 0/1/2)
  - `range_atr_ratio(df, period=20)` : (H-L)/ATR moyen sur 20 barres
  - `body_to_range_ratio(df)` : |close-open| / (high-low)
  - `gap_overnight(df)` : (open - close.shift(1)) / close.shift(1)
  - `consecutive_up(close)` : nombre de jours consécutifs de hausse (vectorisé)
- [ ] Tests anti-look-ahead pour chacune.
- [ ] `scripts/run_h11_bis_features_advanced.py` réentraîne le RF méta-labeling avec features étendues, compare à H11 baseline.
- [ ] Conserver les features qui apportent ≥ +0.1 Sharpe CPCV. Mettre à jour la config retenue dans `JOURNAL.md`.
- [ ] Rapport `docs/v3_hypothesis_11_bis.md`.

## NE PAS FAIRE
- Ne PAS ajouter > 10 features supplémentaires (risque d'overfit + curse of dimensionality avec RF).
- Ne PAS supprimer une feature de H11 sans test.
- Ne PAS optimiser le seuil RF sur val.

## Étapes

### Étape 1 — Implémenter les features
```python
def atr_zscore(df: pd.DataFrame, period: int = 14, lookback: int = 200) -> pd.Series:
    atr_val = atr(df, period)
    return (atr_val - atr_val.rolling(lookback).mean()) / atr_val.rolling(lookback).std()


def return_percentile(close: pd.Series, period: int = 20) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).rank(pct=True)


def consecutive_up(close: pd.Series) -> pd.Series:
    up = (close.diff() > 0).astype(int)
    return up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
```

### Étape 2 — Réentraîner H11 avec features étendues
```python
features_extended = pd.concat([features_h11, advanced_features], axis=1)
# CPCV
sharpe_mean_new, sharpe_std_new = run_cpcv(features_extended, ...)
sharpe_mean_old = ... # depuis JOURNAL.md H11
```

### Étape 3 — Sélection feature par feature (ablation)
Pour chaque feature avancée ajoutée, faire un test ablation : retirer cette seule feature, voir si Sharpe baisse. Si baisse ≥ 0.1, garder. Sinon, supprimer.

## Critères go/no-go
- **GO prompt 13** si : la config finale (H11 + features avancées sélectionnées) reste documentée.
- **NO-GO partiel** : aucune feature avancée n'apporte. Conserver H11 baseline.
