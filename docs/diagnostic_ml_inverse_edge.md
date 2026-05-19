# Diagnostic post-fix — Pourquoi le ML détruit-il l'edge Donchian ?

**Date** : 2026-05-19T06:33:52.907416+00:00
**Question** : après correction des bugs F1+F2+F3, Sharpe portfolio = -5.42.
Le ML inverse-t-il l'edge Donchian, ou Donchian lui-même ne marche plus ?

## Méthodologie

Pour chaque couple Donchian+ML :
1. Backtest Donchian SEUL sur 2024+ (baseline pure).
2. Backtest Donchian + filtre ML (config validation_finale).
3. Décomposition : pour chaque trade baseline, le ML l'a-t-il accepté ou rejeté ?
   Calcul du WR sur acceptés vs rejetés.

Si **WR_rejetés > WR_acceptés + 5pts** → 🔴 le ML inverse l'edge.

## Résultats

| Couple | Donchian seul | ML+Donchian | Acceptés WR | Rejetés WR | Inverse ? |
|---|---|---|---|---|---|
| GBPUSD D1 | Sharpe -4.31, WR 8.8%, n=114 | Sharpe -1.86, WR 9.1%, n=22 | 9.1% (n=22) | 8.7% (n=92) | non |
| EURUSD D1 | Sharpe -4.07, WR 9.7%, n=103 | Sharpe -2.83, WR 8.5%, n=47 | 8.5% (n=47) | 10.7% (n=56) | non |
| USDCHF D1 | Sharpe -2.35, WR 18.1%, n=94 | Sharpe -1.12, WR 19.0%, n=21 | 19.0% (n=21) | 17.8% (n=73) | non |
| ETHUSD D1 | Sharpe -0.58, WR 29.3%, n=99 | Sharpe +0.07, WR 36.4%, n=33 | 37.5% (n=32) | 25.4% (n=67) | non |

## Interprétation

**Donchian seul positif → ML+Donchian négatif** : le filtre ML aggrave les résultats.
C'est la signature d'un **modèle ML défaillant** entraîné sur des features qui
ne se généralisent pas du train (≤ 2022) au test (≥ 2024). Le ML rejette
systématiquement les trades qui *auraient gagné* dans le régime 2024-2026.

**Donchian seul négatif** : la stratégie Donchian elle-même ne fonctionne plus
dans le régime 2024-2026. Probable changement de régime de marché.

**Acceptés WR < Rejetés WR + 5pts** : le ML inverse l'edge directement. Le rejet
ML est plus prédictif d'un winner que l'acceptation.

## Pistes d'amélioration (si Donchian seul a un edge)

1. **Régulariser le ML plus fort** : max_depth=2, min_samples_leaf=50.
2. **Réduire le feature set** : utiliser seulement les 3-5 features les plus stables.
3. **Train plus récent** : remplacer 2010-2022 par 2018-2022 (5 ans plus proches).
4. **Walk-forward** : re-entraîner tous les 6 mois plutôt qu'un modèle figé.
5. **Calibration sur 2023** : utiliser le flag CALIBRATE_THRESHOLD_ON_VAL=True.

## Pistes (si Donchian seul ne marche plus)

1. Passer aux **nouvelles stratégies** (voir plan_v5_amelioration_strategies.md Axe B).
2. Tester **Donchian sur autres timeframes** (H4, H1).
3. Tester **autres paramètres Donchian** (N=10, 30, 50 au lieu de 20).