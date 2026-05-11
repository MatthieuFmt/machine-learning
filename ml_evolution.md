# Historique des Améliorations — Pipeline EURUSD H1

## Date/Version: 1 — 2026-05-11
- **Modification** : Application effective du filtrage `features_dropped` dans `build_ml_ready()` (bug: défini dans `EurUsdConfig` mais jamais appliqué). Ajout de `Range_ATR_ratio` à la liste d'exclusion. Réduction de 19 → 7 features d'entraînement (gardées: RSI_14, ADX_14, Dist_EMA_50, RSI_14_D1, Dist_EMA_20_D1, RSI_D1_delta, Dist_SMA200_D1).
- **Hypothèse** : La réduction drastique de dimensionalité (15 features bruit exclues) diminue l'overfitting en forçant le modèle à se concentrer sur les 5 features significatives identifiées par permutation importance. Les features de régime (ATR_Norm, Volatilite_Realisee_24h, Range_ATR_ratio) restent disponibles pour les filtres backtest mais ne polluent plus l'entraînement.
- **Résultats (pré-fix — baseline)** :
  - 2024 OOS : 216 trades, WR 34.3%, Net −258.7 pips, Sharpe −1.19
  - 2025 OOS : 349 trades, WR 32.7%, Net −659.0 pips, Sharpe −2.62
  - Accuracy OOS 2025 : 0.332 (vs aléatoire 0.333)
  - Δ OOB−OOS : −0.110
  - Biais directionnel : 85% SHORT en marché haussier 2025
- **Résultats (post-fix — 2026-05-11)** :
  - Features actives : 7 (`Dist_EMA_50`, `RSI_14`, `ADX_14`, `Dist_EMA_50_H4`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`)
  - 2024 OOS : 521 trades (+141% vs baseline 216), WR 35.5%, Net −515.5 pips, Sharpe −1.74, Alpha +175.7 pips
  - 2025 OOS : 698 trades (+100% vs baseline 349), WR 32.2%, Net −1365.1 pips, Sharpe −3.41, Alpha −2755.6 pips
  - **Analyse** : ✅ Trades/an ≥ 100 atteint et dépassé (521/698). ❌ Sharpe, profit net, win rate dégradés. L'exclusion des features de régime (ATR_Norm, Volatilite_Realisee_24h, Range_ATR_ratio, Dist_SMA200_D1) a supprimé des signaux de régularisation implicite. `Dist_EMA_50_H4` (borderline dans l'audit) est resté actif et pollue probablement encore. Le biais directionnel SHORT persiste (85%→?).
- **Target Metrics pour v2** : Sharpe OOS ≥ −1.0, Win Rate ≥ 38%, Profit net OOS ≥ 0. Restaurer `Dist_SMA200_D1` comme feature d'entraînement ? Ajouter méta-labeling ?

## Date/Version: 2 — 2026-05-11
- **Modification** : Swap dans `EurUsdConfig.features_dropped` : retrait de `Dist_SMA200_D1`, ajout de `Dist_EMA_50_H4`. Dimensionalité inchangée (7 features actives). Les features d'entraînement deviennent : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`, `Dist_SMA200_D1`.
- **Hypothèse** : Le RandomForest n'avait **aucune information de tendance long-terme** dans ses features (`Dist_SMA200_D1` exclue, `TrendFilter` binary-only trop frustre). Le modèle shortait par défaut (biais historique EURUSD 2010-2023), désastreux en marché haussier 2025. `Dist_EMA_50_H4` était du bruit pur (permutation_mean < 0.007). `Dist_SMA200_D1` encode la tendance ~9 mois en continu, complémentaire au `TrendFilter`. Swap 1-pour-1 → pas de risque d'overfitting supplémentaire.
- **Résultats (pré-fix — v1 post-fix)** :
  - Features actives v1 : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `Dist_EMA_50_H4`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`
  - 2024 OOS : 521 trades, WR 35.5%, Net −515.5 pips, Sharpe −1.74, Alpha +175.7 pips
  - 2025 OOS : 698 trades, WR 32.2%, Net −1365.1 pips, Sharpe −3.41, Alpha −2755.6 pips
- **Résultats (post-fix — à confirmer après run pipeline)** :
  - Features actives v2 : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`, `Dist_SMA200_D1`
  - 2024 OOS : [À MESURER]
  - 2025 OOS : [À MESURER]
- **Target Metrics pour v3** : Sharpe OOS ≥ −1.0, Win Rate ≥ 38%, Réduction biais SHORT < 60%, Profit net OOS ≥ 0.
