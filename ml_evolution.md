# Historique des Améliorations — Pipeline EURUSD H1

## Date/Version: 1
- **Modification** : Application effective du filtrage `features_dropped` dans `build_ml_ready()` (bug: défini dans `EurUsdConfig` mais jamais appliqué). Ajout de `Range_ATR_ratio` à la liste d'exclusion. Réduction de 19 → 7 features d'entraînement (gardées: RSI_14, ADX_14, Dist_EMA_50, RSI_14_D1, Dist_EMA_20_D1, RSI_D1_delta, Dist_SMA200_D1).
- **Hypothèse** : La réduction drastique de dimensionalité (14 features bruit exclues) diminue l'overfitting en forçant le modèle à se concentrer sur les 5 features significatives identifiées par permutation importance. Les features de régime (ATR_Norm, Volatilite_Realisee_24h, Range_ATR_ratio) restent disponibles pour les filtres backtest mais ne polluent plus l'entraînement.
- **Target Metrics** : Δ OOB−OOS ≤ 0.05, Win Rate OOS ≥ 40%, Trades/an ≥ 100, Sharpe OOS ≥ 0.
