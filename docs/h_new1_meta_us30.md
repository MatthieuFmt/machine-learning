# H_new1 — Méta-labeling RF Donchian US30 D1 (pivot v4 B1)

**Date** : 2026-05-16
**Type** : Méta-labeling avec walk-forward retrain 6M
**Actif** : US30 D1
**Stratégie baseline** : DonchianBreakout(N=20, M=20)
**Modèle** : RandomForestClassifier (n_estimators=200, max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=42)

---

## Méthode

Walk-forward avec réentraînement tous les 6 mois (1er janvier / 1er juillet) à partir de 2024-01-01 :

1. Backtest Donchian baseline sur train (≤ retrain_dt − 2j embargo)
2. Extraction des 7 features aux barres d'entrée des trades train
3. Construction de la cible binaire (Pips_Nets > 0 → 1)
4. Entraînement MetaLabelingRF sur train
5. Calibration du seuil sur train (candidates [0.45, 0.50, 0.55, 0.60], rétention ≥ 20%, fallback 0.50)
6. Application du méta-modèle sur le segment OOS
7. Backtest avec signaux filtrés

**Coûts appliqués** : spread=1.5, slippage=0.3, commission=0.0 (US30 XTB)
**Sizing** : 2% risque par trade, capital 10 000 €
**Features** : RSI_14, ADX_14, Dist_SMA_50, Dist_SMA_200, ATR_Norm_14, Log_Return_5, Signal_Donchian

---

## Résultats walk-forward OOS

| Métrique | Valeur |
|----------|--------|
| Sharpe (per_trade) | 0.82 |
| Sharpe (equity, annualisé) | 3.64 |
| DSR | NaN (< 30 obs) |
| Win Rate | 50.0% |
| Max Drawdown | 3.9% |
| Profit net | +578.4 pips (+1 154.72 €) |
| Nombre de trades | 12 |
| Trades/an | 7.2 |
| Buy & Hold | 34 304.85 pips |
| Alpha | −33 726.45 pips |

## Segments walk-forward

| Segment | n_train | n_oos | Sharpe OOS | WR OOS |
|---------|---------|-------|------------|--------|
| 2024-01 → 2024-06 | 377 | 0 | — | — |
| 2024-07 → 2024-12 | 395 | 1 | 0.00 | 100% |
| 2025-01 → 2025-06 | 416 | 7 | −1.98 | 42.9% |
| 2025-07 → 2025-12 | 438 | 2 | −15.87 | 50% |
| 2026-01 → 2026-05 | 455 | 2 | −2.08 | 50% |

## Critères GO/NO-GO

| # | Critère | Seuil | Réalisé | Verdict |
|---|---------|-------|---------|---------|
| 1 | Sharpe annualisé | ≥ 1.0 | 0.82 | ❌ |
| 2 | DSR p < 0.05 | p < 0.05 | NaN | ❌ |
| 3 | Drawdown max | < 15% | 3.9% | ✅ |
| 4 | Win rate | > 35% | 50% | ✅ |
| 5 | Trades/an | ≥ 30 | 7.2 | ❌ |

## Décision

**❌ NO-GO** — 3 critères sur 5 en échec.

Diagnostic :
- Le Sharpe per_trade (0.82) est insuffisant malgré un Sharpe equity (3.64) gonflé par la rareté des trades
- Seulement 12 trades sur ~2.5 ans de test → ~7 trades/an, largement en dessous du seuil de 30
- Le méta-modèle filtre trop agressivement (threshold=0.55) : la plupart des signaux Donchian sont rejetés
- Le DSR est NaN car < 30 observations, rendant tout test de significativité statistique impossible
- Le walk-forward montre une dégradation : Sharpe OOS négatif sur les 3 derniers segments

## Bugs corrigés pendant l'exécution

1. **UnicodeEncodeError CP1252** : `→` et `≥` dans les print() → remplacés par `->` et exécution avec `PYTHONIOENCODING=utf-8`
2. **Timezone mismatch** : `df.index` (tz-aware UTC) vs `test_start_ts` (tz-naive) → normalisation tz dans `pd.date_range()`
3. **Features OOS vides** : `feature_builder(df_oos)` ne contient que ~130 barres, insuffisant pour SMA 200 → passage de `df.loc[:segment_end]` complet puis filtrage `[retrain_dt:segment_end]`

## Prochaines étapes possibles

- Tester un seuil plus bas (0.45) pour augmenter le nombre de trades
- Remplacer SMA 200 par des features moins sensibles au lookback (ou utiliser des features déjà pré-calculées)
- Explorer ATR(20) Trailing Stop au lieu de TP/SL fixes pour US30
- Tester la même approche sur XAUUSD (fenêtre 12h au lieu de 5j, TP/SL adaptés)
