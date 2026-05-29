# EURUSD H4 Mean-Reversion + Méta-Labeling RF

> **Statut** : ✅ GO — Validé OOS 2024-01 → 2026-05  
> **Pipeline ML** : Frozen v4.0.0 — [`app/config/ml_pipeline_v4.py`](../app/config/ml_pipeline_v4.py)  
> **Date de gel** : 2026-05-16  
> **n_trial consommé** : 1 (H_new3) sur 28 cumulés

---

## 1. Résumé

Stratégie mean-reversion sur EURUSD H4 combinant un filtre RSI + Bollinger Bands pour la génération de signaux, avec un méta-modèle RandomForest qui reclassifie chaque trade (take/discard) pour filtrer les faux signaux.

---

## 2. Stratégie primaire — [`app/strategies/mean_reversion.py`](../app/strategies/mean_reversion.py)

| Paramètre | Valeur |
|-----------|--------|
| **Timeframe** | H4 (4 heures) |
| **Actif** | EURUSD |
| **Indicateur 1** | RSI(14), seuils 30 (sursolde achat) / 70 (survente vente) |
| **Indicateur 2** | Bollinger Bands(20, 2) |
| **Signal LONG** | RSI < 30 **ET** prix ≤ BB lower band |
| **Signal SHORT** | RSI > 70 **ET** prix ≥ BB upper band |
| **Anti-look-ahead** | Tous les indicateurs utilisent `.shift(1)` — pas de fuite temporelle |

---

## 3. Méta-modèle — [`app/models/meta_labeling.py`](../app/models/meta_labeling.py)

| Paramètre | Valeur |
|-----------|--------|
| **Algorithme** | RandomForest (sklearn) |
| **n_estimators** | 100 |
| **max_depth** | 6 |
| **min_samples_leaf** | 10 |
| **class_weight** | `balanced` |
| **random_state** | 42 |
| **Seuil de décision** | 0.55 (probabilité minimale pour prendre le trade) |
| **Features** | Top 15 EURUSD H4 ([`app/config/features_selected.py`](../app/config/features_selected.py)) |
| **Triple barrière** | TP/SL/Time — configuré dans le simulateur |
| **Entraînement** | Walk-forward 6 mois, fenêtre glissante, embargo 2 jours |

---

## 4. Top 15 Features EURUSD H4

Issues du ranking bootstrap A6 avec stability (voir [`docs/feature_ranking_v4.md`](../docs/feature_ranking_v4.md)) :

| # | Feature | Catégorie | Stability |
|---|---------|-----------|-----------|
| 1 | `bb_width_20` | Volatility | 0.8 |
| 2 | `kc_width_20` | Volatility | — |
| 3 | `atr_14` | Volatility | — |
| 4 | `adx_14` | Trend | — |
| 5 | `dist_sma_200` | Trend | — |
| 6 | `dist_sma_50` | Trend | — |
| 7 | `ema_12` | Trend | — |
| 8 | `rsi_14` | Oscillator | — |
| 9 | `stoch_k_14` | Oscillator | — |
| 10 | `macd_hist` | Momentum | — |
| 11 | `auto_corr_5` | Statistical | — |
| 12 | `usdchf_ret_1d` | Cross-asset | — |
| 13 | `btcusd_ret_1d` | Cross-asset | — |
| 14 | `xauusd_ret_1d` | Cross-asset | — |
| 15 | `gap_overnight` | Price Action | — |

> **Note** : Les features Economic (9), Sessions (4), cycliques jour (2), Vol Regime (3) et patterns chandeliers rares (3) sont systématiquement exclues (stability 0.0 en A6).

---

## 5. Gestion du risque

| Paramètre | Valeur |
|-----------|--------|
| **Risk per trade** | 2% du capital |
| **Max drawdown cible** | < 15% |
| **Coût total par trade** | 0.9 pip (spread XTB Standard + slippage modélisé) |
| **Sizing** | Dynamique — 2% du capital / distance au stop |
| **Levier max** | À définir selon compte |

---

## 6. Performances OOS (2024-01 → 2026-05)

| Métrique | Valeur | Seuil GO |
|----------|--------|----------|
| **Sharpe per-trade** | +1.73 | ≥ 1.0 |
| **Sharpe annualisé** | +5.39 (validate_edge) | ≥ 1.0 |
| **DSR (Deflated Sharpe Ratio)** | +23.41 | > 0 |
| **p-value (Sharpe > 0)** | 0.000 | < 0.05 |
| **Max drawdown** | 8.1% | < 15% |
| **Win rate** | 53.7% | > 30% |
| **Trades OOS** | 54 (25.2/an) | ≥ 25 (H4) |
| **Profit net** | +56.3% equity | — |
| **Période** | 26 mois | — |

### Walk-forward segments

| Segment | Train | Test | Trades | Sharpe |
|---------|-------|------|--------|--------|
| 2024-01 → 2024-06 | ≤ 2023-12 | 2024 H1 | ~11 | — |
| 2024-07 → 2024-12 | ≤ 2024-06 | 2024 H2 | ~11 | — |
| 2025-01 → 2025-06 | ≤ 2024-12 | 2025 H1 | ~11 | — |
| 2025-07 → 2025-12 | ≤ 2025-06 | 2025 H2 | ~11 | — |
| 2026-01 → 2026-05 | ≤ 2025-12 | 2026 YTD | ~10 | — |

---

## 7. Déploiement — Checklist

### Prérequis techniques
- [ ] Python 3.10+ avec dépendances [`requirements.txt`](../requirements.txt)
- [ ] Accès aux données EURUSD H4 (courtier ou data provider)
- [ ] Compte XTB Standard (spread 0.9 pip calibré) ou équivalent
- [ ] VPS ou machine allumée 24/7 (timeframe H4, vérification toutes les 4h)

### Fichiers à déployer
```
app/
├── config/
│   ├── ml_pipeline_v4.py      ← Pipeline frozen (ne pas modifier)
│   ├── features_selected.py   ← Top 15 features
│   ├── hyperparams_tuned.py   ← Hyperparams RF
│   ├── model_selected.py      ← rf
│   └── instruments.py         ← Coûts calibrés v4
├── features/
│   ├── indicators.py          ← Fonctions d'indicateurs
│   ├── superset.py            ← build_superset()
│   └── ranking.py             ← Optionnel (déjà exécuté)
├── models/
│   ├── meta_labeling.py       ← MetaLabelingRF
│   └── build.py               ← build_locked_model()
├── strategies/
│   └── mean_reversion.py      ← MeanReversionRSIBB
├── backtest/
│   ├── simulator.py           ← Simulateur stateful
│   ├── sizing.py              ← Sizing 2%
│   └── metrics.py             ← Sharpe routing
└── testing/
    └── look_ahead_validator.py
```

### Variables d'environnement
```bash
PYTHONIOENCODING=utf-8
```

### Vérifications avant mise en prod
- [ ] `rtk make verify` — ruff + mypy + pytest (tous verts)
- [ ] `pytest tests/integration/test_pipeline_integrity.py` — 6/6 SHA256 checksums OK
- [ ] `pytest tests/unit/test_mean_reversion_rsi_bb.py` — 6/6
- [ ] `pytest tests/unit/test_meta_labeling_rf.py` — 5/5
- [ ] `pytest tests/unit/test_sharpe_routing.py` — 6/6
- [ ] `pytest tests/unit/test_instruments_costs.py` — 53/53

---

## 8. Monitoring — Métriques à suivre en production

| Métrique | Seuil alerte | Action si dépassé |
|----------|-------------|-------------------|
| Sharpe glissant 20 trades | < 0.5 | Review manuelle, possible pause |
| Drawdown courant | > 10% | Réduire sizing à 1% |
| Win rate glissant 20 trades | < 35% | Vérifier régime de marché |
| Trades par mois | < 1 ou > 6 | Anomalie — vérifier données |
| Slippage réalisé vs modélisé | > 2× modélisé | Recalibrer coûts |
| Délai signal → exécution | > 5 secondes | Problème infrastructure |

---

## 9. Limites connues et risques

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| **Dégradation du mean-reversion** en régime tendanciel fort | Moyenne | Élevé | Le méta-modèle RF filtre les faux signaux ; stop loss intégré |
| **Faible fréquence** (~2 trades/mois) | Élevé (structurel) | Moyen | Accepté — le Sharpe compense. Ne pas forcer plus de trades. |
| **Overfitting du méta-modèle** sur train ≤ 2022 | Faible | Élevé | Walk-forward 6M avec retrain ; test OOS 26 mois clean |
| **Changement de régime macro** (ex: guerre, crise EUR) | Faible | Très élevé | Stop manuel si DD > 15% ; réévaluer le pipeline |
| **Spread XTB variable** (news, rollover) | Moyenne | Faible | Le spread calibré 0.9 pip inclut une marge ; surveiller slippage réel |

---

## 10. Références croisées

| Document | Lien |
|----------|------|
| Rapport B2 complet | [`docs/h_new3_eurusd_h4.md`](../docs/h_new3_eurusd_h4.md) |
| Pipeline gelé | [`app/config/ml_pipeline_v4.py`](../app/config/ml_pipeline_v4.py) |
| Feature ranking A6 | [`docs/feature_ranking_v4.md`](../docs/feature_ranking_v4.md) |
| Coûts calibrés A2 | [`docs/cost_audit_v2.md`](../docs/cost_audit_v2.md) |
| JOURNAL | [`JOURNAL.md`](../JOURNAL.md) |
| Constitution | [`prompts/00_constitution.md`](../prompts/00_constitution.md) |

---

> **Dernière mise à jour** : 2026-05-16 — Pivot v4 Phase B terminé  
> **Prochaine étape** : Phase 5 — Validation finale → Phase 6 — Déploiement VPS
