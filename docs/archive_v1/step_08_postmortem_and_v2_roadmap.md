# Step 08 — Post-Mortem & Roadmap v2

**Date** : 2026-05-13
**Statut** : 🔴 NO-GO final — Fin de la roadmap v1
**Décision** : Abandon de l'approche actuelle (EURUSD H1 + RF + indicateurs techniques). Préparation d'une v2 avec changement structurel.

---

## 1. Synthèse des 7 steps — Tableau de bord

| # | Step | Verdict | EURUSD Sharpe 2024 | EURUSD Sharpe 2025 | BTCUSD Sharpe 2024 | BTCUSD Sharpe 2025 |
|---|------|---------|---------------------|---------------------|---------------------|---------------------|
| Baseline v15 | RF 500 arbres, 7 features | — | −1.19 | −2.62 | — | — |
| [01](step_01_final_report.md) | Redéfinition cible (4 modes) | ❌ NO-GO | −0.72 à −2.74 | −0.11 à −2.71 | — | — |
| [02](step_02_final_report.md) | CPCV + DSR (200 splits) | ❌ NO-GO | — | −0.14 (E[Sharpe]) | — | — |
| [03](step_03_gbm_primary_classifier.md) | LightGBM/XGBoost + Optuna | ⏭️ Bloqué | — | — | — | — |
| [04](step_04_final_report.md) | Features de session | ❌ NO-GO | **+3.07** | **−0.94** | — | — |
| [05](step_05_implementation_spec.md) | Calendrier économique | ❌ NO-GO | — | **−3.15** | — | — |
| [06](step_06_meta_labeling_calibration.md) | Calibration méta-labeling | ⏭️ Partiel | — | — | — | — |
| [07](step_07_cross_asset_validation.md) | Validation BTCUSD | ❌ NO-GO | — | — | **−2.41** | **−1.21** |

**Bilan** : 5 steps exécutés, 5 NO-GO. Aucun Sharpe positif simultané sur train/val/test. Aucun Sharpe positif sur BTCUSD. L'edge n'existe pas.

---

## 2. Analyse des causes racines de l'overfit

### 2.1 Cause racine #1 — Absence d'information prédictive dans les features techniques classiques (poids : 50%)

Les 7 à 20 features utilisées sont exclusivement des **indicateurs techniques classiques** (RSI, EMA, ADX, Bollinger, ATR). Ces indicateurs sont :

- **Lagging** : ils dérivent mathématiquement du prix passé. La distance à l'EMA(t) est une fonction déterministe de Close(t) — il n'y a pas d'information nouvelle.
- **Décorrélés de la cible forward** : le `forward_return` H1 24h est un bruit blanc pour les indicateurs techniques. L'accuracy OOS = 0.332 ≈ aléatoire 3-classes (0.333) le confirme.
- **Redondants** : 3 distances EMA, 3 RSI sur timeframes différentes → forte colinéarité (VIF élevé), aucune diversité informationnelle.

**Preuve empirique** : le CPCV Step 02 montre une distribution de Sharpe entièrement négative sur 200 splits. Le modèle est *consistamment* perdant, pas *aléatoirement* perdant. Il apprend du bruit et généralise négativement.

### 2.2 Cause racine #2 — Data snooping sur 15+ itérations (poids : 30%)

L'historique [`ml_evolution.md`](../ml_evolution.md) documente 15 versions successives, chacune modifiant `features_dropped`, les hyperparamètres, ou les filtres en réaction aux résultats OOS précédents.

Mécanisme classique de **data snooping** (Harvey, Liu & Zhu, 2016) :

1. Itération n : résultat 2025 mauvais → on ajuste un paramètre
2. Itération n+1 : résultat 2025 meilleur → on garde le paramètre
3. En réalité : le paramètre a été sélectionné parce qu'il fonctionnait sur 2025 → ce n'est plus un vrai OOS

Le split temporel (train ≤ 2023, val = 2024, test = 2025) n'a **jamais changé** entre les itérations. Chaque modification a été validée rétrospectivement contre le même test set 2025. C'est du **pseudo-OOS** : on croit tester hors échantillon, mais le test set a déjà été « vu » 15 fois.

### 2.3 Cause racine #3 — Le H1 est un timeframe défavorable au ML supervisé classique (poids : 15%)

Le H1 forex est dominé par :

- **Bruit microstructurel** : le spread typique EURUSD (0.1-0.3 pips) + slippage (0.5 pips) représente une friction de ~0.6 pips par trade, soit 6% du TP=10 pips original. Le ratio signal/bruit est structurellement faible.
- **Régimes non stationnaires** : les relations entre indicateurs techniques et rendements futurs changent selon les politiques monétaires (QE, hausse des taux). Ce qui fonctionne en 2015-2021 (QE massif) échoue en 2022-2025 (resserrement).
- **Efficience intraday** : le FX est le marché le plus liquide au monde. L'information publique est incorporée quasi-instantanément.

Les timeframes supérieurs (H4, D1) ont un meilleur ratio signal/bruit car la friction relative diminue et les fondamentaux macro ont plus de temps pour se refléter dans le prix.

### 2.4 Cause racine #4 — La cible triple_barrier asymétrique (poids : 5%)

Avec TP=30/SL=10 (ratio 3:1), la probabilité de toucher le TP avant le SL dans un mouvement brownien sans drift est de 25% (pas 50%). La cible est intrinsèquement déséquilibrée, ce qui rend la classification plus difficile — le modèle doit identifier des setups avec un avantage directionnel suffisant pour compenser l'asymétrie.

---

## 3. Leçons techniques

| Aspect | Constat |
|--------|---------|
| Framework CPCV | ✅ Fonctionnel, rapide (200 splits en 56s), verdict sans ambiguïté |
| Modularité du pipeline | ✅ BasePipeline → EurUsdPipeline → BtcUsdPipeline : extension simple |
| Gestion anti-leak | ✅ `train_end` pour le SessionVolatilityScaler, `merge_asof backward` pour calendrier |
| Tests unitaires | ✅ 300+ tests passent, zéro régression |
| Cross-asset sans tuning | ✅ Step 07 a respecté la contrainte `features_dropped` identique |
| **Edge réel** | ❌ N'existe pas avec cette approche sur ces instruments/timeframes |

---

## 4. Plan de nettoyage pour préparer la v2

### 4.1 Fichiers à archiver (déplacer dans `archive_v1/`)

```
archive_v1/
├── predictions/
│   ├── metrics_v1_2024.json
│   ├── metrics_v1_2025.json
│   ├── metrics_wf_v14_*.json          (9 fichiers walk-forward)
│   ├── btcusd_metrics_2024.json
│   ├── btcusd_metrics_2025.json
│   ├── edge_validation_wf_v14.json
│   ├── step_01_integration.json
│   ├── cost_aware_comparison_v15.json
│   ├── cpcv_report.md
│   ├── Rapport_Performance_*.md       (5 fichiers)
│   ├── Analyse_et_Pistes_Amelioration.md
│   └── ratio_3_1/
├── run_pipeline_btcusd.py
├── run_step_01_integration.py
├── run_validation_cpcv.py
├── run_pipeline_walk_forward.py
└── compare_cost_aware.py
```

### 4.2 Fichiers à conserver (infrastructure réutilisable)

```
learning_machine_learning/
├── core/                    # exceptions, logging, types → réutilisable
├── config/
│   ├── instruments.py       # structure InstrumentConfig → réutilisable
│   ├── backtest.py          # BacktestConfig → réutilisable
│   └── paths.py             # gestion des chemins → réutilisable
├── data/
│   ├── loader.py            # load_all_timeframes → réutilisable
│   └── calendar_loader.py   # calendrier économique → réutilisable
├── features/
│   ├── calendar.py          # features calendrier → réutilisable
│   └── regime.py            # sessions → réutilisable
├── backtest/
│   ├── simulator.py         # moteur de simulation → réutilisable
│   ├── filters.py           # MomentumFilter, VolFilter, SessionFilter, CalendarFilter
│   └── metrics.py           # calculs de Sharpe, drawdown → réutilisable
├── analysis/
│   ├── cpcv.py              # CPCV → réutilisable
│   └── edge_validation.py   # DSR, PSR → réutilisable
├── pipelines/
│   ├── base.py              # BasePipeline → réutilisable (template)
│   ├── eurusd.py            # à archiver ou à refondre
│   └── btcusd.py            # à archiver ou à refondre
└── model/
    └── training.py          # à refondre
```

### 4.3 Fichiers à supprimer (obsolètes, ne serviront plus)

- `run_pipeline_v1.py` — supprimer (remplacé par les scripts step par step)
- `scripts/_debug_ff.py` — supprimer (debug)
- `scripts/inspect_btc_csv.py` — archiver (utile pour référence d'ingestion)

### 4.4 Structure cible v2

```
learning_machine_learning_v2/
├── core/                    # réimporté de v1, nettoyé
├── config/
│   ├── instruments.py       # nouveaux instruments (indices, H4/D1, multi-actifs)
│   └── backtest.py
├── data/                    # ingestion multi-sources
├── features/
│   ├── alternative/         # nouvelles sources : order flow, COT, sentiment
│   └── transforms/          # transformations non-linéaires, PCA sparse
├── targets/                 # nouvelles cibles (régression, survival analysis)
├── models/                  # architecture propre, config-driven
├── backtest/                # réimporté de v1
├── analysis/                # réimporté de v1
└── pipelines/               # nouveaux pipelines par actif/stratégie
```

---

## 5. Roadmap v2 — Pistes à explorer

### 5.1 Changement de timeframe (priorité 🔴 Haute)

| Timeframe | Avantage | Risque |
|-----------|----------|--------|
| **H4** | Moins de bruit microstructurel, meilleur ratio signal/bruit. ~6 barres/jour × 10 ans = ~15k échantillons. | Moins de trades, nécessite TP/SL plus larges. |
| **D1** | Signal macro dominant, fondamentaux mieux capturés. | Très peu de trades (~250/an), surapprentissage facile. |
| **Multi-TF** | H1 pour l'entrée, H4/D1 pour le filtre de tendance. | Complexité accrue, risque de look-ahead dans le join. |

**Recommandation** : commencer par H4, qui offre le meilleur compromis entre taille d'échantillon et ratio signal/bruit.

### 5.2 Changement d'instruments (priorité 🔴 Haute)

| Classe | Instruments | Justification |
|--------|-------------|---------------|
| **Indices** | US30, GER30, US500 | Régimes de volatilité plus favorables au trend-following. Moins d'efficience que le FX. |
| **Matières premières** | XAUUSD, XAGUSD, USOIL | Corrélation modérée avec le dollar, drivers fondamentaux distincts. |
| **Crypto** | BTCUSD, ETHUSD (H4/D1) | Volatilité élevée = TP larges possibles. Marché moins efficient. |

**Recommandation** : US30 D1 + XAUUSD H4 en première priorité. BTCUSD écarté après Step 07 (Sharpe −1.21/−2.41).

### 5.3 Changement d'approche de modélisation (priorité 🟠 Moyenne)

| Approche | Description | Risque |
|----------|-------------|--------|
| **Règles déterministes** | Identifier des patterns statistiquement robustes sans ML (ex : saisonnalité horaire, effet jour de semaine). Backtester avec CPCV. | Peut ne rien trouver non plus. |
| **Cointégration / paires trading** | Identifier 2+ actifs cointégrés, trader le spread mean-reverting. | Nécessite un univers d'actifs plus large. |
| **Régression de rendements** | Prédire le rendement forward continu plutôt qu'une classe. Régression ridge/kernel avec features parcimonieuses. | Toujours dépendant de la qualité des features. |
| **Survival analysis** | Prédire le temps avant toucher TP/SL plutôt qu'une classe. Modèle de Cox ou Random Survival Forest. | Plus complexe, moins de littérature en trading. |

### 5.4 Protocole anti-snooping strict (priorité 🔴 Haute — non négociable)

**Règle 1 — Un seul regard sur le test set** : le split ≤ 2023 / 2024 / 2025 est brûlé pour EURUSD H1. Tout nouveau développement sur cet instrument/timeframe doit utiliser un **nouveau** split (ex : ≤ 2022 / 2023 / 2024, ou un autre provider de données).

**Règle 2 — Pré-enregistrement des hypothèses** : avant tout run OOS, documenter explicitement :
- La liste exacte des features
- Les hyperparamètres exacts
- La cible
- Le critère de succès/échec

Une fois le run OOS lancé, **aucune modification** n'est permise. Si échec → on passe à l'hypothèse suivante, on ne « corrige » pas.

**Règle 3 — Correction pour comparaisons multiples** : le DSR (Deflated Sharpe Ratio) doit systématiquement être calculé avec `n_trials = nombre d'hypothèses testées sur le même instrument/timeframe`. Pour la roadmap v1, n_trials = 15 (15 versions).

**Règle 4 — Un instrument de validation aveugle** : réserver un instrument « hold-out » (ex : USDCHF) qui n'est jamais utilisé pour le développement. Il n'est testé qu'une seule fois, à la fin de la roadmap v2, comme gate final.

---

## 6. Prochaines actions immédiates

| # | Action | Mode | Priorité |
|---|--------|------|----------|
| 1 | Créer `archive_v1/` et déplacer les fichiers listés en §4.1 | Code | P0 |
| 2 | Créer `learning_machine_learning_v2/` avec la structure §4.4 | Code | P0 |
| 3 | Réimporter `core/`, `config/`, `data/`, `backtest/`, `analysis/` depuis v1 | Code | P0 |
| 4 | Rédiger `docs/v2_hypothesis_01.md` — première hypothèse v2 (instrument, timeframe, features) | Architect | P1 |
| 5 | Ingestion données H4/D1 pour 3 nouveaux instruments candidats | Code | P1 |

---

## 7. Références

- Harvey, C. R., Liu, Y., & Zhu, H. (2016). *… and the Cross-Section of Expected Returns*. Review of Financial Studies.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). *Pseudo-Mathematics and Financial Charlatanism*. Notices of the AMS.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, chapitres 7 (Cross-Validation), 11 (Backtesting), 16 (Feature Importance).
