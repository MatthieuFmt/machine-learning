# Pivot v4 — Audit infrastructure + 4 hypothèses ciblées

> **À LIRE EN PREMIER avant tout fichier de ce dossier.**

## Contexte

À la date du pivot v4, l'état du projet est :

- **22 n_trials consommés**, dont 0 GO en v3 (H06 Donchian multi-actif + H07 4 stratégies alt + H1/H5 pivot post-Phase 2).
- **DD impossiblement élevés** dans tous les rapports v3 : 362 %, 4 829 %, 411 %, 189 %, 92.8 %. Ces valeurs sont **mathématiquement impossibles** sur un compte réel à risque 2 %.
- **Coûts surestimés** dans `app/config/instruments.py` : facteur × 3 à × 80 selon l'actif vs les vrais spreads XTB publics.
- **Le méta-labeling v2 (H05, Sharpe walk-forward +8.84)** n'a **jamais été retesté** correctement avec coûts XTB réels.

## Diagnostic synthétique

| Bug | Localisation | Effet | Sévérité |
|---|---|---|---|
| **Sizing au risque non implémenté** | [app/backtest/metrics.py:181](../../app/backtest/metrics.py#L181) | DD calculé sur PnL brut en pips, pas sur capital en € avec position sizing 2 % → DD > 100 % possible | 🔴 Critique |
| **Coûts XTB sur-estimés** | [app/config/instruments.py:266-337](../../app/config/instruments.py#L266) | US30 8 pts (vrai 1.5), GER30 5 (vrai 1.2), XAUUSD 35 (vrai 0.35), XAGUSD 45 (vrai 0.025) | 🔴 Critique |
| **Confusion pip_size / unité monétaire** | [app/config/instruments.py:302-313](../../app/config/instruments.py#L302) | XAUUSD spread=25 avec pip_size=1.0 = 25 USD au lieu de 0.30 USD | 🔴 Critique |
| **Sharpe daily biaisé** | [app/backtest/metrics.py:81](../../app/backtest/metrics.py#L81) | Sur Donchian D1 (≤ 90 trades/an), 80 % des jours = pct_change(0) → Sharpe écrasé vers 0 | 🟠 Important |

## Conclusion

**Tous les NO-GO v3 (H06, H07, H1, H5) sont à considérer comme INVALIDES**, faute de simulateur fiable. On ne sait actuellement pas si Donchian US30 D1 a un edge ou non — le simulateur ne nous le dit pas correctement.

Le test set 2024+ a été lu pour ces hypothèses → **elles sont brûlées en OOS pour toujours**. On ne peut plus les re-tester sur le test set. Mais on peut :
1. **Auditer + corriger** le simulateur (0 n_trial).
2. **Rejouer train+val** pour observer si l'edge réapparaît avec correction (0 n_trial — c'est de l'info, pas une décision GO/NO-GO).
3. **Tester de NOUVELLES hypothèses** non encore vues du test set, avec simulateur corrigé.

## Stratégie du pivot v4

**Principe directeur** : on arrête de brûler des n_trials sur un simulateur cassé. On corrige d'abord (A1-A4), puis on construit un pipeline ML précis sur train uniquement (A5-A9), puis on teste sur OOS (B1-B4).

> **Insight critique** : la méthodologie correcte selon López de Prado est de fixer **tout** (features, modèle, hyperparams, seuil) sur train AVANT toute lecture du test set. C'est ce que A5-A9 font. Une fois ce pipeline gelé, B1-B4 deviennent de vrais tests OOS au sens DSR.

### Phase A — Audit / Correction simulateur (A1-A4)

| Fichier | Phase | Effort | n_trials | But |
|---|---|---|---|---|
| [01_audit_simulator.md](01_audit_simulator.md) | A1 | 0.5 j | 0 | Corriger sizing au risque 2 %, recalcul DD/Sharpe en € |
| [02_calibration_costs.md](02_calibration_costs.md) | A2 | 0.5 j | 0 | Coûts XTB réels (factor × 3-80 à diviser) |
| [03_sharpe_low_frequency.md](03_sharpe_low_frequency.md) | A3 | 0.3 j | 0 | Fix Sharpe pour stratégies < 100 trades/an |
| [04_replay_h06_h07.md](04_replay_h06_h07.md) | A4 | 0.5 j | 0 | Replay H06/H07 sur train+val (test set interdit) |

### Phase A étendue — Construction du pipeline ML précis (A5-A9, train UNIQUEMENT)

| Fichier | Phase | Effort | n_trials | But |
|---|---|---|---|---|
| [A5_feature_generation.md](A5_feature_generation.md) | A5 | 1 j | 0 | Construire un superset de 50+ features (tech + éco + sessions + régime + price action + cross-asset) |
| [A6_feature_ranking.md](A6_feature_ranking.md) | A6 | 0.5 j | 0 | Ranking train uniquement (mutual info + permutation + bootstrap stability) — top 15 figé par actif |
| [A7_model_selection.md](A7_model_selection.md) | A7 | 1 j | 0 | Comparer RF / HistGBM / Stacking via CPCV train uniquement — modèle figé |
| [A8_hyperparameter_tuning.md](A8_hyperparameter_tuning.md) | A8 | 0.5 j | 0 | Nested CPCV pour hyperparams du modèle retenu — hyperparams figés |
| [A9_pipeline_lock.md](A9_pipeline_lock.md) | A9 | 0.3 j | 0 | Geler tout dans `app/config/ml_pipeline_v4.py` — pipeline immutable pour B1-B4 |

**Total Phase A : 5.1 jours, 0 n_trial. Le test set 2024+ N'EST PAS LU.**

### Phase B — Hypothèses OOS avec pipeline gelé

| Fichier | Phase | Effort | n_trials | Priorité | Précondition |
|---|---|---|---|---|---|
| [05_h_new1_meta_us30.md](05_h_new1_meta_us30.md) | B1 | 1 j | +1 (23) | 🔴 P0 | Phase A complète (A1-A9) |
| [06_h_new3_eurusd_h4_meanrev.md](06_h_new3_eurusd_h4_meanrev.md) | B2 | 1.5 j | +1 (24) | 🟠 P1 | Phase A complète |
| [07_h_new2_walk_forward_rolling.md](07_h_new2_walk_forward_rolling.md) | B3 | 1.5 j | +1 (25) | 🟡 P2 | Conditionnelle : B1 NO-GO |
| [08_h_new4_portfolio.md](08_h_new4_portfolio.md) | B4 | 1 j | +1 (26) | 🟡 P3 | Conditionnelle : ≥ 2 GO en B1/B2/B3 |

**Total max : n_trials = 26 (au lieu de 28 dans le plan initial — économie de 2 trials grâce au pipeline gelé), effort ~11 jours.**

### Pourquoi A5-A9 ne consomment pas de n_trial

DSR pénalise le nombre d'**hypothèses testées sur le test set OOS**. Phase A se passe **uniquement sur train ≤ 2022** :
- Ranking de features : analyse statistique train, pas une décision GO/NO-GO.
- Sélection de modèle : comparaison sur train CPCV, pas d'OOS.
- Hyperparam tuning : nested CPCV train, pas d'OOS.

Une fois le pipeline gelé en A9, le **vrai** test statistique est B1 qui lit le test set UNE seule fois → 1 n_trial unique.

C'est la méthodologie "Sanctity of the Test Set" de Bailey & López de Prado (2014) : tout le tuning sur train, une seule lecture OOS.

## Ordre d'exécution strict

```
A1 → A2 → A3 → A4 → A5 → A6 → A7 → A8 → A9 → B1 → B2 → [B3 si B1+B2 NO-GO] → [B4 si ≥ 2 GO]
```

**Aucun raccourci**. Si A1-A9 ne sont pas tous ✅ Terminés dans `JOURNAL.md`, ne PAS commencer B1.

**Verrouillage progressif** :
- Après A6 : la liste des features est FIGÉE. Toute modification post-A6 = data snooping.
- Après A7 : le modèle ML est FIGÉ. Pas de "essayons XGBoost en plus" sans nouveau n_trial.
- Après A9 : tout le pipeline est immutable. B1-B4 utilisent ce pipeline tel quel.

## Critères d'arrêt définitifs

Le projet **s'arrête** (pivot paradigme ou abandon) si :
- B1 (méta-labeling US30 D1) est NO-GO **ET**
- B2 (EURUSD H4 mean-rev) est NO-GO **ET**
- B3 (walk-forward rolling) est NO-GO

Dans ce cas, conclure : "Les CFD XTB D1/H4 n'ont pas d'edge exploitable par stratégies simples + méta-labeling RF. Pivoter vers options, market-making, ou changer de broker."

## Garde-fous (rappels de la constitution)

- ⚠️ **TEST_SET_LOCK.json reste verrouillé** sur H06/H07/H1/H5/RSI(2) — ces hypothèses sont brûlées.
- ⚠️ La Phase A est uniquement infrastructure → **PAS de modification de stratégie en réaction**.
- ⚠️ La Phase B utilise `read_oos()` à chaque évaluation OOS.
- ⚠️ Pour chaque hypothèse B, le tableau n_trials dans `JOURNAL.md` doit être mis à jour AVANT l'exécution.

## Structure standardisée de chaque fichier suivant

Chaque prompt de ce dossier suit ce template :

```markdown
# Pivot v4 — XX (titre)

## Préalable obligatoire
- Liste numérotée des documents/fichiers à lire dans l'ordre.

## Objectif
- Une phrase mesurable.

## Type d'opération
- Bug fix infrastructure / Audit / Nouvelle hypothèse OOS

## Definition of Done (testable)
- Checklist concrète, avec commandes shell.

## NE PAS FAIRE
- Liste explicite des interdits spécifiques à ce prompt.

## Étapes détaillées
- Numérotées, avec code source / pseudo-code / commandes.

## Tests unitaires associés
- Pour chaque étape, le test qui prouve qu'elle marche.

## Logging obligatoire
- Format exact à ajouter dans JOURNAL.md.

## Critères go/no-go
- Quand passer au prompt suivant vs revenir en arrière.

## Annexes
- Spécifications techniques, formules, références externes.
```

## Vue d'ensemble graphique

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE A — Tout sur train ≤ 2022                       │
│                   0 n_trial consommé                                    │
│                   Test set 2024+ JAMAIS lu                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  A1 ─→ A2 ─→ A3 ─→ A4    │    A5 ─→ A6 ─→ A7 ─→ A8 ─→ A9               │
│  (audit sim. + coûts)    │    (pipeline ML construit sur train)        │
│       ↓                          ↓                                      │
│  simulateur fiable               pipeline ML gelé (features + modèle    │
│  + Sharpe routing                + hyperparams + seuil de calibration)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE B — Tests OOS sur pipeline gelé                      │
│              1 n_trial par hypothèse                                    │
│              Test set 2024+ lu UNE seule fois par actif                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│              ┌────────────────────────┐                                 │
│              │ B1 H_new1 US30 D1      │  (méta-labeling RF gelé A7-A9)  │
│              │ +1 n_trial → 23        │                                 │
│              └────────────────────────┘                                 │
│                          │                                              │
│                          ▼                                              │
│              ┌────────────────────────┐                                 │
│              │ B2 H_new3 EURUSD H4    │  (méta-labeling RF gelé A7-A9)  │
│              │ +1 n_trial → 24        │                                 │
│              └────────────────────────┘                                 │
│                          │                                              │
│         ┌────────────────┼────────────────┐                             │
│         │ B1+B2 NO-GO    │                │ B1 OU B2 GO                 │
│         ▼                                  ▼                            │
│  ┌────────────────┐                ┌────────────────┐                   │
│  │ B3 WF rolling  │                │ B4 portfolio   │                   │
│  │ +1 n_trial 25  │                │ (si ≥ 2 GO)    │                   │
│  └────────────────┘                │ +1 n_trial 26  │                   │
│         │                          └────────────────┘                   │
│         ▼                                  │                            │
│  ┌─────────────────┐                       │                            │
│  │ Tout NO-GO      │                       ▼                            │
│  │ → Plan abandon  │              ┌────────────────┐                    │
│  │ (cf. A5 prompt) │              │ GO production  │                    │
│  └─────────────────┘              │ → prompt 20    │                    │
│                                    └────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Suivant** : [01_audit_simulator.md](01_audit_simulator.md)
