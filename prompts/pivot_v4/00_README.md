# Pivot v4 — Audit infrastructure + 4 hypothèses ciblées

> **À LIRE EN PREMIER avant tout fichier de ce dossier.**

## Carte des fichiers du dossier

Tous les fichiers suivent maintenant la convention uniforme `AX_*.md` (phase A) ou `BX_*.md` (phase B).

| Label | Fichier | Phase | Statut par défaut |
|---|---|---|---|
| **A1** | [A1_audit_simulator.md](A1_audit_simulator.md) | Infrastructure simulateur | ✅ déjà fait |
| **A5** | [A5_feature_generation.md](A5_feature_generation.md) | ML — features (★ priorité) | À faire en premier |
| **A6** | [A6_feature_ranking.md](A6_feature_ranking.md) | ML — ranking | Suit A5 |
| **A7** | [A7_model_selection.md](A7_model_selection.md) | ML — model selection | Suit A6 |
| **A8** | [A8_hyperparameter_tuning.md](A8_hyperparameter_tuning.md) | ML — hyperparams | Suit A7 |
| **A9** | [A9_pipeline_lock.md](A9_pipeline_lock.md) | ML — lock SHA256 | Suit A8, fige tout |
| **A2** | [A2_calibration_costs.md](A2_calibration_costs.md) | Simulateur — coûts XTB | Après A9 |
| **A3** | [A3_sharpe_low_frequency.md](A3_sharpe_low_frequency.md) | Simulateur — Sharpe routing | Après A2 |
| **A4** | [A4_replay_h06_h07.md](A4_replay_h06_h07.md) | Observation (replay) | 🟡 **OPTIONNEL** |
| **B1** | [B1_meta_us30.md](B1_meta_us30.md) | OOS — méta-labeling US30 | +1 n_trial |
| **B2** | [B2_eurusd_h4_meanrev.md](B2_eurusd_h4_meanrev.md) | OOS — EURUSD H4 | +1 n_trial |
| **B3** | [B3_walk_forward_rolling.md](B3_walk_forward_rolling.md) | OOS — WF rolling | Conditionnel |
| **B4** | [B4_portfolio.md](B4_portfolio.md) | OOS — portfolio multi-sleeves | Conditionnel |

> **Ordre canonique révisé** : A1 ✅ → **A5 → A6 → A7 → A8 → A9** → A2 → A3 → [A4 opt] → B1 → B2 → [B3] → [B4]
> Voir section "Ordre d'exécution strict — RÉVISÉ" plus bas pour les dépendances détaillées et les caveats méthodologiques.

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

**Principe directeur** : on arrête de brûler des n_trials sur un simulateur cassé. On corrige d'abord les bugs critiques (A1 ✅), puis on construit le pipeline ML sur train uniquement (A5-A9, priorité utilisateur), puis on finalise le simulateur (A2-A3), puis on teste sur OOS (B1-B4).

> **Insight critique** : la méthodologie correcte selon López de Prado est de fixer **tout** (features, modèle, hyperparams, seuil) sur train AVANT toute lecture du test set. C'est ce que A5-A9 font. Une fois ce pipeline gelé, B1-B4 deviennent de vrais tests OOS au sens DSR.

### Bloc 1 — Infrastructure simulateur critique (A1)

| Fichier | Phase | Effort | n_trials | Type | But |
|---|---|---|---|---|---|
| [A1_audit_simulator.md](A1_audit_simulator.md) | A1 | 0.5 j | 0 | ✅ FAIT | Corriger sizing au risque 2 %, recalcul DD/Sharpe en € — DD bornés [−100 %, 0 %] |

### Bloc 2 — Pipeline ML précis (A5-A9, train UNIQUEMENT, ★ priorité)

| Fichier | Phase | Effort | n_trials | Type | But |
|---|---|---|---|---|---|
| [A5_feature_generation.md](A5_feature_generation.md) | A5 | 1 j | 0 | 🔴 OBLIGATOIRE | Superset de 70+ features (tech + éco + sessions + régime + price action + cross-asset) |
| [A6_feature_ranking.md](A6_feature_ranking.md) | A6 | 0.5 j | 0 | 🔴 OBLIGATOIRE | Ranking train uniquement (mutual info + permutation + bootstrap stability) — top 15 figé par (asset, tf) |
| [A7_model_selection.md](A7_model_selection.md) | A7 | 1 j | 0 | 🔴 OBLIGATOIRE | Comparer RF / HistGBM / Stacking via CPCV train uniquement — modèle figé |
| [A8_hyperparameter_tuning.md](A8_hyperparameter_tuning.md) | A8 | 0.5 j | 0 | 🔴 OBLIGATOIRE | Nested CPCV pour hyperparams + seuil — figés |
| [A9_pipeline_lock.md](A9_pipeline_lock.md) | A9 | 0.3 j | 0 | 🔴 OBLIGATOIRE | Geler tout dans `app/config/ml_pipeline_v4.py` + SHA256 — pipeline immutable pour B1-B4 |

### Bloc 3 — Finition simulateur (A2-A3, après A9)

| Fichier | Phase | Effort | n_trials | Type | But |
|---|---|---|---|---|---|
| [A2_calibration_costs.md](A2_calibration_costs.md) | A2 | 0.5 j | 0 | 🔴 OBLIGATOIRE | Coûts XTB réels (facteur × 3-80 à diviser) — applicable au simulateur utilisé en B1 |
| [A3_sharpe_low_frequency.md](A3_sharpe_low_frequency.md) | A3 | 0.3 j | 0 | 🔴 OBLIGATOIRE | Fix Sharpe routing pour stratégies < 100 trades/an |
| [A4_replay_h06_h07.md](A4_replay_h06_h07.md) | A4 | 0.5 j | 0 | 🟡 OPTIONNEL | Replay H06/H07 sur train+val avec sim corrigé — observation uniquement |

**Total Phase A : 4.6 jours (sans A4) ou 5.1 jours (avec A4), 0 n_trial. Le test set 2024+ N'EST PAS LU.**

### Bloc 4 — Hypothèses OOS avec pipeline gelé + simulateur fiable (B1-B4)

| Fichier | Phase | Effort | n_trials | Priorité | Précondition |
|---|---|---|---|---|---|
| [B1_meta_us30.md](B1_meta_us30.md) | B1 | 1 j | +1 (23) | 🔴 P0 | A1 + A5-A9 + A2 + A3 ✅ |
| [B2_eurusd_h4_meanrev.md](B2_eurusd_h4_meanrev.md) | B2 | 1.5 j | +1 (24) | 🟠 P1 | Idem |
| [B3_walk_forward_rolling.md](B3_walk_forward_rolling.md) | B3 | 1.5 j | +1 (25) | 🟡 P2 | Conditionnelle : B1 NO-GO |
| [B4_portfolio.md](B4_portfolio.md) | B4 | 1 j | +1 (26) | 🟡 P3 | Conditionnelle : ≥ 2 GO en B1/B2/B3 |

**Total max : n_trials = 26 (au lieu de 28 dans le plan initial — économie de 2 trials grâce au pipeline gelé), effort ~11 jours.**

### Pourquoi A5-A9 ne consomment pas de n_trial

DSR pénalise le nombre d'**hypothèses testées sur le test set OOS**. Phase A se passe **uniquement sur train ≤ 2022** :
- Ranking de features : analyse statistique train, pas une décision GO/NO-GO.
- Sélection de modèle : comparaison sur train CPCV, pas d'OOS.
- Hyperparam tuning : nested CPCV train, pas d'OOS.

Une fois le pipeline gelé en A9, le **vrai** test statistique est B1 qui lit le test set UNE seule fois → 1 n_trial unique.

C'est la méthodologie "Sanctity of the Test Set" de Bailey & López de Prado (2014) : tout le tuning sur train, une seule lecture OOS.

## Ordre d'exécution strict — RÉVISÉ (mai 2026)

> **Pivot d'ordre** : à la demande de l'utilisateur, l'ordre original a été modifié pour **prioriser la construction du pipeline ML (A5-A9) AVANT la finition du simulateur (A2-A3)**. Voir section "Pourquoi ce pivot d'ordre" ci-dessous pour les implications méthodologiques.

### Ordre canonique (à suivre)

```
A1 ✅ → A5 → A6 → A7 → A8 → A9 → A2 → A3 → [A4 optionnel] → B1 → B2 → [B3] → [B4]
```

**Découpage** :
- **Bloc ML (A5-A9)** : features + ranking + modèle + hyperparams + lock — tout sur train, classification metrics
- **Bloc simulateur (A2-A3)** : coûts XTB + Sharpe routing — affine le simulateur AVANT les tests OOS
- **Bloc observationnel (A4)** : optionnel, replay H06/H07 avec sim corrigé
- **Bloc OOS (B1-B4)** : lit le test set avec pipeline gelé + simulateur corrigé

### Tableau de dépendances RÉVISÉ

| Phase | Dépend strictement de | Peut tourner sans (avec caveat) | Pourquoi |
|---|---|---|---|
| **A1** ✅ | — | — | Premier prompt, déjà fait |
| **A5** | A1 | A2/A3 | Pure feature engineering, pas de simulateur requis |
| **A6** | A5 | A2/A3 | Ranking par MI + permutation + Spearman — métriques classification, pas trading |
| **A7** | A6 | A2/A3 ⚠️ | Sélection RF/HGBM/Stacking — utilise Sharpe en CPCV (caveat : Sharpe biaisé tant que A2 pas fait, mais **ranking entre modèles est préservé** car le biais coûts est uniforme) |
| **A8** | A7 | A2/A3 ⚠️ | Tuning hyperparams — caveat similaire à A7. Le **seuil de calibration** peut être suboptimal post-A2, mais le modèle reste valide |
| **A9** | A8 | A2/A3 | Lock structurel (features + modèle + hyperparams + seuil) — fige le pipeline ML |
| **A2** | A1, A9 | — | Coûts XTB réels — applicable au simulateur utilisé en B1 |
| **A3** | A2 | — | Sharpe routing — applicable aux métriques finales en B1 |
| **A4** | A3 *(optionnel)* | — | Replay H06/H07 sur sim corrigé — observation uniquement |
| **B1+** | A9 + A2 + A3 | — | Lit le test set 2024+ avec pipeline gelé + simulateur corrigé |

### Pourquoi ce pivot d'ordre ?

**Avantage** : on construit le pipeline ML (A5-A9) sur des **métriques de classification** (F1, AUC, MI, permutation importance) qui sont **insensibles aux coûts de trading**. La sélection du modèle (RF vs HGBM vs Stacking) ne dépend pas de la précision des coûts.

**Caveat méthodologique** : le seuil de calibration choisi en A8 (ex: proba > 0.55) est optimisé sur Sharpe en CPCV. Si les coûts sont sur-estimés (cas actuel : facteur × 3-80), le seuil retenu sera **trop conservateur** (rejette trop de trades). Après A2 (coûts corrigés), le seuil pourrait gagner à être abaissé.

**Mitigation** : la décision GO/NO-GO en B1-B4 ne dépend pas du seuil exact mais du fait que le pipeline produit un edge significatif (Sharpe > 1, DSR > 0). Si B1 est NO-GO uniquement à cause d'un seuil trop conservateur, on pourrait re-tuner. Mais ce serait un nouveau n_trial → à éviter.

**Recommandation** : si possible, faire A2 (0.5j) en parallèle/intercalé avant A8 pour avoir un seuil optimal. Sinon, accepter le caveat.

### Reprise après A1 (si tu as déjà commencé A1)

Si A1 est ✅ Terminé dans `JOURNAL.md`, l'enchaînement à suivre est :

```
[✅ A1 fait]
   ↓
A5 (superset 70+ features)         ← 1 j      ★ priorité utilisateur
   ↓
A6 (ranking + top 15 figé)         ← 0.5 j
   ↓
A7 (sélection RF/HGBM/Stacking)    ← 1 j      ⚠ Sharpe biaisé (coûts non corrigés), ranking préservé
   ↓
A8 (nested CPCV hyperparams)       ← 0.5 j    ⚠ seuil suboptimal, voir caveat ci-dessus
   ↓
A9 (pipeline lock + SHA256)        ← 0.3 j    Pipeline ML figé
   ↓
A2 (calibration coûts XTB)         ← 0.5 j    Simulateur finalisé pour B1
   ↓
A3 (Sharpe routing low-frequency)  ← 0.3 j    Métriques finalisées
   ↓
[A4 optionnel : replay observation]
   ↓
B1 (méta-labeling US30 D1)         ← +1 n_trial
   ↓
B2 (EURUSD H4 mean-rev)            ← +1 n_trial
```

**Aucun raccourci**. Si A2-A9 ne sont pas tous ✅ Terminés dans `JOURNAL.md`, ne PAS commencer B1.

**Verrouillage progressif** :
- Après **A6** : la liste des features est FIGÉE dans `app/config/features_selected.py`. Toute modification post-A6 = data snooping.
- Après **A7** : le modèle ML est FIGÉ dans `app/config/model_selected.py`. Pas de "essayons XGBoost en plus" sans nouveau n_trial.
- Après **A8** : les hyperparams sont FIGÉS dans `app/config/hyperparams_tuned.py`.
- Après **A9** : tout le pipeline est immutable (SHA256 dans `TEST_SET_LOCK.json`). B1-B4 utilisent ce pipeline tel quel.

### Que faire si Deepseek/Roo Code se perd ?

Au début de **CHAQUE** prompt de cette phase, Deepseek doit :
1. Lire [00_README.md](00_README.md) (ce fichier) — section "Tableau de dépendances".
2. Lire `JOURNAL.md` à la racine du projet pour identifier les phases ✅ Terminées.
3. Vérifier que **toutes les dépendances** de la phase courante sont ✅ Terminées.
4. Si une dépendance manque → **STOP**, demander à l'utilisateur si elle doit être faite d'abord.

Exemple concret : si tu lances A7 et que A3 n'est pas dans `JOURNAL.md` comme ✅ Terminé, A7 doit refuser de démarrer.

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

## Vue d'ensemble graphique (ordre RÉVISÉ)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE A — Tout sur train ≤ 2022                       │
│                   0 n_trial consommé                                    │
│                   Test set 2024+ JAMAIS lu                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [A1 ✅] sizing + DD bornés                                              │
│       │                                                                 │
│       ▼                                                                 │
│  ─────────── pipeline ML (★ priorité utilisateur) ──────────            │
│  A5 ─→ A6 ─→ A7 ─→ A8 ─→ A9                                            │
│  (features → ranking → modèle → hyperparams → lock SHA256)              │
│       │                                                                 │
│       ▼                                                                 │
│  pipeline ML immutable (features + modèle + hyperparams + seuil figés)  │
│       │                                                                 │
│       ▼                                                                 │
│  ─────────── finition simulateur (avant tests OOS) ──────────           │
│  A2 ─→ A3      [A4 optionnel : replay observation]                      │
│  (coûts XTB réels + Sharpe routing low-frequency)                       │
│       │                                                                 │
│       ▼                                                                 │
│  pipeline ML gelé + simulateur fiable                                   │
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

**Suivant (puisque A1 ✅)** : [A5_feature_generation.md](A5_feature_generation.md)
