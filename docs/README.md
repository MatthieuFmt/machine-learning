# Roadmap stratégique — Fiabiliser l'edge EURUSD H1

**Date de création** : 2026-05-12
**Contexte** : Après 15 itérations documentées dans [`ml_evolution.md`](../ml_evolution.md), l'edge atteint Sharpe +0.53 sur 2025 (v8 méta-labeling) mais **n'est PAS confirmé statistiquement** : p(Sharpe>0) = 0.29, DSR = -1.97 sur la distribution de test 2025. L'accuracy primaire reste ≈ 0.332 (aléatoire pour 3 classes).

**Objectif de la roadmap** : passer la barre de fiabilité statistique (DSR > 0, p-value < 0.05 sur OOS multi-année) en attaquant les causes racines plutôt qu'en empilant des optimisations cosmétiques.

---

## Vue d'ensemble des 7 steps

| # | Step | Catégorie | Priorité | Effort | Dépend de |
|---|---|---|---|---|---|
| [01](step_01_target_redefinition.md) | Redéfinition cible (régression / binaire pure) | Rupture | 🔴 Haute | 2-3 j | — |
| [02](step_02_robust_validation_framework.md) | CPCV + DSR (validation robuste) | Méthodologie | 🔴 Haute | 2-3 j | — |
| [03](step_03_gbm_primary_classifier.md) | LightGBM/XGBoost + Optuna | Modèle | 🟠 Moyenne | 2-3 j | 02 (validation), idéalement 01 |
| [04](step_04_session_aware_features.md) | Features de session (Tokyo/Londres/NY) | Feature engineering | 🟠 Moyenne | 1-2 j | 03 (catégorielles GBM) |
| [05](step_05_economic_calendar_integration.md) | Calendrier macro (NFP, CPI, FOMC, BCE) | Feature exogène | 🟠 Moyenne | 3-5 j | — |
| [06](step_06_meta_labeling_calibration.md) | Calibration Platt/isotonique + seuil breakeven | Optimisation | 🟠 Moyenne | ½ - 1 j | 02 (CPCV pour mode robuste) |
| [07](step_07_cross_asset_validation.md) | Validation GBPUSD / USDJPY / XAUUSD | Validation | 🟡 Basse | 2-3 j | 01-06 (config "production candidate") |

---

## Ordre d'exécution recommandé

### Phase 1 — Causes racines (parallélisable)
- **step_01** (cible) + **step_02** (validation) **en parallèle**. Ces deux pistes attaquent les blocages fondamentaux indépendamment.

### Phase 2 — Optimisations conditionnelles
- **step_03** (GBM) UNIQUEMENT si step_01 ou step_02 a montré un potentiel exploitable.
- **step_06** (calibration) en quick win, dès que step_02 fournit le framework CPCV.

### Phase 3 — Enrichissement informationnel
- **step_04** (sessions) et **step_05** (calendrier) après la config "candidate" stabilisée. Ces deux ajoutent de la variance dans le signal mais ne créent pas un edge ex nihilo.

### Phase 4 — Validation finale
- **step_07** (multi-actifs) en gate final avant production. Si échec, retour Phase 1.

---

## Critères go/no-go inter-étapes

### Après step_01 (target redefinition)
- ✅ **Si** accuracy OOS 2025 > 0.36 (toute variante) → **GO step_03**
- ❌ **Sinon** → reconsidérer la viabilité du projet sur EURUSD H1, envisager TF ou instrument alternatif (cf. step_07 inversé)

### Après step_02 (validation CPCV)
- ✅ **Si** DSR > 0 et % splits profitables > 60 % sur la baseline v15 actuelle → **GO Phase 2-3** (l'edge existant est mesurable)
- ❌ **Sinon** → l'edge n'est pas réel, retour à step_01 obligatoire avant tout autre investissement

### Après steps 03-06
- ✅ **Si** Sharpe OOS 2025 > 0.50 ET DSR 2025 > 0 → **GO step_07**
- ❌ **Sinon** → audit complet : quelle piste a aidé, laquelle a nui ? Ablation study.

### Après step_07 (multi-actifs)
- ✅ **Si** Sharpe > 0 sur ≥ 2 actifs / 4 → **GO production** (paper trading 3 mois recommandé d'abord)
- ❌ **Sinon** → overfit EURUSD confirmé, retour au labo (step_01)

---

## Métriques de référence — Baseline v15

| Métrique | Valeur |
|---|---|
| Accuracy OOS 2024 | 0.355 |
| Accuracy OOS 2025 | 0.332 (≈ aléatoire 3-classes) |
| Sharpe OOS 2024 | +0.49 (v13 bootstrap : DSR +5.94) |
| Sharpe OOS 2025 | +0.04 (v13 bootstrap : DSR -1.97, p=0.29) |
| WR OOS 2025 | 33.3 % (v8 méta-labeling) |
| Breakeven WR (TP=30, SL=10, fr=1.5) | 27.7 % |
| Biais directionnel 2025 | 75 % SHORT |
| Features actives | 7 + XAU_Return |

---

## Convention des fichiers `step_NN_*.md`

Chaque fichier suit le template :

1. **Hypothèse mathématique** : formulation formelle du problème ou de l'amélioration attendue
2. **Méthodologie d'implémentation** : fichiers à modifier, choix techniques, précautions anti-leak
3. **Métriques de validation** : tableau cible + métriques secondaires + critère d'arrêt
4. **Risques & dépendances** : risques identifiés et mitigations
5. **Références** : papiers, livres (López de Prado), liens code

Pas de pseudocode complet — focus stratégique. L'implémentation se fait après lecture du step + discussion avec l'utilisateur.

---

## Hors-scope explicite

Les pistes suivantes ont été identifiées mais **volontairement exclues** de cette roadmap (faible ROI estimé ou complexité disproportionnée) :

- **HMM / régime-switching models** : trop spécialisé, dépendance `hmmlearn`, gains historiques modestes en forex H1.
- **Deep learning (LSTM, Transformer)** : taille de dataset (~80k samples) insuffisante pour battre les arbres correctement régularisés. Coût compute disproportionné.
- **Microstructure (order flow, tick imbalance)** : nécessite données tick-by-tick que le projet n'a pas. Inapplicable au TF H1 actuel.
- **Position sizing avancé (Kelly, vol-targeting)** : utile en aval, sans edge confirmé d'abord c'est mettre la charrue avant les bœufs.

Ces pistes peuvent être réintégrées dans une roadmap v17+ si la fiabilité de l'edge est confirmée et qu'on cherche des leviers marginaux supplémentaires.


écrit un rapport final sur ce les résultats qu'à apporté la step une fois terminée 