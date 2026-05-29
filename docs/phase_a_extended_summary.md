# Phase A étendue — Bilan global (Pivot v4 + C1-C5)

**Date** : 2026-05-18
**Pipeline version** : v4.1.0-extended
**n_trials consommés en Phase C** : 0
**Test set ≥ 2024 lu** : NON (préservé)
**n_trials cumul actuel** : 28

## 1. Couverture

| Catégorie | Compte |
|---|---|
| Couples cibles | 21 (7 actifs × 3 TF) |
| Couples avec données (C1) | 18 (BTCUSD, ETHUSD ×3 TF, EURUSD D1/H4, GBPUSD D1/H4, USDCHF D1/H4 + US30 D1, XAUUSD D1) |
| Couples indisponibles | 3 (US30 H1/H4, XAUUSD H1/H4 — données absentes) |
| Couples rankés (C2) | 9/9 nouveaux couples |
| Couples shortlist C3 (stab ≥ 0.5) | 9/9 → tous passent en C3 |
| Couples shortlist C4 (Sharpe CPCV ≥ 0.5) | 10 (2 A7 + 8 C3) |
| Couples tunés C4 | 8 (6 nested CPCV + 2 stacking defaults) |
| **Couples figés en pipeline (LOCKED_COUPLES)** | **11** (3 A9 + 8 C4) |
| **Shortlist C5 qualifiés (Sharpe outer ≥ 0.5, gap < 1.0)** | **6** (2 A8 + 4 C4) |
| Couples exclus (insufficient_trades / stab / Sharpe < 0.5 / gap ≥ 1.0) | 7 |

## 2. Tableau global des couples figés (LOCKED_COUPLES = 11)

| Actif | TF | Modèle | Params | Threshold | Sharpe outer | WR outer | Stab top 15 | Gap inner-outer | Statut shortlist C5 | Phase B testé | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| US30 | D1 | rf | md=3, msl=10, ne=100 | 0.55 | +1.913 | 57.5% | 0.72 | 0.16 | ✅ | B1 ❌ NO-GO | A9 |
| EURUSD | H4 | rf | md=6, msl=10, ne=100 | 0.55 | +0.592 | 51.5% | 0.59 | 0.31 | ✅ | B2 ✅ GO | A9 |
| XAUUSD | D1 | stacking | defaults | 0.50 | 0.000 | — | 0.56 | — | ❌ | — | A9 |
| ETHUSD | D1 | hgbm | lr=0.05, md=3, mln=15, msl=20 | 0.50 | +1.700 | 65.0% | 0.56 | 0.19 | ✅ | — | C5 |
| ETHUSD | H4 | hgbm | lr=0.05, md=3, mln=15, msl=50 | 0.60 | +0.388 | 44.4% | 0.61 | — | ❌ Sharpe <0.5 | — | C5 |
| ETHUSD | H1 | hgbm | lr=0.05, md=6, mln=15, msl=50 | 0.50 | +1.812 | 47.4% | 0.71 | 0.02 | ✅ | — | C5 |
| EURUSD | D1 | stacking | defaults | 0.50 | 0.000 | — | 0.60 | — | ❌ stacking | — | C5 |
| GBPUSD | D1 | rf | md=3, msl=10, ne=200 | 0.50 | +7.818 | 79.3% | 0.60 | 1.92 | ❌ gap ≥1.0 | — | C5 |
| GBPUSD | H4 | rf | md=10, msl=10, ne=100 | 0.50 | +3.451 | 53.3% | 0.63 | 0.50 | ✅ | — | C5 |
| USDCHF | D1 | stacking | defaults | 0.50 | 0.000 | — | 0.55 | — | ❌ stacking | — | C5 |
| USDCHF | H4 | rf | md=3, msl=10, ne=100 | 0.60 | +1.166 | 29.8% | 0.61 | 0.32 | ✅ | — | C5 |

### Couples non figés (dans features/model mais pas hyperparams)

| Actif | TF | Raison exclusion |
|---|---|---|
| BTCUSD | D1 | Sharpe CPCV 0.28 < 0.5 (C3) → non tuné en C4 |

## 3. Patterns par classe d'actif

### Indices
- **US30 D1** : seul indice avec données. Sharpe outer +1.913, RF fonctionne bien. Déjà testé B1 → NO-GO (12 trades OOS insuffisants).

### Forex majeures
- **EURUSD** : H4 validé B2 ✅ (Sharpe OOS +1.73, 25.2 trades/an). D1 stacking non tunable. Efficace en H4 via mean-reversion + méta-labeling.
- **GBPUSD** : H4 excellent candidat Phase B (Sharpe outer +3.45, gap 0.50). D1 overfitting (gap 1.92). Le RF capture bien les régimes forex.
- **USDCHF** : H4 candidat Phase B (Sharpe outer +1.17), WR faible (29.8%) mais Sharpe solide. D1 stacking non tunable.

### Métaux
- **XAUUSD D1** : stacking placeholder. 85 trades train, WR 11.8%. Insuffisant pour CPCV fiable.

### Crypto
- **ETHUSD** : classe dominante C4. 2/3 TF shortlist C5 (D1 +1.70, H1 +1.81). H4 exclu (Sharpe 0.39). HGBM systématiquement meilleur que RF sur crypto.
- **BTCUSD D1** : exclu en C3 (Sharpe CPCV 0.28). HGBM meilleur que RF mais insuffisant.

## 4. Décision Phase B — couples candidats prioritaires

Couples shortlist finale C5 non testés, triés par Sharpe outer décroissant :

| Rang | Couple | Modèle | Sharpe outer | Gap | WR outer | Coût n_trial | Recommandation |
|---|---|---|---|---|---|---|---|
| 1 | GBPUSD H4 | rf | +3.45 | 0.50 | 53.3% | +1 | ★ Tester en priorité |
| 2 | ETHUSD H1 | hgbm | +1.81 | 0.02 | 47.4% | +1 | Second |
| 3 | ETHUSD D1 | hgbm | +1.70 | 0.19 | 65.0% | +1 | Third |
| 4 | USDCHF H4 | rf | +1.17 | 0.32 | 29.8% | +1 | WR faible, Sharpe correct |

**Cadre méthodologique** : chaque test Phase B = lecture définitive du test set 2024+ pour ce couple = +1 n_trial cumul.

- n_trials_cumul actuel : 28
- Si on teste les 4 candidats : n_trials_cumul → 32
- DSR pénalise par √(log(n_trials)) → impact croissant sur le seuil de significativité

## 5. Recommandations utilisateur

Trois options possibles :

### Option A — Aller en Phase B sélective (1 à 3 couples max)
- Pertinent : GBPUSD H4 (Sharpe outer +3.45, le plus prometteur), ETHUSD H1 (+1.81, gap quasi nul)
- Cible : compléter le portefeuille (actuellement single-sleeve EURUSD H4)
- Coût méthodologique acceptable : 28 → 29-31 n_trials
- Risque : chaque test consomme 1 n_trial définitif

### Option B — Faire la vérification spreads démo d'abord
- L'utilisateur a explicitement demandé cette vérification après Phase C
- Mise à jour de `ASSET_CONFIGS` avec les vrais coûts XTB capturés en démo
- Si gros écart → retour en C2-C4 pour quelques couples (les coûts impactent la cible binaire winner)
- Bénéfice : Phase B avec coûts réels, résultats plus crédibles
- **Rappel** : BTCUSD, ETHUSD, GBPUSD, USDCHF ont des coûts PROVISOIRES (marqués dans C1)

### Option C — Aller en prompt 18 (validation finale) sur l'existant
- Le portfolio single-sleeve (EURUSD H4) doit être confronté à Buy-and-Hold + Monte Carlo
- Si ça passe → production (prompt 20). Sinon → retour Phase C avec stratégies alternatives.
- N'invalide pas la Phase C : les nouveaux couples restent dans le pipeline gelé pour usage futur.

**Recommandation par défaut** : Option B (spreads démo) puis Option A si gros candidat émerge, sinon Option C.

## 6. Annexes techniques

### Fichiers figés (SHA256 après exécution C5)
- `app/config/features_selected.py` : (recalculé par `run_c5_pipeline_lock_extended.py`)
- `app/config/model_selected.py` : (recalculé par `run_c5_pipeline_lock_extended.py`)
- `app/config/hyperparams_tuned.py` : (recalculé par `run_c5_pipeline_lock_extended.py`)
- `app/config/ml_pipeline_v4.py` : (recalculé par `run_c5_pipeline_lock_extended.py`)

### Couples écartés (raison détaillée)

| Couple | Étape | Raison |
|---|---|---|
| BTCUSD D1 | C3 | Sharpe CPCV 0.28 < 0.5 — pas d'edge méta-labeling détectable |
| ETHUSD H4 | C4 | Sharpe outer 0.39 < 0.5 — HGBM sous-performe en H4 crypto |
| EURUSD D1 | C3/C4 | Stacking non tunable — CPCV instable, defaults conservés |
| GBPUSD D1 | C4 | Gap inner-outer 1.92 ≥ 1.0 — overfitting sévère, Sharpe outer +7.82 suspect |
| USDCHF D1 | C3/C4 | Stacking non tunable — CPCV instable, defaults conservés |
| US30 H1/H4 | C1 | Données absentes |
| XAUUSD H1/H4 | C1 | Données absentes |

### Historique complet Phase A + Phase C

| Phase | Action | n_trial | Sorti dans |
|---|---|---|---|
| A1 | Audit simulateur + sizing 2% | 0 | `simulator.py` + `sizing.py` |
| A2 | Coûts XTB réels | 0 | `instruments.py` + `cost_audit_v2.md` |
| A3 | Sharpe routing par fréquence | 0 | `metrics.py` |
| A4 | Replay H06/H07 train+val | 0 | `_replay.md` |
| A5 | Superset ~70 features | 0 | `superset.py` |
| A6 | Top 15 features ×3 actifs | 0 | `features_selected.py` (3 entrées) |
| A7 | Modèle retenu ×3 actifs | 0 | `model_selected.py` (3 entrées) |
| A8 | Hyperparams + seuil ×3 actifs | 0 | `hyperparams_tuned.py` (3 entrées) |
| A9 | Pipeline lock + checksums | 0 | `ml_pipeline_v4.py` v4.0.0-locked |
| B1-B4 | Phase B (4 hypothèses) | +6 | `predictions/h_new*.json` |
| C1 | Inventory multi-actifs | 0 | `ASSET_CONFIGS` + `run_c1_inventory.py` |
| C2 | Feature ranking ×9 couples | 0 | `features_selected.py` (+9 entrées) |
| C3 | Model selection ×9 couples | 0 | `model_selected.py` (+9 entrées) |
| C4 | Hyperparams tuning ×8 couples | 0 | `hyperparams_tuned.py` (+8 entrées) |
| C5 | Pipeline lock étendu + bilan | 0 | `ml_pipeline_v4.py` v4.1.0-extended |
| **Total Phase C** | | **0 n_trials** | Pipeline étendu de 3 → 11 couples |
| **Total Phase A+B+C** | | **28 n_trials** | 6 hypothèses B testées, 11 couples figés |

### Évolutions futures du pipeline (post-C5)

Toute extension ultérieure devra :
1. Repartir de C2-C4 pour les nouveaux couples (jamais modifier les couples figés).
2. Bumper la version (v4.2.x, v4.3.x…).
3. Documenter dans une nouvelle section de ce fichier.

Tant qu'**aucun couple existant n'est modifié**, ces extensions consomment 0 n_trial.
