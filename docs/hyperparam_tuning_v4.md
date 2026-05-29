# Hyperparam Tuning v4 (pivot v4 A8)

**Date** : 2026-05-15
**Méthode** : Nested CPCV (outer=5, inner=3, embargo=1%)
**Périmètre** : train ≤ 2022-12-31 UNIQUEMENT
**n_trials** : inchangé

## Résumé

Le module `app/models/nested_tuning.py` implémente la nested cross-validation de López de Prado :
- **Outer CV** : évalue la performance honnête (jamais vu pendant le tuning)
- **Inner CV** : sélectionne les meilleurs hyperparams sur le train de chaque outer fold
- **Vote majoritaire** : params et seuil retenus = ceux qui apparaissent le plus souvent sur les outer folds

## Résultats par actif

### US30 D1 (RF)

Grid : `n_estimators ∈ {100, 200, 400}`, `max_depth ∈ {3, 4, 6}`, `min_samples_leaf ∈ {5, 10, 20}` → 27 combos × 3 thresholds = 81 essais par outer fold.

| Paramètre | Valeur retenue |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 3 |
| `min_samples_leaf` | 10 |
| `threshold` | 0.55 |
| **Sharpe outer** | **+1.913** |
| Sharpe outer std | ±2.005 |
| WR outer | 57.5% |
| n_kept moyen | 21.6 |

**Analyse** : Forte variance inter-fold (Sharpe de −1.32 à +4.76). Le fold 1 (−1.32) est problématique mais compensé par les folds 3-5. Le vote majoritaire donne `max_depth=3, min_samples_leaf=10, n_estimators=100` (2/5 folds) et `threshold=0.55` (2/5 folds).

### EURUSD H4 (RF)

Même grid RF que US30 D1.

| Paramètre | Valeur retenue |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `min_samples_leaf` | 10 |
| `threshold` | 0.55 |
| **Sharpe outer** | **+0.592** |
| Sharpe outer std | ±0.713 |
| WR outer | 51.5% |
| n_kept moyen | 26.8 |

**Analyse** : 2 folds négatifs (−0.28, −0.14), 3 positifs (+1.63, +0.82, +0.93). Sharpe outer 0.59 au-dessus du seuil GO mais modeste. Le vote majoritaire donne `max_depth=6, min_samples_leaf=10, n_estimators=100` (2/5 folds) et `threshold=0.55` (4/5 folds).

### XAUUSD D1 (Stacking)

Stacking non tuné (trop lent). Defaults A7 conservés. `params: {}`, `threshold: 0.50`, Sharpe outer = 0.000 placeholder.

## Sharpe "outer" vs Sharpe "inner"

| Actif | Sharpe inner (biaisé, A7) | Sharpe outer (honnête, A8) | Écart |
|---|---|---|---|
| US30 D1 | 1.75 | 1.91 | 0.16 ✅ |
| EURUSD H4 | 0.90 | 0.59 | 0.31 ✅ |

L'écart < 1.0 pour les deux actifs tunés = pas d'overfit hyperparams détecté. Note : US30 outer > inner (1.91 > 1.75), suggérant que le seuil 0.55 + params simplifiés (max_depth=3) généralisent mieux que le seuil fixe 0.50 de A7.

## Go/No-Go

| Critère | US30 D1 | EURUSD H4 | XAUUSD D1 | Seuil |
|---|---|---|---|---|
| Sharpe outer ≥ 0.5 | +1.91 ✅ | +0.59 ✅ | 0.00 ❌ | ≥ 0.5 |
| Écart inner-outer < 1.0 | 0.16 ✅ | 0.31 ✅ | N/A | < 1.0 |
| **Verdict** | **GO** | **GO** | **NO-GO** | |

**Décision** : GO pour US30 D1 et EURUSD H4. XAUUSD D1 stacking non tunable — le splitting temporel strict (train ≤ 2022, 85 trades, WR 11.8%) rend le CPCV inapplicable. À réévaluer en walk-forward ou sur H4.

## Décision de gel

Hyperparams + seuil FIGÉS dans `app/config/hyperparams_tuned.py`.
**Aucune modification post-A8 autorisée sans nouveau pivot.**

## Limites

- Grid de 27 combos × 3 thresholds = 81 essais par outer fold = 405 fits par actif. Compromis coût/exhaustivité.
- inner_k=3 plutôt petit. Pour plus de stabilité, 5 serait mieux mais 2× plus lent.
- Le vote majoritaire peut masquer un désaccord entre folds → loggé dans `outer_folds`.
- XAUUSD D1 exclu du tuning car Stacking (trop lent à fitter en nested CV).

## Tests

9 tests unitaires dans `tests/unit/test_nested_tuning.py` :
1. `test_expand_grid_produces_cartesian` — produit cartésien correct
2. `test_expand_grid_single_value` — 1 valeur par axe = 1 combo
3. `test_nested_tuning_runs` — tourne sans erreur sur synthétique
4. `test_threshold_below_05_raises` — AssertionError si seuil < 0.50
5. `test_threshold_mixed_with_below_05_raises` — même avec un seul seuil < 0.50 dans la liste
6. `test_reproducible` — même seed → même résultat
7. `test_n_combos_correct` — comptage correct
8. `test_tuning_result_fields` — structure TuningResult valide
9. `test_no_data_leak_outer_to_inner` — pas de fuite entre outer_train et outer_test
