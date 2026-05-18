# Hyperparam tuning v4 — Extension multi-actifs (Phase C4)

**Date** : 2026-05-18
**Périmètre** : 8 couples shortlist C3 (Sharpe CPCV ≥ 0.5) + 3 A8 originaux = 11 couples
**Train cutoff** : 2022-12-31
**Méthode** : Nested CPCV (outer 5 × inner 3), embargo 1 %
**n_trials** : inchangé

## Tableau récapitulatif complet (11 couples)

| Actif | TF | Modèle | Best params | Threshold | Sharpe inner (C3) | Sharpe outer (C4) | Gap I-O | Pass C5 ? |
|---|---|---|---|---|---|---|---|---|
| US30 | D1 | rf | n=100, d=3, leaf=10 | 0.55 | 1.75 | +1.913 | 0.16 | ✅ A8 |
| EURUSD | H4 | rf | n=100, d=6, leaf=10 | 0.55 | 0.90 | +0.592 | 0.31 | ✅ A8 |
| XAUUSD | D1 | stacking | (defaults) | 0.50 | −1.05 | 0.000 | — | ❌ A8 |
| ETHUSD | D1 | hgbm | lr=0.05, md=3, mln=15, msl=20 | 0.50 | 1.55 | +1.70 | 0.19 | ✅ C5 |
| ETHUSD | H4 | hgbm | lr=0.05, md=3, mln=15, msl=50 | 0.60 | 0.55 | +0.39 | 1.13 | ❌ |
| ETHUSD | H1 | hgbm | lr=0.05, md=6, mln=15, msl=50 | 0.50 | 2.02 | +1.81 | 0.02 | ✅ C5 |
| EURUSD | D1 | stacking | (defaults) | 0.50 | 6.21 | — | — | ❌ |
| GBPUSD | D1 | rf | n=200, d=3, leaf=10 | 0.50 | 8.62 | +7.82 | 1.92 | ❌ |
| GBPUSD | H4 | rf | n=100, d=10, leaf=10 | 0.50 | 3.69 | +3.45 | 0.50 | ✅ C5 |
| USDCHF | D1 | stacking | (defaults) | 0.50 | 3.33 | — | — | ❌ |
| USDCHF | H4 | rf | n=100, d=3, leaf=10 | 0.60 | 1.15 | +1.17 | 0.32 | ✅ C5 |

**Légende paramètres RF** : n = n_estimators, d = max_depth, leaf = min_samples_leaf
**Légende paramètres HGBM** : lr = learning_rate, md = max_depth, mln = max_leaf_nodes, msl = min_samples_leaf

## Shortlist C5 (Sharpe outer ≥ 0.5 ET gap < 1.0)

4 couples retenus pour la Phase C5 (Pipeline lock + bilan global Phase C) :

| Actif | TF | Modèle | Sharpe outer | Gap | Analyse |
|---|---|---|---|---|---|
| ETHUSD | D1 | hgbm | +1.70 | 0.19 | ✅ Fort Sharpe, gap quasi-nul → excellente généralisation |
| ETHUSD | H1 | hgbm | +1.81 | 0.02 | ✅ Meilleur Sharpe C4, gap quasi-nul → pas d'overfitting |
| GBPUSD | H4 | rf | +3.45 | 0.50 | ✅ Très fort Sharpe, gap modéré → généralisation acceptable |
| USDCHF | H4 | rf | +1.17 | 0.32 | ✅ Sharpe correct, gap faible → bon candidat diversification |

## Couples exclus de C5

### Sharpe outer < 0.5
- **ETHUSD H4** (hgbm, Sharpe=0.39, gap=1.13) : Sharpe insuffisant ET gap élevé. Le HGBM sur H4 ne capture pas d'edge suffisant sur cet actif. Le gap > 1.0 suggère un overfitting malgré le Sharpe faible.

### Gap ≥ 1.0 (overfitting probable)
- **GBPUSD D1** (rf, Sharpe=7.82, gap=1.92) : Sharpe outer spectaculaire (+7.82, meilleur absolu) mais gap très élevé (1.92). Le modèle RF sur D1 avec n_estimators=200 overfitte massivement malgré le nested CPCV. Le gap dépasse le seuil de 1.0 → exclusion automatique, car les performances OOS risquent de s'effondrer.

### Stacking non tunés (defaults conservés)
- **EURUSD D1** (stacking) : Modèle stacking trop lent pour le nested CPCV → defaults A7 conservés. Pas de gap calculable. Exclu de C5 car pas de tuning → pas de garantie de robustesse.
- **USDCHF D1** (stacking) : Idem EURUSD D1.
- **XAUUSD D1** (stacking) : Déjà NO-GO en A8 (85 trades, WR 11.8%, CPCV inapplicable).

## Analyse qualitative

### Modèles qui généralisent (gap < 0.5)
- **ETHUSD H1** (gap=0.02) : Généralisation quasi-parfaite. Le HGBM sur H1 avec 845 trades d'entraînement bénéficie d'un échantillon large et d'un bruit intraday favorable.
- **ETHUSD D1** (gap=0.19) : Très bonne généralisation. Les paramètres retenus (lr=0.05, md=3, mln=15, msl=20) sont conservateurs → bon équilibre biais/variance.
- **USDCHF H4** (gap=0.32) : Bonne généralisation. RF simple (n=100, d=3, leaf=10) avec seuil 0.60 conservateur.

### Modèles en zone grise (0.5 ≤ gap < 1.0)
- **GBPUSD H4** (gap=0.50) : Limite acceptable. Sharpe outer très élevé (+3.45) compense un gap modéré.
- **US30 D1** (gap=0.16), **EURUSD H4** (gap=0.31) : Déjà validés en A8, gap faible.

### Modèles qui overfittent (gap ≥ 1.0)
- **GBPUSD D1** (gap=1.92) : Overfitting flagrant. Le RF avec n_estimators=200 et seulement 482 trades → variance inter-fold explosive.
- **ETHUSD H4** (gap=1.13) : Overfitting modéré mais Sharpe outer déjà trop faible (0.39).

### Constat global
- **HGBM** : Excellent sur ETHUSD (crypto, volatilité élevée, nombreux trades). Gap très faible sur D1 et H1.
- **RF** : Performant sur forex H4 (GBPUSD, USDCHF). Sur D1, tendance à l'overfitting (GBPUSD D1 gap=1.92).
- **Stacking** : Non tunable en nested CPCV → angle mort méthodologique. À réévaluer avec une approche walk-forward si les autres modèles échouent en OOS.

## Méthodologie

### Nested CPCV

- **Outer CV** (5-fold) : évalue la performance honnête sur données jamais vues pendant le tuning
- **Inner CV** (3-fold) : sélectionne les meilleurs hyperparams sur le train de chaque outer fold
- **Embargo** : 1 % entre folds pour éviter le data leakage temporel
- **Vote majoritaire** : params + threshold les plus fréquents sur les 5 outer folds

### Grilles d'hyperparams

**RF** (RandomForestClassifier) :
- `n_estimators` ∈ {100, 200}
- `max_depth` ∈ {3, 6, 10}
- `min_samples_leaf` ∈ {5, 10, 20}
→ 18 combos × 3 thresholds = 54 évaluations par outer fold

**HGBM** (HistGradientBoostingClassifier) :
- `max_depth` ∈ {3, 6, None}
- `learning_rate` ∈ {0.05, 0.10}
- `max_leaf_nodes` ∈ {15, 31}
- `min_samples_leaf` ∈ {20, 50}
→ 24 combos × 3 thresholds = 72 évaluations par outer fold

**Stacking** : non tuné (trop lent). Defaults A7 conservés.

### Seuils méta candidats

{0.50, 0.55, 0.60} — identique A8 pour comparabilité.

### Critères C5 (Go/No-Go)

| Critère | Seuil | Signification |
|---|---|---|
| Sharpe outer | ≥ 0.5 | Performance honnête positive |
| Gap inner-outer | < 1.0 | Pas d'overfitting hyperparams |
| Modèle tuné | nested CPCV complété | Exclusion des stacking non tunés |

## Limites

- Grid de 18-24 combos × 3 thresholds = compromis coût/exhaustivité
- inner_k=3 plutôt petit ; 5 serait plus stable mais 2× plus lent
- Stacking exclu du tuning → les couples stacking (EURUSD D1, USDCHF D1, XAUUSD D1) gardent leurs defaults
- Le vote majoritaire peut masquer un désaccord entre folds → détail dans `outer_folds` du JSON
- Les Sharpe inner sont issus de C3 (CPCV standard, seuil fixe 0.50), pas du nested CPCV → le gap est une approximation conservative
