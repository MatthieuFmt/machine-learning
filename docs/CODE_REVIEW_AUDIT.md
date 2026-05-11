# Audit de Code — Pipeline ML/Trading EURUSD

**Date :** 2026-05-11  
**Périmètre :** 23 fichiers analysés (scripts racine + package `learning_machine_learning/`)  
**Méthode :** Revue statique exhaustive, comparaison ancien/nouveau code, détection de régressions

---

## 1. Bugs Logiques (P0 — Critique)

### B1 — [`features/triple_barrier.py:82-116`](../learning_machine_learning/features/triple_barrier.py#L82) : SL hit n'arrête pas la direction

**Sévérité :** P0 — Résultats de labellisation incorrects

**Description :** La version refactorisée fusionne les deux boucles LONG/SHORT originales en une seule boucle qui teste les deux directions simultanément. Quand le SL est touché pour LONG (`curr_low <= long_sl`), le code fait `pass` au lieu de marquer la direction comme morte. Sur les itérations suivantes de la boucle, si le TP est touché, `long_win` devient `True` — ce qui est faux : le SL a déjà été atteint avant.

**Code actuel (buggé) :**
```python
# Ligne 89-91
if not long_win:
    if curr_low <= long_sl:
        pass  # BUG: n'empêche pas long_win = True plus tard
    elif curr_high >= long_tp:
        long_win = True
```

**Code original (correct) :** Deux boucles séparées, le `break` sur SL empêche toute victoire ultérieure.

**Impact :** Labels `1` ou `-1` attribués à tort quand le SL est touché avant le TP. Surestimation de la win rate réelle.

**Correctif :** Ajouter un flag `long_dead` / `short_dead`. Une fois le SL touché pour une direction, on ne la teste plus :

```python
long_dead = False
short_dead = False
for j in range(1, window + 1):
    idx = i + j
    curr_high, curr_low = highs[idx], lows[idx]
    
    if not long_dead and not long_win:
        if curr_low <= long_sl:
            long_dead = True
        elif curr_high >= long_tp:
            long_win = True
    
    if not short_dead and not short_win:
        if curr_high >= short_sl:
            short_dead = True
        elif curr_low <= short_tp:
            short_win = True
    
    if (long_dead or long_win) and (short_dead or short_win):
        break  # optimisation: les deux directions sont résolues
```

---

### B2 — [`backtest/simulator.py:129,157-160`](../learning_machine_learning/backtest/simulator.py#L129) : Timeout assume la perte maximale (SL)

**Sévérité :** P1 — Métriques de backtest pessimistes pour les timeouts

**Description :** Quand un trade expire sans toucher TP ni SL, le code attribue `pips_brut = -sl_pips - spread_cost` (ligne 129), ce qui est la perte maximale. Or un timeout devrait calculer le PnL réel : `(Close_final - entry_price) / pip_size - spread_cost`. Le prix final peut être n'importe où entre SL et TP.

**Code actuel (lignes 129, 160) :**
```python
pips_brut = -sl_pips - spread_cost  # initialisé pour loss_sl ET loss_timeout
...
else:
    i += window
    result_type = "loss_timeout"
# pips_brut reste à -sl_pips - spread_cost !
```

**Correctif :** Pour les timeouts, calculer le PnL réel à partir du Close à `i + window` :
```python
else:
    exit_price = closes[i + window]
    if signal == 1:
        pips_brut = (exit_price - entry_price) / pip_size - spread_cost
    else:
        pips_brut = (entry_price - exit_price) / pip_size - spread_cost
    i += window
    result_type = "loss_timeout"
```

**Note :** Le bug est hérité de [`backtest_utils.py:219`](../backtest_utils.py#L219). Il était déjà présent dans l'ancien code.

---

### B3 — [`model/prediction.py:50-54`](../learning_machine_learning/model/prediction.py#L50) : Extraction des probas fragile (classes hardcodées)

**Sévérité :** P1 — Crash si `model.classes_` n'est pas `[-1.0, 0.0, 1.0]`

**Description :** Le mapping `class_map` est construit dynamiquement (ligne 50), mais l'extraction utilise des clés hardcodées `-1.0`, `0.0`, `1.0` (lignes 52-54). Si une classe est absente des données (ex : aucun `-1` dans l'échantillon d'entraînement), `model.classes_` sera `[0.0, 1.0]` et `class_map[-1.0]` lèvera une `KeyError`.

**Correctif :** Utiliser `class_map.get(-1.0, 0)` avec fallback, ou itérer sur `model.classes_` :

```python
proba_baisse = probas[:, class_map.get(-1.0, 0)] if -1.0 in class_map else np.zeros(len(probas))
proba_neutre = probas[:, class_map.get(0.0, 0)] if 0.0 in class_map else np.zeros(len(probas))
proba_hausse = probas[:, class_map.get(1.0, 0)] if 1.0 in class_map else np.zeros(len(probas))
```

---

## 2. Régression Fonctionnelle (P1)

### R1 — [`backtest/simulator.py:162-169`](../learning_machine_learning/backtest/simulator.py#L162) : Perte de la traçabilité `filter_rejected` par trade

**Sévérité :** P1 — Information de diagnostic perdue

**Description :** L'ancien code (`backtest_utils.py:124,160,176`) traçait quel filtre avait rejeté chaque signal dans la colonne `Filter_Rejected`. Le nouveau code délègue au `FilterPipeline` mais ne propage pas l'information de rejet par barre jusqu'au DataFrame de trades. Résultat : tous les trades ont `filter_rejected=""`.

**Correctif :** Le `FilterPipeline.apply()` doit retourner un masque de rejet par filtre, ou le simulateur doit reconstruire l'information à partir des masques avant/après filtrage.

---

### R2 — [`pipelines/base.py:97`](../learning_machine_learning/pipelines/base.py#L97) : Fonction de sizing incorrecte

**Sévérité :** P1 — Résultats de backtest différents de l'original

**Description :** `run_backtest()` utilise `weight_linear` (seuil 0.45, range 0.5-1.5) au lieu de `weight_centered` (seuil 0.35, range 0.8-1.2) qui est la fonction utilisée par [`4_backtest_triple_barrier.py:26`](../4_backtest_triple_barrier.py#L26).

---

### R3 — [`features/merger.py:86-93`](../learning_machine_learning/features/merger.py#L86) : Crash potentiel si feat_h4 ou feat_d1 est un DataFrame vide

**Sévérité :** P1 — Plantage si un timeframe n'a pas de features

**Description :** [`features/pipeline.py:150-151`](../learning_machine_learning/features/pipeline.py#L150) passe `pd.DataFrame()` quand `feat_h4` est None. Dans `merge_features`, `feat_h4.sort_index().reset_index()` sur un DataFrame vide produira un DataFrame sans colonnes, et `feat_h4["Time"]` lèvera une `KeyError`.

**Correctif :** Vérifier `feat_h4.empty` avant d'appeler `merge_features`, ou gérer le cas vide dans `merge_features`.

---

## 3. Code Dupliqué (P2)

| # | Duplication | Emplacement 1 | Emplacement 2 | Action |
|---|-----------|--------------|--------------|--------|
| D1 | `log_row_loss()` | [`features/merger.py:39`](../learning_machine_learning/features/merger.py#L39) | [`backtest_utils.py:42`](../backtest_utils.py#L42) | Supprimer version `backtest_utils`, utiliser `merger.log_row_loss` |
| D2 | `_pips_to_return()` | [`backtest/metrics.py:18`](../learning_machine_learning/backtest/metrics.py#L18) | [`backtest_utils.py:37`](../backtest_utils.py#L37) | Déjà migré, `backtest_utils.py` est du code mort |
| D3 | `_normalize_seuil()` | [`backtest/simulator.py:22`](../learning_machine_learning/backtest/simulator.py#L22) | [`backtest_utils.py:80`](../backtest_utils.py#L80) | Déjà migré, `backtest_utils.py` est du code mort |
| D4 | Calcul `proba_max` | [`backtest/simulator.py:65-69`](../learning_machine_learning/backtest/simulator.py#L65) | [`backtest_utils.py:119-121`](../backtest_utils.py#L119) | Déjà migré |
| D5 | Extraction BBands colonnes | [`features/pipeline.py:108-112`](../learning_machine_learning/features/pipeline.py#L108) | [`features/technical.py:133-137`](../learning_machine_learning/features/technical.py#L133) | `pipeline.py` devrait appeler `calc_bb_width()` au lieu de dupliquer |

---

## 4. Code Mort (P2)

| # | Élément | Emplacement | Action |
|---|---------|------------|--------|
| M1 | Scripts racine numérotés | `1_clean_data.py`, `2_master_feature_engineering.py`, `3_model_training.py`, `4_backtest_triple_barrier.py`, `5_analyze_losses.py` | Supprimer ou remplacer par des points d'entrée minces |
| M2 | `backtest_utils.py` | Racine | Entièrement remplacé par `backtest/*`, supprimer |
| M3 | `optimize_sizing.py` | Racine | Remplacé par `backtest/sizing.py` |
| M4 | `config.py` | Racine | Remplacé par `config/*`, mais encore importé par les vieux scripts |
| M5 | `class MetricDict(dict): pass` | [`core/types.py:68`](../learning_machine_learning/core/types.py#L68) | Classe vide, aucune valeur ajoutée |
| M6 | `_diag_direction.py` | Racine | Déjà migré vers [`analysis/diagnostics.py`](../learning_machine_learning/analysis/diagnostics.py) |
| M7 | Import `numpy as np` dans fonction | [`features/pipeline.py:53`](../learning_machine_learning/features/pipeline.py#L53) | `np.log` utilisé une seule fois, `np` déjà importé via `numpy` au niveau module ? Non, c'est un lazy import. OK mais inélégant. |

---

## 5. Performance (P2)

| # | Problème | Emplacement | Impact |
|---|---------|------------|--------|
| P1 | Boucle `while` Python pure dans `simulate_trades` | [`backtest/simulator.py:114-171`](../learning_machine_learning/backtest/simulator.py#L114) | ~10s pour 4 ans H1. Acceptable pour EURUSD (~35K barres/an). Deviendra problématique pour BTCUSD ou timeframes inférieurs. |
| P2 | `dropna()` global après merge | [`features/merger.py:123`](../learning_machine_learning/features/merger.py#L123) | Supprime toutes les lignes avec AU MOINS un NaN. Pourrait être plus sélectif (ne dropper que si feature critique manquante). |
| P3 | `permutation_importance` sur tout le val_year | [`model/evaluation.py:85`](../learning_machine_learning/model/evaluation.py#L85) | Utilize déjà `n_jobs=-1`. OK. |
| P4 | Appels `ta.ema()` multiples dans `build_ml_ready` | [`features/pipeline.py:96-98`](../learning_machine_learning/features/pipeline.py#L96) | Chaque appel recalcule l'EMA. Pourrait calculer EMAs une fois et réutiliser. |

---

## 6. Testabilité (P3)

| # | Problème | Emplacement | Action |
|---|---------|------------|--------|
| T1 | `simulate_trades` non testé unitairement | [`backtest/simulator.py`](../learning_machine_learning/backtest/simulator.py) | Ajouter `tests/unit/test_simulator.py` couvrant: SL touché, TP touché, timeout, stateful, 0 trade |
| T2 | `triple_barrier` — le bug B1 n'est pas couvert par les tests | Pas de test existant pour ce cas | Ajouter test: SL touché puis TP touché dans la même fenêtre → doit être 0 ou -1, pas 1 |
| T3 | `FilterPipeline` non testé en isolation | Pas de test unitaire | Tester chaque filtre individuellement + composite |
| T4 | `merge_features()` non testé | Pas de test unitaire | Tester avec DataFrames vides, avec/sans offset |
| T5 | Tests existants ne couvrent pas les cas limites | [`tests/unit/`](../tests/unit/) | Pandera valide la forme, pas la sémantique des features |

---

## 7. Qualité de Code (P3)

| # | Problème | Emplacement |
|---|---------|------------|
| Q1 | `FilterPipeline.__init__` type annotation faible | [`backtest/filters.py:144`](../learning_machine_learning/backtest/filters.py#L144) : `list` au lieu de `list[TrendFilter | VolFilter | SessionFilter]` |
| Q2 | `compute_target_series` supprime les NaN sans expliquer pourquoi | [`features/triple_barrier.py:143`](../learning_machine_learning/features/triple_barrier.py#L143) |
| Q3 | `build_ml_ready()` a 182 lignes — trop de responsabilités | [`features/pipeline.py`](../learning_machine_learning/features/pipeline.py) : fait labelling + features H1 + features H4/D1 + macro + merge + sélection |
| Q4 | `features/pipeline.py:55-69` imports lazy dans la fonction | 15 lignes d'imports au milieu d'une fonction de 180 lignes |

---

## 8. Plan de Correction — Priorisé

### Phase 1 : Bugs Critiques (P0)

| Étape | Fichier | Action |
|-------|---------|--------|
| 1.1 | `features/triple_barrier.py` | Corriger B1 : ajouter flags `long_dead`/`short_dead` |
| 1.2 | `backtest/simulator.py` | Corriger B2 : PnL réel pour les timeouts |
| 1.3 | `model/prediction.py` | Corriger B3 : extraction robuste des probas |

### Phase 2 : Régressions (P1)

| Étape | Fichier | Action |
|-------|---------|--------|
| 2.1 | `backtest/simulator.py` + `backtest/filters.py` | Restaurer traçabilité `filter_rejected` par trade (R1) |
| 2.2 | `pipelines/base.py` | Remplacer `weight_linear` par `weight_centered` (R2) |
| 2.3 | `features/merger.py` | Gérer le cas DataFrame vide (R3) |

### Phase 3 : Duplication & Code Mort (P2)

| Étape | Fichier | Action |
|-------|---------|--------|
| 3.1 | `features/pipeline.py` | Remplacer calcul BBands inline par appel à `calc_bb_width()` (D5) |
| 3.2 | `core/types.py` | Supprimer `MetricDict` vide (M5) |
| 3.3 | Racine | Supprimer les scripts obsolètes ou les remplacer par des wrappers minces |

### Phase 4 : Tests (P3)

| Étape | Fichier | Action |
|-------|---------|--------|
| 4.1 | `tests/unit/` | Ajouter `test_triple_barrier.py` (cas SL puis TP) |
| 4.2 | `tests/unit/` | Ajouter `test_simulator.py` |
| 4.3 | `tests/unit/` | Ajouter `test_filters.py` (déjà partiellement existant?) |
| 4.4 | `tests/unit/` | Ajouter `test_merger.py` |

---

## 9. Résumé

| Catégorie | Count | Sévérité max |
|-----------|-------|-------------|
| Bugs logiques | 3 | P0 |
| Régressions | 3 | P1 |
| Duplications | 5 | P2 |
| Code mort | 7 | P2 |
| Performance | 4 | P2 |
| Testabilité | 5 | P3 |
| Qualité | 4 | P3 |

**Conclusion :** La migration vers le package `learning_machine_learning/` est structurellement saine mais a introduit 3 bugs de régression (B1 critique, R1-R3). Le bug B1 dans `triple_barrier.py` est le plus grave : il fausse les labels d'entraînement et donc tout le pipeline aval.
