# Plan de Refactoring — Exécution Priorisée

**Date :** 2026-05-11  
**Source :** [`docs/CODE_REVIEW_AUDIT.md`](CODE_REVIEW_AUDIT.md) + revue fraîche du code actuel  
**Statut :** Prêt pour exécution en mode Code

---

## Synthèse de l'état actuel

L'audit initial (CODE_REVIEW_AUDIT.md) identifiait 3 bugs P0, 3 régressions P1, 5 duplications, 7 code mort, 4 perf, 5 testabilité, 4 qualité.  
Après vérification du code actuel (git status montre 10 fichiers modifiés), plusieurs correctifs sont déjà appliqués :

| Issue | Statut | Note |
|-------|--------|------|
| B1 — SL n'arrête pas la direction | ✅ Corrigé | `long_dead`/`short_dead` flags présents |
| B2 — Timeout assume perte max | ✅ Corrigé | PnL réel calculé sur Close final (lignes 166-173) |
| B3 — Extraction probas fragile | ✅ Corrigé | `_get_col()` avec fallback zéros |
| R1 — Traçabilité filter_rejected | ✅ Corrigé | `FilterPipeline.apply()` retourne `rejection_reason` Series |
| R2 — Fonction sizing incorrecte | ✅ Corrigé | `weight_centered` utilisé dans `base.py:97` |
| R3 — Crash DataFrame vide | ✅ Corrigé | `feat_h4.empty` vérifié dans `merger.py:92` |
| M5 — `MetricDict` vide | ✅ Supprimé | N'existe plus dans `core/types.py` |

**Restent 7 problèmes non résolus** — listés ci-dessous par priorité.

---

## Phase 1 — Correctifs critiques (à exécuter en premier)

### P1.1 — [`simulator.py:133`](../learning_machine_learning/backtest/simulator.py#L133) : `pips_brut` non initialisé → `UnboundLocalError` en bordure de données

**Sévérité :** P0 (crash)  
**Description :** Ligne 133 déclare `pips_brut: float` (annotation de type, pas une initialisation). Quand `idx >= n` dans la boucle forward (ligne 138-140), le `break` sort du `for` sans exécuter le `else`, et `pips_brut` est référencé ligne 178 sans avoir été assigné.

**Correctif :**
```python
# Remplacer ligne 133 :
pips_brut = 0.0
result_type = "loss_timeout"  # valeur par défaut sécuritaire
```

Ajouter un test unitaire : signal sur les 3 dernières barres → pas de crash.

---

### P1.2 — [`pipeline.py:108-112`](../learning_machine_learning/features/pipeline.py#L108) : Duplication du calcul BB_Width (D5)

**Sévérité :** P2 (duplication)  
**Description :** `build_ml_ready()` recalcule les BBands inline (lignes 108-112) alors que [`calc_bb_width()`](../learning_machine_learning/features/technical.py#L120) existe déjà et fait exactement la même chose.

**Correctif :** Remplacer les lignes 108-112 par :
```python
h1["BB_Width"] = calc_bb_width(h1[["Close"]])
```

Note : `calc_bb_width` attend un DataFrame avec colonne `'Close'`, il faut adapter l'appel.

---

### P1.3 — [`pipeline.py:96-98`](../learning_machine_learning/features/pipeline.py#L96) : 3 appels `ta.ema()` redondants (P1)

**Sévérité :** P3 (micro-optim)  
**Description :** 3 appels `ta.ema(close, length=9)`, `ta.ema(close, length=21)`, `ta.ema(close, length=50)` calculent chacun une EMA indépendamment. On peut les remplacer par un appel à [`calc_ema_distance()`](../learning_machine_learning/features/technical.py#L55) qui existe déjà, ou au moins les grouper.

**Correctif :** Remplacer les lignes 96-98 par :
```python
ema_dists = calc_ema_distance(h1, periods=(9, 21, 50))
h1["Dist_EMA_9"] = ema_dists["Dist_EMA_9"]
h1["Dist_EMA_21"] = ema_dists["Dist_EMA_21"]
h1["Dist_EMA_50"] = ema_dists["Dist_EMA_50"]
```

---

### P1.4 — [`pipeline.py:55-69`](../learning_machine_learning/features/pipeline.py#L55) : Imports lazy au milieu de la fonction (Q4)

**Sévérité :** P3 (qualité)  
**Description :** 15 lignes d'imports au milieu de `build_ml_ready()` (lignes 55-69). Ces imports lazy sont là pour éviter des imports circulaires. Solution : déplacer au niveau module et résoudre la circularité.

**Correctif :** Déplacer tous les imports en haut du fichier. Si circularité, extraire les fonctions utilisées dans un module utilitaire ou utiliser `from __future__ import annotations` (déjà présent) + `TYPE_CHECKING` si nécessaire.

---

## Phase 2 — Robustesse et testabilité

### P2.1 — Créer [`tests/unit/test_simulator.py`](../tests/unit/test_simulator.py)

**Sévérité :** P2 (testabilité)  
**Tests requis :**
| Test | Description |
|------|-------------|
| `test_tp_touche` | LONG avec TP touché → win, pips_bruts = tp_pips - spread_cost |
| `test_sl_touche` | LONG avec SL touché → loss_sl, pips_bruts = -sl_pips - spread_cost |
| `test_timeout_pnl_reel` | Timeout → PnL basé sur Close final (B2) |
| `test_stateful_saut_barres` | Après un trade, on saute les barres consommées |
| `test_zero_signal` | DataFrame sans signal → 0 trades |
| `test_signal_fin_data` | Signal proche de la fin → pas de crash (P1.1) |
| `test_filtres_appliques` | Avec FilterPipeline → n_filtres_appliques peuplé |
| `test_spread_cost` | Coût de spread correctement appliqué |

---

### P2.2 — Renforcer [`tests/unit/test_triple_barrier.py`](../tests/unit/test_triple_barrier.py)

Tests déjà bons. Ajouter :
| Test | Description |
|------|-------------|
| `test_long_dead_short_win` | Déjà présent (test_sl_touche_puis_tp_reste_perdant) ✅ |
| `test_short_dead_long_win` | Symétrique du précédent |
| `test_both_dead` | Les deux SL touchés → 0 |
| `test_both_win` | Les deux TP touchés → 0 (ambigu) |
| `test_very_small_window` | window=1 → seules les barres n-1 sont NaN |

---

### P2.3 — [`pipeline.py:52-53`](../learning_machine_learning/features/pipeline.py#L52) : Import `numpy as np` redondant

**Sévérité :** P3  
**Description :** `numpy` est déjà importé en haut de `pipeline.py` via les dépendances transitives ? Non, vérifions : le fichier n'importe pas `numpy` au niveau module. Mais `np.log` est utilisé ligne 95. L'import est donc nécessaire mais devrait être au niveau module, pas à l'intérieur de la fonction.

**Correctif :** Déplacer `import numpy as np` et `import pandas_ta as ta` en haut du fichier.

---

## Phase 3 — Qualité de code

### P3.1 — [`simulator.py:65-69`](../learning_machine_learning/backtest/simulator.py#L65) : Extraction `proba_max` inline

**Sévérité :** P3  
**Description :** Le calcul de `proba_max` est fait inline. Pourrait être extrait dans une fonction pure `_compute_proba_max(df)` → testable.

---

### P3.2 — [`merger.py:126-128`](../learning_machine_learning/features/merger.py#L126) : `dropna()` global

**Sévérité :** P3  
**Description :** `dropna(inplace=True)` supprime toute ligne avec au moins un NaN. Pourrait être plus sélectif — dropper seulement si une feature critique est NaN. Pour l'instant, acceptable car l'alternative complexifierait sans bénéfice clair.

---

## Résumé des modifications à effectuer

| # | Fichier | Action | Sévérité |
|---|---------|--------|----------|
| 1 | [`simulator.py`](../learning_machine_learning/backtest/simulator.py#L133) | Initialiser `pips_brut = 0.0` + `result_type` | P0 |
| 2 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py#L108-L112) | Remplacer BBands inline par `calc_bb_width()` | P2 |
| 3 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py#L96-L98) | Remplacer 3× EMA par `calc_ema_distance()` | P3 |
| 4 | [`pipeline.py`](../learning_machine_learning/features/pipeline.py#L52-L69) | Remonter imports au niveau module | P3 |
| 5 | [`tests/unit/test_simulator.py`](../tests/unit/test_simulator.py) | Créer fichier de test (8 cas) | P2 |
| 6 | [`tests/unit/test_triple_barrier.py`](../tests/unit/test_triple_barrier.py) | Ajouter 3 cas symétriques | P3 |

**Total :** 6 interventions, temps estimé ~30 min.
