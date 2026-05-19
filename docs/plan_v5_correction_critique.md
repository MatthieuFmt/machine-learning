# Plan v5 — Corrections critiques avant tout nouveau test

**Objectif** : fiabiliser le pipeline pour que les Sharpe affichés correspondent à des Sharpe **réels et reproductibles**. Aucune nouvelle stratégie n'est ajoutée tant que ces corrections ne sont pas effectives.

**Lien** : voir [`audit_v4_findings.md`](audit_v4_findings.md) pour le détail des bugs.

---

## Phase 1 — Bloqueurs méthodologiques (PRIORITÉ 1)

### Action 1.1 — Réparer la rupture de distribution train/test (F1)

**Cible** : `scripts/run_phase_b_c5_*.py` + `scripts/run_validation_finale.py`.

**Choix** :

#### Option A (méta-labeling pur, fidèle à H05 v2)

- **TRAIN** : Donchian génère les signaux → backtest produit les trades → meta-RF apprend `P(winner | features at entry)`.
- **TEST** : Donchian génère les signaux → on FILTRE par `meta_proba > seuil` → on garde le sous-ensemble. **Direction = Donchian, jamais features de tendance.**

C'est la définition de López de Prado §3 et l'esprit de H05.

#### Option B (modèle directionnel)

Si on veut un modèle qui prédit la direction (pas méta-labeling), il faut redéfinir la target en train :
- `y = sign(forward_return(horizon))` (1, -1, 0)
- entraîner un classifieur multi-classe
- prédire en test sur les mêmes barres
- **Pas de gating par "trend_sign" sorti du chapeau**

Les deux sont valides ; **mélanger les deux dans le même script ne l'est pas**.

→ Choisir Option A pour rester compatible avec la roadmap C5.

### Action 1.2 — Corriger le Sharpe (F2)

**Cible** : `app/backtest/metrics.py` — fonction `sharpe_annualized` et appels en aval.

**Diagnostic** :
- Sizing utilise `capital_eur=10000` constant.
- `compute_metrics` compose `capital + cumsum(pnl)` puis `pct_change()` → calcul de Sharpe en composé.
- Incohérence.

**Trois corrections possibles** (à débattre, ne PAS coder sans décision) :

1. **Sharpe linéaire** : `daily_pnl_eur.cumsum() / capital_eur_initial` → `diff()` → vrais retours linéaires. Cohérent avec sizing à capital fixe. Le plus simple.
2. **Capital adaptatif** : recalculer les lots à chaque trade avec `capital_eur = capital_initial + pnl_cumul`. Sharpe pct_change devient cohérent. Plus réaliste mais demande de modifier `compute_position_size`.
3. **Volatility targeting** : sizing inversement proportionnel à la vol récente. Sharpe pct_change naturel.

**Recommandation** : pour l'audit, choisir (1) — le moins invasif, le plus interprétable.

À ajouter : un **test unitaire** qui vérifie que pour un set de trades constants, `sharpe_compute_metrics ≈ sharpe_backtest` à ±10 %.

### Action 1.3 — Fixer le TP-prime → SL-prime (F3)

**Cible** : [`app/backtest/deterministic.py:132-156`](../app/backtest/deterministic.py#L132-L156).

Remplacer l'ordre TP→SL par SL→TP en cas de touche same-bar. Corriger aussi le commentaire mensonger ("conservateur"). Aligner sur le simulateur stateful.

**Impact attendu** :
- WR baisse de 1–5 % selon l'actif (BTCUSD, US30, ETHUSD : effet fort ; EURUSD : effet faible).
- Sharpe random Monte Carlo baisse de ~30 % (plus de cohérence).

**Test à ajouter** : un test unitaire avec une bougie synthétique où high>tp ET low<sl, vérifier que le résultat = `loss_sl`.

---

## Phase 2 — Look-ahead et statistiques (PRIORITÉ 2)

### Action 2.1 — Stacking sans look-ahead (F4)

**Cible** : [`app/models/candidates.py:42-59`](../app/models/candidates.py#L42-L59).

Remplacer :
```python
StackingClassifier(estimators=..., cv=5)
CalibratedClassifierCV(stacking, method="isotonic", cv=3)
```

par :
```python
from sklearn.model_selection import TimeSeriesSplit
StackingClassifier(estimators=..., cv=TimeSeriesSplit(n_splits=5))
CalibratedClassifierCV(stacking, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
```

ou utiliser des CV custom *purged*, comme `purged_kfold_cv` déjà présent.

### Action 2.2 — DSR : aligner n_trials sur les vrais reads (F5)

**Cible** : `scripts/run_validation_finale.py:61` + autres consommateurs de `n_trials`.

Décision méthodologique :
- **Option stricte** : `N_TRIALS_CUMUL = len(read_history)` = 44 → DSR sous-déflaté résolu, mais c'est conservateur.
- **Option Bailey** : compter les **configurations testées indépendantes** (hyperparams uniques × seed × actif × TF), pas les runs. Calculer ce N depuis JOURNAL.md.

Documenter le choix dans un encart "Méthodologie n_trials".

### Action 2.3 — Monte Carlo benchmark représentatif (F6)

**Cible** : `monte_carlo_random_benchmark` ligne 625.

Remplacer :
- Random sur **1 seul actif** US30 → random sur les **5 actifs** réellement utilisés.
- `signal_freq=0.05` hardcoded → calibrer sur la **fréquence réelle** observée pour chaque actif.
- **Inclure le bug F3 fixé** (sinon le P95 reste gonflé).
- Augmenter `n_iter` à 1000–5000.

Sortir P5, P50, P95, P99 pour donner une vision distributionnelle.

### Action 2.4 — Bootstrap block (F15)

**Cible** : [`app/analysis/edge_validation.py:123-160`](../app/analysis/edge_validation.py#L123-L160).

Remplacer le bootstrap iid par un **stationary bootstrap** (Politis-Romano) avec block_size moyen = 10. Implémentation : `arch.bootstrap.StationaryBootstrap` (lib `arch`), ou implémentation native via geometric block sampling.

### Action 2.5 — Supprimer le code mort (F16)

**Cible** : `_compute_sharpe_from_returns` et `validate_edge_distribution` dans edge_validation.py.

Soit supprimer, soit corriger pour utiliser `sharpe_ratio()` (la version annualisée). Ajouter une note dans le module.

---

## Phase 3 — Robustesse features (PRIORITÉ 3)

### Action 3.1 — Cross-asset : assertion de convention timestamp (F7)

**Cible** : [`app/features/superset.py:340-348`](../app/features/superset.py#L340-L348) + `app/data/loader.py`.

À l'import d'un fichier D1, **inspecter explicitement** le premier timestamp pour deviner la convention :
- Si `00:00:00` → start-of-day (le close de D1 représente le close à la fin de la journée).
- Si `23:00:00` ou `23:59:59` → end-of-day.

Documenter dans `app/data/loader.py` la convention attendue, et **lever DataValidationError** si mismatch.

Pour la reindex H1, **toujours faire `shift(1)` avant `ffill`** pour garantir que la valeur D1 de la journée J n'apparaît qu'à partir de J+1 00:00.

### Action 3.2 — Validation look-ahead générique active (F8)

**Cible** : `tests/unit/test_indicators_look_ahead.py`.

Retirer les `except Exception: pytest.skip(...)` larges. Si une fonction ne peut pas être auto-testée, **marquer un xfail explicite** avec raison documentée. Au moins, **fail le test** si plus de 20 % des fonctions sont skip.

Ajouter des tests **manuels** ciblés pour les fonctions skippées :
- `cross_asset_features`
- `economic_features_for_index`
- `session_features`

Au-delà du décorateur, ajouter un test qui parcourt **chaque feature du superset** une par une.

### Action 3.3 — Fenêtre window_hours fiable (F18)

**Cible** : [`app/backtest/deterministic.py:76-81`](../app/backtest/deterministic.py#L76-L81).

Calculer `window_bars` à partir du **bar-spacing modal** (le plus fréquent diff hors weekends), pas la moyenne :
```python
diffs_hours = df.index.to_series().diff().dt.total_seconds() / 3600.0
typical_hours = float(diffs_hours.mode().iloc[0])    # le plus fréquent
window_bars = max(1, int(round(window_hours / typical_hours)))
```

→ Pour H1, on obtient 1.0 h/bar (et donc window_bars = 120) au lieu de 1.5.

---

## Phase 4 — Cosmétique & dette (PRIORITÉ 4)

### Action 4.1 — Format WR/max_dd dans le rapport (F11)

**Cible** : [`scripts/run_validation_finale.py:932-935`](../scripts/run_validation_finale.py#L932-L935).

`wr` est déjà en %, ne pas refaire `{:.1%}`. Utiliser `{:.1f}%` ou diviser par 100.

### Action 4.2 — Tracer max_dd_pct = -1549 % (F12)

Repérer pourquoi `compute_metrics` retourne des valeurs hors [-100, 0]. Suspect : chemin legacy pris quand `position_size_lots` est absent ; `_pips_to_return` avec mauvais pip_value_eur.

Ajouter une **assertion** : `assert max_dd_pct in [-100, 0]` à la sortie de `compute_metrics`.

### Action 4.3 — Nettoyer les `.bak` (F19)

**Cible** : `tests/unit/*.bak`.

Soit ré-activer les tests, soit supprimer définitivement avec note dans le commit. La règle :
- Si le code testé existe encore → réactiver.
- Si le code a disparu → supprimer.

Ne jamais laisser un `.bak` traîner — c'est un "fantôme" qui ment au lecteur.

### Action 4.4 — Inclure 2023 dans le workflow (F14)

**Décision méthodologique** : 2023 doit servir de validation set.
- Phase de tuning : utiliser 2023 pour calibrer hyperparams et seuils.
- Phase de test : 2024-2026 reste intouché.
- Le compteur n_trials s'incrémente sur 2023, pas sur 2024+.

À refléter dans la constitution (`prompts/00_constitution.md` §3).

---

## Phase 5 — Sanity checks à ajouter

Une suite de tests qui doit passer **avant** toute nouvelle hypothèse :

1. **Sanity test simulateur** :
   - 1 trade synthétique avec TP=20, SL=10 → vérifier `pips_net = 20 - cost` (win) ou `-10 - cost` (loss).
   - Bougie large-range avec TP+SL touchés → vérifier `result = loss_sl` (post F3).

2. **Sanity test sharpe** :
   - Trades générés artificiellement avec mean=0.1%, std=1% sur 252 jours → Sharpe doit être ≈ 0.1/1 × √252 ≈ 1.59 (à ±10 %).

3. **Sanity test cross-asset** :
   - Charger BTCUSD D1, vérifier que la valeur sur EURUSD H1 à `J 00:00` est la close de **J-1**, jamais J.

4. **Sanity test consistance Sharpe** :
   - Sur les mêmes trades, `sharpe_backtest` et `sharpe_compute_metrics` doivent différer de moins de 10 % (post F2).

5. **Sanity test DSR** :
   - DSR(SR=2, N=10, n_obs=500, normal returns) → vérifier valeur attendue ≈ 5.5, p ≈ 0.

---

## Ordre d'exécution recommandé

```
Phase 1 (bloqueurs)       Phase 2 (stats)        Phase 3 (features)
  ├─ Action 1.1  ──┐         ├─ Action 2.1            ├─ Action 3.1
  ├─ Action 1.2  ──┼──> Refaire validation ──> Action 2.2 → Action 2.3
  └─ Action 1.3  ──┘                            ├─ Action 2.4
                                                └─ Action 2.5

Phase 4 (cosmétique)    Phase 5 (sanity)
  Indépendantes          À chaque action 1-3
```

**Estimation** :
- Phase 1 : ~1 journée de code + 1 rejeu validation.
- Phase 2 : ~2 jours.
- Phase 3 : ~1 jour.
- Phase 4 : ~2 heures.
- Phase 5 : ~1 jour de tests.

Total : **5–7 jours de travail** avant de pouvoir conclure si le pipeline donne des Sharpe réels.

---

## Critères de sortie de la Phase Correction

Avant de passer à de nouvelles stratégies (voir [`plan_v5_amelioration_strategies.md`](plan_v5_amelioration_strategies.md)), il faut :

- [ ] `sharpe_backtest` et `sharpe_compute_metrics` cohérents à ±10 % sur les 6 couples.
- [ ] Test SL-prime same-bar passe.
- [ ] Validation finale ré-exécutée avec pipeline corrigé.
- [ ] Tous les chiffres dans `predictions/validation_finale_v5.json` sont dans des plages physiques (WR ∈ [0, 1], max_dd ∈ [-1, 0]).
- [ ] Le P95 Monte Carlo est < 5 (signe que F3 est résolu).
- [ ] Au moins 1 stratégie passe le DSR avec p<0.05 ET n_trials=44+, OU on accepte que NO-GO est correct.

Si **aucune** stratégie ne survit après correction → c'est un signal fort qu'il faut **changer d'approche** (voir le plan d'amélioration).
