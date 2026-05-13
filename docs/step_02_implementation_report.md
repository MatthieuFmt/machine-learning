# Step 02 — Rapport d'implémentation : Framework de validation robuste (CPCV + DSR)

**Date** : 2026-05-12
**Version** : v1.0
**Fichier stratégique** : [`step_02_robust_validation_framework.md`](step_02_robust_validation_framework.md)

---

## Résumé

Implémentation complète du Combinatorial Purged Cross-Validation (CPCV) et du Deflated Sharpe Ratio (DSR) distributionnel selon López de Prado (Advances in Financial ML, ch. 11–12). Cinq modifications atomiques couvrant 2 fichiers modifiés et 3 fichiers créés.

---

## Fichiers modifiés/créés

| Fichier | Type | Lignes | Description |
|---|---|---|---|
| [`learning_machine_learning/analysis/edge_validation.py`](../learning_machine_learning/analysis/edge_validation.py) | Modifié | +173 | Ajout `psr_from_returns()`, `deflated_sharpe_ratio_from_distribution()`, `validate_edge_distribution()` |
| [`learning_machine_learning/model/training.py`](../learning_machine_learning/model/training.py) | Modifié | +15 | `train_test_split_purge` accepte `train_mask`/`test_mask` arbitraires |
| [`learning_machine_learning/analysis/cpcv.py`](../learning_machine_learning/analysis/cpcv.py) | Créé | 592 | Module CPCV : splits, backtest parallèle, agrégation |
| [`tests/unit/test_cpcv.py`](../tests/unit/test_cpcv.py) | Créé | 281 | 20 tests unitaires (tous passent) |
| [`run_validation_cpcv.py`](../run_validation_cpcv.py) | Créé | 440 | Script orchestrateur standalone |

---

## Architecture

```
run_validation_cpcv.py (orchestrateur)
│
├── EurUsdPipeline.load_data() → ohlcv_h1
├── EurUsdPipeline.build_features() → ml_data_full
│
├── generate_cpcv_splits()
│   ├── 48 groupes sur 2024-2025 (2 semaines/groupe)
│   ├── k_test=12 groupes (6 mois) par split
│   ├── purge bidirectionnelle 48h avant ET après chaque groupe test
│   └── 200 combinaisons aléatoires (seed=42)
│
├── run_cpcv_backtest()  [joblib.Parallel(n_jobs=-1)]
│   └── _run_one_split() × 200
│       ├── train_model / train_regressor (selon target_mode)
│       ├── predict + simulate_trades / simulate_trades_continuous
│       └── métriques (Sharpe, win_rate, drawdown, profit_net)
│
├── aggregate_cpcv_metrics()
│   └── E[SR], σ[SR], médiane, CI 95%, % profitables, coverage
│
├── deflated_sharpe_ratio_from_distribution()
│   ├── SR₀* = √Var({SRᵢ}) · ((1−γ)Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))
│   ├── DSR = (ŜR − SR₀*) / σ(SR)
│   └── PSR(SR*=0) simplifié
│
├── validate_edge_distribution() [split principal train≤2023, test=2025]
│   ├── validate_edge() (tests classiques)
│   ├── psr_from_returns() (Bailey PSR)
│   └── DSR CPCV
│
└── Rapport Markdown → predictions/cpcv_report.md
    + CSV → predictions/cpcv_results.csv
```

---

## Détail des modifications

### Mod 1 — [`edge_validation.py`](../learning_machine_learning/analysis/edge_validation.py)

**`psr_from_returns(returns, sr_benchmark=0.0) → float`**
- Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).
- Formule complète avec skewness (`γ̂₃`) et kurtosis (`γ̂₄`).
- `Φ( (ŜR − SR*) · √(n−1) / √(1 − γ̂₃·ŜR + (γ̂₄−1)/4 · ŜR²) )`

**`deflated_sharpe_ratio_from_distribution(observed_sr, sharpe_distribution) → dict`**
- Calcule `SR₀*` (seuil déflaté) à partir de la distribution CPCV empirique.
- Utilise la formule EVT (γ d'Euler-Mascheroni) :
  `E[max Z] = (1−γ)Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))`
- DSR = `(ŜR − SR₀*) / σ(SR)`
- Retourne un dict complet avec toutes les statistiques descriptives.

**`validate_edge_distribution(trades_df, backtest_cfg, sharpe_distribution=None) → dict`**
- Wrapper combinant `validate_edge()` + `psr_from_returns()` + DSR distributionnel.
- Point d'entrée unique pour le reporting.

### Mod 2 — [`training.py`](../learning_machine_learning/model/training.py)

`train_test_split_purge()` accepte maintenant `train_mask`/`test_mask` keyword-only :
- `train_mask: pd.Series | None = None`
- `test_mask: pd.Series | None = None`
- Si fournis → split arbitraire par masque booléen (utilisé par CPCV).
- Sinon → comportement inchangé (split par `train_end_year`).

### Mod 3 — [`cpcv.py`](../learning_machine_learning/analysis/cpcv.py)

**`generate_cpcv_splits(index, n_groups, k_test, purge_hours, n_samples, random_state) → Iterator`**
- Partitionne le `DatetimeIndex` en groupes contigus (~égaux).
- Échantillonne `n_samples` combinaisons de `k_test` groupes-test.
- Purge bidirectionnelle : exclut `[t_start − purge, t_start)` et `(t_end, t_end + purge]`.
- Vérifications d'invariants programmatiques (pas de chevauchement).

**`_run_one_split(split_id, train_idx, test_idx, ...) → dict`**
- Pipeline complet sur UN split : train → predict → simulate → metrics.
- Supporte `triple_barrier` (classifieur + probas) et `forward_return` (régresseur).
- Applique les mêmes filtres (MomentumFilter, VolFilter, SessionFilter) que `BasePipeline.run_backtest()`.

**`run_cpcv_backtest(ml_data, ohlcv_h1, splits, model_factory, ...) → DataFrame`**
- Parallélise via `joblib.Parallel(n_jobs=n_jobs)`.
- `model_factory` doit garantir `n_jobs=1` pour le RandomForest (prévention fork-bomb).

**`aggregate_cpcv_metrics(results_df) → dict`**
- Agrège les 200 lignes du DataFrame en stats descriptives.
- Calcule la couverture temporelle par mois.

### Mod 4 — [`test_cpcv.py`](../tests/unit/test_cpcv.py)

20 tests, 4 classes :
- `TestGenerateCpcvSplits` (7) : pas de chevauchement, purge bidirectionnelle, n_samples respecté, ordre temporel, couverture, index non-DatetimeIndex, n_groups < 2.
- `TestPsrFromReturns` (5) : returns positifs, distribution normale, 0 mean, distribution dégénérée, benchmark non nul.
- `TestDeflatedSharpeRatioDistribution` (4) : distribution profitable, non-profitable, haute variance, < 2 splits.
- `TestAggregateCpcvMetrics` (4) : DataFrame vide, structure, colonnes requises, coverage.

### Mod 5 — [`run_validation_cpcv.py`](../run_validation_cpcv.py)

Script standalone exécutable :
```bash
python run_validation_cpcv.py
```

Flux :
1. Charge données EUR/USD via `EurUsdPipeline`
2. Génère CPCV splits sur période OOS 2024-2025
3. Exécute 200 backtests parallèles
4. Agrège métriques → DSR distributionnel → rapport Markdown
5. Lance le split principal (train≤2023, test=2025) pour `validate_edge_distribution`
6. Verdict GO/NO-GO basé sur 3 critères (DSR > 0, % profitables > 60%, E[SR] > 0)

Sorties :
- `predictions/cpcv_results.csv` — DataFrame 200 lignes
- `predictions/cpcv_report.md` — Rapport Markdown structuré

---

## Invariants anti-leak vérifiés

Chaque split CPCV (`generate_cpcv_splits`) garantit :

1. **Index monotonic** : `index.is_monotonic_increasing`
2. **Pas de chevauchement** : `len(np.intersect1d(train_idx, test_idx)) == 0`
3. **Purge avant test** : `max(train_ts) + purge_delta < min(test_ts)` quand train précède test
4. **Purge après test** : les groupes train après un groupe test sont exclus s'ils tombent dans la zone `(test_end, test_end + purge]`

---

## Rétrocompatibilité

- `train_test_split_purge()` : appel existant avec `train_end_year` inchangé.
- `validate_edge()` : API inchangée.
- Tests existants : `test_edge_validation.py` (13/13 ✓), `test_training.py` (9/9 ✓).

---

## Exécution

```bash
# Lancer le CPCV complet (200 splits, ~10-30 min selon CPU)
python run_validation_cpcv.py

# Tests unitaires
pytest tests/unit/test_cpcv.py -v
```

---

## Critères GO/NO-GO

| Critère | Seuil | Implémenté dans |
|---|---|---|
| DSR > 0 | `deflated_sharpe_ratio_from_distribution()` > 0 | `run_validation_cpcv.py:_print_verdict()` |
| % splits profitables > 60% | `aggregate_cpcv_metrics()["pct_profitable"]` > 60 | idem |
| E[Sharpe] > 0 | `aggregate_cpcv_metrics()["sharpe"]["mean"]` > 0 | idem |

Les 3 critères doivent être vrais pour un verdict **GO**.
