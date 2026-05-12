# Step 02 — Framework de validation robuste (CPCV + DSR)

**Catégorie** : Méthodologie
**Priorité** : 🔴 Haute
**Effort estimé** : 2-3 jours
**Dépendances** : aucune (peut être fait en parallèle de step_01)

---

## 1. Hypothèse mathématique

### Problème actuel
La validation actuelle (`ml_evolution.md` v14) repose sur :
- 1 split fixe : train ≤ 2023 / val 2024 / test 2025
- Walk-forward 5 folds (36 mois glissants)

Le Sharpe observé sur 2025 (+0.04, [ml_evolution.md v13](../ml_evolution.md)) a une **variance d'estimation inconnue**. Le test bootstrap donne $p(\text{Sharpe} > 0) = 0.29$ — on ne peut pas rejeter $H_0: \text{Sharpe} = 0$.

### Formalisation

Soit $\hat{SR}$ le Sharpe ratio empirique observé sur OOS. On veut tester :
$$H_0 : SR_{vrai} \le 0 \quad \text{vs} \quad H_1 : SR_{vrai} > 0$$

**Probabilistic Sharpe Ratio (PSR)** (Bailey & López de Prado 2012) :
$$\text{PSR}(SR^*) = \Phi\left( \frac{(\hat{SR} - SR^*) \sqrt{n-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \hat{SR}^2}} \right)$$

où $\hat{\gamma}_3, \hat{\gamma}_4$ sont skewness et kurtosis empiriques, $SR^* = 0$ pour le test, $n$ = nombre d'observations.

**Deflated Sharpe Ratio (DSR)** : corrige le PSR pour le nombre $N$ de stratégies testées (multiple comparison) :
$$SR_0^* = \sqrt{\text{Var}(\{SR_i\})} \cdot \left( (1-\gamma) \Phi^{-1}(1 - 1/N) + \gamma \Phi^{-1}(1 - 1/(N \cdot e)) \right)$$
$$\text{DSR} = \text{PSR}(SR_0^*)$$

### Combinatorial Purged CV (CPCV) — López de Prado ch.12
Partitionner la timeline en $N$ groupes contigus. Pour chaque combinaison $\binom{N}{k}$ de $k$ groupes-test (les $N-k$ autres = train), entraîner et tester avec purge_hours embargo. On obtient $\binom{N}{k}$ Sharpe → distribution empirique → DSR final.

---

## 2. Méthodologie d'implémentation

### Fichiers concernés

- **Créer** [`learning_machine_learning/analysis/cpcv.py`](../learning_machine_learning/analysis/) avec :
  - `generate_cpcv_splits(index, n_groups, k_test, purge_hours) -> Iterator[tuple[train_idx, test_idx]]` : générateur des combinaisons train/test avec embargo
  - `run_cpcv_backtest(df, model_factory, splits, backtest_cfg) -> pd.DataFrame` : exécute le pipeline complet sur chaque split, retourne un DataFrame avec colonnes (`split_id`, `train_period`, `test_period`, `sharpe`, `pnl`, `n_trades`, `wr`)
  - `aggregate_cpcv_metrics(results_df) -> dict` : E[SR], σ[SR], min/max/median, % profitables, distributions

- **Créer** [`learning_machine_learning/analysis/edge_validation.py`](../learning_machine_learning/analysis/) avec :
  - `probabilistic_sharpe_ratio(returns, sr_benchmark=0.0) -> float` : PSR
  - `deflated_sharpe_ratio(sharpe_distribution, observed_sr) -> float` : DSR à partir d'une distribution CPCV
  - `bootstrap_sharpe_pvalue(returns, n_iter=10000) -> dict` : p-value bootstrap + IC 95 %
  - `validate_edge(trades_df, backtest_cfg, sharpe_distribution=None) -> dict` : wrapper appelant les 3 ci-dessus + breakeven_wr

- **Modifier** [`model/training.py:24-66`](../learning_machine_learning/model/training.py) : généraliser `train_test_split_purge` pour accepter `train_index` et `test_index` arbitraires (pas seulement une coupe par année).

- **Créer** `run_validation_cpcv.py` à la racine : script orchestrateur qui lance CPCV sur 12 groupes (2 mois chacun, 2 ans test_window), produit `predictions/cpcv_results.csv` et `predictions/cpcv_report.md`.

### Choix techniques

- **Granularité CPCV** : $N = 36$ groupes (12 mois × 3 ans = 36 mois roulants), $k = 6$ groupes-test (6 mois) → $\binom{36}{6} = 1\,947\,792$ combinaisons. **Trop**. Pratique : utiliser un sampling stratifié de 200 combinaisons aléatoires (toujours dominant statistiquement).
- **Purge entre groupes** : `purge_hours=48` (cohérent avec `ModelConfig.purge_hours`).
- **Test parallélisable** : `joblib.Parallel(n_jobs=-1)` sur les splits — chaque split est indépendant.
- **Cache** : sauvegarder le modèle entraîné par fold dans `results/cpcv_models/fold_{i}.pkl` si on veut réutiliser pour méta-analyse.

### Anti-leak / précautions
- **Purge bidirectionnelle obligatoire** : pour un groupe-test au milieu, purger `purge_hours` AVANT et APRÈS (pas seulement avant comme dans le split linéaire actuel).
- **Pas de calibration globale** : tout scaler/calibrateur (Platt, isotonic) doit être refitté à chaque split sur la partie tardive du train.
- **Vérification programmatique** : pour chaque split, asserter `max(train_index) + purge_hours < min(test_index)` ET `max(test_index) + purge_hours < min(train_index_after_test)`.

---

## 3. Métriques de validation

### Métriques cibles
| Métrique | Baseline v13/v14 | Objectif step_02 |
|---|---|---|
| p(Sharpe > 0) bootstrap 2025 | 0.29 | **< 0.05** sur ≥ 50 % des splits CPCV |
| DSR 2025 (1 split) | -1.97 | **DSR sur distribution CPCV > 0** |
| E[Sharpe] sur CPCV | NA | **> 0.3** |
| σ[Sharpe] sur CPCV | NA | **< 1.0** (stabilité) |
| % splits profitables | NA | **> 60 %** |

### Métriques secondaires
- **Type I error rate** : sur des **données shufflées** (label permutation), DSR doit rester < 1 dans ≥ 95 % des cas. Sanity check méthodologique.
- **Couverture temporelle** : chaque mois doit apparaître au moins 5 fois comme test à travers les splits.
- **Variance par période macro** : grouper les splits par contexte macro (haussier / baissier / range EURUSD) et reporter Sharpe conditionnel.

### Critère d'arrêt
- Si la baseline v8 (méta-labeling) n'atteint pas **DSR > 0 ET % profitables > 60 %** sur CPCV → l'edge n'existe pas, pivot stratégique requis (changer cible/TF/instrument, cf. step_01 ou abandon).
- Si la baseline les atteint → on tient un edge mesurable, step_03 et suivants deviennent légitimes.

---

## 4. Risques & dépendances

- **R1 — Coût compute** : 200 splits × ~30s de pipeline = 100 minutes par run. Mitigation : exécuter `run_validation_cpcv.py` uniquement à chaque step majeure (pas à chaque commit).
- **R2 — Faux positifs de leak** : avec $N$ groupes contigus, certains splits ont un test entouré de train — la purge bidirectionnelle doit être stricte. Mitigation : test unitaire `test_cpcv_split_no_temporal_overlap` qui vérifie ces invariants.
- **R3 — Sur-corrélation des Sharpe** : deux splits qui partagent 90 % de leur train donnent des Sharpe corrélés ⇒ variance sous-estimée. Mitigation : dans `deflated_sharpe_ratio`, utiliser un facteur de correction $N_{eff}$ basé sur la corrélation moyenne des splits.
- **R4 — Dépendance circulaire avec step_06** : la calibration du méta-modèle se fait sur val_year — en CPCV, il faut redéfinir "val" comme la tranche tardive de chaque train. Documenter dans step_06.

---

## 5. Références

- Bailey, D. H., & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*. The Journal of Risk.
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio*. The Journal of Portfolio Management.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 12 (Backtesting through Cross-Validation).
- Code existant à généraliser : [`learning_machine_learning/model/training.py:24-66`](../learning_machine_learning/model/training.py).
