# Step 03 — Remplacement RandomForest primaire par GBM + Optuna

**Catégorie** : Modèle
**Priorité** : 🟠 Moyenne (haute si step_01 valide une cible robuste)
**Effort estimé** : 2-3 jours
**Dépendances** : step_02 (pour valider via CPCV), idéalement step_01 (la cible définitive doit être choisie avant l'optim)

---

## 1. Hypothèse mathématique

### Diagnostic du choix actuel

[`ModelConfig`](../learning_machine_learning/config/model.py) configure :
```
n_estimators=500, max_depth=6, min_samples_leaf=50, class_weight='balanced'
```

RandomForest minimise l'erreur **par bagging indépendant** : chaque arbre est entraîné sur un échantillon bootstrap, indépendamment des autres. La variance est réduite par moyennage. Mais :

1. **Biais de représentation de classe** : avec `class_weight='balanced'`, le RF rebalance via pondération des échantillons, mais cela n'empêche pas la **mémorisation de la distribution conditionnelle apprise sur 2010-2023** (régime majoritairement baissier/range). En 2025 (régime haussier), le modèle prédit SHORT à 85 % (`ml_evolution.md` v3) malgré le rebalancing.
2. **Pas d'early stopping** : RF ne s'arrête jamais d'ajouter des arbres — pas de signal sur "j'ai capturé tout le signal exploitable".
3. **Régularisation par profondeur fixe** : `max_depth=6` est arbitraire et identique pour tous les arbres.

### Formalisation

Soit $L(\theta) = \mathbb{E}_{X, Y}[\ell(f_\theta(X), Y)]$ la perte espérée. RF estime $\hat{L}_{RF}$ par moyennage indépendant :
$$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$

GBM construit une approximation **séquentielle** par descente de gradient fonctionnel :
$$\hat{f}_{GBM}^{(m)}(x) = \hat{f}_{GBM}^{(m-1)}(x) + \eta \cdot h_m(x)$$
où $h_m$ minimise le résidu de l'étape $m-1$.

**Hypothèse $H_1$** : sur une cible à signal faible (accuracy théorique max ≈ 0.40-0.50), GBM avec `early_stopping_rounds=20` sur `val_loss` détecte plus tôt le sur-apprentissage que RF. **Hypothèse $H_2$** : la flexibilité de la fonction de perte (focal loss, asymmetric loss, custom loss alignée sur PnL) ouvre des leviers que RF ne permet pas.

---

## 2. Méthodologie d'implémentation

### Fichiers concernés

- **Modifier** [`learning_machine_learning/config/model.py:8-25`](../learning_machine_learning/config/model.py) : ajouter
  - `model_backend: Literal["rf", "xgb", "lgbm", "catboost"] = "rf"` (défaut conservateur)
  - `optuna_n_trials: int = 50`
  - `optuna_objective: Literal["log_loss", "neg_sharpe", "neg_brier"] = "log_loss"`
  - `gbm_early_stopping_rounds: int = 20`
  - Validation `__post_init__` : si `model_backend != "rf"`, vérifier que la lib est importable.

- **Modifier** [`learning_machine_learning/model/training.py:69-93`](../learning_machine_learning/model/training.py) : extraire un `build_model(backend, params) -> Any` qui retourne l'instance appropriée. Les 4 backends doivent exposer l'interface sklearn (`fit`, `predict`, `predict_proba`, `feature_importances_`).

- **Créer** [`learning_machine_learning/model/hyperopt.py`](../learning_machine_learning/model/) avec :
  - `run_optuna_search(X_train, y_train, model_backend, n_trials, objective, inner_cv_purge_hours) -> dict[str, Any]` : retourne les meilleurs hyperparam
  - `objective_neg_sharpe(trial, X, y, model_backend, backtest_cfg)` : objectif qui lance un mini-backtest sur les folds inner CV et retourne `-Sharpe` (à minimiser)

- **Étendre** `requirements.txt` (ou `pyproject.toml`) avec : `xgboost>=2.0`, `lightgbm>=4.0`, `optuna>=3.5`. CatBoost optionnel.

- **Modifier** [`pipelines/base.py:37-52`](../learning_machine_learning/pipelines/base.py) : `train_model` route via `build_model(self.model_cfg.model_backend, params)`.

### Choix techniques

- **Backend par défaut recommandé** : **LightGBM** — plus rapide que XGBoost sur ~80k samples, gestion native des catégorielles (utile pour step_04 `session_id`), bonne API early_stopping.
- **Espace de recherche Optuna** (LightGBM exemple) :
  - `num_leaves ∈ [15, 127]` (∼2^max_depth)
  - `min_child_samples ∈ [20, 200]`
  - `learning_rate ∈ [0.01, 0.2]` (log scale)
  - `feature_fraction ∈ [0.5, 1.0]`
  - `bagging_fraction ∈ [0.5, 1.0]`
  - `reg_alpha, reg_lambda ∈ [0.0, 10.0]` (log scale)
  - `class_weight ∈ {None, "balanced"}`
- **Inner CV pour Optuna** : `TimeSeriesSplit(n_splits=3, gap=48)` sur le train uniquement. JAMAIS toucher OOS.
- **Objectif** : commencer par `"log_loss"` (rapide, dense en signal). Si log_loss optimal donne un mauvais Sharpe en backtest, passer à `"neg_sharpe"` (coûteux mais aligné business).
- **Reproductibilité** : `optuna.create_study(sampler=TPESampler(seed=42))`.

### Anti-leak / précautions
- **Optuna inner CV strictement sur train** : pas une seule ligne de val_year/test_year dans le `fit` ou l'évaluation Optuna.
- **Early stopping** : la validation set pour early stopping doit être la **tranche tardive** du train (ex : derniers 20 % chronologiquement), pas un random split.
- **Si focal loss / asymmetric loss** : les paramètres ($\alpha, \gamma$ pour focal) doivent être déterminés sur train uniquement.
- **Sauvegarde des hyperparam** : écrire `results/best_params_{backend}.json` après chaque search Optuna pour traçabilité.

---

## 3. Métriques de validation

### Métriques cibles
| Métrique | Baseline RF (v15) | Objectif step_03 |
|---|---|---|
| Log-loss inner CV | ~1.09 (3 classes uniform) | **< 1.05** |
| Accuracy OOS 2025 | 0.332 | **> 0.36** |
| Brier score | NA | **< 0.22** (calibration) |
| Sharpe OOS 2025 backtest | +0.04 | **> 0.50** |
| Sharpe OOS 2024 backtest | +0.49 | **> 0.50** (préserver) |
| Biais directionnel 2025 | 75 % SHORT | **< 60 %** |

### Métriques secondaires
- **Feature importance permutation** post-Optuna : doit révéler ≥ 4 features avec `permutation_mean > 2 × std`.
- **Stabilité** : variance des accuracy entre les inner CV folds — devrait être < 0.02.
- **Temps d'inférence** : doit rester < 5ms par prédiction pour live trading futur.

### Critère d'arrêt
- Si après Optuna le meilleur GBM ne dépasse pas la baseline RF de **+5 % en accuracy OOS** et **+0.2 en Sharpe**, considérer que le bottleneck n'est pas le modèle (donc cible — step_01 — ou features — step_04/05).
- Si le GBM dépasse → l'adopter, passer step_06 (calibration sur le nouveau modèle).

---

## 4. Risques & dépendances

- **R1 — Surapprentissage Optuna sur inner CV** : avec 50 trials et un signal faible, on peut overfitter les hyperparam sur le train. Mitigation : **double cross-validation** (Optuna sur 2/3 du train, validation finale sur le 1/3 restant) OU régulariser fort via prior bayésien.
- **R2 — Coût compute Optuna sur Sharpe** : 50 trials × 3 inner folds × ~30s = 75 min. Acceptable mais batch overnight.
- **R3 — Dépendance LightGBM/Optuna** : nouvelles libs à installer. Mitigation : marquer optionnel dans pyproject (`extras_require={"gbm": ["lightgbm", "optuna"]}`).
- **R4 — Tests unitaires existants** : 204 tests existants supposent peut-être `RandomForestClassifier`. Auditer `tests/unit/test_training.py` et adapter (utiliser `model_backend="rf"` par défaut pour rétro-compat).
- **R5 — Class imbalance** : LightGBM gère via `is_unbalance=True` ou `scale_pos_weight`, différent de l'API RF. Tester soigneusement.

---

## 5. Références

- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 6-7 (Ensemble Methods, Cross-Validation).
- Code à généraliser : [`learning_machine_learning/model/training.py`](../learning_machine_learning/model/training.py), [`config/model.py`](../learning_machine_learning/config/model.py).
