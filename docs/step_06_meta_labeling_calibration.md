# Step 06 — Calibration des probas méta + sweep robuste du seuil

**Catégorie** : Optimisation (quick win)
**Priorité** : 🟠 Moyenne
**Effort estimé** : ½ - 1 jour
**Dépendances** : step_02 (CPCV pour le sweep multi-fold), idéalement step_03 si nouveau backend

---

## 1. Hypothèse mathématique

### Diagnostic du seuil actuel

`ml_evolution.md` v8-v9 :
- v8 (seuil méta fixé à **0.55** sur 2025 par observation) : Sharpe +0.53
- v9 (seuil optimisé sur val_year 2024 → **0.50**) : Sharpe **−0.64** sur 2025

**Conclusion** : le seuil 0.55 a probablement été trouvé par sur-optimisation sur la seule année test 2025. Le sweep sur 2024 donne un seuil différent qui ne généralise pas. C'est un cas d'école de **fragilité hors-sample**.

### Cause racine probable

Les probabilités du méta-classifieur (RandomForest) sont **mal calibrées** : RF a tendance à pousser les probas vers 0/1, créant des "trous" autour de 0.5. Le seuil optimal change donc selon la distribution exacte des probas, qui n'est pas stable d'une année à l'autre.

### Formalisation

Soit $p(X) = P(\text{trade profitable} | X)$ la proba estimée par le méta-RF. La proba **vraie** $\pi(X) = P(\text{réalité} | X)$ est inconnue. On dit que $p$ est **bien calibré** si :
$$P(\text{trade profitable} | p(X) = q) = q \quad \forall q \in [0, 1]$$

**Calibration Platt** : ajuster une sigmoïde $\sigma(a \cdot p + b)$ avec $a, b$ estimés par maximum likelihood sur un set de calibration.

**Calibration isotonique** : ajuster une fonction monotone non-paramétrique $g$ sur les paires $(p, y)$ du set de calibration.

Une fois calibré, le **seuil optimal** correspond directement à un break-even attendu :
$$\text{seuil}^* = \frac{|\text{loss}|}{|\text{loss}| + |\text{gain}|} = \frac{SL + \text{friction}}{SL + TP}$$

Avec TP=30, SL=10, friction=1.5 : $\text{seuil}^* = 11.5 / 41.5 \approx 0.277$. Très différent de 0.55 actuel — preuve indirecte que les probas sont mal calibrées.

### Hypothèse

**$H_1$** : après calibration isotonique, le seuil break-even ~0.28 donne un Sharpe **plus stable** entre 2024 et 2025 qu'un seuil sweepé à 0.55. Stabilité = critère de production-readiness.

**$H_2$** : la calibration **améliore** aussi l'expected calibration error (ECE) et le Brier score, fournissant des probas exploitables pour du position sizing ultérieur (Kelly).

---

## 2. Méthodologie d'implémentation

### Fichiers concernés

- **Localiser ou créer** [`learning_machine_learning/model/meta_labeling.py`](../learning_machine_learning/model/) (selon état actuel post-v8) — si la classe `MetaLabeler` existe :
  - Wrapper autour du méta-classifieur avec `fit(X, y)` et `predict_proba(X)`
  - Ajouter méthode `calibrate(X_cal, y_cal, method: Literal["sigmoid", "isotonic"] = "isotonic")` qui fitte un `CalibratedClassifierCV(self.base_model, method=method, cv='prefit')` sur le set de calibration

- **Étendre** [`pipelines/base.py`](../learning_machine_learning/pipelines/base.py) (ou la pipeline méta-labeling spécifique) :
  - Découper le train en `train_inner` (80 %) + `cal` (20 % tardif) chronologiquement
  - Fitter le méta sur `train_inner`, puis calibrer sur `cal`
  - Evaluer sur OOS (val_year, test_year)

- **Modifier** [`backtest/simulator.py:71`](../learning_machine_learning/backtest/simulator.py) (`seuil_confiance`) : devient implicitement le seuil méta calibré. Le défaut config doit refléter le seuil break-even calculé depuis TP/SL/friction.

- **Créer** [`learning_machine_learning/analysis/calibration_report.py`](../learning_machine_learning/analysis/) :
  - `plot_reliability_diagram(y_true, y_proba, n_bins=10) -> matplotlib.Figure`
  - `compute_brier_score(y_true, y_proba) -> float`
  - `compute_ece(y_true, y_proba, n_bins=10) -> float` (Expected Calibration Error)
  - Génération d'un rapport `predictions/calibration_report_{year}.md` avec ces métriques + reliability diagram

- **Modifier** [`config/backtest.py`](../learning_machine_learning/config/backtest.py) :
  - `meta_calibration_method: Literal["none", "sigmoid", "isotonic"] = "isotonic"`
  - `meta_threshold_mode: Literal["fixed", "breakeven", "robust_cpcv"] = "breakeven"`
  - `meta_threshold_value: float = 0.55` (utilisé si mode="fixed")
  - `meta_threshold_safety_margin: float = 0.05` (ajouté au seuil breakeven pour conservatisme)

### Choix techniques

- **Méthode de calibration** : **isotonique** par défaut. Justification : non paramétrique, capture des miscalibrations non-sigmoïdes (typique du RF). Platt convient mieux à des SVM.
- **Split train/calibration** : 80 % chronologique des données train + 20 % derniers mois pour calibration (jamais un split aléatoire — leak temporel sinon).
- **Mode `robust_cpcv`** (recommandé après step_02) : pour chaque fold CPCV, calculer le seuil optimal qui maximise le Sharpe. Le seuil retenu = **médiane** des seuils optimaux par fold (robuste aux outliers).
- **Sanity check** : si le seuil calibré tombe en dehors de [0.20, 0.60], probable bug — raise warning.

### Anti-leak / précautions
- **Split chronologique strict** pour calibration : `cal = X_train.iloc[int(0.8 * n):]` (PAS de `train_test_split` aléatoire).
- **`cv='prefit'`** : `CalibratedClassifierCV` ne refit pas le base estimator. Si `cv > 1`, sklearn refit le base estimator → invalide pour séries temporelles.
- **Le set de calibration ne doit JAMAIS contenir des données de val_year ou test_year**.
- **Si CPCV (step_02)** : refitter la calibration à chaque fold (jamais réutiliser un calibrateur fitté sur d'autres folds).

---

## 3. Métriques de validation

### Métriques cibles
| Métrique | Baseline v8/v9 | Objectif step_06 |
|---|---|---|
| Brier score méta-classifieur OOS 2025 | NA | **< 0.22** (uniforme = 0.25) |
| ECE 10 bins OOS 2025 | NA | **< 0.05** |
| Sharpe OOS 2024 avec seuil calibré | +0.49 (v13) | **> 0.40** (préservation) |
| Sharpe OOS 2025 avec seuil calibré | +0.04 (v13) | **> 0.30** |
| Écart Sharpe(2024) − Sharpe(2025) | -0.45 | **< 0.30 en absolu** (stabilité inter-année) |
| Trades pris OOS 2025 | 153 (v8) | **150-300** (filtre robuste, pas trop strict) |

### Métriques secondaires
- **Reliability diagram** avant/après : visuellement la courbe doit se rapprocher de la diagonale.
- **Distribution des probas méta** OOS : histogramme avant/après calibration. Devrait passer de bimodal (poussé vers 0 et 1) à plus continue.
- **Seuil break-even calculé** : doit matcher analytiquement le ratio TP/SL/friction.

### Critère d'arrêt
- Si Brier score post-calibration **ne baisse pas de ≥ 0.02** vs pré-calibration → le base classifieur est déjà raisonnablement calibré (rare pour un RF) OU la cible est trop bruitée. Garder uniquement le sweep multi-fold sans calibration.
- Si la stabilité Sharpe(2024) ≈ Sharpe(2025) est atteinte → quick win confirmé, livrer en production.

---

## 4. Risques & dépendances

- **R1 — Calibration sur-fittée sur cal set** : si cal set est petit (< 1000 samples), isotonique peut surapprendre. Mitigation : minimum 2000 samples pour cal, sinon fallback Platt.
- **R2 — Dépendance CPCV** : le mode `robust_cpcv` est puissant mais nécessite step_02 fonctionnel. Sans CPCV, fallback au mode `breakeven` (déterministe).
- **R3 — Seuil breakeven dépend de TP/SL** : si TP/SL changent (step_01 ou tuning futur), le seuil bouge mécaniquement. Documenter la formule et la recalculer à chaque changement de config.
- **R4 — Si le méta-modèle change** (step_03 : GBM au lieu de RF), refaire la calibration depuis zéro. Les hyperparam de calibration ne se transfèrent pas entre backends.

---

## 5. Références

- Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*. ALMC.
- Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities With Supervised Learning*. ICML.
- Guo, C. et al. (2017). *On Calibration of Modern Neural Networks*. ICML. (pertinent pour ECE)
- scikit-learn docs : [`sklearn.calibration.CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- `ml_evolution.md` v8 / v9 / v12 : motivent l'analyse de fragilité du seuil 0.55.
