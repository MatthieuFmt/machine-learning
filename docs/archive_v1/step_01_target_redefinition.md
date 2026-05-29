# Step 01 — Redéfinition de la cible (cible continue ou binaire purifiée)

**Catégorie** : Rupture
**Priorité** : 🔴 Haute
**Effort estimé** : 2-3 jours
**Dépendances** : aucune

---

## 1. Hypothèse mathématique

Soit $X_t \in \mathbb{R}^d$ le vecteur de features à $t$, et $Y_t$ la cible à prédire.

**Cible actuelle** (`apply_triple_barrier`, [features/triple_barrier.py:108-114](../learning_machine_learning/features/triple_barrier.py)) :
$$Y_t^{TB} = \begin{cases} +1 & \text{si LONG win seul} \\ -1 & \text{si SHORT win seul} \\ 0 & \text{sinon (timeout, ambigu, ou les deux dead)} \end{cases}$$

**Distribution mesurée** (`ml_evolution.md` v15) : `P(Y=0) ≈ 36%, P(Y=+1) ≈ 32%, P(Y=-1) ≈ 32%`.

**Problème** : la classe $Y=0$ regroupe des situations sémantiquement disjointes (timeout, ambiguïté bidirectionnelle, double SL). Un classifieur multi-classes apprend à prédire $Y=0$ par défaut quand il manque de confiance ⇒ dilution du signal directionnel.

**Hypothèses alternatives** :
- $H_a$ (régression) : $Y_t^{FR}(h) = \log(C_{t+h} / C_t)$ est continu, $E[Y_t^{FR} | X_t]$ est plus informatif que la version discrétisée.
- $H_b$ (binaire pure) : $Y_t^{BIN}(h) = \mathbb{1}\{|\log(C_{t+h}/C_t)| > \theta_{noise}\} \cdot \text{sign}(\cdot)$ — élimine la zone de bruit.
- $H_c$ (cost-aware optimisée) : $Y_t^{COST} = \mathbb{1}\{\text{pnl\_réel}(t) - \text{friction} > k \cdot \text{ATR}(t)\} \cdot \text{sign}(\cdot)$.

---

## 2. Méthodologie d'implémentation

### Fichiers concernés
- **Modifier** [`learning_machine_learning/features/triple_barrier.py`](../learning_machine_learning/features/triple_barrier.py) : ajouter trois fonctions parallèles à `apply_triple_barrier` :
  - `compute_forward_return_target(df, horizon_hours, pip_size) -> np.ndarray` (régression)
  - `compute_directional_clean_target(df, horizon_hours, noise_threshold_atr) -> np.ndarray` (binaire pure)
  - `compute_cost_aware_target_v2(df, tp_pips, sl_pips, window, friction_pips, k_atr) -> np.ndarray` (variante de la v15)
- **Modifier** [`config/instruments.py`](../learning_machine_learning/config/instruments.py) : ajouter `target_mode: Literal["triple_barrier", "forward_return", "directional_clean", "cost_aware_v2"]` dans `InstrumentConfig`.
- **Modifier** [`features/pipeline.py:81-86`](../learning_machine_learning/features/pipeline.py) : router vers la fonction de target selon `instrument.target_mode`.
- **Modifier** [`model/training.py`](../learning_machine_learning/model/training.py) : si `target_mode == "forward_return"`, instancier `RandomForestRegressor` (ou GBM régresseur) au lieu de `Classifier`.
- **Modifier** [`backtest/simulator.py:74-79`](../learning_machine_learning/backtest/simulator.py) : adapter `mask_long`/`mask_short` selon le type de prédiction (continue → `predicted_return > seuil`, binaire → `predicted_class == 1`).

### Choix techniques
- **Horizon de prédiction $h$** : tester $h \in \{8, 24, 48\}$ heures. Justification : doit être ≥ `window` du backtest (24h) pour cohérence label/exécution, ≤ 48h pour éviter trop de bruit macro intercurrent.
- **Seuil de bruit $\theta_{noise}$ pour $H_b$** : exprimé en multiples d'ATR (ex : $0.5 \cdot \text{ATR}_{14}$) pour rester invariant au régime de volatilité.
- **Modèle pour régression** : GBM (`HistGradientBoostingRegressor` ou XGBRegressor) plutôt que RF — meilleure gestion des queues de distribution.
- **Décision finale en backtest** : pour cible continue, signal = $\text{sign}(\hat{Y})$ si $|\hat{Y}| > \tau$ (seuil dérivé du quantile 70 % de $\hat{Y}$ sur val_year).

### Anti-leak / précautions
- Les fonctions de target utilisent `closes[i:i+h+1]` ou `highs/lows[i+1:i+window+1]` — les `window`/`h` dernières lignes du DataFrame restent NaN (déjà géré par `dropna(subset=["Target"])`).
- Le seuil $\tau$ (quantile sur $\hat{Y}$) doit être calibré sur **val_year uniquement**, jamais sur test_year — sinon look-ahead.
- Si calibration de $\theta_{noise}$ sur ATR, calculer la médiane d'ATR sur train uniquement, jamais sur le futur.

---

## 3. Métriques de validation

### Métriques cibles (à dépasser baseline v15)
| Métrique | Baseline v15 | Objectif step_01 |
|---|---|---|
| Accuracy multi-class OOS 2025 | 0.332 | **NA** (pas de multi-class si on passe en régression/binaire) |
| Pour régression : Spearman $\rho(\hat{Y}, Y_{réalisé})$ OOS 2025 | NA | **> 0.15** |
| Pour binaire pure : Accuracy 2 classes | NA | **> 0.55** |
| Sharpe OOS 2025 (backtest converti) | +0.04 | **> 0.50** |
| DSR 2025 | -1.97 | **> 0** |
| WR 2025 | 33.3 % | **> 38 %** (breakeven 27.7 % + marge) |
| Biais directionnel (% SHORT) | 75 % | **< 60 %** |

### Métriques secondaires
- **MSE** ou **MAE** de la régression sur val_year (pour comparer les 3 horizons $h$).
- **AUC-ROC** sur cible binaire pure (élimine le problème de seuil).
- **Distribution des prédictions** par direction (LONG/SHORT/NEUTRE → ne pas tomber dans le piège du biais directionnel persistant).

### Critère d'arrêt
- Si après benchmark des 3 variantes ($H_a$, $H_b$, $H_c$), aucune ne dépasse **Sharpe > 0.30 sur 2025** et **biais SHORT < 65 %** → la cible n'est pas le bottleneck, abandonner et passer à step_02.
- Si une variante atteint les objectifs → la promouvoir comme cible par défaut, passer step_02 puis step_03.

---

## 4. Risques & dépendances

- **R1 — Risque de leak via le seuil $\tau$** : si calibration globale au lieu de train-only, contamine OOS. Mitigation : test unitaire `test_target_threshold_no_leak`.
- **R2 — Régresseur RF mal calibré sur queues** : un `RandomForestRegressor` peut sous-estimer les grands mouvements. Mitigation : utiliser HistGradientBoostingRegressor avec `loss="quantile"` ou `"huber"`.
- **R3 — Le backtest existant attend des classes {-1, 0, 1}** : adapter `simulate_trades` pour accepter une cible continue. Risque de régression sur tests existants — couvrir par `test_simulator_with_continuous_signal`.
- **R4 — Distribution déséquilibrée pour $H_b$** : avec un seuil $\theta_{noise}$ élevé, la classe 0 (no trade) peut dominer. Mitigation : `class_weight='balanced'` et stratifier la cross-validation.

---

## 5. Références

- López de Prado, *Advances in Financial Machine Learning*, ch. 3 (Labelling) et ch. 5 (Fractionally Differentiated Features).
- Bailey & López de Prado, *The Sharpe Ratio Efficient Frontier* (2012) — pour le seuil de promotion d'une stratégie.
- `ml_evolution.md` v15 (cost-aware labeling) : motive l'extension cost-aware v2 ([d:\Documents\learning-machine-learning\ml_evolution.md](../ml_evolution.md)).
