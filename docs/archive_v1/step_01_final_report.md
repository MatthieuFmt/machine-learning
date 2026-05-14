# Step 01 — Rapport Final : Redéfinition de la Cible

**Date** : 2026-05-13
**Version** : 1
**Verdict** : ❌ **NO-GO — Le problème n'est pas la cible, mais les features**

---

## 1. Résumé d'exécution

| Mode | Val Sharpe (2024) | Test Sharpe (2025) | Distribution cible | Verdict |
|------|-------------------|---------------------|--------------------|---------|
| `forward_return` | **-1.19** | **-2.71** | continue (log-return) | ❌ |
| `directional_clean` | **-1.92** | **-2.03** | -1=43.9%, 0=12.4%, 1=43.7% | ❌ |
| `cost_aware_v2` | **-2.74** | **-1.94** | -1=74.8%, 0=2.7%, 1=22.5% | ❌ |
| `triple_barrier` (baseline v15) | -0.72 | -0.11 | -1=33.3%, 0=33.3%, 1=33.3% | ❌ |

**Aucun mode ne produit de Sharpe positif.** Le CPCV est inutile : DSR ≪ 0 garanti.

---

## 2. Analyse des résultats

### 2.1 `forward_return` (régression)

- **Prédictions** : `mean_pred ≈ 0`, `std_pred` 2024 = 0.0006, 2025 = 0.0012
- Le GBM régresseur produit des prédictions quasi-centrées sur zéro → aucun signal directionnel
- 314 trades en 2024, 499 en 2025 — le seuil `continuous_signal_threshold=0.0005` laisse passer trop de bruit
- Sharpe 2025 = **-2.71** : le modèle est anti-informatif en OOS

### 2.2 `directional_clean` (binaire avec seuil ATR)

- Distribution cible quasi-équilibrée (44/12/44) — ✅ excellente
- 815 trades en 2025, WR 26.6% — le RandomForest ne trouve pas de structure
- Le seuil de bruit ATR filtre correctement le bruit dans la cible, mais les features ne permettent pas de prédire la direction

### 2.3 `cost_aware_v2` (cost-aware avec seuil ATR adaptatif)

- Distribution très déséquilibrée vers SHORT (75%) — ⚠️ le biais est structurel
- 832 trades, WR 26.4% — aucune amélioration vs directional_clean

---

## 3. Diagnostic racine

### 3.1 Ce n'est PAS la cible

Les 4 modes de cible (dont 3 nouveaux) produisent tous des Sharpe négatifs. La cible `directional_clean` a une distribution quasi-parfaite (44/12/44) mais le modèle ne parvient pas à la prédire. Le problème est en amont.

### 3.2 Les features techniques classiques n'ont pas de pouvoir prédictif sur EURUSD H1

Le set de features actuel (12 colonnes après `features_dropped`) :

| Feature | Source | Type |
|---------|--------|------|
| `Log_Return` | H1 | Momentum court |
| `Dist_EMA_50` | H1 | Tendance |
| `RSI_14` | H1 | Surachat/survente |
| `ADX_14` | H1 | Force de tendance |
| `BB_Width` | H1 | Volatilité |
| `Hour_Sin`, `Hour_Cos` | H1 | Saisonnalité horaire |
| `Range_ATR_ratio` | H1 | Régime |
| `RSI_14_H4` | H4 | Tendance MT |
| `Dist_EMA_20_H4`, `Dist_EMA_50_H4` | H4 | Tendance MT |
| `RSI_14_D1`, `Dist_EMA_20_D1`, `Dist_EMA_50_D1` | D1 | Tendance LT |
| `RSI_D1_delta`, `Dist_SMA200_D1` | D1 | Régime LT |

Ces features sont :
1. **Redondantes** — multiples variantes d'EMA distance et RSI sur différentes timeframes
2. **Non-informatives** — les indicateurs techniques classiques n'ont pas de pouvoir prédictif démontré sur le FX H1
3. **Décorrélées de la cible** — le `forward_return` 24h sur EURUSD est quasi-imprédictible avec ces seules features

### 3.3 Confirmation par le CPCV Step 02

Le CPCV sur `triple_barrier` a montré :
- Accuracy = 0.332 (≈ aléatoire 3-classes = 0.333)
- DSR = -5.15
- CI 95% du Sharpe entièrement négatif

---

## 4. Plan d'action

### 4.1 Ce qu'il faut ARRÊTER

- ❌ Step 03 (GBM primary classifier) — changer de modèle ne résout pas l'absence de signal dans les features
- ❌ Optimiser les hyperparamètres — overfitting sur du bruit

### 4.2 Ce qu'il faut FAIRE

| Priorité | Action | Justification |
|----------|--------|---------------|
| **P0** | Step 04 — Features session-aware | Les sessions (London/NY/Asia) sont le principal déterminant de la volatilité et direction EURUSD |
| **P0** | Step 05 — Calendrier économique | Les annonces macro (NFP, CPI, FOMC) sont les seuls vrais drivers directionnels |
| **P1** | Analyse de corrélation feature→target | Mutual information, Spearman par feature, pour identifier les features qui portent un minimum de signal |
| **P1** | Réduire la colinéarité | 3 variantes d'EMA distance + 3 RSI → garder la plus informative par timeframe |
| **P2** | Step 06 — Meta-labeling | Une fois qu'un edge primaire existe |

### 4.3 Hypothèse de travail

> Les features de **régime** (session, calendrier économique, flux d'ordres) sont plus informatives que les features de **prix** (indicateurs techniques) pour prédire la direction EURUSD H1.

---

## 5. Code livré

| Fichier | Statut | Description |
|---------|--------|-------------|
| `tests/unit/test_target_regression.py` | ✅ 25/25 | Tests des 3 nouvelles fonctions cibles |
| `run_step_01_integration.py` | ✅ | Script d'intégration testant les 3 modes |
| `predictions/step_01_integration.json` | ✅ | Résultats bruts |

Tous les autres fichiers Step 01 (`triple_barrier.py`, `instruments.py`, `pipeline.py`, `training.py`, `prediction.py`, `simulator.py`, `base.py`, `model.py`) étaient déjà implémentés avant cette session.

**292 tests unitaires passent, 0 échec.**

---

## 6. Prochaine étape

**Step 04 — Features session-aware** : ajouter des features basées sur les sessions de trading (London open, NY open, overlap, Asian range). Ces features capturent des effets de microstructure bien documentés dans la littérature FX.
