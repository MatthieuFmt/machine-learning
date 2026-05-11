# 🔬 Analyse Post-Backtest et Pistes d'Amélioration

**Date :** 2026-05-10  
**Pipeline re-généré :** scripts 1→4 + audits  
**Configuration actuelle :** TP=20p / SL=10p / Window=24h / Seuil=0.45 / Commission=0.5p / Slippage=1.0p

---

## 📊 Synthèse des Résultats

| Année | Trades | WR | Profit (pips) | Sharpe | Alpha vs B&H |
|-------|--------|-----|---------------|--------|---------------|
| 2022 | 975 | 64.4% | +7674 | 11.91 | +8344 ⚠️ in-sample |
| 2023 | 787 | 67.9% | +7146 | 12.17 | +6802 ⚠️ in-sample |
| 2024 | 46 | 43.5% | +59 | 0.62 | +750 ✅ OOS |
| 2025 | 33 | 39.4% | −5 | −0.05 | −1395 ❌ OOS |

> ⚠️ 2022-2023 sont **in-sample** (train ≤ 2023), donc leurs Sharpe ~12 ne sont PAS exploitables.  
> Seuls 2024-2025 comptent comme OOS.

---

## 🔍 5 Problèmes Identifiés

### Problème 1 : Écart Train↔Test Massif (Overfitting)

| Métrique | Valeur |
|----------|--------|
| OOB score (train ≤ 2023) | 0.442 |
| Accuracy 2024 OOS | 0.355 (Δ −0.094) |
| Accuracy 2025 OOS | 0.332 (Δ −0.110) |
| Accuracy aléatoire (3 classes) | 0.333 |

Le modèle ne bat le hasard que de **2 points** en 2024 et **0 point** en 2025. L'OOB est gonflé par la structure temporelle des données (même avec purge 48h).

**Solution :** Simplifier drastiquement le modèle (régularisation plus forte, moins de features, shallow trees).

---

### Problème 2 : Confiance du Modèle en Chute Libre

| Année | proba_max moyenne | ≥ 0.45 | Trades générés |
|-------|------------------|--------|----------------|
| 2022 (in-sample) | 0.422 | 1612 | 975 |
| 2023 (in-sample) | 0.419 | 1511 | 787 |
| 2024 (OOS) | 0.373 | 257 | 46 |
| 2025 (OOS) | 0.364 | 101 | 33 |

La combinaison `proba_max < seuil` filtre 98.5% des barres en 2025 (101/6226). Le modèle n'est « sûr de rien » en OOS — c'est un signal d'overfitting. Les 33-46 trades annuels sont trop peu pour une stratégie H1.

**Solution :** Baisser le seuil à 0.38-0.40, ou utiliser un sizing qui ne dépend pas du seuil.

---

### Problème 3 : Biais Directionnel Majeur

| Direction | Trades 2025 | Profit | Win Rate |
|-----------|-------------|--------|----------|
| SHORT | 28 (85%) | +25.3 pips | 42.9% |
| LONG | 5 (15%) | −30.2 pips | 20.0% |

En 2025, l'EURUSD a monté de +1390 pips (B&H positif). Le modèle a massivement shorté dans un marché haussier → désastre. Le RandomForest sur-apprend la tendance baissière historique d'EURUSD (2010-2023) et n'arrive pas à s'adapter au changement de régime 2025.

**Solution :** Neutraliser la tendance long-terme (features de momentum normalisé, filtre de régime).

---

### Problème 4 : Features Non Discriminantes

Sur 9 features actives, la permutation importance montre :

| Niveau | Features |
|--------|----------|
| ✅ Significatif (3) | RSI_14_D1, Dist_EMA_20_D1, ADX_14 |
| ⚠️ Borderline (4) | Dist_EMA_50, RSI_14, Dist_EMA_50_H4, ATR_Norm |
| ❌ Bruit (2) | Dist_EMA_20_H4, RSI_14_H4 |

Seules les features D1 + ADX_14 ont un pouvoir prédictif réel. Les features H1/H4 sont du bruit déguisé (le modèle les utilise via impurity mais la permutation le révèle). **Les features H4 sont quasi-inutiles** (permutation_mean < 0.007, std comparable).

**Solution :** Garder uniquement les 5 meilleures features, ajouter des features de régime (volatilité réalisée, range ATR, distance à SMA 200).

---

### Problème 5 : 100% des Pertes Sont des SL

| Issue | Nombre |
|-------|--------|
| loss_sl | 20 (61%) |
| loss_timeout | 0 (0%) |
| win | 13 (39%) |

Zéro timeout : le marché touche toujours SL ou TP dans la fenêtre de 24h. Le ratio TP/SL 20/10 est serré. Avec un win rate de 39%, le payoff est `0.39×20 − 0.61×10 = 1.7 pips` → à peine positif avant friction. Avec comm+slip = 1.5p : `0.39×18.5 − 0.61×11.5 = 7.2 − 7.0 = +0.2 pips` → espérance quasi nulle.

**Solution :** Ajuster le ratio TP/SL, ou filtrer les trades par une métrique de volatilité/régime.

---

## 🎯 Plan d'Action Priorisé

### 🔴 Priorité 1 — Réduire l'Overfitting (Impact : ÉLEVÉ, Effort : FAIBLE)

1. **Augmenter `min_samples_leaf`** de 5 → 50 : force des splits plus robustes
2. **Réduire `max_depth`** de 12 → 6 : empêche d'apprendre des patterns de bruit
3. **Augmenter `n_estimators`** de 200 → 500 : stabilise avec plus d'arbres simples
4. **Option : remplacer RandomForest par GradientBoosting** (meilleure généralisation sur données financières)

### 🟠 Priorité 2 — Améliorer les Features (Impact : ÉLEVÉ, Effort : MOYEN)

1. **Dropper RSI_14_H4 et Dist_EMA_20_H4** (permutation non significative)
2. **Ajouter des features de régime** :
   - `Volatilite_Realisee_24h` = écart-type des returns H1 sur 24 barres
   - `Range_ATR_ratio` = (High−Low) / ATR(14) — expansion vs contraction
   - `Dist_SMA200` = distance de la Close à la SMA 200 (tendance long-terme)
   - `RSI_D1_delta` = variation du RSI D1 sur 3 jours (momentum macro)
3. **Normaliser les features** (StandardScaler) pour RandomForest — optionnel mais utile si on passe à un autre algo

### 🟡 Priorité 3 — Adapter le Seuil et le Sizing (Impact : MOYEN, Effort : FAIBLE)

1. **Tester seuil ∈ {0.36, 0.38, 0.40, 0.42}** sur 2024 (validation) et mesurer sur 2025
2. **Option : supprimer le seuil** — prendre tous les signaux (1 ou −1) et laisser le sizing faire le tri
3. **Sizing basé sur la volatilité** : réduire la taille dans les régimes à haute volatilité (ATR_Norm)

### 🟢 Priorité 4 — Filtre de Régime (Impact : MOYEN, Effort : MOYEN)

1. **Filtre tendance** : si Close > SMA(200), n'autoriser que les LONG ; si Close < SMA(200), n'autoriser que les SHORT
2. **Filtre volatilité** : ne pas trader si ATR_Norm > 2× médiane glissante (marché chaotique)
3. **Filtre session** : exclure les rolls/liquidité faible (22h-01h GMT)

### 🔵 Priorité 5 — Ajuster TP/SL (Impact : MOYEN, Effort : FAIBLE)

1. **Tester TP=20/SL=20** (ratio 1:1) — plus facile à gagner, WR devrait monter à 50-55%
2. **Tester TP=30/SL=10** (ratio 3:1) — WR baissera mais payoff peut s'améliorer si features discriminantes
3. **TP/SL dynamiques** basés sur ATR (ex: `N × ATR(14)`)

---

## 📋 Fichiers à Modifier

| Fichier | Modifications |
|---------|--------------|
| `config.py` | `RF_PARAMS` (depth=6, leaf=50, n=500), ajuster `FEATURES_DROPPED`, ajouter `SEUIL_CONFIANCE_ALTERNATIFS` |
| `2_master_feature_engineering.py` | Ajouter features de régime, normalisation optionnelle |
| `3_model_training.py` | Option GradientBoosting, feature importance élargie |
| `4_backtest_triple_barrier.py` | Filtre de régime, sizing volatilité |
| `backtest_utils.py` | Logique de filtre trend/vol/session |
| `config.py` | Nouvelles valeurs TP/SL à tester |

---

## ✅ Critères de Succès

Après améliorations, un backtest sain sur 2025 devrait afficher :
- **Trades** ≥ 100/an (contre 33 actuellement)
- **Win Rate** ∈ [45%, 60%]
- **Sharpe** ∈ [0.3, 2.0]
- **Alpha positif** vs B&H sur les deux années de test
- **Écart OOB−Test** ≤ 0.05
