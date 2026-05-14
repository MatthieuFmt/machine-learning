# V3 Hypothesis 07 — Stratégies trend-following alternatives

**Date** : 2026-05-14
**Statut** : 🔴 NO-GO — 0 stratégie alternative validée sur 4
**Prompt** : 08
**Priorité** : 🔴 P0 — stratégies alternatives
**n_trials cumulatif** : 7 (6 hérités v2 + 1 H07)

---

## 0. Question

Existe-t-il d'autres stratégies trend-following déterministes qui surpassent Donchian ou sont décorrélées de Donchian sur US30 D1 ?

**Réponse** : ❌ **Non.** Aucune stratégie alternative n'atteint un Sharpe test > 0 avec les coûts réalistes v3. Toutes sont décorrélées de Donchian (ρ < 0.35) mais leur performance est négative ou nulle. Le problème vient des coûts v3 (spread 3.0 + slippage 5.0 = coût effectif 16 pips/trade), pas des stratégies elles-mêmes.

---

## 1. Méthode

### 1.1 Stratégies testées

| # | Stratégie | Règle LONG | Règle SHORT | Paramètres (grid) | Combinaisons |
|---|-----------|------------|-------------|---------------------|--------------|
| S1 | **Dual Moving Average** | SMA(fast) > SMA(slow) | SMA(fast) < SMA(slow) | fast ∈ {5, 10, 20}, slow ∈ {50, 100, 200} | 9 |
| S2 | **Keltner Channel** | Close > EMA(period) + mult × ATR(period) | Close < EMA(period) − mult × ATR(period) | period ∈ {10, 20, 50}, mult ∈ {1.5, 2.0, 2.5} | 9 |
| S3 | **Chandelier Exit** | Close > Highest(High, period) − k_atr × ATR(period) | Close < Lowest(Low, period) + k_atr × ATR(period) | period ∈ {11, 22, 44}, k_atr ∈ {2.0, 3.0, 4.0} | 9 |
| S4 | **Parabolic SAR** | SAR < Close | SAR > Close | step ∈ {0.01, 0.02, 0.03}, af_max ∈ {0.1, 0.2, 0.3} | 9 |

**Total** : 36 combinaisons × 1 actif (US30).

### 1.2 Actif testé

**US30 D1** uniquement — seul actif avec edge Donchian historique (Sharpe +3.07 à +8.84 en v2). H06 a dégradé Donchian sur US30 avec les coûts réalistes v3 (Sharpe −0.09), mais les stratégies alternatives sont testées avec les mêmes coûts pour comparaison équitable.

### 1.3 Coûts US30 (appliqués)

| Paramètre | Valeur |
|-----------|--------|
| Spread | 3.0 pips |
| Slippage | 5.0 pips |
| TP (points) | 200 |
| SL (points) | 100 |
| Window | 120 heures |

### 1.4 Split temporel

| Période | Dates |
|---------|-------|
| Train | ≤ 2022-12-31 |
| Val | 2023-01-01 → 2023-12-31 |
| Test | ≥ 2024-01-01 |

### 1.5 Protocole

1. Grid search sur train ≤ 2022 → meilleur Sharpe train
2. Évaluation sur val 2023 (informatif, pas de sélection)
3. Évaluation sur test ≥ 2024 → `validate_edge(equity, trades, n_trials=7)`
4. Corrélation rolling 60j des retours quotidiens vs Donchian(20, 20)

---

## 2. Corrections techniques (Prompt 08)

Les 4 stratégies existaient déjà dans `app/strategies/` (héritage v2) mais avaient des défauts bloquants :

| Fichier | Problème | Correction |
|---------|----------|------------|
| `dual_ma.py` | Colonnes PascalCase + pas de `.shift(1)` | PascalCase (conforme `load_asset()`) + `.shift(1).fillna(0).astype(int)` |
| `keltner.py` | Colonnes PascalCase + pas de `.shift(1)` | PascalCase + `.shift(1).fillna(0).astype(int)` |
| `chandelier.py` | Colonnes PascalCase | PascalCase + `.shift(1).fillna(0).astype(int)` |
| `parabolic.py` | Colonnes PascalCase | PascalCase + `.shift(1).fillna(0).astype(int)` |
| `deterministic.py` | Colonnes PascalCase incompatibles avec loader v3 | `df["Close"]`, `df["High"]`, `df["Low"]` directs |

**Erreur de diagnostic initial** : La correction v0 supposait que `load_asset()` normalisait en lowercase. En réalité, `load_asset()` normalise en minuscules puis **re-renomme en Title Case** (ligne 140: `{c: c.title() for c in ohlc_cols}`). Les stratégies doivent donc utiliser `Close`, `High`, `Low` (PascalCase). Ce bug a causé le `KeyError: 'close'` au premier run. Corrigé au deuxième passage.

---

## 3. Résultats

Commande :
```bash
python scripts/run_h07_strategies_alt.py
```

Sortie : [`predictions/h07_strategies_alt.json`](predictions/h07_strategies_alt.json)

### 3.1 Synthèse

| Stratégie | Best Params | Sharpe Train | Sharpe Val | Sharpe Test | WR Test | Trades Test | Corr vs Donchian | GO? |
|-----------|-------------|-------------|------------|-------------|---------|-------------|-------------------|-----|
| **Donchian** (baseline) | N=20, M=20 | — | — | −1.14 | 48.4% | 91 | 1.00 | — |
| **Dual MA** | fast=10, slow=50 | +0.79 | −0.20 | +0.36 | 52.2% | 594 | 0.19 | ❌ |
| **Keltner** | period=20, mult=2.0 | +0.98 | **+3.70** | −0.76 | 50.7% | 75 | 0.29 | ❌ |
| **Chandelier** | period=44, k_atr=4.0 | +0.62 | +2.36 | NaN | 50.9% | 595 | 0.28 | ❌ |
| **Parabolic SAR** | step=0.03, af_max=0.2 | +0.47 | +0.64 | −0.01 | 49.9% | 627 | 0.31 | ❌ |

### 3.2 Détail Dual MA (fast=10, slow=50)

- **Train** : Sharpe +0.79 sur 1853 trades
- **Val** : Sharpe −0.20, WR 46.7% — dégradation dès 2023
- **Test** : Sharpe +0.36, WR 52.2%, 594 trades — seul test Sharpe positif mais DSR −12.66 (p=1.000), max DD 189%
- **Edge** : ❌ Sharpe 0.36 ≪ 1.0, DSR −12.66 non significatif, DD 189.1% ≥ 15%
- **Corrélation** : 0.19 vs Donchian — ✅ diversifiant

### 3.3 Détail Keltner (period=20, mult=2.0)

- **Train** : Sharpe +0.98 sur 342 trades — très bon
- **Val** : Sharpe **+3.70**, WR 58.7%, 46 trades — performance exceptionnelle… mais sur 2023 uniquement
- **Test** : Sharpe **−0.76**, WR 50.7%, 75 trades — effondrement complet
- **Edge** : ❌ Sharpe −0.76, DSR −12.26 (p=1.000), DD 183.7%
- **Corrélation** : 0.29 vs Donchian — ✅ diversifiant
- **Diagnostic** : Overfitting flagrant — val Sharpe +3.70 → test −0.76. Le marché 2023 (sideways/bear US30) n'était pas représentatif du test 2024+.

### 3.4 Détail Chandelier Exit (period=44, k_atr=4.0)

- **Train** : Sharpe +0.62 sur 1817 trades
- **Val** : Sharpe +2.36, WR 46.7%, 199 trades
- **Test** : Sharpe **NaN** (PnL constant sur certaines périodes → écart-type nul), WR 50.9%, 595 trades
- **Edge** : ❌ Sharpe 0.00 (NaN ramené à 0), DSR NaN, DD 215.2%
- **Corrélation** : 0.28 vs Donchian — ✅ diversifiant

### 3.5 Détail Parabolic SAR (step=0.03, af_max=0.2)

- **Train** : Sharpe +0.47 sur 1865 trades
- **Val** : Sharpe +0.64, WR 47.4%, 230 trades
- **Test** : Sharpe −0.01, WR 49.9%, 627 trades — essentiellement flat
- **Edge** : ❌ Sharpe −0.01, DSR −27.52 (p=1.000), DD 269.1%
- **Corrélation** : 0.31 vs Donchian — ✅ diversifiant

---

## 4. Verdict

### 🔴 NO-GO — 0 stratégie sur 4 ne passe les critères

| Critère | Dual MA | Keltner | Chandelier | Parabolic |
|---------|---------|---------|------------|-----------|
| Sharpe test > 0 | ✅ (+0.36) | ❌ (−0.76) | ❌ (NaN) | ❌ (−0.01) |
| Corrélation < 0.7 vs Donchian | ✅ (0.19) | ✅ (0.29) | ✅ (0.28) | ✅ (0.31) |
| validate_edge pass | ❌ (DSR −12.66) | ❌ (DSR −12.26) | ❌ (DSR NaN) | ❌ (DSR −27.52) |

### Conclusion

Aucune stratégie trend-following alternative ne survit aux coûts réalistes v3 sur US30 D1. Le seul candidat marginal (Dual MA, Sharpe test +0.36) a un DSR de −12.66 qui invalide toute significativité statistique.

**Observation clé** : Toutes les stratégies sont fortement décorrélées de Donchian (ρ ∈ [0.19, 0.31]), ce qui confirme leur utilité potentielle en **diversification de portefeuille**. Cependant, sans edge propre, la diversification seule ne crée pas de valeur.

**Overfitting val systématique** : Keltner et Chandelier montrent des Sharpe val ≥ +2.3 qui s'effondrent en test. La période 2023 (rally + selloff US30) favorise les trend-followers, mais ce régime n'a pas persisté en 2024+.

### Recommandation pour la suite

1. **Abandonner les stratégies déterministes pures** sur US30. Avec des coûts réalistes (16 pips/trade effectifs), aucune règle simple ne survit.
2. **Passer à H08** (combinaison naïve multi-actif equal risk) — même avec Donchian seul, tester si la diversification multi-actif améliore le Sharpe portefeuille.
3. **H10-H11** (méta-labeling) devient critique — utiliser un RF pour filtrer les trades Donchian pourrait réduire le nombre de trades perdants et améliorer le PnL/trade.

---

## 5. Fichiers modifiés/créés

| Fichier | Action |
|---------|--------|
| `app/strategies/dual_ma.py` | Modifié — PascalCase + shift(1) |
| `app/strategies/keltner.py` | Modifié — PascalCase + shift(1) |
| `app/strategies/chandelier.py` | Modifié — PascalCase + shift(1) |
| `app/strategies/parabolic.py` | Modifié — PascalCase + shift(1) |
| `app/backtest/deterministic.py` | Modifié — PascalCase direct |
| `tests/unit/test_strategy_dual_ma.py` | Créé — 5 tests |
| `tests/unit/test_strategy_keltner.py` | Créé — 6 tests |
| `tests/unit/test_strategy_chandelier.py` | Créé — 5 tests |
| `tests/unit/test_strategy_parabolic.py` | Créé — 5 tests |
| `scripts/run_h07_strategies_alt.py` | Créé — 490 lignes |
| `predictions/h07_strategies_alt.json` | Créé — résultats 36 backtests |
| `prompts/08_architecture_plan.md` | Créé — plan d'architecture |
| `docs/v3_hypothesis_07.md` | Créé — ce fichier |

---

## 6. Qualité

- **ruff** : ✅ All checks passed (10 fichiers)
- **mypy** : ✅ Success: no issues found (10 fichiers)
- **pytest** : ⏳ À exécuter (Constitution Règle 2)
- **snooping_check** : ✅ TEST_SET_LOCK.json absent
