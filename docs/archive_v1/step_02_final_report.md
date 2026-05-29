# Step 02 — Rapport final d'analyse : CPCV + DSR — EURUSD H1

**Date** : 2026-05-13
**Statut** : 🔴 NO-GO
**Fichiers** : [`step_02_robust_validation_framework.md`](docs/step_02_robust_validation_framework.md) | [`step_02_implementation_report.md`](docs/step_02_implementation_report.md)
**Résultats bruts** : [`predictions/cpcv_report.md`](predictions/cpcv_report.md) | [`predictions/cpcv_results.csv`](predictions/cpcv_results.csv)

---

## 1. Résumé exécutif

Le framework CPCV + DSR a été implémenté et exécuté sur la baseline v15 (RandomForest 500 arbres, triple_barrier, TP=30/SL=10). **Les 3 critères GO sont en échec** :

| Critère | Seuil | Mesuré | Verdict |
|---|---|---|---|
| DSR > 0 | > 0 | **−5.15** | ❌ |
| % splits profitables > 60% | > 60% | **1.0%** | ❌ |
| E[Sharpe] > 0 | > 0 | **−0.138** | ❌ |

**Le modèle n'a pas d'edge mesurable sur EURUSD H1.** La distribution CPCV est centrée en territoire négatif avec une variance très faible — le modèle est *consistamment* perdant, pas *aléatoirement* perdant.

---

## 2. Détail des résultats

### 2.1 Distribution CPCV — 200 splits (2024-2025 OOS)

| Statistique | Valeur |
|---|---|
| Splits valides | 200/200 (100%) |
| E[Sharpe] | −0.1378 |
| σ[Sharpe] | 0.0579 |
| Sharpe médian | −0.1435 |
| Sharpe min | −0.2760 |
| Sharpe max | +0.0245 |
| Sharpe CI 95% | [−0.2352, −0.0257] |
| % profitables | 1.00% (2 splits / 200) |
| Trades/split (μ ± σ) | 341.3 ± 70.0 |
| Trades totaux | 68 266 |

**Interprétation** :
- La distribution est **étroite et entièrement négative** (σ = 0.058, max = +0.025). Le modèle produit des résultats négatifs quelle que soit la période de test OOS.
- Seulement **2 splits sur 200** ont un Sharpe > 0 — et le max est à +0.0245, négligeable.
- Le CI 95% est **entièrement négatif** [−0.235, −0.026] : avec 95% de confiance, le vrai Sharpe est négatif.
- Ce n'est pas un problème de variance ou de surapprentissage — c'est un **problème d'absence d'edge fondamental**.

### 2.2 DSR Distributionnel

| Métrique | Valeur |
|---|---|
| DSR | **−5.1462** |
| PSR(SR*=0) | 0.0086 |
| SR₀* (seuil déflaté) | 0.1601 |
| E[max SR] sous H₀ (EVT) | 0.1601 |
| Var[SR] CPCV | 0.0034 |

**Interprétation** :
- Le SR₀* (seuil que le Sharpe doit dépasser pour être significatif après correction des comparaisons multiples) est de **0.16** — relativement bas car la variance des splits est faible.
- Le DSR est à **−5.15** : le Sharpe observé (−0.138) est à **5.15 écarts-types en-dessous du seuil de significativité**. C'est un rejet massif de H₁.
- Le PSR(SR*=0) est à **0.009** : probabilité < 1% que le vrai Sharpe soit > 0. Autrement dit, on a > 99% de confiance que le vrai Sharpe est ≤ 0.

### 2.3 Split Principal (train ≤ 2023, test = 2025)

| Métrique | Valeur |
|---|---|
| N trades | 853 |
| Sharpe observé | −0.1099 |
| p(Sharpe>0) bootstrap | 0.9997 |
| PSR (Bailey) | 0.0013 |
| DSR distributionnel | −4.6645 |
| Breakeven WR | 28.8% |
| WR observé | 25.7% |
| t-statistique | −3.21 (p=0.0014) |

**Performance 2025** :

| Métrique | Valeur |
|---|---|
| Sharpe | −2.63 |
| Return total | −13.44% |
| Max drawdown | −14.49% |
| Win rate | 25.67% |

**Interprétation** :
- Le WR observé (25.7%) est **inférieur** au breakeven WR (28.8%) → structurellement non rentable même sans friction supplémentaire.
- Le t-test rejette H₀: E[P&L]=0 avec p=0.0014 — mais dans la **mauvaise direction** (P&L moyen négatif).
- La probabilité bootstrap que le Sharpe > 0 est de **0.9997** — le bootstrap confirme que le Sharpe est quasi-certainement négatif.
- Perte de **−13.4%** du capital en 2025, avec un drawdown de −14.5%.

---

## 3. Diagnostic racine

Le CPCV établit sans ambiguïté que le problème n'est pas le surapprentissage (overfitting). Les causes probables par ordre de vraisemblance :

### 3.1 Le signal primaire est nul (Accuracy ≈ aléatoire)

L'accuracy OOS du classifieur 3-classes est de **0.332** — presque exactement l'aléatoire pour 3 classes (0.333). Le RandomForest n'apprend **rien** de structurel dans les features actuelles.

**Features actives** (7) : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`, `Dist_SMA200_D1`.

Ces 7 features sont toutes des **indicateurs techniques classiques** — il est probable qu'elles ne contiennent tout simplement pas d'information prédictive exploitable sur les mouvements H1 de l'EURUSD après prise en compte des coûts de friction.

### 3.2 La cible triple_barrier est trop difficile

Avec TP=30, SL=10, la cible est asymétrique : pour être classé "LONG gagnant", il faut un mouvement de +30 pips en ≤ 24h sans toucher −10 pips d'abord. Même si le modèle prédisait parfaitement la direction, le ratio TP/SL de 3:1 rend la tâche de classification intrinsèquement difficile.

### 3.3 Les filtres ne suffisent pas

Le MomentumFilter (v4) a réduit les trades 2025 de 848→552 (−35%) et amélioré le Sharpe de −4.35→−3.18, mais l'edge sous-jacent reste négatif. Les filtres réduisent le bruit sans créer de signal.

---

## 4. Recommandations

### 4.1 Immédiat (avant toute nouvelle itération)

1. **Ne pas lancer step_03 (GBM)**. Changer de modèle sans changer les features ou la cible ne résoudra pas le problème fondamental : il n'y a pas d'information prédictive dans les données actuelles.

2. **Exécuter step_01 (target redefinition)**. La cible `triple_barrier` avec TP=30/SL=10 produit des labels quasi-aléatoires. Les variantes à explorer :
   - `forward_return` (régression continue) : prédire le rendement forward plutôt qu'une classe discrète
   - `directional_clean` : classification binaire (HAUSSE/BAISSE) sans barrière de temps — plus simple, potentiellement plus de signal
   - `cost_aware` v2 : intégrer les coûts directement dans la fonction de perte

3. **Envisager un changement de timeframe**. Le H1 est notoirement difficile en forex — le bruit microstructurel domine. Le H4 ou D1 pourraient offrir un ratio signal/bruit plus favorable.

### 4.2 Si step_01 échoue également

- **Changer d'instrument** : tester la même approche sur XAUUSD (or) ou un indice (US30, GER30) qui ont des régimes de volatilité plus favorables aux stratégies trend-following.
- **Abandonner l'approche ML supervisé classique** sur cet instrument/timeframe et pivoter vers une approche par règles (mean-reversion identifiée par cointégration, ou arbitrage statistique multi-paires).

### 4.3 Note méthodologique positive

Le framework CPCV lui-même a **parfaitement fonctionné** :
- 200 splits générés et exécutés en 56 secondes
- Distribution propre, sans artefact
- DSR calculé correctement
- Le verdict est **sans ambiguïté** — c'est exactement ce qu'on attend d'un framework de validation robuste : éviter de poursuivre une fausse piste.

---

## 5. Leçons techniques

| Aspect | Constat |
|---|---|
| Performance CPCV | 200 splits en 56s sur CPU (joblib, n_jobs=-1). Scaling correct. |
| Purge bidirectionnelle | Fonctionnelle — 0 chevauchement train/test vérifié. |
| DSR distributionnel | Formule EVT correcte, SR₀* cohérent avec la variance empirique. |
| Tests unitaires | 42/42 passent (dont 20 nouveaux CPCV). Zéro régression. |
| Robustesse du NON | Le verdict NO-GO est massif (−5.15 DSR) — pas une décision borderline. |

---

## 6. Prochaines étapes

```
┌─────────────────────────────────────────────────────────────┐
│ ÉTAT ACTUEL : Step 02 terminé → NO-GO                       │
│                                                             │
│ SUIVANT (obligatoire) : Step 01 — Target Redefinition       │
│   - forward_return (régression)                             │
│   - directional_clean (binaire)                             │
│   - cost_aware v2                                           │
│                                                             │
│ BLOQUÉ JUSQU'À GO STEP 01/02 :                             │
│   - Step 03 (GBM)                                           │
│   - Step 04 (Session features)                              │
│   - Step 05 (Economic calendar)                             │
│   - Step 06 (Meta-labeling calibration)                     │
│   - Step 07 (Cross-asset validation)                        │
└─────────────────────────────────────────────────────────────┘
```

**Référence roadmap** : [`docs/README.md`](docs/README.md:47) — critère après step_02 : *« Si DSR > 0 et % splits profitables > 60% → GO Phase 2-3. Sinon → retour à step_01 obligatoire. »*
