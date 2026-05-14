# Roadmap v3 — Portfolio Multi-Actif Trend Following visant Sharpe ≥ 1.0

**Date** : 2026-05-13
**Statut** : 📋 Planifié
**Version** : 3.0.0
**Contexte** : v1 (EURUSD H1 + RF) → 5 NO-GO. v2 (H01–H05) → 2 GO sur US30 D1 (Donchian pur Sharpe +3.07, Donchian + RF méta-labeling Sharpe +8.61), 3 NO-GO (US30 RF, XAUUSD H4 RF, stratégies alternatives sur XAUUSD).
**n_trials hérité de v2** : 5 (H01–H05). Compteur repris en v3.

---

## 0. Synthèse des acquis v1/v2

### Ce qui a échoué (ne pas répéter)

| Pattern | Occurrences | Cause racine |
|---------|-------------|--------------|
| RF sur features techniques OHLC brutes | v1 entière, H01, H02 | Les indicateurs techniques (RSI, ADX, EMA) ne contiennent **aucune information prédictive** forward — ce sont des transformations déterministes du prix passé |
| RF mono-instrument sans signal préalable | v1, H01, H02 | Le RF ne trouve pas de structure là où il n'y en a pas. Le ML n'est pas une baguette magique. |
| TP/SL fixes sur actifs volatils | v1 EURUSD H1, H02 XAUUSD H4 | Le ratio TP/SL fixe ignore le régime de volatilité. Les stops sont touchés par le bruit microstructurel. |
| Optimisation de seuil sur un seul split | H04 (seuil 0.65 → 0 trade en val 2023) | Overfit classique : le seuil qui maximise le Sharpe train élimine tous les signaux OOS. |
| Boucle de rétroaction data-snooping | v1 (15 itérations) | Modification de `features_dropped` en réaction aux résultats OOS → le test set n'était plus out-of-sample. |

### Ce qui a fonctionné (amplifier)

| Pattern | Occurrences | Sharpe OOS | Leçon |
|---------|-------------|------------|-------|
| Donchian Breakout sur US30 D1 | H03 | +3.07 | Les stratégies trend-following **déterministes** fonctionnent sur les indices parce que les indices ont des régimes de tendance persistants |
| Donchian + RF méta-labeling | H04, H05 | +8.61 / +8.84 | Le ML en **surcouche** (méta-labeling) sur un signal déjà rentable améliore le Sharpe. Le ML ne crée pas l'edge, il le filtre. |
| Grid search déterministe systématique | H03 | — | Tester 164 combinaisons stratégie × actif sans ML a révélé l'edge Donchian que 4 tentatives ML avaient manqué. |
| Split temporel strict | Toute la v2 | — | Aucun leak. Train ≤ 2022, val = 2023, test ≥ 2024. Méthodologie robuste. |
| CPCV | H04 | — | A confirmé l'amélioration du méta-labeling (Sharpe moyen 5.79 vs 3.24) mais a aussi révélé l'instabilité (std ±10.03). |

### Principe directeur v3

> **Le ML ne remplace pas un edge. Il l'amplifie. Chercher d'abord l'edge déterministe, puis superposer le ML.**

---

## 1. Architecture cible v3

```
┌─────────────────────────────────────────────────────────────────────┐
│                        V3 PORTFOLIO ENGINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────┐ │
│  │  ASSET   │   │  STRATEGY    │   │    REGIME     │   │  META   │ │
│  │ UNIVERSE │   │  FACTORY     │   │   DETECTOR    │   │ LABELING│ │
│  │          │   │              │   │               │   │         │ │
│  │ US30 D1  │   │ Donchian(N,M)│   │ Trending vs   │   │ RF par  │ │
│  │ GER30 D1 │   │ SMA Cross    │   │ Ranging par   │   │ actif   │ │
│  │ US500 D1 │   │ Dual MA      │──▶│ actif          │──▶│ OU      │ │
│  │ XAUUSD   │   │ Keltner      │   │               │   │ RF multi│ │
│  │ XAGUSD   │   │ ...          │   │               │   │ -actif  │ │
│  │ USOIL    │   │              │   │               │   │         │ │
│  │ BUND     │   │              │   │               │   │         │ │
│  └──────────┘   └──────────────┘   └───────────────┘   └─────────┘ │
│       │                │                   │                  │     │
│       └────────────────┼───────────────────┼──────────────────┘     │
│                        ▼                   ▼                        │
│               ┌─────────────────────────────────────┐               │
│               │      PORTFOLIO CONSTRUCTOR          │               │
│               │  • Volatility targeting (10% ann.)  │               │
│               │  • Equal risk weight per strategy   │               │
│               │  • Max correlation cap = 0.7        │               │
│               │  • Daily rebalance                  │               │
│               └─────────────────────────────────────┘               │
│                        │                                            │
│                        ▼                                            │
│               ┌─────────────────────────────────────┐               │
│               │      WALK-FORWARD ENGINE            │               │
│               │  • Rolling retrain (6M)             │               │
│               │  • Realistic costs (spread+slip)    │               │
│               │  • Equity curve + Sharpe daily      │               │
│               └─────────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Plan d'hypothèses — Ordonnancement par priorité et coût

Chaque hypothèse est numérotée H06–H18 (suite de v2 H01–H05). Le DSR cumulatif intègre n_trials = 5 + numéro v3.

### Phase 1 — Expansion de l'univers déterministe (H06–H08)

**Objectif** : Trouver ≥ 3 stratégies décorrélées avec Sharpe > 0 OOS sur ≥ 3 actifs décorrélés.
**Coût** : Faible. Réutilisation du moteur `deterministic.py` + `grid_search.py` de v2.
**Risque** : Moyen. Tous les actifs ne sont pas trend-friendly.

| # | Hypothèse | Priorité | Effort | Dépend de |
|---|-----------|----------|--------|-----------|
| H06 | Extension univers CFD : Donchian grid search sur GER30, US500, USOIL, XAGUSD, BUND | 🔴 P0 | 1 jour | H03 |
| H07 | Stratégies trend-following additionnelles : Dual MA, Keltner Channel, Chandelier Exit | 🔴 P0 | 1 jour | H03 |
| H08 | Combinaison naïve multi-actif : equal risk weight, volatilité cible 10% | 🟠 P1 | 1 jour | H06, H07 |

### Phase 2 — Régime et filtrage avancé (H09–H12)

**Objectif** : Améliorer le Sharpe unitaire par stratégie via filtrage conditionnel au régime de marché.
**Coût** : Moyen. Nécessite implémentation de `RegimeDetector` et features cross-asset.
**Risque** : Élevé. Le régime detection est lui-même sujet à l'overfit.

| # | Hypothèse | Priorité | Effort | Dépend de |
|---|-----------|----------|--------|-----------|
| H09 | Régime detection par actif : ADX + volatilité + distance SMA200 pour classifier Trending/Ranging | 🟠 P1 | 2 jours | H06 |
| H10 | Trend following conditionnel : ne trader Donchian/SMA Cross qu'en régime Trending | 🟠 P1 | 0.5 jour | H09 |
| H11 | Méta-labeling RF par actif : entraîner un RF par actif pour filtrer les signaux de la stratégie gagnante | 🟡 P2 | 1.5 jour | H06, H07 |
| H12 | Méta-labeling RF multi-actif : un seul RF entraîné sur tous les actifs pour filtrer les signaux | 🟡 P2 | 1.5 jour | H11 |

### Phase 3 — Portfolio construction avancée (H13–H15)

**Objectif** : Maximiser le Sharpe portfolio via diversification et allocation dynamique.
**Coût** : Moyen-élevé. Mathématique plus complexe, validation plus exigeante.
**Risque** : Élevé. L'optimisation de portefeuille sur petit échantillon est notoirement instable.

| # | Hypothèse | Priorité | Effort | Dépend de |
|---|-----------|----------|--------|-----------|
| H13 | Correlation-aware weighting : downweight les stratégies dont la corrélation rolling 60j > 0.7 | 🟡 P2 | 1 jour | H08 |
| H14 | Volatility targeting adaptatif : cible 10% annuelle avec scaling basé sur la volatilité réalisée 20j | 🟡 P2 | 0.5 jour | H08 |
| H15 | Allocation dynamique mensuelle : réallouer le capital selon Sharpe rolling 12M par stratégie | 🟢 P3 | 1 jour | H13 |

### Phase 4 — Timeframe stacking et signaux alternatifs (H16–H17)

**Objectif** : Ajouter des sources de signal indépendantes pour diversification.
**Coût** : Élevé. Ingestion de données supplémentaires, complexité de pipeline.
**Risque** : Élevé. Multiplier les signaux = multiplier le risque d'overfit.

| # | Hypothèse | Priorité | Effort | Dépend de |
|---|-----------|----------|--------|-----------|
| H16 | Timeframe stacking : D1 pour direction, H4 pour timing d'entrée + stop placement adaptatif | 🟢 P3 | 2 jours | H10 |
| H17 | Signaux alternatifs : COT report (positioning institutionnel), backwardation/contango pour commodities | 🟢 P3 | 3 jours | — |

### Phase 5 — Walk-forward continu et déploiement (H18)

**Objectif** : Pipeline automatisé de réentraînement et déploiement périodique.
**Coût** : Élevé. Infrastructure, monitoring, gestion des états.
**Risque** : Faible. La technologie existe déjà dans `walk_forward.py`.

| # | Hypothèse | Priorité | Effort | Dépend de |
|---|-----------|----------|--------|-----------|
| H18 | Walk-forward continu automatisé : réentraînement 6M, déploiement automatique de la meilleure config, rapport mensuel | 🟢 P3 | 2 jours | H15, H16 |

---

## 3. Hypothèses détaillées — Phase 1 (H06–H08)

### H06 — Extension univers CFD : Grid Search Donchian multi-actif

**Question** : Le Donchian Breakout (qui fonctionne sur US30 D1) fonctionne-t-il sur d'autres CFD décorrélés ?

**Actifs testés** :

| Actif | Timeframe | Justification | Corrélation attendue vs US30 | Données |
|-------|-----------|---------------|------------------------------|---------|
| US30 (baseline) | D1 | Déjà GO en v2 | 1.0 (référence) | ✅ `data/raw/` |
| GER30 (DAX) | D1 | Indice européen, corrélation modérée avec US30 | ~0.6 | ⚠️ À sourcer |
| US500 (S&P 500) | D1 | Indice US large, corrélé mais pas identique | ~0.8 | ⚠️ À sourcer |
| XAUUSD (Or) | D1 | Métal précieux, anti-dollar, décorrélé des indices | ~0.0 à −0.3 | ✅ `data/raw/` |
| XAGUSD (Argent) | D1 | Métal précieux + industriel, partiellement corrélé à l'or | ~0.5 vs XAUUSD | ⚠️ À sourcer |
| USOIL (WTI) | D1 | Énergie, drivers fondamentaux distincts | ~0.2 | ⚠️ À sourcer |
| BUND (Obligataire allemand) | D1 | Taux, décorrélé des actions | ~−0.3 à 0.2 | ⚠️ À sourcer |

**Stratégie** : Donchian Breakout uniquement — grid search N ∈ {20, 50, 100}, M ∈ {10, 20, 50} (9 combinaisons).

**Split** : train ≤ 2022, val = 2023, test ≥ 2024 (identique v2).

**Backtest** : `run_deterministic_backtest` avec coûts réalistes par actif.

**Critères GO** :
- Sharpe test ≥ 0 ET WR > breakeven WR (dépend du ratio TP/SL par actif)
- Au moins 2 nouveaux actifs GO (en plus d'US30)

**Critères NO-GO** :
- Aucun nouvel actif ne produit Sharpe > 0
- → L'univers trend-following se limite à US30. On passe en mono-actif avec H09 (régime).

**Coûts réalistes par actif** (à calibrer avec le broker) :

| Actif | Spread indicatif | Slippage | Commission | TP (points) | SL (points) | Window |
|-------|------------------|----------|------------|-------------|-------------|--------|
| GER30 | 2.0 | 3.0 | 0 | 400 | 200 | 120h |
| US500 | 1.5 | 2.0 | 0 | 200 | 100 | 120h |
| XAUUSD | 25 | 10 | 0 | 600 | 300 | 120h |
| XAGUSD | 30 | 15 | 0 | 150 | 75 | 120h |
| USOIL | 4 | 3 | 0 | 200 | 100 | 120h |
| BUND | 3 | 2 | 0 | 150 | 75 | 120h |

> Note : TP/SL en points pour indices/or/pétrole. Ratio 2:1 conservé. À ajuster selon l'ATR(20) moyen de chaque actif (TP ≈ 2 × ATR, SL ≈ 1 × ATR).

**n_trials DSR** : 6 (5 hérités + H06)

---

### H07 — Stratégies trend-following additionnelles

**Question** : Existe-t-il d'autres stratégies trend-following déterministes qui surpassent Donchian ou qui sont décorrélées de Donchian ?

**Stratégies testées** :

| # | Stratégie | Règle LONG | Règle SHORT | Paramètres (grid) | Combinaisons |
|---|-----------|------------|-------------|---------------------|--------------|
| S6 | **Dual Moving Average** | SMA(fast) > SMA(slow) | SMA(fast) < SMA(slow) | fast ∈ {5,10,20}, slow ∈ {50,100,200} | 9 |
| S7 | **Keltner Channel** | Close > KC_upper(20, 2.0) | Close < KC_lower(20, 2.0) | period ∈ {10,20,50}, mult ∈ {1.5, 2.0, 2.5} | 9 |
| S8 | **Chandelier Exit** | Close > Highest(High, 22) − 3×ATR(22) | Close < Lowest(Low, 22) + 3×ATR(22) | period ∈ {11,22,44}, atr_mult ∈ {2.0, 3.0, 4.0} | 9 |
| S9 | **Parabolic SAR** | PSAR flips below price | PSAR flips above price | step ∈ {0.01, 0.02, 0.03}, max ∈ {0.1, 0.2, 0.3} | 9 |

**Actifs** : US30 D1 (baseline) + tous les actifs GO de H06.

**Protocole** : Même grid search que H03. Meilleur paramétrage sélectionné sur train ≤ 2022, évalué sur val 2023 + test ≥ 2024.

**Critères GO** :
- Au moins 1 nouvelle stratégie avec Sharpe test > 0 sur au moins 1 actif
- Corrélation des retours quotidiens < 0.7 avec Donchian sur le même actif (sinon, c'est la même stratégie déguisée)

**Critères NO-GO** :
- Aucune stratégie alternative ne produit Sharpe > 0
- → On conserve Donchian comme unique stratégie trend-following

**n_trials DSR** : 7

---

### H08 — Combinaison naïve multi-actif equal risk

**Question** : La combinaison de N stratégies × M actifs décorrélés en portefeuille equal risk weight produit-elle un Sharpe portfolio ≥ 1.0 ?

**Méthode** :

```
Pour chaque stratégie S_i sur actif A_j validée GO :
  1. Calculer la volatilité réalisée 60j des retours quotidiens σ_ij
  2. Position size = (target_vol × capital) / (σ_ij × √252)
  3. Allocation = capital / (N_stratégies × M_actifs_valides) → equal risk weight
  4. Rebalance quotidien
  5. Sharpe portfolio = mean(daily_returns) / std(daily_returns) × √252
```

**Paramètres** :
- `target_vol` = 10% annuelle
- `leverage_max` = 2.0
- `correlation_cap` = 0.7 (si deux stratégies ont une corrélation rolling 60j > 0.7, on garde la meilleure)

**Configuration de test** :
- Période : 2024-01-01 → 2025-05-13 (walk-forward avec réentraînement 6M)
- Coûts réels par actif (spread + slippage + commission)
- Pas de régime filter, pas de méta-labeling (baseline naïve)

**Critères GO** :
- **Sharpe portfolio test ≥ 1.0** ET **Sharpe portfolio val ≥ 0**
- → Passage en Phase 2 pour optimisation

**Critères NO-GO** :
- Sharpe portfolio < 0.5
- → Rester en mono-actif US30. Objectif Sharpe ≥ 1.0 via régime + méta-labeling.

**n_trials DSR** : 8

---

## 4. Hypothèses détaillées — Phase 2 (H09–H12)

### H09 — Régime detection par actif

**Question** : Peut-on classifier de manière robuste le régime de marché (Trending vs Ranging) pour conditionner l'exécution des stratégies trend-following ?

**Méthode** :

```python
def classify_regime(df: pd.DataFrame, period: int = 60) -> pd.Series:
    """
    Régime binaire : 1 = Trending, 0 = Ranging/Choppy.
    Basé sur 3 indicateurs :
    1. ADX(14) > 25 → condition nécessaire
    2. |Close − SMA(200)| / ATR(14) > 2.0 → tendance établie
    3. Efficiency Ratio = |Close − Close.shift(period)| / sum(|Close − Close.shift(1)|) sur `period` jours > 0.3
    """
```

**Split** : Le régime est calculé en **pure forward** : à la barre t, le régime utilise uniquement l'information ≤ t.

**Validation** :
- Distribution des régimes sur train ≤ 2022
- Stabilité du régime en OOS (pas de flip-flap intraday)
- Pourcentage de temps en Trending par actif (doit être entre 30% et 70% — si 5% ou 95%, le classifieur est inutile)

**Critères GO** :
- Le régime Trending contient ≥ 70% des trades gagnants de Donchian sur train
- Le filtre Trending réduit le nombre de trades de ≤ 50% (sinon, le filtre est trop agressif)
- Stabilité OOS : le ratio Trending/Ranging en val 2023 est similaire à train (±20%)

**Critères NO-GO** :
- Le régime detector ne discrimine pas les trades gagnants/perdants mieux que le hasard
- → Abandon du régime filtering. Le trend-following tourne non conditionnel.

**n_trials DSR** : 9

---

### H10 — Trend following conditionnel au régime

**Question** : Ne trader Donchian/SMA Cross qu'en régime Trending améliore-t-il le Sharpe par rapport au non-conditionnel ?

**Méthode** :
- Pour chaque stratégie × actif validé GO en Phase 1 :
  - Backtest avec `regime_filter = True` : seuls les signaux émis en régime Trending sont tradés
  - Comparaison A/B : Sharpe conditionnel vs Sharpe non-conditionnel

**Critères GO** (par stratégie × actif) :
- Sharpe conditionnel > Sharpe non-conditionnel
- ET réduction du max drawdown

**Critères NO-GO** :
- Le filtre dégrade le Sharpe sur ≥ 50% des stratégies × actifs
- → Abandon du régime filtering.

**n_trials DSR** : 10

---

### H11 — Méta-labeling RF par actif

**Question** : Un RF entraîné par actif pour filtrer les signaux de la stratégie gagnante améliore-t-il le Sharpe unitaire ?

**Réplication de H04** sur chaque nouvel actif GO :
- Features : RSI(14), ADX(14), Dist_SMA50, Dist_SMA200, ATR_Norm, Log_Return_5d, Signal_Strategy
- Cible méta-labeling : 1 si trade gagnant, 0 sinon
- RF 200 arbres, max_depth=4, min_samples_leaf=10
- Seuil calibré sur train (grille [0.45, 0.50, 0.55, 0.60])

**Différence clé vs H04** :
- Seuil **plancher** à 0.50 maximum (leçon H04 : 0.65 → 0 trade en val)
- Si le seuil optimal train élimine > 80% des trades val → fallback à 0.50

**Critères GO** (par actif) :
- Sharpe méta-labeling > Sharpe baseline déterministe
- CPCV : Sharpe moyen méta-labeling > Sharpe moyen baseline

**Critères NO-GO** (par actif) :
- Méta-labeling n'améliore pas → conserver la baseline déterministe pour cet actif

**n_trials DSR** : 11

---

### H12 — Méta-labeling RF multi-actif

**Question** : Un seul RF entraîné sur les signaux de TOUS les actifs combinés généralise-t-il mieux qu'un RF par actif ?

**Hypothèse** : Plus de données d'entraînement (signaux de 5-7 actifs vs 1) → meilleure généralisation. Les patterns de faux signaux trend-following peuvent être universels (faux breakout dans un range, divergence RSI, etc.).

**Méthode** :
- Dataset = concaténation des signaux de tous les actifs GO
- Feature supplémentaire : `Asset_ID` encodé en one-hot
- Même protocole que H11 (seuil plancher 0.50, CPCV)

**Split** : temporel strict par actif. Pas de mélange inter-temporel.

**Critères GO** :
- Sharpe moyen multi-actif > Sharpe moyen H11 (RF par actif)
- OU meilleure stabilité CPCV (écart-type plus faible)

**Critères NO-GO** :
- RF multi-actif ≤ RF par actif
- → Conserver H11

**n_trials DSR** : 12

---

## 5. Hypothèses détaillées — Phase 3 (H13–H15)

### H13 — Correlation-aware weighting

**Question** : Downweight les stratégies fortement corrélées améliore-t-il le Sharpe portfolio ?

**Méthode** :
- Matrice de corrélation rolling 60j des retours quotidiens par stratégie × actif
- Si ρ_ij > 0.7 → on conserve la stratégie avec le meilleur Sharpe rolling 6M, on désactive l'autre
- Rebalance hebdomadaire

**Critères GO** :
- Sharpe portfolio ≥ Sharpe H08 (equal weight naïf) + 0.1
- OU réduction du max drawdown ≥ 20%

**n_trials DSR** : 13

---

### H14 — Volatility targeting adaptatif

**Question** : Ajuster la taille de position dynamiquement selon la volatilité réalisée récente améliore-t-il le Sharpe ?

**Méthode** :
- `position_size = (target_vol × capital) / (realized_vol_20d × √252)`
- `target_vol` = 10% annuelle
- `leverage_cap` = 2.0
- Comparaison A/B : fixed size vs volatility-targeted size

**Critères GO** :
- Sharpe portfolio ≥ Sharpe fixed-size + 0.05
- OU réduction de la volatilité réalisée du portefeuille ≥ 10%

**n_trials DSR** : 14

---

### H15 — Allocation dynamique mensuelle

**Question** : Réallouer le capital mensuellement selon le Sharpe rolling 12M par stratégie améliore-t-il le Sharpe long-terme ?

**Méthode** :
- Chaque mois, calculer le Sharpe rolling 12M de chaque stratégie × actif
- Allocation ∝ max(0, Sharpe_12M) → les stratégies à Sharpe négatif récent reçoivent 0 capital
- Lissage exponentiel pour éviter les à-coups : `weight_t = 0.7 × weight_{t-1} + 0.3 × target_weight_t`

**Critères GO** :
- Sharpe portfolio ≥ Sharpe equal-weight + 0.15

**Critères NO-GO** :
- Allocation dynamique introduit du turnover excessif (> 50% de turnover mensuel)
- OU Sharpe inférieur à equal-weight

**n_trials DSR** : 15

---

## 6. Hypothèses détaillées — Phase 4 (H16–H17)

### H16 — Timeframe stacking D1/H4

**Question** : Utiliser le signal D1 pour la direction et H4 pour le timing d'entrée + stop placement améliore-t-il le Sharpe ?

**Méthode** :
- Signal D1 : Donchian(20,20) sur D1 → direction LONG/SHORT/FLAT
- Timing H4 : entrée sur pullback dans la direction D1 (Close H4 < SMA20 H4 en tendance D1 haussière)
- Stop H4 : ATR(14) H4 × 1.5 au lieu de SL fixe
- TP : 2 × stop (ratio risque/récompense dynamique)

**Actifs testés** : US30 (priorité), GER30, US500 (si données H4 disponibles).

**Critères GO** :
- Sharpe stacking > Sharpe D1-only

**n_trials DSR** : 16

---

### H17 — Signaux alternatifs : COT + term structure

**Question** : Les données de positionnement institutionnel (COT) et la structure par terme (commodities) ajoutent-elles un pouvoir prédictif incrémental ?

**Sources** :
- **COT (Commitments of Traders)** : publié chaque vendredi par la CFTC. Positions nettes des Commercials, Non-Commercials, Non-Reportable sur futures.
- **Term structure** : backwardation/contango pour USOIL (futures WTI).

**Méthode** :
- Feature COT : `COT_Commercial_Net_Change_4W` (variation sur 4 semaines de la position nette Commercial)
- Feature Term Structure : `Roll_Yield = (Front_Month − Next_Month) / Next_Month`
- Ajoutées aux features du RF méta-labeling (H11/H12)

**Actifs concernés** : XAUUSD, XAGUSD, USOIL (ceux pour lesquels le COT est disponible).

**Critères GO** :
- Sharpe avec COT > Sharpe sans COT (pour les actifs concernés)

**n_trials DSR** : 17

---

## 7. Hypothèses détaillées — Phase 5 (H18)

### H18 — Walk-forward continu automatisé

**Question** : Un pipeline de réentraînement périodique automatique maintient-il le Sharpe ≥ 1.0 en conditions walk-forward continues ?

**Méthode** :
- Réentraînement tous les 6 mois (01/01 et 01/07)
- Pour chaque réentraînement :
  1. Toutes les données ≤ date sont disponibles
  2. Grid search des paramètres de stratégie sur train (expanding window)
  3. Recalibration du seuil méta-labeling (si applicable)
  4. Recalcul de la matrice de corrélation
  5. Rapport automatique : Sharpe par stratégie, Sharpe portfolio, drawdown
- Simulation walk-forward identique à H05 mais sur le portefeuille complet

**Critères GO** :
- **Sharpe portfolio walk-forward ≥ 1.0** sur la période test (2024-2025)
- **Max drawdown ≤ 25%**
- **DSR > 0** avec n_trials cumulatif final (≥ 13)
- → 🟢 **DÉPLOIEMENT EN PAPER TRADING LIVE**

**Critères NO-GO** :
- Sharpe < 0.5
- → Retour à la Phase 1 avec de nouveaux actifs/timeframes

**n_trials DSR** : 18

---

## 8. Protocole anti-snooping — Règles v3

### Règle 1 — Split temporel strict, jamais modifié
- Train ≤ 2022-12-31, Val = 2023-01-01 → 2023-12-31, Test ≥ 2024-01-01
- **Ce split est le dernier** avant passage en walk-forward rolling (H18)
- Si une hypothèse utilise un sous-ensemble de dates (données indisponibles avant 2018), le documenter AVANT le run

### Règle 2 — DSR cumulatif depuis v2
- Compteur initial = 5 (H01–H05 v2)
- Chaque hypothèse v3 incrémente le compteur
- Le DSR final (H18) utilise n_trials = 5 + nombre d'hypothèses v3 exécutées

### Règle 3 — Pré-enregistrement obligatoire
Avant chaque run OOS, documenter dans l'hypothèse :
- Liste exacte des actifs
- Paramètres de stratégie exacts
- Features (si ML)
- Hyperparamètres (si ML)
- Coûts modélisés
- Critères GO/NO-GO explicites

### Règle 4 — Un seul regard par hypothèse
Les résultats OOS sont lus une seule fois. Aucune modification de paramètres en réaction.

### Règle 5 — Instrument hold-out
USDCHF D1 est réservé comme gate final. Il ne sera testé qu'à la fin de la roadmap, avec la config figée, pour une validation truly out-of-sample.

### Règle 6 — Sharpe corrigé
Le Sharpe est TOUJOURS calculé comme :
```python
daily_returns = equity_curve.pct_change().dropna()
sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
```
**Jamais** PnL/trade × √252. Cette erreur a été commise en v1 et ne sera pas répétée.

### Règle 7 — Coûts réalistes avant validation
Les coûts (spread + slippage + commission) sont modélisés dans le backtest stateful, pas soustraits a posteriori. Le slippage inclut une composante aléatoire uniforme.

---

## 9. Calendrier estimé

| Phase | Hypothèses | Effort total | Durée calendaire |
|-------|------------|--------------|------------------|
| 1 — Expansion univers | H06, H07, H08 | 3 jours | 1 semaine |
| 2 — Régime et filtrage | H09, H10, H11, H12 | 5 jours | 1.5 semaines |
| 3 — Portfolio avancé | H13, H14, H15 | 2.5 jours | 1 semaine |
| 4 — Stacking et alternatif | H16, H17 | 5 jours | 1.5 semaines |
| 5 — Walk-forward continu | H18 | 2 jours | 0.5 semaine |
| **Total** | **13 hypothèses** | **17.5 jours** | **5-6 semaines** |

---

## 10. Go/No-Go global v3

### GO final (après H18) si :
- **Sharpe portfolio walk-forward ≥ 1.0** après coûts
- **DSR > 0** avec n_trials cumulatif ≥ 13
- **Max drawdown ≤ 25%**
- **≥ 3 actifs** contribuent positivement au Sharpe
- **≥ 2 stratégies décorrélées** (ρ < 0.5)
- → 🟢 **Déploiement en paper trading live** avec capital réel limité

### NO-GO final si :
- Sharpe < 0.5 après tous les efforts
- → Conclusion : le trend-following multi-actif sur CFD D1 avec stops fixes ne produit pas de Sharpe ≥ 1.0. Pivot vers :
  - Exécution discrétionnaire assistée par les signaux
  - Options / produits structurés
  - Timeframes plus longs (hebdomadaire)
  - Changement de paradigme (market making, arbitrage statistique)

---

## 11. Fichiers à créer — Structure v3

```
learning_machine_learning_v3/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── instruments.py          # GER30, US500, XAGUSD, USOIL, BUND configs
│   ├── backtest.py             # BacktestConfig étendu (multi-actif)
│   └── portfolio.py            # PortfolioConfig (target_vol, max_correlation, etc.)
├── core/
│   ├── __init__.py
│   ├── exceptions.py           # Réimporté de v2
│   ├── logging.py              # Réimporté de v2
│   └── types.py                # Réimporté de v2
├── data/
│   ├── __init__.py
│   └── loader.py               # load_all_assets() → dict[str, pd.DataFrame]
├── features/
│   ├── __init__.py
│   ├── technical.py            # Indicateurs techniques (minimal, explicite)
│   ├── regime.py               # classify_regime() — Trending/Ranging
│   └── alternative/
│       ├── __init__.py
│       ├── cot.py               # Features COT (Commitments of Traders)
│       └── term_structure.py    # Roll yield pour commodities
├── strategies/
│   ├── __init__.py
│   ├── base.py                 # BaseStrategy (réimporté v2)
│   ├── donchian.py             # DonchianBreakout (réimporté v2)
│   ├── dual_ma.py              # DualMovingAverage
│   ├── keltner.py              # KeltnerChannel
│   ├── chandelier.py           # ChandelierExit
│   └── parabolic.py            # ParabolicSAR
├── backtest/
│   ├── __init__.py
│   ├── deterministic.py        # Moteur backtest stateful (réimporté v2)
│   ├── grid_search.py          # Grid search multi-stratégie (réimporté v2)
│   ├── meta_labeling.py        # Méta-labeling (réimporté v2)
│   ├── cpcv.py                 # CPCV (réimporté v2)
│   ├── filters.py              # Filtres (VolFilter, RegimeFilter)
│   ├── metrics.py              # Calculs Sharpe, drawdown (réimporté v2)
│   └── walk_forward.py         # Walk-forward continu (réimporté v2, étendu)
├── portfolio/
│   ├── __init__.py
│   ├── constructor.py          # build_portfolio() — equal risk weight, vol targeting
│   ├── correlation.py          # rolling_correlation_matrix(), filter_correlated()
│   ├── allocation.py           # allocate_dynamic(), allocate_equal_risk()
│   └── rebalance.py            # daily_rebalance(), apply_costs()
├── models/
│   ├── __init__.py
│   └── meta_rf.py              # Méta-labeling RF (réimporté v2)
├── pipelines/
│   ├── __init__.py
│   ├── base.py                 # BasePipeline (réimporté v2)
│   └── multi_asset.py          # MultiAssetPipeline — orchestre N actifs
├── analysis/
│   ├── __init__.py
│   ├── edge_validation.py      # DSR, PSR (réimporté v2)
│   └── cross_asset.py          # Corrélation PnL, heatmap
scripts/
├── inspect_{asset}_csv.py      # Diagnostic données par actif
run_v3_phase1.py                # H06–H08 orchestrateur
run_v3_phase2.py                # H09–H12 orchestrateur
run_v3_phase3.py                # H13–H15 orchestrateur
run_v3_phase4.py                # H16–H17 orchestrateur
run_v3_walk_forward.py          # H18 orchestrateur final
docs/
├── v3_roadmap.md               # Ce document
├── v3_hypothesis_06.md         # Rapport H06
├── v3_hypothesis_07.md         # Rapport H07
├── ...                         # Un doc par hypothèse
└── v3_final_report.md          # Rapport final
```

---

## 12. Dépendances de données — Plan de sourcing

| Actif | Timeframes | Source | Priorité |
|-------|------------|--------|----------|
| USA30IDXUSD | D1, H4 | ✅ Déjà disponible | — |
| XAUUSD | D1, H4 | ✅ Déjà disponible | — |
| GER30 | D1, H4 | Dukascopy / FXCM CSV export | 🔴 P0 |
| US500 | D1, H4 | Dukascopy / FXCM CSV export | 🔴 P0 |
| XAGUSD | D1 | Dukascopy / FXCM CSV export | 🟠 P1 |
| USOIL | D1 | Dukascopy / FXCM CSV export | 🟠 P1 |
| BUND | D1 | Dukascopy / FXCM CSV export | 🟡 P2 |
| COT (tous) | Hebdo | CFTC website (gratuit) | 🟢 P3 |

---

## 13. Risques et mitigations

| Risque | Probabilité | Impact | Mitigation |
|--------|------------|--------|------------|
| Données insuffisantes pour nouveaux actifs (< 2018) | Moyenne | Retard | Commencer par US30+XAUUSD (déjà dispo), ajouter les autres au fil de l'eau |
| Aucun nouvel actif ne passe Donchian (H06 NO-GO) | Moyenne | Réduction du scope à US30 | Accepter le portefeuille mono-actif. Objectif Sharpe > 0.5 avec régime+méta-labeling, pas 1.0. |
| Régime detector overfit (H09) | Élevée | Perte de temps | Critères de stabilité OOS stricts. Si instable → abandon rapide. |
| Méta-labeling instable (H11/H12) | Élevée | Faux espoirs | CPCV obligatoire. Si std > Sharpe moyen → rejeter. Seuil plancher à 0.50. |
| Corrélation entre actifs augmente en crise | Élevée | Drawdown simultané | Correlation cap + vol targeting. Accepter que les crises rendent tout corrélé à 1. |
| Overfit cumulatif sur le split ≤ 2022/2023/2024 | Élevée | DSR non significatif | Le split est FIXE. Après 13 hypothèses, le DSR corrige. Si DSR final < 0 → tout rejeter, nouveau split. |
| Coûts de transaction sous-estimés | Moyenne | Sharpe réel < Sharpe backtest | Utiliser des coûts conservateurs (spread × 1.5). Paper trading live = validation ultime. |
