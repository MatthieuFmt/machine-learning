# Audit v6 — Angles morts : Méthodologie

**Périmètre** : approches ML alternatives, features non explorées, méthodes de validation manquantes, sizing/portfolio, robustesse.

---

## 1. ML — ce qui n'a pas été essayé

### 1.1 Modèles tabulaires SOTA

| Modèle | Statut projet | Pourquoi tester |
|---|---|---|
| **LightGBM** | ❌ Exclu volontairement (CLAUDE.md, plan v5) | SOTA tabulaire 2017-2024. Gain typique +5-10 % vs HGBM sur dataset moyen. Optuna tuning trivial. |
| **XGBoost** | ❌ Exclu | Idem LightGBM. Plus mature, mêmes gains. |
| **CatBoost** | ❌ Exclu | Excellent sur features mixtes catégorielles/numériques. Natif sur dates et booléens. |
| **TabPFN** (Hollmann 2023) | ❌ | Foundation model tabulaire, performances bluffantes sur petits datasets (<10k samples). Pas besoin de tuning. |
| **TabNet** (Arik & Pfister 2019) | ❌ | DL tabulaire interprétable. Probablement overkill pour notre taille de dataset. |

**Décision projet** : aucune dépendance lourde. C'est défendable, mais HGBM (HistGradientBoostingRegressor sklearn) n'égale pas LightGBM en pratique. **Coût d'ajout** : 1 ligne dans `requirements.txt`, ~50 lignes pour wrapping cohérent avec l'API sklearn.

**Recommandation** : ajouter LightGBM comme candidat dans `app/models/candidates.py`, comparer en CPCV sur EURUSD H4 et ETHUSD H1. Si gain > 0.3 Sharpe, conserver.

### 1.2 Calibration probabiliste

| Méthode | Statut | Pourquoi |
|---|---|---|
| **Platt scaling** | ⚠️ F14 partiellement fixé (val 2023 dispo) | Calibre les probas vers de "vraies" probabilités fréquentistes. Critique pour le seuil méta-labeling. |
| **Isotonic regression** | ⚠️ Partiel | Plus flexible que Platt, monotone. |
| **Beta calibration** | ❌ | Variante moderne, performe mieux sur petits datasets. |

**Action** : sur la validation 2023, comparer 3 calibrations × 3 modèles → choisir celle qui maximise le **Sharpe net après seuil** sur 2023.

### 1.3 Stratégies d'apprentissage spécifiques au déséquilibre

Le winrate sur signaux Donchian primaires est ~40-50 %, soit binaire ~équilibré. Mais sur certaines stratégies (RsiContrarian, Bollinger fades), le WR peut être 60-70 % → déséquilibre.

| Technique | Statut | Pourquoi tester |
|---|---|---|
| **Class weighting** | ⚠️ Partiel (`class_weight=balanced` par défaut sklearn) | Compensation simple. |
| **SMOTE / ADASYN** | ❌ | Sur-échantillonnage synthétique de la classe minoritaire. Sur séries temporelles, à utiliser avec précaution (look-ahead potentiel). |
| **Focal loss** | ❌ | Fonction de perte avec hard-example mining. Adaptée déséquilibre extrême. |
| **Cost-sensitive learning** | ❌ | Poids différentiel basé sur le PnL réel par trade (pas juste la classe binaire). **Très adapté** au trading car winners et losers ont des amplitudes différentes. |

### 1.4 Approches dynamiques

| Approche | Statut | Description |
|---|---|---|
| **Online learning** (passive-aggressive, SGD partial_fit) | ❌ | Pour walk-forward continu : pas de re-entraînement complet à chaque pas. |
| **Hidden Markov Model (régime)** | ❌ | Inférence d'état caché (trend/range/vol-high), conditionner les prédictions. |
| **Mixture of Experts** | ❌ | N modèles spécialisés par régime, gating par classifieur séparé. |
| **Meta-learning (model selection)** | ❌ | Apprendre quel modèle utiliser selon le régime courant. |

### 1.5 Recommandations ML

Priorité (effort croissant) :
1. **LightGBM en alternative HGBM** (1 j) — gain marginal mais quasi-gratuit.
2. **Calibration isotonique systématique sur val 2023** (1 j) — déjà presque en place.
3. **Cost-sensitive learning** (2-3 j) — poids ∝ |PnL_attendu| par trade.
4. **HMM régime detection + stratégie conditionnée** (5-7 j) — H09 du roadmap original.

---

## 2. Features — données externes manquantes

### 2.1 Features actuelles (résumé du superset)

~70 colonnes réparties en 10 catégories :
- Trend (12), Momentum (6), Oscillators (5), Volatility (4), Price Action (10).
- Statistical Rolling (10), Market Regime (7).
- Economic (9 events), Sessions (8 features).
- Cross-asset (3 : USDCHF, XAUUSD, BTCUSD returns 5-bar).

**Type dominant** : indicateurs techniques classiques sur le prix de l'actif lui-même.

### 2.2 Features de classes absentes

#### 2.2.1 Macro structurelle (gratuit, sous-exploité)

| Feature | Source | Pertinence |
|---|---|---|
| **DXY index** (Dollar Index) | FRED `DTWEXBGS` ou yfinance `DX-Y.NYB` | Direction USD agrégée. Filtre fondamental pour tout forex. |
| **Yield curve slope** (2y-10y, 3m-10y) | FRED `T10Y2Y`, `T10Y3M` | Recession indicator. Slope < 0 → 70 % proba récession dans 12 mois (Estrella 1991). |
| **VIX** | FRED ou yfinance `^VIX` | Régime risk-on/off. VIX > 30 → forex risk-off (long JPY/CHF). |
| **VVIX, SKEW** | yfinance, CBOE | Régime second ordre. Crash-risk indicator. |
| **TED spread** | FRED | Stress crédit bancaire. |
| **High-Yield credit spread** | FRED `BAMLH0A0HYM2` | Stress crédit corporate. |
| **DGS10 (10-yr yield)** | FRED | Niveau absolu taux US. |
| **CPI YoY, PPI YoY** | FRED `CPIAUCSL`, `PPIACO` | Inflation regime. |
| **Unemployment rate** | FRED `UNRATE` | Cycle économique. |

**Toutes gratuites via [FRED API](https://fred.stlouisfed.org/docs/api/fred/)** (clé API gratuite, illimité). Pas d'excuse pour les ignorer.

#### 2.2.2 Sentiment / positioning (publique)

| Feature | Source | Pertinence |
|---|---|---|
| **COT report** (CFTC, hebdo) | [cftc.gov](https://www.cftc.gov/MarketReports/CommitmentsofTraders) | Position nette spéculateurs vs hedgers. Niveau extrême = signal contraire. |
| **AAII Sentiment Survey** | aaii.com | Bull/bear ratio retail. Niveau extrême = signal contraire. |
| **Put/Call ratio** | CBOE | Fear gauge actions. |
| **Crypto Fear & Greed Index** | alternative.me API | Sentiment crypto agrégé. |
| **Bitcoin Funding rates** | Binance API public | Long/short skew dérivés crypto. |
| **Open Interest** | Future data, parfois exposé via Polygon | Total positions ouvertes. |

#### 2.2.3 Inter-market spreads

| Feature | Formule |
|---|---|
| **Gold/Silver ratio** | XAUUSD / XAGUSD — niveau extrême signale régime |
| **Brent-WTI spread** | UKOIL - USOIL — divergence supply/demand |
| **ES-NQ spread (tech vs broad)** | (US500 - US100) / US500 — risk-on/off intra-actions |
| **SOX (semi vs SP500)** | ratio semi-conducteurs / SP500 — momentum tech |
| **Copper/Gold ratio** | Pro-cyclical indicator (Druckenmiller) |
| **Yen carry** | rate_USD - rate_JPY (proxy via différentiels yields) |

#### 2.2.4 News sentiment (gratuit avec effort)

- [NewsAPI.org](https://newsapi.org/) : gratuit 100 req/jour. Récupère headlines.
- Scoring sentiment via dictionnaire VADER, FinBERT, ou simple polarity TextBlob.
- Agrégation : daily news count, daily mean sentiment, count of "rate hike" / "rate cut" mentions.

**Effort** : 3-5 j pour MVP. Effet attendu : modeste mais directionnel sur Forex/indices.

#### 2.2.5 Time series sophistiquées

| Feature | Description |
|---|---|
| **Hurst exponent** (rolling) | Mesure de persistance des retours. H > 0.5 → trendy, H < 0.5 → mean-reverting. |
| **Autocorrelation lag-k** | Persistance directe. Lag 1, 5, 22 sur returns. |
| **Volatility clustering (GARCH proxy)** | rolling std ratio short/long. |
| **Skewness rolling** | Asymétrie returns sur 60 jours. |
| **Kurtosis rolling** | Tail risk. |
| **Distance to N-day high/low** (Donchian-like comme **feature**, pas signal) | Position relative dans le range. |
| **Ranks cross-sectional** | Sur univers multi-asset : "Cette barre, EURUSD est-il dans le top quartile de momentum du forex ?" |

### 2.3 Recommandation features

Priorité :
1. **DXY + VIX + Yield curve** (1-2 j) — gratuit via FRED, gain potentiel massif sur forex/indices.
2. **COT report** (2-3 j) — scrap CFTC, hebdo. Pertinent pour majors forex.
3. **Inter-market spreads** (1 j) — features dérivées, déjà-utilise nos data.
4. **Hurst + autocorrelation rolling** (1 j) — pure feature engineering, zéro data nouvelle.

---

## 3. Validation et méthodologie

### 3.1 Walk-forward roulant continu (F13)

**Statut** : F13 marqué OUVERT par audit v5.

**Constat** : actuellement, split unique train ≤ 2022 / val 2023 / test ≥ 2024 (figé Constitution §3). C'est conservateur mais **élimine la possibilité d'apprendre du régime récent**.

**Proposition Pardo (1992)** :
- Train initial : 2010-2018 (8 ans).
- Re-entraînement annuel ou semestriel sur fenêtre glissante (10 ans rolling).
- Test = chaque période suivant le retrain (1 an forward).
- Métriques : 12 tests indépendants → écart-type Sharpe inter-période → mesure de stabilité.

**Variantes** :
- Expanding window (jamais ne réduit la taille du train).
- Rolling window (taille constante).
- Anchored walk-forward (avec point d'ancrage à 2010).

**Effort** : 3-5 j (le module `app/pipelines/walk_forward_rolling.py` existe — testé en H_new2 et abandonné car NO-GO. À ressusciter avec re-entraînement plus fréquent).

**Pourquoi prometteur** : si l'edge existe mais varie selon régime (USD bull 2014-2016, USD bear 2017-2018, COVID 2020-2021, Fed cycle 2022-2024), un walk-forward fréquent **capture les changements** alors que le split fixe ne le peut pas.

### 3.2 CPCV multi-asset

Actuellement CPCV intra-asset (folds temporels sur un seul actif). Lopez de Prado (2018) propose CPCV multi-asset avec **purging cross-asset** :
- Si on apprend conjointement sur 5 actifs, leur cointégration crée un look-ahead inter-asset.
- Solution : purger les barres adjacentes dans le temps **sur tous les actifs**.

**Effort** : 5-7 j. Pertinent uniquement si on bascule sur multi-asset learning (cross-sectional momentum, pairs).

### 3.3 Stress tests régimes

Découper le test 2024-2026 par régime :
- **Trend USD bull** (2024-Q1) : Sharpe par sleeve.
- **Range USD** (2024-Q3) : Sharpe.
- **Trend USD bear** (2025-Q4) : Sharpe.
- **Vol high** (selon VIX > 25) : Sharpe.
- **Vol low** (VIX < 15) : Sharpe.

Si Sharpe < 0 dans plus de 50 % des sous-périodes → modèle pas robuste.

**Effort** : 2-3 j (script analytique, pas de nouveau modèle).

### 3.4 Robustesse aux coûts

Pas de sensitivity analysis dans le projet. À ajouter :
- Test "coût × 1.5" sur tous les actifs.
- Test "slippage × 2".
- Test "spread = max XTB observed (worst case)".

Si Sharpe passe sous 0 avec coût × 1.5 → fragile.

**Effort** : 1 j (paramétrer `AssetConfig` × 1.5).

### 3.5 Stress test événements

Périodes critiques à tester isolément :
| Période | Événement | Pertinence |
|---|---|---|
| 2010-05-06 | Flash Crash | Test risk control |
| 2015-01-15 | SNB removes EURCHF floor | EURCHF -25 % en 1 min |
| 2016-06-24 | Brexit | GBPUSD -10 % overnight |
| 2020-03 | COVID | Tous actifs -30 % |
| 2022-Q1 | Russia/Ukraine | Energies + 50 % |
| 2022 crypto winter | BTC -80 % | Crypto only |
| 2025-Q4 | Fed cycle reversal | À confirmer |

Si la stratégie blow-up sur ≥ 1 de ces périodes → risk control insuffisant.

**Effort** : 2 j.

---

## 4. Position sizing — au-delà du risque fixe 2 %

### 4.1 Constat

Sizing actuel = `compute_position_size(risk_eur=200, sl_pips, pip_value_eur)` → quantité fixe 2 % du capital.

Variantes non testées :

### 4.2 Kelly fractionnaire

```
f* = (p × R - q) / R   où p = WR, q = 1-p, R = TP/SL
```

Kelly pur = très volatile. **Kelly/4** ou **Kelly/2** est l'usage standard (Thorp 1969).

Effet : sizing **augmente** sur les setups à WR élevé / R élevé, **réduit** sur les marginaux. Améliore le geometric return.

**Effort** : 1 j (formule simple + tests).

### 4.3 Volatility targeting

```
lots = (capital × target_vol_annual) / (atr_pips × pip_value × √252)
```

Effet : sizing **réduit** en régime volatil, **augmente** en régime calme. **Stabilise le Sharpe** (l'objectif est un Sharpe = constant en fonction du régime).

Plan v5 (axe A3) mentionne déjà cette idée mais elle n'a pas été implémentée.

**Effort** : 1-2 j.

### 4.4 Drawdown-control sizing

```
sizing_multiplier = 1 - (current_drawdown / max_acceptable_dd)
```

Effet : sizing **réduit** après pertes, **rétablit** après récupération. Évite ruin lors de séquences de pertes.

**Effort** : 1 j.

### 4.5 Risk parity multi-sleeve

Actuellement equal-weight. Risk parity = chaque sleeve contribue **également au risque** (pas au capital) :
```
weight_i = (1 / vol_i) / sum(1 / vol_j)
```

Plus mature : **HRP (Hierarchical Risk Parity, Lopez de Prado 2016)** clustering + risk parity intra-clusters.

**Effort** : 2-3 j (lib `hrp` ou scratch).

### 4.6 Recommandation sizing

Priorité (effort × impact) :
1. **Volatility targeting** (A3 plan v5) — 1-2 j, gain probable +0.2 Sharpe.
2. **Drawdown control** — 1 j, gain en stabilité.
3. **Risk parity portfolio** — 2 j, pertinent quand ≥ 2 sleeves GO.

---

## 5. Détection de régime — H09 jamais implémenté

### 5.1 Concept

Détecter automatiquement l'état du marché :
- **Trending** : ADX > 25, |slope_sma_50| élevé, autocorrelation returns > 0.
- **Range** : ADX < 20, prix oscille ±1 σ autour moyenne.
- **Vol high** : ATR_pct > percentile 80.
- **Vol low** : ATR_pct < percentile 20.

→ Activer la **stratégie adaptée** :
- Trending → Donchian / TsMomentum.
- Range → BollingerBands fade / RsiContrarian.
- Vol high → reduce sizing ou no-trade.
- Vol low → ORB / NR4 breakout.

### 5.2 Méthodes

| Méthode | Avantages | Inconvénients |
|---|---|---|
| **Règles déterministes** (ADX > 25) | Simple, interprétable | Seuils arbitraires, transitions abruptes |
| **Hidden Markov Model (HMM)** | Probabiliste, transitions douces | Calibration sur train, risque overfitting |
| **Clustering K-means sur features** | Pas de pré-supposition labels | Choix N clusters arbitraire |
| **Markov-switching GARCH** (Hamilton 1989) | Standard académique | Lourd, lent |
| **Change-point detection** (ruptures) | Détecte cassures structurelles | Latency : détecte après le changement |

### 5.3 Recommandation

MVP avec **règles déterministes** :
- Train sur 2010-2022 : étiqueter chaque barre selon (ADX_14, ATR_pct_14) → labels {0=range_low_vol, 1=range_high_vol, 2=trend_low_vol, 3=trend_high_vol}.
- Filtrer signaux primaires par label :
  - Donchian → label ∈ {2, 3} (trend).
  - Bollinger fade → label ∈ {0, 1} (range).
- Comparer Sharpe filtré vs non-filtré.

Si MVP montre gain ≥ 0.3 Sharpe → passer à HMM.

**Effort** : 3-5 j MVP, 7-10 j version HMM.

---

## 6. Tableau récap angles morts méthodologiques

| Catégorie | Item | Effort | Probabilité gain |
|---|---|---|---|
| **ML** | LightGBM comparison | 1 j | ★ marginal |
| ML | Cost-sensitive learning | 2-3 j | ★★ modéré |
| ML | Calibration isotonique systématique | 1 j | ★★ modéré |
| ML | HMM regime states | 5-7 j | ★★ modéré-élevé |
| **Features** | DXY + VIX + Yield curve | 1-2 j | ★★★ élevé |
| Features | COT report | 2-3 j | ★★ modéré |
| Features | Inter-market spreads | 1 j | ★★ modéré |
| Features | News sentiment NLP | 3-5 j | ★ spéculatif |
| **Validation** | Walk-forward roulant continu (F13) | 3-5 j | ★★ modéré |
| Validation | Stress test régimes | 2-3 j | ★★ identifie fragilité |
| Validation | Robustesse coûts | 1 j | ★★★ critique (identifie sur-fit aux coûts) |
| **Sizing** | Volatility targeting | 1-2 j | ★★ stabilise Sharpe |
| Sizing | Drawdown control | 1 j | ★ marginal mais protecteur |
| Sizing | Risk parity / HRP | 2-3 j | ★★ pertinent multi-sleeve |
| **Régime** | Régime detection (règles) | 3-5 j | ★★★ élevé (dispatch stratégies) |
| Régime | HMM regime states | 5-7 j | ★★ modéré |

★★★ = très élevé, ★★ = modéré, ★ = spéculatif.

---

## 7. Conclusion méthodologique

Le projet a une **base méthodologique solide** (CPCV, DSR, bootstrap, walk-forward sur sleeve unique, look-ahead validation, snooping guard). C'est ce qui a permis l'audit v5 de détecter les bugs honnêtement.

Mais il manque **3 outils centraux** :
1. **Régime detection + dispatch stratégies** (H09 du roadmap original, jamais implémenté).
2. **Features macro externes** (DXY/VIX/yield curve) — gratuit, sous-exploité.
3. **Walk-forward roulant** (F13) — abandonné prématurément en H_new2.

Ces 3 ajouts cumulés (~10-15 j) transforment radicalement la **capacité d'apprentissage** du pipeline. Sans eux, on cherche un edge constant à travers tous les régimes — un constraint dur que peu de stratégies satisfont.
