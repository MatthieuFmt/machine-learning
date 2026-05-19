# Audit v6 — Angles morts : Données

**Périmètre** : univers d'actifs testés vs disponibles XTB, qualité des historiques, sources alternatives, coûts manquants.

---

## 1. Univers actuel — 9 actifs, < 15 % de XTB

### 1.1 Ce que le projet a aujourd'hui

| Catégorie | Actifs présents (`data/raw/`) | D1 | H4 | H1 | Couverture historique |
|---|---|---|---|---|---|
| Forex majors | EURUSD, GBPUSD, USDCHF | ✅ | ✅ | ✅ (tronqué) | 16 ans D1, ~11 ans H1 |
| Indices | US30 (USA30IDXUSD), US500, GER30 | ✅ | ✅ sauf GER30 | ✅ sauf GER30 | Variable |
| Métaux | XAUUSD, XAGUSD | ✅ | ✅ | ✅ | ~14 ans |
| Crypto | BTCUSD (depuis 2017), ETHUSD | ✅ | ✅ | ✅ | 8-9 ans BTC, moins ETH |
| Énergies | USOIL | ✅ | ❌ | ❌ | D1 seulement |
| Obligations | BUND | ❌ vide | ❌ | ❌ | Échec download 2026-05-14 |

**Total** : 11 fichiers D1, 8 fichiers H4, 8 fichiers H1 = 27 séries OHLCV.

### 1.2 Ce que XTB offre (recherche 2026-05)

D'après [xtb.com](https://www.xtb.com/int/instrument-specification) et reviews :
- **5400+ instruments** au total
- **57-70 paires forex** (majors, minors, exotics)
- **25 indices** globaux
- **24 commodities**
- **40 cryptomonnaies**
- **2541 stocks** CFD
- **1831 ETF cash**

Le projet a testé **0.5 % du catalogue forex** (3/70), **12 % des indices** (3/25), **8 % des commodities** (2/24), **5 % des cryptos** (2/40), **0 % des stocks**.

---

## 2. Actifs XTB à fort potentiel non testés (priorité)

### 2.1 Forex — pairs JPY (priorité 1)

| Pair | Pourquoi prometteur | Spread XTB typ. |
|---|---|---|
| **USDJPY** | Tendance carry-trade structurelle 2012-2024. Major le plus "trendy". | 1.4 pips |
| **EURJPY** | Cross JPY, volatilité ~1.3× USDJPY, tendances persistantes. | 1.8 pips |
| **GBPJPY** | "The widow-maker" — volatilité élevée, breakouts profonds. | 2.5 pips |
| **AUDJPY** | Risk-on/off proxy, corrélé S&P/AUD. Carry positif structurel. | 1.7 pips |
| **NZDJPY** | Plus haut carry positif, mais moins liquide. | 2.0 pips |
| **CHFJPY** | Cross JPY/CHF, faible corrélation avec autres pairs. | 2.0 pips |

**Hypothèse à tester** : Donchian D1 (20,20) ou TsMomentum_60 sur ces 6 pairs, sur train ≤ 2022 puis OOS 2024+. Coût ~2-3 j (script existe déjà, juste downloader + nouveaux configs).

### 2.2 Forex — autres majors et crosses (priorité 2)

| Pair | Hypothèse | Spread XTB typ. |
|---|---|---|
| **AUDUSD** | Commodity currency (corrélée or, fer, AAPL/risk). | 1.3 pips |
| **NZDUSD** | Idem AUDUSD mais plus mince. | 2.0 pips |
| **USDCAD** | Corrélée WTI/Brent → pair commodity-driven. | 1.8 pips |
| **EURGBP** | Range historique étroit → mean-reversion. | 1.6 pips |
| **EURCHF** | Avant 2015 SNB cap = range parfait, après = trending. Régime split. | 2.0 pips |
| **EURAUD** | Cross volatil, peu corrélé EURUSD. | 2.5 pips |

### 2.3 Forex — exotiques (priorité 3, mais EUR-base accessibles XTB)

| Pair | Hypothèse | Note |
|---|---|---|
| **USDTRY** | Tendance secular faiblesse TRY 2018+, volatilité ATR 5-10 % | Spread élevé, swap énorme |
| **USDMXN** | Carry trade positif structurel | Disponible XTB |
| **USDPLN** | Liquide chez XTB (broker polonais), tendance EUR/PLN | Cheap chez XTB |
| **USDZAR** | High volatility, corrélé or | Disponible |

⚠️ Les exotiques ont swap énorme — bien tester avec swap modélisé (voir §4).

### 2.4 Indices — actions globales (priorité 1)

| Index | Symbole XTB approx. | Pourquoi |
|---|---|---|
| **Nasdaq 100** | US100 | Plus volatile que S&P, breakouts tech historiques |
| **Nikkei 225** | JAP225 | Tendance secular 2012-2024 +250 % |
| **FTSE 100** | UK100 | Range-bound, mean-reversion plus marqué |
| **CAC 40** | FRA40 | Cousine GER30, peu corrélée US |
| **ASX 200** | AUS200 | Asie-Pacifique, peu corrélée US/EU |
| **Hang Seng** | HKComp ou similaire | Bear market secular 2021+, opportunité short |

### 2.5 Commodities — non testées (priorité 2)

| Commodity | Symbole | Pourquoi |
|---|---|---|
| **Brent** | UKOIL ou BRENT.CASH | Différent de WTI (déjà testé en D1) |
| **Natural Gas** | NATGAS | Volatilité extrême, breakouts ATR énormes (saisonnalité hiver) |
| **Copper** | COPPER | Indicateur macro chinois/industriel |
| **Coffee** | COFFEE | Tendance saisonnière (gel Brésil), faibles corrélations |
| **Corn / Wheat / Soybean** | CORN/WHEAT/SOYBEAN | Saisonnalité USDA reports |
| **Sugar / Cocoa / Cotton** | SUGAR/COCOA/COTTON | Soft commodities, tendances faim/climat |
| **Palladium** | PALLADIUM | Très volatil 2018-2022, peu corrélé autres |

### 2.6 Crypto — diversification

XTB offre 40 cryptos. Testées : 2 (BTC, ETH). À ajouter : **LTCUSD, XRPUSD, BCHUSD, SOLUSD, ADAUSD, DOTUSD, LINKUSD, AVAXUSD**. Toutes ont périodes trendy distinctes.

### 2.7 Stocks individuels — totalement vierge

2541 stocks CFD chez XTB. Le projet n'a testé **aucune action individuelle**. Pourtant :
- Stocks individuels ont mean-reversion intraday plus marqué que les indices (overnight gap → fill, intraday range → fade).
- Earnings drift (post-earnings announcement drift, PEAD) est un des plus vieux edges documentés (Bernard & Thomas 1989).
- Momentum cross-sectional (long top 20 % / short bottom 20 %) est l'edge le plus robuste de la littérature (Jegadeesh & Titman 1993).

C'est un univers entier qui mérite au minimum un test exploratoire sur top 30 mega-caps US (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, JPM, BAC, ...).

---

## 3. Qualité et couverture des données existantes

### 3.1 Constat sur EURUSD H1

```
wc -l data/raw/EURUSD/EURUSD_H1.csv
99999 lignes
```

**99 999 = limite par défaut export MT5** (10⁵). Cela signifie ~11 ans seulement de H1, alors que D1 couvre 16 ans (2010-2026). On a **perdu 5 ans** d'historique H1 simplement à cause de la limite d'export.

**Fix** : ré-télécharger via Dukascopy (gratuit, OHLC ou tick depuis 2003 sur EURUSD).

### 3.2 Constat sur les "data invalides H4/H1" mentionnées par l'utilisateur

Les fichiers existent physiquement. Les "gaps" sont probablement :
- **Weekends** (forex/indices ferment vendredi 22:00 UTC) — gap normal, géré par `load_asset()` via `is_normal_gap()`.
- **Holidays** US (Thanksgiving, Christmas) — gap normal.
- **Crypto** (BTC/ETH) n'a pas de weekend → si gap, c'est anormal.

À traiter au cas-par-cas. Sans accès aux data, je ne peux pas trancher — mais le pipeline a un `MAX_GAP_HOURS = {"D1": 7×24, "H4": 3×24, "H1": 2×24}` qui devrait rejeter les anomalies.

**Hypothèse à valider** : sont-ce des gaps de qualité (data manquante) ou des gaps légitimes (weekend forex) ?

### 3.3 Recommandation : ré-télécharger via Dukascopy

[Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/) offre **gratuitement** :
- **OHLC tick-by-tick → M1 → M5 → M15 → M30 → H1 → H4 → D1 → W1 → MN1**
- **Depuis 2003 sur EURUSD**, plus tard sur exotiques
- **1000+ instruments** : Forex, commodities, indices, CFDs, crypto, ETFs, stocks
- **CSV ou binary** via outil JForex, ou Python (`dukascopy-python`, `dukascopy-node`).

Bénéfices :
- Source unifiée, qualité institutionnelle.
- Tick data → permet vrais tests M1/M5 (jamais faits).
- Cohérence inter-actifs (même fuseau UTC, même définition pip).

Effort : 1-2 j pour télécharger 30 actifs × 7 timeframes via script Python.

### 3.4 Sources alternatives complémentaires

| Source | Forces | Faiblesses | Recommandé pour |
|---|---|---|---|
| **Dukascopy** | Tick + OHLC, 1000+, gratuit, API Python | Régionalisé Europe, pas de stocks US complets | Forex, commodities, indices liquides |
| **HistData.com** | Forex M1 gratuit, csv direct | Forex seulement | EURUSD/USDJPY/etc. M1 pour ORB |
| **Polygon.io** | Stocks US + crypto + forex, free tier 5 req/min | Pas commodities/indices intl | Stocks individuels |
| **Alpha Vantage** | API gratuite, daily OHLC | 25 req/jour gratuit, lent | Backup léger |
| **Yahoo Finance** (via yfinance) | Stocks/indices/crypto large | Données suspectes sur intraday | Backup uniquement |
| **CFTC COT** (cftc.gov) | Commitment of Traders hebdo gratuit | CSV brut, peu structuré | Feature COT |
| **FRED** (federalreserve) | Macro (yield curve, GDP, CPI) | Daily/weekly seulement | Features macro |

### 3.5 Recommandation prioritaire

1. **Phase F1** : script `scripts/download_dukascopy_full.py` pour 30 actifs × 4 TFs (H1, H4, D1, W1).
2. **Phase F2** : valider qualité (check gaps anormaux, prix négatifs, dupes index).
3. **Phase F3** : enregistrer dans `app/config/instruments.py` (AssetConfig) avec coûts XTB **validés** (voir §4).

---

## 4. Coûts manquants — 🔴 BUG STRUCTUREL

### 4.1 Constat : swap absent du simulateur

Dans [`app/backtest/deterministic.py:73-74`](../app/backtest/deterministic.py#L73-L74) :
```python
cost_per_side = commission_pips + slippage_pips  # entrée + sortie
cost_total = cost_per_side * 2                   # entrée ET sortie
```

`cost_total` est calculé une seule fois à l'entrée du trade, indépendamment de sa durée. Si le trade dure 5 nuits, **aucune charge overnight n'est ajoutée**.

**Conséquences** :
1. Tous les Sharpe Donchian D1 (trades typiquement 3-10 jours) **surestiment le résultat**.
2. Pour BTCUSD long position 1 semaine chez XTB : swap typique ~0.05 %/nuit = -0.35 %/semaine = -350 € sur 100k position. Sur 50 trades/an, c'est -17500 € de coûts manqués.
3. Pour USDJPY long : swap POSITIF ~+0.5 pip/nuit (carry favorable). **C'est un mini-edge gratuit** qui n'est jamais comptabilisé → Donchian long USDJPY pourrait être plus rentable que ce que le pipeline montre.
4. Pour exotiques (USDTRY, USDZAR) : swap énorme (5-15 pips/nuit) — comptabiliser ou abandonner.

### 4.2 Order de grandeur de l'impact

| Pair | Swap typ. long (pips/nuit) | Swap typ. short (pips/nuit) | Sur 5 nuits |
|---|---|---|---|
| EURUSD | -0.5 (carry défavorable) | +0.2 | -2.5 long / +1 short |
| USDJPY | +0.6 (carry favorable) | -0.8 | +3 long / -4 short |
| GBPUSD | -0.4 | +0.1 | -2 long / +0.5 short |
| AUDJPY | +1.5 (carry très positif) | -2.0 | +7.5 long / -10 short |
| XAUUSD | -0.10 USD/nuit | +0.05 USD | minime |
| BTCUSD | -0.05 %/nuit (≈ -25 USD) | +0.02 % | -125 USD long / +50 short |
| US30 | -0.5 USD/nuit | +0.2 USD | -2.5 long |

Sur Donchian D1 EURUSD avec TP=20/SL=10, swap -2.5 pips long = **12.5 % du TP** ou **25 % du SL**. C'est l'équivalent d'**augmenter le spread de 2.5 pips** sur chaque trade long.

### 4.3 F10 toujours OUVERT — coûts provisoires non validés

[`audit_v5_execution_status.md`](audit_v5_execution_status.md) ligne 24 : F10 marqué `🟡 OUVERT (à valider XTB démo)`.

Coûts actuels dans [`app/config/instruments.py`](../app/config/instruments.py:419-447) :
```
BTCUSD : spread 30 USD, slippage 30 USD → 60 USD round-trip
ETHUSD : spread 3 USD, slippage 3 USD → 6 USD round-trip
GBPUSD : spread 0.9 + slippage 0.2 = 1.1 pip
USDCHF : spread 1.0 + slippage 0.2 = 1.2 pip
```

À comparer aux specs réelles XTB (à confirmer en démo) :
- BTCUSD spread XTB Standard = 30-100 USD (variable). Slippage XTB = variable selon vol marché.
- ETHUSD spread XTB ≈ 2-5 USD.
- GBPUSD spread XTB Standard ≈ 2.2 pips (selon [bestbrokers.com](https://www.bestbrokers.com/reviews/xtb/spreads-fees-and-commissions/) — pas 0.9 !).
- USDCHF spread XTB ≈ 1.9 pips (pas 1.0).

🚨 **Si les spreads réels XTB sont 2× ceux configurés, tous les Sharpe forex sont sur-estimés de 5-10 %**.

### 4.4 Recommandation

**Bloqueur** avant toute exploration v6 :
1. Ouvrir compte démo XTB MT5.
2. Récupérer "Symbol Specifications" pour les 30 actifs cibles → renseigner dans un fichier `docs/xtb_specs_demo_<date>.csv`.
3. Ajouter `swap_long_pips_per_night` et `swap_short_pips_per_night` à `AssetConfig`.
4. Modifier `_simulate_stateful_core()` pour ajouter `nights_held * swap_pips_per_night` au `pips_net` final.
5. Modifier le simulateur déterministe idem.
6. Ajouter test unitaire `test_swap_overnight_impacts_pnl.py`.
7. Re-runner tous les Sharpe historiques pour mesurer l'impact (probable -10 à -30 %).

Effort total estimé : 1-2 j.

---

## 5. Tableau récap actifs à télécharger (XTB-tradables)

Pour le plan d'attaque, voici la liste cible :

```
Forex majors (déjà):   EURUSD, GBPUSD, USDCHF
Forex majors (manque): USDJPY, AUDUSD, NZDUSD, USDCAD                    (4)
Forex JPY crosses:     EURJPY, GBPJPY, AUDJPY, NZDJPY, CADJPY, CHFJPY    (6)
Forex EUR/GBP crosses: EURGBP, EURCHF, EURAUD, GBPAUD, GBPCAD            (5)
Forex exotiques:       USDPLN, USDTRY, USDMXN, USDZAR                    (4)
Indices (déjà):        US30, US500, GER30
Indices (manque):      US100 (Nasdaq), JAP225, UK100, FRA40, AUS200      (5)
Métaux (déjà):         XAUUSD, XAGUSD
Métaux (manque):       PALLADIUM, PLATINUM, COPPER                       (3)
Énergies (déjà):       USOIL (D1 only)
Énergies (manque):     UKOIL (Brent), NATGAS                             (2)
Soft commodities:      COFFEE, CORN, WHEAT, SOYBEAN, SUGAR, COCOA, COTTON (7)
Crypto (déjà):         BTCUSD, ETHUSD
Crypto (manque):       LTCUSD, XRPUSD, BCHUSD, SOLUSD, ADAUSD, DOTUSD    (6)
Stocks (manque tout):  AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA + 20 autres méga-caps (top 30)
                                                                          (30+)
─────────────────────────────────────────────────────────────────────────
TOTAL CIBLE Phase G/H :  ~70 nouveaux instruments
```

**Effort téléchargement** : 1 j (script Dukascopy).
**Effort intégration `app/config/instruments.py`** : 1-2 j (création AssetConfig par actif + validation pip_size/pip_value).

---

## 6. Conclusion sur les données

Le projet a travaillé sur **9 actifs / 5400 disponibles**. Le verdict "rien ne marche" basé sur 0.2 % de l'univers tradable est statistiquement faible.

Avec **70 instruments × 4 TF × 13 stratégies × 4 ratios TP/SL = 14560 backtests train** vs les ~900 backtests actuels (×16), la probabilité de trouver au moins 1 edge marginal augmente énormément — pas par chance pure (le DSR pénalise correctement), mais par exposition à des familles structurellement différentes (carry, mean-rev intraday, calendar, stat-arb).
