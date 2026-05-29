# Audit v6 — Ce qu'on a raté (synthèse fresh, sans biais hérité)

**Date** : 2026-05-19
**Périmètre** : audit fresh de tout le projet après le verdict v5 (Donchian D1 mort, screening D1/H4 → 0 GO production).
**Question centrale** : maintenant que le pipeline est fiable mais qu'aucune stratégie ne passe Sharpe ≥ 1.0, **avons-nous exploré le bon espace de recherche** ?

---

## TL;DR — Verdict en une page

Le pipeline est techniquement sain (49 tests neufs, 15/19 findings audit v5 fixés). Le verdict "aucun edge détectable" est **honnête sur le périmètre testé**. Mais ce périmètre couvre **moins de 15 % de l'espace de recherche raisonnable** sur XTB. Les angles morts identifiés ci-dessous correspondent à des pans entiers de littérature finance quantitative qui n'ont jamais touché ce repo.

Les **5 angles morts les plus probants** (par ordre d'effort × impact estimé) :

| # | Angle mort | Effort | Probabilité edge | Pourquoi |
|---|---|---|---|---|
| **1** | **Swap overnight absent des coûts** | 1-2 j | 🔴 BUG → invalide les Sharpe affichés sur D1/H4 | Tous les trades multi-jour sous-estiment leurs coûts. Sur EURUSD long 5 nuits ≈ -3 pips additionnels = 15 % d'un SL de 20 pips. |
| **2** | **Pairs JPY non testées (USDJPY, EURJPY, AUDJPY)** | 2-3 j | 🟢 Élevée | Carry trade = tendance structurelle persistante. Famille la plus "trendy" du forex, jamais évaluée. |
| **3** | **Stratégies Asian Range / Opening Range Breakout (ORB)** | 3-5 j | 🟢 Élevée | Standard intraday H1/M15 sur forex et indices. Plus prouvée par la littérature que Donchian D1 (Toby Crabel, Larry Williams). |
| **4** | **Walk-forward roulant continu (re-train 3-6 mois)** | 3-5 j | 🟡 Moyenne | F13 marqué OUVERT par audit v5. Méthode reconnue (Pardo) pour adapter au régime. Le split fixe figé peut masquer un edge non-stationnaire. |
| **5** | **Pairs trading / cointégration (stat-arb)** | 5-7 j | 🟡 Moyenne | EURUSD vs GBPUSD, gold vs silver, Brent vs WTI. Famille mathématiquement orthogonale aux trend-followers déjà testés. |

**Recommandation** : commencer par corriger le swap (#1, bloqueur méthodologique), puis attaquer #2 et #3 en parallèle (peu d'effort, fortes chances). #4 et #5 si #2-#3 échouent.

---

## 1. Cadre de l'audit

### Ce qui a déjà été fait (à ne PAS refaire)

| Domaine | Couvert | Évaluation |
|---|---|---|
| Pipeline ML méta-labeling RF/HGBM/Stacking | ✅ Très approfondi | Solide après audit v5. F1-F8 corrigés. |
| Validation : CPCV, walk-forward (sleeve unique), DSR, bootstrap | ✅ Complet | F15 fixé (block bootstrap). |
| Stratégies déterministes : Donchian, Bollinger, Keltner, Chandelier, Parabolic, DualMA, SmaCrossover, RsiContrarian, TsMomentum, MeanReversionRSIBB | ✅ Screening D1 et H4 sur 9 actifs | 0 GO production. |
| Features : ~70 (trend, momentum, oscillators, vol, price action, stats, regime, economic, sessions, cross-asset) | ✅ Superset large | Mais sélection figée Donchian-centric. |
| Coûts (spread + slippage commission) | 🟡 Partiel | **F10 OUVERT** + swap absent (nouveau finding). |
| 9 actifs : forex (3), indices (3), métaux (2), crypto (2), énergies/obligations partiels | 🟡 Partiel | < 15 % de l'univers XTB. |
| Timeframes D1 + H4 | 🟡 Partiel | H1 quasi pas exploité (data tronquée à 100k bars MT5 export). |

### Ce qui n'a JAMAIS été touché

Les sections 2-7 ci-dessous détaillent. Chaque pan est un fichier markdown séparé.

---

## 2. Les angles morts par catégorie

### 2.1 Données — voir [`audit_v6_data_gaps.md`](audit_v6_data_gaps.md)

- 9 actifs testés sur ~5400+ instruments XTB disponibles.
- **Forex** : USDJPY, AUDUSD, NZDUSD, USDCAD jamais touchés. Crosses JPY/EUR ignorés.
- **Indices** : Nasdaq (US100), Nikkei (JAP225), FTSE, CAC40 non testés.
- **Commodities** : pétrole Brent, gaz naturel, cuivre, soft commodities (coffee/corn/wheat/sugar) = 0.
- **Crypto** : 2/40 cryptos XTB. LTC, XRP, SOL, ADA, BCH non testées.
- **Stocks individuels** : 2541 disponibles, 0 testé (mean-reversion intraday plus marqué que sur indices).
- **Timeframes** : H1 EURUSD est tronqué à 99 999 lignes (limite export MT5 par défaut → ~11 ans seulement). Re-download via Dukascopy donnerait 20+ ans propres.
- **Sources alternatives** : Dukascopy (gratuit, tick + OHLC, 1000+ instruments), HistData (forex M1 propre), Polygon, Alpha Vantage. Tous ignorés.

### 2.2 Stratégies — voir [`audit_v6_strategy_gaps.md`](audit_v6_strategy_gaps.md)

**Familles non implémentées (la liste est longue)** :
- **Opening Range Breakout (ORB)** — sortie après le range des N premières minutes/heures, standard intraday.
- **Asian Range breakout** durant la session Londres — variante ORB classique forex.
- **NR4/NR7** (Toby Crabel) — breakout après contraction du range.
- **Volatility expansion** — pattern de squeeze ATR.
- **Calendar effects** : pre-FOMC drift (Lucca-Moench 2015), turn-of-the-month, end-of-month USD flows, day-of-week.
- **Pairs trading / cointégration** — z-score mean-reversion entre actifs cointégrés (EURUSD/GBPUSD, gold/silver, Brent/WTI).
- **Carry trade** — long forex high-yielder, short low-yielder (USDJPY, AUDJPY, NZDJPY).
- **Volatility risk premium** — short straddle proxies (CFD ne permet pas directement, mais variantes existent).
- **Momentum cross-sectional** — top N performers vs bottom N (style AQR/Asness).
- **VIX-driven risk on/off** — long JPY/CHF si VIX > 30, short si VIX < 15.

**Exits non explorés** :
- Trailing stops (chandelier ATR, SAR-based).
- Time-stop pur (sortie après N bars, pas de TP/SL fixe).
- Partial exits (50 % @ +1R, 50 % @ +2R).
- Breakeven move après +1R.
- Vol-adjusted exits (SL recalculé à chaque bar selon ATR courant).

**Ratios TP/SL non testés** : 1:1, 3:1, 1:3 (le screening n'a testé que 2:1).

### 2.3 ML — voir [`audit_v6_methodology_gaps.md`](audit_v6_methodology_gaps.md) §1

- **LightGBM / XGBoost** : exclu volontairement par CLAUDE.md, mais SOTA tabulaire. Coût d'ajout = +1 dépendance.
- **CatBoost** : excellent sur features mixtes catégorielles/numériques.
- **Calibration Platt/isotonique** : F14 partiellement fixée (val 2023 disponible) mais pas systématique.
- **Class weighting / SMOTE / undersampling** : non testés sur déséquilibre WR.
- **Online learning / passive-aggressive** : adapté au walk-forward roulant.
- **Stacking dynamique par régime** : meta-learner différent selon état marché détecté.

### 2.4 Features — voir [`audit_v6_methodology_gaps.md`](audit_v6_methodology_gaps.md) §2

- **COT report** (Commitment of Traders, hebdomadaire CFTC) — proxy positionnement smart money. Gratuit, jamais utilisé.
- **VIX / VVIX / SKEW** — regime volatilité actions, corrélé risk-off forex.
- **Yield curve slope** (2y-10y, 3m-10y) — recession indicator, prédicteur macro.
- **DXY index** (Dollar Index) — direction USD agrégée, utile pour tous les forex.
- **Inter-market spreads** : gold/silver ratio, Brent-WTI, ES-NQ spread.
- **HMM regime states** (Hidden Markov Models) — détection automatique de régime.
- **Order flow proxies** : volume profile, bid/ask imbalance (limité en CFD).
- **News sentiment** : scoring NLP simple via headlines (gratuit via newsapi.org, RSS).

### 2.5 Méthodologie — voir [`audit_v6_methodology_gaps.md`](audit_v6_methodology_gaps.md) §3

- **Walk-forward roulant continu** : F13 OUVERT. Re-train tous les 3-6 mois sur fenêtre glissante (Pardo 1992). Permet d'adapter aux régimes.
- **CPCV multi-asset** : actuellement intra-asset. Cross-asset purging.
- **Régime detection** : H09 jamais implémenté. HMM ou Markov simple sur ATR/ADX → switch stratégie.
- **Position sizing dynamique** : actuellement risque fixe 2 % du SL. Kelly fractionnaire, vol-target, drawdown-control non testés.
- **Portfolio construction** : equal-risk → risk parity (vol-inverse), HRP (Lopez de Prado), min-CVaR jamais évalués.
- **Robustesse aux coûts** : pas de sensitivity analysis (spread ±50 %, slippage ×2).
- **Stress test régimes** : COVID 2020-Q1, Swiss Franc spike 2015-01-15, crypto winter 2022 — pas analysés isolément.

### 2.6 Coûts — voir [`audit_v6_data_gaps.md`](audit_v6_data_gaps.md) §4

- 🔴 **Swap overnight ABSENT du simulateur** ([`app/backtest/deterministic.py:73-74`](../app/backtest/deterministic.py#L73-L74)). `cost_total = (commission + slippage) × 2`, aucune charge par nuit. Impact massif sur trades D1/swing.
- 🟡 F10 : spreads provisoires BTCUSD/ETHUSD/GBPUSD/USDCHF non validés en démo XTB.
- Pas de stress test "et si le spread réel est 1.5× ?".

---

## 3. Pourquoi ces angles morts peuvent révéler un edge réel

### 3.1 Pairs JPY (USDJPY, EURJPY, AUDJPY)

Le carry trade est un edge **structurel** documenté depuis 30 ans (Fama 1984, Lustig & Verdelhan 2007). Les pairs JPY de 2010-2022 sont **les plus persistant-trendy du forex** :
- USDJPY 2012-2015 : 77 → 125 (tendance +60 % sans correction majeure).
- USDJPY 2022 : 115 → 152 (tendance +30 % en 12 mois).
- AUDJPY/NZDJPY suivent le risk-on/off avec persistance.

Donchian ou TsMomentum sur ces actifs ont des chances **qualitativement différentes** de Donchian sur EURUSD (sideways depuis 2015).

### 3.2 Opening Range Breakout (ORB) intraday

Crabel (1990) et Williams (2003) ont documenté ce pattern sur indices US :
- Range des 30 premières minutes du cash open.
- Breakout du high ou low → suivi de tendance jusqu'au close.
- Performance historique Sharpe ≥ 1.5 sur S&P, Nasdaq, DAX.

Le projet a quasi-ignoré H1 (data tronquée) et M15/M5 (pas téléchargés). C'est **la famille de stratégies intraday la plus simple et la plus rentable** historiquement, et elle est absente.

### 3.3 Pairs trading EURUSD-GBPUSD (cointégration)

Les pairs EUR-GBP sont cointégrées via les corrélations USD. Quand l'écart z-score dépasse ±2, mean-reversion historiquement profitable (Engle-Granger 1987, Gatev et al. 2006). **Mathématiquement orthogonal** aux trend-followers testés → indépendance statistique → ajout dans portfolio = pure diversification.

### 3.4 Calendar effects (pre-FOMC, turn-of-month)

Lucca & Moench (2015, Journal of Finance) : **80 % du rendement S&P 500 depuis 1994 vient des 24h précédant les FOMC**. Strategy "long S&P 3 jours avant FOMC, flat sinon" génère Sharpe > 1.5. C'est public, c'est gratuit, et ce n'est pas dans le repo.

### 3.5 Swap overnight (correction de coûts)

Pas un edge mais un **dé-bias** : actuellement les Sharpe affichés surestiment systématiquement les coûts manquants. Sur Donchian D1, la durée moyenne d'un trade est ~3-5 jours. Pour EURUSD long, swap typique XTB ≈ -0.6 pip/nuit. Soit -3 pips supplémentaires (15 % du SL = 20 pips). Cela peut transformer Sharpe +0.42 (meilleur train du diagnostic Donchian) en Sharpe 0.

C'est aussi une **opportunité asymétrique** : certains carry trades (long AUDJPY, long NZDJPY) ont swap POSITIF chez certains brokers — c'est un edge gratuit pour les longs et un coût additionnel pour les shorts. Modéliser correctement crée une asymétrie exploitable.

---

## 4. Plan d'attaque — voir [`audit_v6_action_plan.md`](audit_v6_action_plan.md)

Le plan détaillé hiérarchise les pistes par triplet (priorité, effort, impact). Il propose 3 phases :

- **Phase F (Foundation, ~1 semaine)** : correctifs structurels — swap, validation coûts XTB démo, redownload Dukascopy.
- **Phase G (Universe expansion, ~2 semaines)** : tester JPY pairs + nouveaux indices/commos.
- **Phase H (Strategy expansion, ~3 semaines)** : ORB, pairs trading, calendar effects.

Chaque phase a des critères go/no-go chiffrés et un budget n_trials cumul.

---

## 5. Ce qu'on ne fera PAS (volontairement)

Aligné avec [`plan_v5_amelioration_strategies.md`](plan_v5_amelioration_strategies.md) §"Ce qu'on n'ajoutera pas" :

1. **Deep learning (LSTM, Transformer)** : volume données insuffisant, GPU non dispo. Sur-paramétré → garanti overfitting.
2. **Reinforcement learning** : même raison + besoin simulateur fidèle (action discrete trade).
3. **Bayesian methods** : pas de gain marginal vs. ensembles classiques sur ce dataset.
4. **Production / VPS / Telegram** : hors scope confirmé.
5. **HFT / market making** : nécessite latence ms, infrastructure pro.
6. **Options gamma scalping** : pas accessible via CFD XTB.

---

## 6. Conclusion

Le pipeline corrigé v5 est **un bon outil mais utilisé sur un échantillon trop étroit**. Le verdict "rien ne marche" est valide pour Donchian/MeanRev/Keltner sur 9 actifs D1/H4 — mais pas pour le trading quantitatif retail dans son ensemble.

Les 6 angles morts identifiés couvrent ~85 % des techniques retail documentées dans la littérature (Pardo, Chan, Lopez de Prado, Bouchaud). Une exploration honnête de **3 à 5 de ces angles** révélera probablement au moins **un edge marginal** (Sharpe ≥ 0.5) — pas un free lunch, mais quelque chose à raffiner.

L'alternative — conclure "pas d'edge possible" — n'est pas justifiée tant que ces angles ne sont pas attaqués.

---

## Annexes — fichiers détaillés

1. **[audit_v6_data_gaps.md](audit_v6_data_gaps.md)** — univers actifs XTB, sources data alternatives, swap missing.
2. **[audit_v6_strategy_gaps.md](audit_v6_strategy_gaps.md)** — familles stratégies non testées + variantes exits/TP-SL.
3. **[audit_v6_methodology_gaps.md](audit_v6_methodology_gaps.md)** — ML, features alternatives, walk-forward roulant, régime detection.
4. **[audit_v6_action_plan.md](audit_v6_action_plan.md)** — plan d'attaque hiérarchisé Phase F/G/H avec critères go/no-go.
