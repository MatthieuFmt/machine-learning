# Audit v6 — Angles morts : Stratégies

**Périmètre** : familles de stratégies absentes du repo, variantes TP/SL/exits non testées, justifications théoriques.

---

## 1. Inventaire — ce qui existe vs ce qui manque

### 1.1 Stratégies présentes dans `app/strategies/`

| Stratégie | Type | Testée |
|---|---|---|
| `donchian.py` | Trend-following breakout N-bars | ✅ Approfondi (et disqualifié) |
| `mean_reversion.py` (RSI+BB) | Mean-reversion oscillator | ✅ EURUSD H4 |
| `bollinger.py` | Mean-rev sur bandes BB | ✅ Screening D1/H4 |
| `chandelier.py` | Trailing stop ATR | ⚠️ Bug test (à fixer) |
| `dual_ma.py` | Trend-following SMA crossover | ✅ Screening |
| `keltner.py` | Trend / mean-rev sur Keltner channels | ✅ Screening |
| `parabolic.py` | Trailing stop Parabolic SAR | ✅ Screening |
| `rsi_contrarian.py` | Mean-rev RSI extrême | ✅ Screening |
| `sma_crossover.py` | Crossover lent/rapide | ✅ Screening (1 GO faiblement) |
| `ts_momentum.py` | Time series momentum (Hurst-like) | ✅ Screening |

**Verdict global** : toutes appartiennent à 2 familles — **trend-following** et **mean-reversion oscillator**. C'est ~30 % de l'espace de stratégies retail documentées.

### 1.2 Familles totalement absentes

| Famille | Représentants types | Référence académique |
|---|---|---|
| **Opening Range Breakout (ORB)** | Crabel ORB, Williams ORB | Crabel 1990 |
| **Asian Range Breakout** | Tokyo range → London breakout | folklore + Lo 2017 |
| **Volatility expansion (NR4/NR7)** | Narrow range setup | Crabel 1990 |
| **Calendar effects** | Pre-FOMC drift, turn-of-month | Lucca & Moench 2015 (JF) |
| **Carry trade** | Long high-yielder / short low-yielder | Fama 1984, Lustig & Verdelhan 2007 |
| **Pairs trading (cointégration)** | EURUSD-GBPUSD z-score | Engle-Granger 1987, Gatev 2006 |
| **Statistical arbitrage** | Multi-asset z-score | Avellaneda & Lee 2010 |
| **Cross-sectional momentum** | Top N / bottom N rebalance | Jegadeesh & Titman 1993 |
| **Volatility risk premium** | Short vol via straddle proxies | Bollerslev 2009 (non-CFD direct) |
| **Cross-asset trend (TSMOM Moskowitz)** | Univers 50+ futures, signed momentum | Moskowitz, Ooi, Pedersen 2012 |
| **Earnings momentum (PEAD)** | Post-earnings drift sur stocks | Bernard & Thomas 1989 |
| **HMM regime-switching strategies** | Stratégie différente par état caché | Hamilton 1989 |

---

## 2. Stratégies prioritaires à implémenter

### 2.1 Opening Range Breakout (ORB) — priorité 1

**Concept** :
- À l'ouverture d'une session (ex: 14:30 UTC pour US cash), capturer le range des N premières minutes (ex: 30 min ou 60 min).
- Breakout du high ou low du range → entrée dans la direction.
- TP = X × range_size, SL = range_size (ou MAE adapté).
- Sortie au close du jour si pas de TP/SL touché (time-stop).

**Variantes** :
- ORB sur US30/US500/US100 indices D1 → H1 (range 14:30-15:30 UTC, breakout 15:30+).
- ORB sur Forex EURUSD/GBPUSD H1 (range London open 08:00-09:00 UTC).
- ORB intraday M15 (range 30 min, breakout 30-180 min après).

**Pourquoi prometteur** :
- Pattern le plus documenté pour intraday indices retail (livre Crabel 1990, "Day Trading with Short Term Price Patterns").
- Capture le "directional break" qui suit l'accumulation d'ordres pré-ouverture.
- Profil de payoff asymétrique : SL serré, TP large.

**Effort** : 2-3 j (stratégie simple, exige H1 ou M15 propre = pré-requis Phase F).

### 2.2 Asian Range Breakout — priorité 1

**Concept** :
- Tokyo session = faible volatilité (00:00-08:00 UTC).
- Mesurer le range Tokyo.
- À l'ouverture Londres (08:00 UTC), breakout du high ou low du range Tokyo.
- TP = 1× range, SL = 0.5× range.

**Pourquoi prometteur** :
- Folklore forex bien établi.
- Confirme empiriquement (Lo & MacKinlay 1999) : la séance asiatique a une volatilité significativement plus faible que Londres → range Tokyo = "compression" → break Londres = "expansion".
- Spécifique au forex (et donc différent de ce qui marche sur indices).

**Effort** : 2-3 j (besoin H1 propre + tag session déjà dispo dans `app/features/` regime/sessions).

### 2.3 Calendar effects — priorité 1 (low effort)

#### 2.3.1 Pre-FOMC drift

Lucca & Moench (2015, *Journal of Finance*, "The Pre-FOMC Announcement Drift") :
> "Since 1994, 80 % of the equity premium on US stocks has been earned over the 24 hours preceding scheduled FOMC announcements."

**Stratégie** :
- 24h avant un FOMC : long S&P 500 (US500) ou Nasdaq (US100).
- Close à l'announcement (14:00 EST = 18:00 UTC).
- ~8 FOMC/an → 8 trades/an. Petit volume mais robuste.

**Implémentation** : utiliser `app/features/calendar.py` ou `app/features/economic.py` (déjà chargé), filtrer signaux long uniquement dans la fenêtre [-24h, -1h] pré-FOMC.

**Effort** : 1-2 j (le calendrier économique est déjà chargé !).

#### 2.3.2 Turn-of-month effect

Ariel (1987), Lakonishok & Smidt (1988) :
> "S&P 500 returns concentrate on the last trading day of month + first 3 trading days."

**Stratégie** :
- Long US500/US100 du jour-2 fin de mois au jour+2 début de mois.
- 12 cycles/an × 5 jours = 60 jours de position/an.

**Effort** : 1 j (filtre date simple).

#### 2.3.3 NFP pre/post

NFP (Non-Farm Payrolls) = 1er vendredi du mois, 13:30 UTC. Volatilité énorme sur USD pairs.

**Stratégie A — pre-NFP straddle proxy** : entrer long ou short selon trend 3h avant NFP, SL serré, take profit sur le spike (TP/SL = 3:1).

**Stratégie B — post-NFP mean-reversion** : entrer contre le spike 1h après NFP si déplacement > 1.5 ATR.

**Effort** : 2 j.

### 2.4 Pairs trading EURUSD-GBPUSD — priorité 2

**Concept** :
- Calculer le spread = EURUSD - β × GBPUSD (β estimé par OLS sur train).
- Z-score = (spread - μ) / σ rolling 60 jours.
- Quand |z| > 2 → entrer mean-reversion : long EUR, short GBP si z < -2 (et inversement).
- Sortie quand |z| < 0.5.

**Variantes** :
- gold/silver (XAUUSD vs XAGUSD).
- Brent/WTI (UKOIL vs USOIL).
- ES/NQ (US500 vs US100).
- BTC/ETH si cointégrés (à vérifier).

**Pourquoi prometteur** :
- Mathématiquement orthogonal aux trend-followers (corrélation ≈ 0).
- Ajouté dans portfolio : pure diversification (Sharpe combiné > Sharpe individuel).
- Bien documenté (Gatev et al. 2006, Avellaneda-Lee 2010).

**Effort** : 4-5 j (besoin code cointégration via `statsmodels.tsa.stattools.coint`, test Engle-Granger ou Johansen).

### 2.5 Carry trade systematic — priorité 2

**Concept** :
- Univers de 6-10 forex pairs.
- Chaque mois, ranking par swap rate (taux de change overnight broker).
- Long top 3 swap+ , short bottom 3 swap- .
- Rebalance mensuel.

**Référence** : "Common Risk Factors in Currency Markets" (Lustig, Roussanov, Verdelhan 2011).

**Effort** : 5-7 j (besoin du swap correctement modélisé — prérequis Phase F).

### 2.6 Cross-sectional momentum (style AQR) — priorité 3

**Concept** :
- Univers de N actifs (ex: 9 forex + 5 indices + 3 commodities + 5 crypto = 22).
- Chaque mois, calculer le return 12 mois (avec skip-1 month).
- Long top quintile / short bottom quintile.
- Rebalance mensuel ou trimestriel.

**Référence** : Asness, Moskowitz, Pedersen (2013, *Journal of Finance*, "Value and Momentum Everywhere").

**Effort** : 5-7 j (changement de paradigme — il faut un loop multi-asset, pas single-asset).

### 2.7 Volatility breakout (NR4/NR7) — priorité 3

**Concept** :
- NR4 = "Narrow Range 4" = barre dont le range est le plus petit des 4 dernières.
- Le lendemain, breakout du high/low de la NR4 = setup d'expansion.
- TP = 2× range moyen, SL = range NR4.

**Référence** : Crabel (1990).

**Effort** : 1-2 j (filtre simple sur indicateurs existants).

### 2.8 Earnings momentum (PEAD) — priorité 4 (stocks)

**Concept** :
- Après publication earnings (EPS surprise > X %), long le stock pour 60 jours.
- Performance documentée stable depuis 1968.

**Référence** : Bernard & Thomas (1989).

**Effort** : élevé — nécessite univers stocks + calendrier earnings (à scraper Polygon ou yfinance).

---

## 3. Variantes d'exits / TP-SL non testées

### 3.1 Constat — un seul ratio testé

Le screening D1 et H4 a testé :
- **TP/SL = 2:1** uniquement (TP = 2 × SL).
- **SL ∈ {0.5×ATR, 0.7×ATR, 1.0×ATR, 1.5×ATR}**.

**Tout le reste de l'espace exits est inexploré** :

### 3.2 Ratios TP/SL non testés

| Ratio | Breakeven WR | Stratégies adaptées |
|---|---|---|
| 1:1 (TP = SL) | 50 % | Mean-reversion (signal très précis), scalping |
| 1:2 (TP < SL) | ~67 % | Mean-reversion intense, high-WR strats |
| 1:3 (TP = SL/3) | ~75 % | Très high-WR rare (martingale-like, dangereux) |
| 3:1 (TP = 3×SL) | ~25 % | Trend-following avec edge directionnel fort (carry, breakout) |
| 5:1 (TP = 5×SL) | ~17 % | Long-tail breakouts (crypto, energies, NR4 expansion) |

**Hypothèse** : Donchian sur USDJPY (trend persistant) avec ratio **3:1** ou **5:1** + SL serré pourrait fonctionner alors que 2:1 a échoué.

### 3.3 Exits dynamiques

#### Trailing stops
- **Chandelier exit** : SL = high(N) - k × ATR(N), suivant le prix. La stratégie [`chandelier.py`](../app/strategies/chandelier.py) existe mais a un bug (test échoué). **À fixer en priorité** car c'est le trailing stop le plus utilisé.
- **Parabolic SAR** : SL = niveau SAR courant. Testé via [`parabolic.py`](../app/strategies/parabolic.py), mais comme stratégie d'entrée. À tester comme **exit only** (entrée différente + exit SAR).
- **N-bar high/low** : SL = low(N) sur les N dernières barres. Variante Donchian-stop.
- **Volatility stop** (Wilder) : SL recalculé chaque bar = close - k × ATR.

#### Time-stop pur (no TP/SL)
- Entrée → sortie après N bars (ex: 24h = 24 H1 bars), pas de stop, juste fermeture forcée.
- Forces : capture pur effet directionnel, élimine biais TP/SL.
- Faiblesses : exposition gros DD si gros mouvement contraire.
- Recommandé pour : pre-FOMC drift, NFP straddle, calendar effects.

#### Partial exits + breakeven move
- Position size 1 lot.
- À +1R (= TP1) : fermer 50 %.
- À +1R : déplacer SL au breakeven (élimine risque).
- À +2R (= TP2) : fermer tout.
- Variante : laisser courir avec trailing après TP2.

**Effort** : 3-5 j (refonte du simulateur stateful pour supporter partial fills).

#### Profit lock (breakeven)
- À +1R atteint, SL → entry price.
- Garantit zéro perte sur les trades qui décollent puis reviennent.
- Effet : transforme WR ~40 % en WR ~50 % au coût d'un peu d'expectancy.

**Effort** : 2 j (extension simulateur stateful).

### 3.4 Tableau récap — espace exits à explorer

```
Dimension 1 — ratio TP/SL          {0.5:1, 1:1, 2:1, 3:1, 5:1}                   = 5
Dimension 2 — SL en ATR units      {0.3, 0.5, 0.7, 1.0, 1.5, 2.0}                = 6
Dimension 3 — type de SL           {fixed, trailing ATR, trailing N-bar, SAR}    = 4
Dimension 4 — TP (si fixed)        {fixed atomic, trailing partial, time-stop}   = 3
────────────────────────────────────────────────────────────────────────────────
Combinaisons totales :                                                            360
Testées actuellement :                                                            4
Pourcentage couvert :                                                             1.1 %
```

98.9 % de l'espace exits est inexploré.

---

## 4. Réflexion sur l'erreur fondamentale du projet

### 4.1 Sur-spécialisation Donchian

Tout le pipeline ML/méta-labeling a été construit autour de **Donchian breakout** comme générateur de signal primaire. Après audit v5, Donchian est mort. Mais le pipeline ne peut pas facilement basculer sur ORB, pairs trading ou calendar — chaque famille demande son propre générateur de signal, ses propres features de contexte, ses propres exits.

**Recommandation** : restructurer `app/strategies/` pour qu'une nouvelle famille (ex: `OrbStrategy`, `PairsStrategy`, `CalendarStrategy`) puisse s'insérer dans le pipeline méta-labeling sans réécrire `scripts/run_phase_b_*.py`. Le `BasePipeline` (`app/pipelines/base.py`) est censé permettre ça mais n'a jamais été testé sur autre chose que Donchian.

### 4.2 Sur-spécialisation D1

D1 = 1 bar par jour = ~250 bars/an = ~3000 bars sur 12 ans. C'est **trop peu** pour distinguer un edge réel d'une chance après ajustement DSR.

H1 (24×250 = 6000/an, 72000 sur 12 ans) ou même M15 (24×4×250 = 24000/an) offrent **10× à 50×** plus d'échantillons. La signification statistique se gagne à coût de complexité acceptable.

**Recommandation** : changer le centre de gravité du projet de D1 → H1/H4 mixed. D1 reste utile pour les filtres de régime, mais les signaux primaires viennent du TF inférieur.

### 4.3 Sur-spécialisation single-asset

Le projet teste chaque actif **isolément**. Or beaucoup d'edges retail sont cross-asset :
- Pairs trading nécessite ≥ 2 actifs cointégrés.
- Stat-arb nécessite ≥ 5 actifs.
- Cross-sectional momentum nécessite ≥ 10 actifs.
- Risk-on/off (long JPY/CHF, short AUD/NZD) nécessite ≥ 4 pairs.

Le module `app/portfolio/` existe mais n'est utilisé que pour combiner des sleeves préfabriquées (mode single-sleeve fallback dans H_new4). Il faudrait un module `app/strategies/cross_asset/` qui ingère un univers et émet des signaux multi-asset simultanés.

---

## 5. Hiérarchie des paris stratégies (effort × probabilité succès)

```
Effort faible (≤ 3 j) :
  ★★★ Pre-FOMC drift (calendar)           — 8 trades/an, robuste
  ★★★ Turn-of-month                        — 12 cycles/an
  ★★  NR4/NR7 breakout                     — quotidien, simple
  ★★  Asian Range breakout forex H1        — quotidien, needs H1 propre
  ★★  ORB indices H1                       — quotidien, needs H1 propre

Effort moyen (3-7 j) :
  ★★★ Pairs trading EURUSD-GBPUSD          — orthogonal aux trend-followers
  ★★★ JPY pairs Donchian/TsMomentum        — needs data download
  ★★  ORB intraday M15 sur indices         — needs M15 download + tests
  ★★  Carry trade systematic               — needs swap modélisé
  ★   Partial exits + breakeven move       — refonte simulateur

Effort élevé (≥ 1 semaine) :
  ★★★ Cross-sectional momentum (AQR style) — change paradigme single→multi
  ★★  Régime detection HMM + strat-switch  — H09 jamais fait
  ★★  Walk-forward roulant continu         — F13 OUVERT
  ★   Earnings momentum (PEAD)             — needs univers stocks complet
```

★★★ = très prometteur, ★★ = prometteur, ★ = spéculatif.

---

## 6. Conclusion sur les stratégies

Le projet a creusé profondément **2 familles sur ~12 documentées dans la littérature** (Pardo, Chan "Algorithmic Trading", Lopez de Prado, Bouchaud, AQR papers).

Les familles absentes ne sont pas exotiques :
- **Calendar effects** : article académique central (Lucca-Moench 2015), implémentation 1 j.
- **ORB / Asian range** : standard intraday retail depuis 30 ans.
- **Pairs trading** : enseigné dans tous les cours de finance quantitative (Ernie Chan, Carver).

Une exploration de 3-4 de ces familles (priorité haute du tableau §5) à coût d'effort cumulé ~2 semaines a une probabilité raisonnable de révéler **au moins 1 edge marginal Sharpe ≥ 0.7**. L'alternative — conclure que l'edge n'existe pas — manque de fondement empirique sans avoir testé ces familles.
