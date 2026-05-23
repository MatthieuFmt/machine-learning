# Phase G — Bilan et bascule Phase H

**Date** : 2026-05-22
**Statut** : ❌ **NO-GO sur le seul candidat — pivot Phase H validé**
**n_trials cumulés consommés** : +1 (lecture OOS G2 ETHUSD)

---

## 1. Périmètre exécuté

### G-light : screening 13 stratégies × 9 actifs × 4 ratios × 3 TF

Total : **1404 backtests** sur train ≤ 2022-12-31, swap réaliste appliqué (F6).

| TF | Backtests | Skipped (gaps / signaux insuffisants) | Candidats |
|---|---:|---:|---:|
| D1 | 468 | 0 | 0 |
| H4 | 468 | 52 | 0 |
| H1 | 468 | 72 | **1** |
| **Total** | **1404** | 124 | **1** |

Critère candidat : Sharpe ≥ 0.5, WR ≥ 35%, n_trades ≥ 30 sur train.

Sources :
- [predictions/screen_strategies_train.json](../predictions/screen_strategies_train.json)
- [predictions/screen_strategies_train_h4.json](../predictions/screen_strategies_train_h4.json)
- [predictions/screen_strategies_train_h1.json](../predictions/screen_strategies_train_h1.json)

### G2 : test OOS unique sur le candidat survivant

**ETHUSD H1 — SmaCrossover(24, 120), SL=1.5×ATR_train, TP=2:1**

| Métrique | Train (≤ 2022-12-31) | OOS (2024-01 → 2026-05) | Δ |
|---|---:|---:|---:|
| Sharpe | +0.75 | **−0.39** | **−1.14** |
| Win rate | 41.6% | **31.2%** | −10.4 pp |
| n_trades | 303 | 224 | — |
| Total PnL (pips) | +123 868 | **−43 506** | — |
| Max DD (pips) | −34 000 | **−86 875** | ×2.5 pire |

Source : [predictions/test_oos_g2_ethusd_smacross_h1.json](../predictions/test_oos_g2_ethusd_smacross_h1.json)

**Verdict** : NO-GO sur 3 critères sur 4. L'edge n'existait pas — c'était un artefact statistique attendu (1 candidat parmi 1404 essais).

---

## 2. Lecture lucide

### Ce que disent les chiffres

Avec 1404 backtests indépendants, le **seuil Sharpe ajusté pour multiple testing** (DSR / Bonferroni) pour rejeter le bruit à p=5% est d'environ Sharpe > 2.0, pas 0.5. Le candidat unique à Sharpe +0.75 sur train était donc statistiquement compatible avec l'hypothèse "aucun edge réel", et l'OOS le confirme.

### Patterns observés malgré l'échec

Quelques régularités intéressantes pour la suite, **à valeur d'intuition seulement** (pas exploitables directement) :

1. **ETHUSD domine les podiums sur les 3 TF** — confirme F4 (cryptos = 37-42% trend). Reste l'actif où le trend-following classique tient le mieux, mais marche est dur en environnement bull (faux breakouts fréquents).
2. **Mean-reversion forex/indices** apparaît systématiquement dans le top 10-20 sans franchir le seuil. Cohérent avec la littérature : edge réel mais petit, mangé par les coûts.
3. **Plus on monte en fréquence (D1 → H1), plus c'est sélectif** — H1 est la résolution où les patterns apparaissent vraiment, mais aussi où le bruit + le coût relatif dominent.
4. **Six actifs sur neuf produisent zéro candidat sur les 3 TF** : GBPUSD, USDCHF, US30, GER30, XAUUSD, BTCUSD. Plus probable que ces actifs n'ont aucun edge exploitable avec les stratégies TA classiques.

### Ce que l'échec valide

- Le **pipeline lui-même** : ingestion data, calcul de coûts (swap + spread + slippage), discipline train/test, validate_edge, snooping_guard — tout fonctionne et a produit un résultat **fiable**.
- La **discipline** : on a pas insisté sur le candidat marginal, on a fait l'OOS, on a accepté le verdict.
- La **lucidité acquise** : le screening exhaustif sur TA classique + OHLCV ne mène nulle part. C'est une donnée, pas une opinion.

---

## 3. Décision : bascule Phase H avec architecture cascade

### Hypothèses reconsidérées

1. Le edge ne viendra **pas d'un pattern technique classique sur OHLCV brut** (testé, mort).
2. Le edge peut venir d'**événements timés** (FOMC, earnings, calendar) avec un mécanisme connu.
3. Le edge peut venir de **données non-structurées** (texte, news, transcripts) sous-exploitées par retail.
4. Le edge peut venir de **niches structurelles** (funding rates crypto, CoT positioning, pairs trading) — à explorer plus tard.

### Architecture cible : cascade en 3 niveaux

```
┌──────────────────────────────────────────────┐
│ NIVEAU 1 : Stratégie déterministe            │
│  - Signal généré par règle simple            │
│  - Ex : long US500 entre FOMC-24h et FOMC-1h │
│  - Backtestable, n_trial = 0 si bien posé    │
└──────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────┐
│ NIVEAU 2 : ML méta-labeling                  │
│  - Features : VIX, DXY, yield slope, régime  │
│  - Output : P(trade_winner) calibrée         │
│  - Filtre take/discard sur signaux niveau 1  │
└──────────────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────────────┐
│ NIVEAU 3 : Veto LLM (optionnel)              │
│  - LLM lit news/contexte 24h avant le trade  │
│  - Output : TAKE / VETO + raison             │
│  - Asymétrique : ne peut que désapprouver    │
└──────────────────────────────────────────────┘
```

Apport mesurable de chaque niveau via comparaison :
- Baseline 1 = Niveau 1 seul
- Baseline 2 = Niveau 1 + Niveau 2 (mesure apport ML)
- Final = Niveau 1 + Niveau 2 + Niveau 3 (mesure apport LLM)

Si un niveau n'améliore pas, on le retire.

### Premier livrable Phase H : Pre-FOMC drift sur US500

**Pourquoi en premier** :
1. **Edge documenté** (Lucca-Moench 2015) — pas un signal data-miné, donc zéro multiple testing à compenser.
2. **Calendar dispo** : `data/raw/economic_calendar/` contient déjà les dates FOMC.
3. **Implémentation rapide** : 3-4h pour le Niveau 1 déterministe.
4. **Échantillon limité** mais effet attendu fort : ~96 FOMC train, ~16 OOS.
5. **Bon banc d'essai cascade** : permet de plugger Niveau 2 (ML) puis Niveau 3 (LLM tone Fed minutes) progressivement.

### Stratégies suivantes envisagées (Phase H+)

- H2 : Asian Range Breakout sur EURUSD/GBPUSD/USDJPY H1
- H3 : NR4/NR7 (volatility contraction breakout)
- H4 : Pairs trading EURUSD-GBPUSD (cointégration)
- H5 : Funding rate arbitrage crypto (spot vs perp)

---

## 4. Comptabilité n_trials

| Source | n_trial consommé | Cumul |
|---|---:|---:|
| Avant Phase F | — (historique) | ~50 |
| Phase B C5 b1/b2/b3/b4 | 4 (historiques) | 54 |
| Phase G2 (ETHUSD SmaCross) | **+1** (2026-05-22) | **55** |

Budget cible Phase H : ≤ 10 lectures OOS supplémentaires pour rester sous Sharpe DSR-target raisonnable (~2.0+ pour la prochaine stratégie validée).

---

## 5. Synthèse 1-ligne

> **G-light + G2 = échec attendu du screening TA classique. Pivot validé vers
> architecture cascade event-driven, première stratégie : Pre-FOMC drift US500.**
