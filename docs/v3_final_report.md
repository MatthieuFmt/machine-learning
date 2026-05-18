# Rapport de validation finale — Prompt 18

**Date** : 2026-05-18T20:18:34.342012+00:00
**n_trials cumulé** : 29
**Verdict** : ❌ NO-GO — RETOUR EN RECHERCHE

## Stratégies / Sleeves retenus

| Actif | TF | Modèle | Sharpe | WR | Trades | Max DD |
|---|---|---|---|---|---|---|
| GBPUSD | D1 | rf | 5.17 | 6583.9% | 483 | -507.9% |
| EURUSD | D1 | stacking | 4.01 | 5809.5% | 630 | -1549.4% |
| USDCHF | D1 | stacking | 3.29 | 5413.7% | 423 | -1307.3% |
| ETHUSD | D1 | hgbm | 2.14 | 4390.2% | 328 | -781.6% |
| GBPUSD | H4 | rf | 3.19 | 4995.5% | 1103 | -1723.6% |
| EURUSD | H4 | rf | 1.73 | 5370.4% | 54 | -807.8% |

## Critères de la constitution

| Critère | Cible | Observé | Verdict |
|---|---|---|---|
| Sharpe | ≥ 1.0 | 4.97 | ✅ |
| DSR | > 0, p < 0.05 | 19.47 (p=0.000) | ✅ |
| Max DD | < 15% | 2.7% | ✅ |
| WR | > 30% | 54.2% | ✅ |
| Trades/an | ≥ 30 | 1310.5 | ✅ |

## Benchmarks

| Benchmark | Cible | Observé | Verdict |
|---|---|---|---|
| Beat B&H+0.3 | Sharpe > 0.82 | 4.97 | ✅ |
| Beat P95 random | Sharpe > 9.96 | 4.97 | ❌ |

## Verdict

### ❌ NO-GO — Itération requise

**Raisons** :
- Portfolio Sharpe 4.97 <= P95 random (9.96)

**Actions recommandées** :
- Le portfolio ne bat pas des signaux aléatoires