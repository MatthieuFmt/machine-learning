# Rapport de validation finale — Prompt 18

**Date** : 2026-05-19T06:25:31.755259+00:00
**n_trials cumulé** : 44
**Verdict** : ❌ NO-GO — RETOUR EN RECHERCHE

## Stratégies / Sleeves retenus

| Actif | TF | Modèle | Sharpe | WR | Trades | Max DD |
|---|---|---|---|---|---|---|
| GBPUSD | D1 | rf | -3.16 | 9.1% | 22 | -35.3% |
| EURUSD | D1 | stacking | -4.78 | 8.5% | 47 | -78.0% |
| USDCHF | D1 | stacking | -1.57 | 19.0% | 21 | -24.1% |
| ETHUSD | D1 | hgbm | 0.15 | 35.3% | 34 | -12.5% |
| GBPUSD | H4 | rf | -0.94 | 29.6% | 71 | -33.9% |
| EURUSD | H4 | rf | -1.22 | 24.4% | 41 | -34.3% |

## Critères de la constitution

| Critère | Cible | Observé | Verdict |
|---|---|---|---|
| Sharpe | ≥ 1.0 | -5.42 | ❌ |
| DSR | > 0, p < 0.05 | -18.29 (p=1.000) | ❌ |
| Max DD | < 15% | 31.3% | ❌ |
| WR | > 30% | 22.5% | ❌ |
| Trades/an | ≥ 30 | 103.1 | ✅ |

## Benchmarks

| Benchmark | Cible | Observé | Verdict |
|---|---|---|---|
| Beat B&H+0.3 | Sharpe > 0.82 | -5.42 | ❌ |
| Beat P95 random | Sharpe > -0.98 | -5.42 | ❌ |

## Verdict

### ❌ NO-GO — Itération requise

**Raisons** :
- Sharpe -5.42 < 1.0
- DSR=-18.29 (p=1.000) non significatif
- Max DD 31.3% >= 15%
- WR 22.5% <= 30%
- Portfolio Sharpe -5.42 < B&H+0.3 (0.82)
- Portfolio Sharpe -5.42 <= P95 random (-0.98)

**Actions recommandées** :
- Prompt 14 (vol targeting) ou prompt 11 (méta-labeling)
- Revoir n_trials, réduire hypothèses testées
- Prompt 15 (vol targeting) ou prompt 14 (corrélation)
- Prompt 10 (régime) ou prompt 11 (méta-labeling)
- Le portfolio n'apporte pas d'edge vs buy-and-hold
- Le portfolio ne bat pas des signaux aléatoires