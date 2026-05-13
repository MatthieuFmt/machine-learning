# Rapport CPCV — EURUSD H1

**Date** : 2026-05-13 06:41
**Temps d'exécution** : 56s

## Configuration

| Paramètre | Valeur |
|---|---|
| n_groups | 48 |
| k_test | 12 |
| n_samples (demandé) | 200 |
| n_splits (réalisés) | 200 |
| purge_hours | 48h |
| target_mode | triple_barrier |
| Model | RandomForest (n=500, depth=6) |
| TP/SL | 30.0 / 10.0 pips |
| Commission | 0.50 pips |
| Slippage | 1.00 pips |
| Filtres | Momentum+Vol+Session |

## Métriques CPCV (distribution sur les splits)

| Métrique | Valeur |
|---|---|
| Splits valides | 200 / 200 |
| % splits profitables | 1.0% |
| E[Sharpe] | -0.1378 |
| σ[Sharpe] | 0.0579 |
| Sharpe médian | -0.1435 |
| Sharpe min | -0.2760 |
| Sharpe max | 0.0245 |
| Sharpe CI 95% | [-0.2352, -0.0257] |
| Trades/split (moyenne) | 341.3 ± 70.0 |
| Trades totaux | 68266 |

## DSR Distributionnel (López de Prado 2014)

| Métrique | Valeur |
|---|---|
| DSR | -5.1462 |
| PSR(SR*=0) | 0.0086 |
| SR₀* (seuil déflaté) | 0.1601 |
| E[max SR] sous H₀ | 0.1601 |
| Var[SR] CPCV | 0.0034 |
| N splits pour DSR | 200 |
| % profitables (DSR) | 1.0000% |
| E[SR] distrib | -0.1378 |
| σ[SR] distrib | 0.0579 |
| SR min distrib | -0.2760 |
| SR max distrib | 0.0245 |
| SR médian distrib | -0.1435 |
| CI 95% distrib | [-0.2352, -0.0257] |

## Split Principal (train≤2023, test=2025)

| Métrique | Valeur |
|---|---|
| N trades | 853 |
| Sharpe observé | -0.1099 |
| p(Sharpe>0) bootstrap | 0.9997 |
| PSR (Bailey) | 0.0013 |
| DSR distributionnel | -4.6645 |
| Breakeven WR | 28.8% |
| WR observé | 25.7% |
| t-statistique | -3.2099 (p=0.0014) |

### Performance 2025

| Métrique | Valeur |
|---|---|
| Sharpe | -2.6315 |
| Return total (%) | -13.4407 |
| Max drawdown (%) | -14.4917 |
| Win rate (%) | 25.6741 |
| N trades | 853 |

## Verdict

| Critère | Seuil | Valeur | OK? |
|---|---|---|---|
| DSR > 0 | > 0 | -5.1462 | ❌ |
| % profitables > 60% | > 60% | 1.0% | ❌ |
| E[Sharpe] > 0 | > 0 | -0.1378 | ❌ |

### 🔴 **NO-GO — Améliorer l'edge avant de continuer**
