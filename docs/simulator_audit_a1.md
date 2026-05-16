# Simulator Audit — Pivot v4

## A1 — Equity € & Sizing 2 % (terminé)

Calcul de l'equity en EUR via `position_size_lots`, drawdown borné [−100 %, 0 %], blowup detection.

## A2 — Calibration des coûts (terminé)

Coûts XTB Standard Account calibrés : US30 1.8 pts, EURUSD 0.9 pip, XAUUSD 0.35 USD, etc.

## A3 — Sharpe routing par fréquence

Routing automatique selon `trades_per_year` (tpy) :

| Régime | Méthode | Annual factor |
|---|---|---|
| tpy ≥ 100 | resample daily | √252 |
| 30 ≤ tpy < 100 | resample weekly | √52 |
| tpy < 30 | per-trade | √tpy |

### Justification

- **Daily** : valable si au moins 1 trade tous les 2-3 jours en moyenne. Sinon le ffill écrase la variance.
- **Weekly** : valable pour Donchian D1 (~30-90 trades/an = ~0.6-1.7 trades/semaine).
- **Per-trade** : valable pour Donchian H4 D1 multi-actif où chaque actif fait 10-30 trades/an. La méthode standard suppose i.i.d. ce qui est raisonnable pour des trades indépendants.

### Impact sur résultats v3

Donchian US30 D1 H06 avait 75 trades sur 16 mois = 56 trades/an = méthode weekly avec ce routing. L'ancien Sharpe daily était écrasé. Avec weekly, il devrait remonter de ~0.3.

### Métadonnées

- **Fonction** : `sharpe_annualized()` dans `app/backtest/metrics.py`
- **Intégration** : `compute_metrics()` → `"sharpe_method"` dans le dict retour
- **Tests** : `tests/unit/test_sharpe_routing.py` (6 tests)
- **Rétrocompatibilité** : `sharpe_ratio()` et `sharpe_daily_from_trades()` conservés
