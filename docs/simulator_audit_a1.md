# Pivot v4 A1 — Audit Simulator : sizing + DD + Sharpe

> **Date** : 2026-05-15 | **n_trials consommés** : 0 | **Type** : Bug fix infrastructure

## Bug corrigés

| # | Bug | Localisation | Avant | Après |
|---|-----|-------------|-------|-------|
| 1 | Sizing au risque non implémenté | [`app/backtest/simulator.py`](../../app/backtest/simulator.py) | `Pips_Nets = pips_brut × weight` (poids probabiliste 0-1) | `position_size_lots` calculé par `compute_position_size()` → risque 2% exact |
| 2 | DD calculé sur PnL en pips | [`app/backtest/metrics.py`](../../app/backtest/metrics.py) | `max_dd_pct = dd_pips × pip_value / capital × 100` → pouvait dépasser −100% | `max_dd_pct = min(equity/cummax - 1) × 100` → borné [−100%, 0%] |
| 3 | Sharpe sur retours en pips | [`app/backtest/metrics.py`](../../app/backtest/metrics.py) | `daily_returns = pips → € linéaire` | `equity_daily.pct_change()` sur equity en € |
| 4 | Pas de détection blow-up | [`app/backtest/metrics.py`](../../app/backtest/metrics.py) | Equity peut devenir négative | `equity.clip(lower=0.01)` + flag `blowup_detected` |

## Comparaison avant/après sur H06 US30 D1 (théorique)

| Métrique | Avant A1 | Après A1 (attendu) |
|----------|----------|---------------------|
| DD test | −362 % | [−100 %, 0 %] |
| Sharpe test | −0.09 | Recalculé sur equity € |
| PnL par trade | Variable (poids ML) | Proportionnel au risque 2% |
| Blowup | Non détecté | Détecté si equity < 0.01 € |

> **Note** : Les valeurs "après" seront confirmées en A4 quand H06 sera rejoué avec le nouveau simulateur.

## Fichiers créés/modifiés

| Fichier | Action |
|---------|--------|
| [`app/backtest/sizing.py`](../../app/backtest/sizing.py) | Créé — `compute_position_size()`, `expected_pnl_eur()` |
| [`app/backtest/simulator.py`](../../app/backtest/simulator.py) | Modifié — injection `asset_cfg`, `capital_eur`, `risk_pct` dans `_simulate_stateful_core` + propagation wrappers |
| [`app/backtest/metrics.py`](../../app/backtest/metrics.py) | Modifié — mode A1 (equity €, DD borné, blowup detection) + mode legacy préservé |
| [`tests/unit/test_simulator_sizing.py`](../../tests/unit/test_simulator_sizing.py) | Créé — 12 tests |

## Formule de sizing

```
risk_eur = capital × risk_pct
distance_pts = |entry - sl| / pip_size
loss_1lot_eur = distance_pts × pip_value_eur
lots = risk_eur / loss_1lot_eur
lots = clamp(round(lots, 2), min_lot, max_lot)
```

## Tests unitaires

```
tests/unit/test_simulator_sizing.py - 12 passed
```

| Test | Vérification |
|------|-------------|
| `test_sizing_us30_100pt_sl` | 2.17 lots pour 200€ risque |
| `test_sizing_eurusd_10pip_sl` | 2.0 lots pour 200€ risque |
| `test_sizing_invalid_sl_equals_entry` | ValueError |
| `test_sizing_min_lot_clamp` | Clamp à 0.01 |
| `test_trade_sl_exact_minus2pct` | −200€ ≈ −2% |
| `test_trade_tp_exact_4pct` | +400€ ≈ +4% (R:R 2:1) |
| `test_dd_bounded_minus_100` | DD ≥ −100%, blowup détecté |
| `test_sharpe_equity_plate_returns_zero` | Sharpe = 0.0 |
| `test_10_sl_compound_dd` | DD ∈ [−22%, −18%] |
| `test_equity_positive_after_wins` | Equity > capital initial |
| `test_legacy_no_asset_cfg` | Rétrocompatibilité |
| `test_missing_position_size_lots_raises` | ValueError |

## Prochaine étape

→ [A2 — Calibration coûts XTB réels](../prompts/pivot_v4/02_calibration_costs.md)
