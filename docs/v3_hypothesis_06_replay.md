# Hypothesis 06 — Replay pivot v4 (train + val uniquement)

**Date** : 2026-05-16
**Type** : Audit informatif. **Aucun verdict GO/NO-GO**.
**n_trials** : 22 (inchangé, c'est un replay du même n_trial H06=1).

## Comparaison train + val (test set non touché)

| Actif | Best (N,M) v3 | Best (N,M) v4 | Sharpe train v3 | Sharpe train v4 | Sharpe val v3 | Sharpe val v4 | WR val v3 | WR val v4 | Trades val v3 | Trades val v4 | Coût total v3 (pips) | Coût total v4 (pips) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EURUSD | (non testé v3) | (20, 20) | — | **+1.08** | — | **+0.98** | — | **67.4%** | — | **43** | — | **0.90** |
| US30 | (100, 10) | (20, 20) | +0.35 | **+0.75** | +0.58 | **+1.87** | 47.1% | **57.5%** | 34 | **40** | 8.0 | 1.8 |
| XAUUSD | (100, 20) | (50, 50) | +1.13 | **+0.57** | 0.00 | **−0.89** | 0.0% | **31.3%** | 15 | **16** | 35.0 | 0.35 |
| GER30 | (50, 10) | (100, 50) | +0.29 | **+0.20** | +1.86 | **+1.47** | 26.3% | **37.5%** | 19 | **8** | 5.0 | 1.2 |
| US500 | (50, 50) | (100, 50) | +0.62 | **+0.69** | +1.62 | **+0.90** | 43.8% | **71.4%** | 16 | **21** | 3.5 | 0.6 |
| XAGUSD | (20, 10) | (20, 10) | 0.00 | **+0.93** | 0.00 | **+0.97** | 0.0% | **42.2%** | ? | **45** | 45.0 | 0.035 |
| USOIL | — | ❌ erreur | — | ❌ | — | ❌ | — | ❌ | — | ❌ | 7.0 | 0.07 |

*Note : USOIL exclu (prix négatifs dans le CSV raw). EURUSD n'était pas testé en v3, ajouté ici car disponible.*

## Détail coûts v3 → v4

| Actif | Spread v3 | Spread v4 | Slippage v3 | Slippage v4 | Ratio coût v4/v3 |
|-------|----------|----------|-------------|-------------|------------------|
| US30 | 3.0 | 1.5 | 5.0 | 0.3 | 22.5% |
| XAUUSD | 25.0 | 0.30 | 10.0 | 0.05 | 1.0% |
| GER30 | 2.0 | 1.0 | 3.0 | 0.2 | 24.0% |
| US500 | 1.5 | 0.5 | 2.0 | 0.1 | 17.1% |
| XAGUSD | 30.0 | 0.025 | 15.0 | 0.01 | 0.08% |
| USOIL | 4.0 | 0.05 | 3.0 | 0.02 | 1.0% |

## Interprétation

### Points positifs (encourageants)
- **US30** : Sharpe train multiplié par 2.1× (0.35→0.75), Sharpe val multiplié par 3.2× (0.58→1.87). L'edge Donchian US30 était bien masqué par les coûts excessifs. WR val passe de 47.1% à 57.5%.
- **XAGUSD** : Passe de Sharpe train 0.00 (v3) à +0.93 (v4). La correction de coût 45→0.035 pips change totalement la donne.
- **EURUSD** (nouveau) : Sharpe train +1.08, val +0.98, bonne stabilité train→val.
- **US500** : Sharpe val +0.90, WR val 71.4% — solide malgré la baisse vs v3 (+1.62→+0.90).

### Points négatifs
- **XAUUSD** : Sharpe v4 négatif en val (−0.89) malgré la correction massive des coûts (35→0.35). L'edge n'existe pas sur cet actif.
- **GER30** : 8 trades val seulement, non significatif statistiquement.

### Constats clés
- **Sharpe train v4 ≥ Sharpe train v3 + 1.0** sur **1 actif** (XAGUSD +0.93). Pas 3 actifs → la correction seule n'explique pas tout.
- **Sharpe val v4 ≥ +0.90 sur 4 actifs** (EURUSD, US30, US500, XAGUSD) — signe d'un edge réel sur le Donchian multi-actif avec coûts corrigés.
- **Sharpe val v4 reste négatif** sur XAUUSD (−0.89) — l'edge n'existe pas partout.

## Décision

Cette section sert à informer la Phase B, **pas à statuer**. Le test set 2024+ n'est plus disponible pour ces hypothèses (brûlé). Les vrais GO/NO-GO viendront de H_new1, H_new2, H_new3 sur leur propre test set.

**Implication Phase B** : Le Donchian mérite d'être retesté en hypothèse fraîche (H_new) avec coûts corrigés sur EURUSD, US30, US500, XAGUSD.

## Fichiers

- Script : `scripts/run_pivot_a4_replay.py`
- Résultats : `predictions/pivot_a4_replay.json`
- Tests : `tests/unit/test_pivot_a4_cutoff.py`
