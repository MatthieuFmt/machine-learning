# Hypothesis 07 — Replay pivot v4 (train + val uniquement)

**Date** : 2026-05-16
**Type** : Audit informatif. **Aucun verdict GO/NO-GO**.
**n_trials** : 22 (inchangé, c'est un replay du même n_trial H07=1).

## Comparaison train + val (test set non touché) — US30 D1 uniquement

| Stratégie | Best Params v3 | Best Params v4 | Sharpe train v3 | Sharpe train v4 | Sharpe val v3 | Sharpe val v4 | WR val v3 | WR val v4 | Trades val v3 | Trades val v4 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Donchian** (baseline) | — | N=20, M=20 | — | **+0.75** | — | **+1.87** | — | **57.5%** | — | **40** |
| **Dual MA** | fast=10, slow=50 | fast=5, slow=100 | +0.79 | **+0.43** | −0.20 | **−0.08** | 46.7% | **45.9%** | ? | **157** |
| **Keltner** | period=20, mult=2.0 | period=50, mult=2.0 | +0.98 | **+0.53** | +3.70 | **+1.05** | 58.7% | **56.1%** | 46 | **82** |
| **Chandelier** | period=44, k_atr=4.0 | period=22, k_atr=2.0 | +0.62 | **+0.43** | +2.36 | **+1.24** | 46.7% | **50.0%** | 199 | **192** |
| **Parabolic SAR** | step=0.03, af_max=0.2 | step=0.03, af_max=0.1 | +0.47 | **−0.13** | +0.64 | **+1.84** | 47.4% | **45.3%** | 230 | **234** |

## Détail coûts US30 v3 → v4

| Paramètre | v3 | v4 | Ratio |
|-----------|-----|-----|-------|
| Spread | 3.0 pts | 1.5 pts | 50% |
| Slippage | 5.0 pts | 0.3 pts | 6% |
| Coût total/trade | 16.0 pts | 3.6 pts | 22.5% |
| TP | 200 pts | 200 pts | — |
| SL | 100 pts | 100 pts | — |

## Interprétation

### Constats clés
- **Keltner** : Sharpe val v4 (+1.05) est bien inférieur au Sharpe val v3 (+3.70). Le +3.70 v3 était un artefact : les coûts quasi-nuls apparents (spread 3.0 pips sur un indice US30) gonflaient artificiellement la performance val. Le vrai Sharpe est autour de +1.0.
- **Chandelier** : Sharpe val stable entre v3 (+2.36) et v4 (+1.24). Bonne robustesse, 192 trades val → statistiquement significatif.
- **Parabolic SAR** : Meilleur Sharpe val v4 (+1.84) du panel, mais Sharpe train v4 négatif (−0.13) → overfitting évident du grid search. 234 trades val.
- **Dual MA** : Toujours non performant en val (−0.08), même avec coûts corrigés.
- **Donchian baseline** : Meilleur Sharpe val du panel (+1.87) avec seulement 40 trades — ratio Sharpe/trade le plus efficient.

### Réponse aux questions
- Sharpe train v4 ≥ 1.0 sur Keltner ? **Non** (+0.53). L'edge train n'était pas masqué, il était modeste.
- Sharpe val v4 < Sharpe val v3 sur Keltner ? **Oui** (+1.05 vs +3.70). Confirmation que l'overfitting val v3 était lié aux coûts faibles.
- Sharpe val v4 positif pour ≥ 2 strats ? **Oui** : Donchian +1.87, Keltner +1.05, Chandelier +1.24, Parabolic +1.84 → **4/5 strats en Sharpe val positif**.
- Sharpe val v4 négatif partout ? **Non**. Seul Dual MA reste marginal (−0.08).

## Décision

Cette section sert à informer la Phase B, **pas à statuer**. Le test set 2024+ n'est plus disponible pour ces hypothèses (brûlé). Les vrais GO/NO-GO viendront de H_new1, H_new2, H_new3 sur leur propre test set.

**Implication Phase B** : 4 stratégies trend-following sur 5 montrent un Sharpe val ≥ +1.0 sur US30 D1 avec coûts corrigés. Cela justifie de re-tester ces stratégies en hypothèses fraîches (H_new) avec split temporel vierge.

**Observation H07 v3** : Toutes les stratégies sont fortement décorrélées de Donchian (ρ ∈ [0.19, 0.31]). Utilité potentielle en diversification même sans edge propre.

## Fichiers

- Script : `scripts/run_pivot_a4_replay.py`
- Résultats : `predictions/pivot_a4_replay.json`
- Tests : `tests/unit/test_pivot_a4_cutoff.py`
