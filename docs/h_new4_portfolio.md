# H_new4 — Portfolio des sleeves GO (pivot v4 B4)

**Date** : 2026-05-16
**n_trials** : 28
**Sleeves combinés** : 1 (EURUSD H4 uniquement)

## Question
Le portfolio equal-risk weight + filtre corrélation des sleeves GO produit-il
un Sharpe portfolio strictement supérieur (+0.2) au max des Sharpe individuels ?

## Sleeves GO disponibles

| Sleeve | Source | Sharpe individuel | Trades/an | DD | Verdict |
|---|---|---|---|---|---|
| H_new1 US30 D1 | predictions/h_new1_meta_us30.json | 0.82 | 5.6 | 3.9% | ❌ NO-GO |
| H_new2 WF rolling | predictions/h_new2_walk_forward_rolling.json | US30 0.60, XAUUSD 1.65 | < 30 | — | ❌ NO-GO |
| **H_new3 EURUSD H4** | **predictions/h_new3_eurusd_h4.json** | **+1.73** | **25.2** | **8.1%** | **✅ GO** |

## Résultat

**Single-sleeve fallback automatique** — 1 seul sleeve GO. Le portfolio = l'unique sleeve EURUSD H4.

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| Sharpe portfolio | ≥ max + 0.2 | +1.73 (= max) | ✗ |
| Sharpe portfolio absolu | ≥ 1.2 | +1.73 | ✓ |
| DSR (n=28) | > 0 (p<0.05) | +23.41 (p=0.0) | ✓ |
| Max DD | < 12 % | 8.1% | ✓ |
| Trades/an | ≥ 60 (somme) | 25.2 | ✗ |
| Corrélation moyenne | < 0.5 | N/A (1 sleeve) | ✗ |

## Décision

❌ **NO-GO portfolio** — single-sleeve fallback. Production = EURUSD H4 seule.

## Causes de l'échec du portfolio

- **1 seul sleeve GO sur 3 hypothèses Phase B.** H_new1 (US30 D1, Sharpe 0.82, 12 trades) et H_new2 (walk-forward rolling, Sharpe < 1.0, < 30 trades/an) n'ont pas atteint les critères GO.
- La diversification nécessite ≥ 2 sleeves GO. Le module `app/portfolio/constructor.py` est prêt pour usage futur.

## Prochaine étape

Production single-sleeve : `prompts/20_signal_engine.md` — déploiement EURUSD H4 mean-reversion + méta-labeling RF.
