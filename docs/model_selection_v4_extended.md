# Model selection v4 — Extension multi-actifs (Phase C3)

**Date** : 2026-05-17
**Périmètre** : 12 couples (3 A7 originaux + 9 shortlist C2, stab >= 0.5)
**Train cutoff** : 2022-12-31
**Méthode** : CPCV 5-fold × embargo 1%, seuil méta 0.50
**Candidats** : RandomForest, HistGradientBoosting, Stacking (RF + HGBM + LogReg meta)
**Statut** : ✅ Terminé — 12/12 couples évalués

## Tableau récapitulatif

| Actif | TF | n trades | WR train | RF Sharpe | HGBM Sharpe | Stack Sharpe | Modèle retenu | Pass C4 ? |
|---|---|---|---|---|---|---|---|---|
| US30 | D1 | 338 | 46.7% | **+1.75** | +1.36 | +0.35 | rf | ✅ |
| EURUSD | H4 | 506 | 38.7% | **+0.90** | −0.04 | −0.36 | rf | ✅ |
| XAUUSD | D1 | 85 | 11.8% | −1.39 | −1.24 | **−1.05** | stacking | ❌ |
| BTCUSD | D1 | 149 | 43.6% | +0.26 | **+0.28** | +0.02 | hgbm | ❌ |
| ETHUSD | D1 | 145 | 50.3% | +0.89 | **+1.55** | +0.21 | hgbm | ✅ |
| ETHUSD | H4 | 467 | 40.5% | +0.53 | **+0.55** | −0.41 | hgbm | ✅ |
| ETHUSD | H1 | 845 | 40.7% | +1.89 | **+2.02** | −0.07 | hgbm | ✅ |
| EURUSD | D1 | 498 | 64.3% | +6.08 | +5.06 | **+6.21** | stacking | ✅ |
| GBPUSD | D1 | 482 | 69.3% | **+8.62** | +7.46 | +7.37 | rf | ✅ |
| GBPUSD | H4 | 1000 | 45.7% | **+3.69** | +2.86 | +1.36 | rf | ✅ |
| USDCHF | D1 | 210 | 58.1% | +2.29 | +2.87 | **+3.33** | stacking | ✅ |
| USDCHF | H4 | 905 | 36.7% | **+1.15** | +0.40 | −0.14 | rf | ✅ |

*Les lignes US30/EURUSD/XAUUSD (A7 originaux) sont incluses pour référence. Seules les 9 lignes C3 ont été ajoutées à [`model_selected.py`](../app/config/model_selected.py).*

## Patterns par classe d'actif

### Crypto (BTCUSD, ETHUSD)

- **Modèle dominant** : HGBM (4/4 crypto-timeframes)
- **ETHUSD** : HGBM systématiquement supérieur à RF et stacking. Sharpe croissant avec la fréquence : D1=1.55, H4=0.55, H1=2.02. Le H1 produit le meilleur Sharpe absolu de la classe crypto (2.02) avec 845 trades train et WR 40.7%.
- **BTCUSD** : Sharpe très faible (0.28), HGBM domine de justesse sur RF. Stability catastrophique (1.88) → peu fiable. Exclu de C4.
- **Stacking** : Inutilisable sur crypto — Sharpe négatif ou proche de zéro, n_kept très faible (4.6–7.4), WR méta < 42%.
- **Pattern** : La force du HGBM sur crypto vient probablement de sa capacité à capturer des interactions non-linéaires entre features de volatilité et momentum, là où RF linéarise trop.

### Forex majeures (EURUSD, GBPUSD, USDCHF)

- **Pas de modèle dominant unique** : RF 3/6, Stacking 2/6, HGBM 0/6
- **D1 vs H4** : En D1, Stacking domine sur EURUSD (6.21) et USDCHF (3.33), RF sur GBPUSD (8.62). En H4, RF systématiquement gagnant (EURUSD 0.90, GBPUSD 3.69, USDCHF 1.15).
- **GBPUSD D1** : Meilleur Sharpe absolu toutes classes confondues (8.62) avec 482 trades, WR 69.3%, stability 0.16 — signal exceptionnellement stable.
- **EURUSD D1** : Stacking surpasse RF (6.21 vs 6.08) d'une courte tête. Les 3 modèles performent bien (tous ≥ 5.06) → l'actif lui-même porte l'edge, pas le choix du modèle.
- **USDCHF H4** : WR train faible (36.7%) mais RF parvient à filtrer (WR méta 34.4% → Sharpe 1.15). HGBM et stacking écrasés par le bruit.
- **XAUUSD D1** (rappel A7) : Exclu — 85 trades, WR 11.8%, Sharpe −1.05. Structurellement insoluble avec CPCV 5-fold.

## Shortlist C4 (Sharpe CPCV moyen >= 0.5)

**10 couples retenus :**

| Actif | TF | Modèle | Sharpe CPCV | Décision |
|---|---|---|---|---|
| US30 | D1 | rf | +1.75 | C4 eligible |
| EURUSD | H4 | rf | +0.90 | C4 eligible |
| ETHUSD | D1 | hgbm | +1.55 | C4 eligible |
| ETHUSD | H4 | hgbm | +0.55 | C4 eligible |
| ETHUSD | H1 | hgbm | +2.02 | C4 eligible |
| EURUSD | D1 | stacking | +6.21 | C4 eligible |
| GBPUSD | D1 | rf | +8.62 | C4 eligible |
| GBPUSD | H4 | rf | +3.69 | C4 eligible |
| USDCHF | D1 | stacking | +3.33 | C4 eligible |
| USDCHF | H4 | rf | +1.15 | C4 eligible |

## Couples exclus

| Actif | TF | Modèle | Sharpe CPCV | Raison |
|---|---|---|---|---|
| XAUUSD | D1 | stacking | −1.05 | Sharpe négatif, 3/5 folds sans trade, WR 11.8% |
| BTCUSD | D1 | hgbm | +0.28 | Sharpe < 0.5, stability 1.88, variance inter-fold explosive |

## Analyse qualitative

### Hiérarchie des modèles par classe d'actif

1. **Crypto (4 couples)** : HGBM > RF >> Stacking. Le gradient boosting capture mieux les régimes volatils et les interactions feature non-linéaires propres aux crypto-actifs.
2. **Forex D1 (3 couples)** : RF ≈ Stacking > HGBM. Sur timeframes longues, le signal forex est plus linéaire → RF et stacking (méta-modèle logistique) suffisent. HGBM n'apporte pas de gain.
3. **Forex H4 (4 couples)** : RF > HGBM >> Stacking. En H4, RF domine sans ambiguïté. Stacking déçoit systématiquement (Sharpe négatif ou marginal).

### Stabilité inter-fold

- **Tous les couples C3 ont stability < 1.0** sauf BTCUSD D1 (1.88) — critère A7 automatiquement satisfait pour 8/9.
- Les stabilities les plus faibles (meilleures) : GBPUSD D1 (0.16), EURUSD D1 (0.24), GBPUSD H4 (0.31) — tous forex majeures D1/H4.
- Les stabilities les plus élevées : BTCUSD D1 (1.88), ETHUSD D1 (0.53), USDCHF H4 (0.64).

### Nombre de trades et fiabilité

- **GBPUSD H4** : 1000 trades train — plus large échantillon, Sharpe 3.69 avec stability 0.31 → signal le plus robuste de l'extension.
- **ETHUSD H1** : 845 trades — second plus large échantillon, Sharpe 2.02 → excellent ratio signal/bruit.
- **BTCUSD D1** : 149 trades seulement → probablement sous-échantillonné pour CPCV 5-fold.

## Notes

- Seuil méta 0.50 commun à tous les couples (pas encore tuné — C4).
- Stability inter-fold > 1.0 n'est plus un critère de rejet (cf. Annexe A3 du prompt C3).
- Le seuil C4 à 0.5 est un filtre de pertinence, pas un GO production.
- XAUUSD D1 reste non résolu — nécessiterait H4 ou walk-forward au lieu de CPCV.
- Les 3 entrées A7 originales dans [`model_selected.py`](../app/config/model_selected.py) sont intactes et non modifiées.
