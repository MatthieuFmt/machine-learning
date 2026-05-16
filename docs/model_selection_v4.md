# Model Selection v4 (pivot v4 A7)

**Date** : 2026-05-15
**Périmètre** : train ≤ 2022-12-31 UNIQUEMENT
**n_trials** : inchangé (0 consommé)
**Méthode** : CPCV 5 folds × embargo 1 %, seuil fixe 0.50
**Statut** : ❌ NO-GO — stability > 1.0 sur tous les actifs, XAUUSD Sharpe négatif

## Résultats par actif

### US30 D1

| Modèle | Sharpe moyen | Sharpe std | Stability | WR moyen | n_kept |
|---|---|---|---|---|---|
| RF | **+1.75** | 2.03 | 1.16 | 54.4% | 29.0 |
| HGBM | +1.36 | 2.36 | 1.74 | 52.7% | 29.2 |
| Stacking | +0.35 | 1.56 | 4.43 | 40.0% | 12.4 |

**Modèle retenu** : RF
**Sharpe fold** : [−1.26, +1.02, +1.17, +3.05, +4.76]

### EURUSD H4

| Modèle | Sharpe moyen | Sharpe std | Stability | WR moyen | n_kept |
|---|---|---|---|---|---|
| RF | **+0.90** | 1.10 | 1.23 | 53.9% | 34.4 |
| HGBM | −0.04 | 0.66 | 16.58 | 40.6% | 38.4 |
| Stacking | −0.36 | 0.71 | 2.00 | 6.7% | 5.8 |

**Modèle retenu** : RF
**Sharpe fold** : [+1.89, +1.24, −0.38, −0.42, +2.17]

### XAUUSD D1

| Modèle | Sharpe moyen | Sharpe std | Stability | WR moyen | n_kept |
|---|---|---|---|---|---|
| RF | −1.39 | 1.97 | 1.42 | 8.7% | 3.2 |
| HGBM | −1.24 | 1.90 | 1.53 | 10.2% | 3.6 |
| Stacking | **−1.05** | 2.09 | 2.00 | 2.0% | 2.0 |

**Modèle retenu** : stacking (meilleur Sharpe parmi les négatifs, mais non utilisable)
**Sharpe fold** : [0.0, 0.0, 0.0, 0.0, −5.23]

## Critères go/no-go

| Critère | US30 D1 | EURUSD H4 | XAUUSD D1 | Seuil | Verdict |
|---|---|---|---|---|---|
| Sharpe ≥ 0.5 | +1.75 ✅ | +0.90 ✅ | −1.05 ❌ | ≥ 0.5 | ❌ |
| Stability < 1.0 | 1.16 ❌ | 1.23 ❌ | 2.00 ❌ | < 1.0 | ❌ |
| make verify OK | ✅ | ✅ | ✅ | — | ✅ |

→ **NO-GO confirmé** — les 3 actifs échouent le critère de stabilité (< 1.0). XAUUSD échoue également le Sharpe.

## Interprétation

- **US30 D1** : Sharpe moyen solide (+1.75) mais forte variance inter-fold (fold 1 = −1.26, fold 5 = +4.76). Le RF capte du signal mais les folds CPCV produisent des résultats très hétérogènes → overfitting probable sur certaines périodes.
- **EURUSD H4** : RF seul viable (+0.90). HGBM et Stacking écrasés par le bruit. 506 trades train avec WR 38.7% → le RF parvient à filtrer correctement (WR méta 53.9%) mais l'instabilité demeure.
- **XAUUSD D1** : Échantillon trop faible (85 trades train, WR 11.8%). 3 folds sur 5 produisent 0 trade → CPCV inapplicable. Le stacking est « sélectionné » par défaut (moins pire Sharpe), mais inutilisable en l'état.

## Cause racine

1. **Stabilité** : CPCV 5-fold avec ~17-68 trades/test par fold génère une variance extrême sur les folds à faible nombre de trades. La métrique stability = Sharpe_std / Sharpe_mean pénalise mécaniquement les petits échantillons.
2. **XAUUSD** : n_train = 85 trades, WR 11.8% → le problème est structurellement non soluble avec CPCV 5-fold. Il faut soit plus de données (H4 au lieu de D1), soit un split walk-forward au lieu de CPCV.
3. **Seuil fixe 0.50** : Le seuil de méta-labeling est calibré de façon fixe, sans optimisation par actif. Une calibration adaptative (A8) pourrait améliorer le Sharpe.

## Décision de gel

Les modèles retenus par actif sont FIGÉS dans [`app/config/model_selected.py`](../app/config/model_selected.py) :
- US30 D1 → RF
- EURUSD H4 → RF
- XAUUSD D1 → stacking (non utilisable)

**Aucun changement jusqu'à fin Phase B.**

## Limites

- Seuil 0.50 fixe pour A7. Calibration en A8 peut décaler Sharpe ± 0.2.
- n_splits=5 = compromis vitesse/stabilité. 10 plus rigoureux mais 2× plus lent.
- Le Sharpe per-trade × √n est une approximation ; B1+ utilisera le vrai Sharpe walk-forward.
- XAUUSD avec seulement 85 trades train → CPCV folds de ~17 trades test → 3 folds sans trade → variance explosive.
- Stability > 1.0 ne signifie pas nécessairement que le modèle est inutilisable, mais que la variance inter-fold dépasse la performance moyenne — signe de surapprentissage ou d'échantillon insuffisant.
