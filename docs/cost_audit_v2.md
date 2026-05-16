# Audit des coûts v4 (pivot)

**Date** : 2026-05-15
**Source** : XTB Standard Account — valeurs publiques documentées (cf prompt A2 §1)
**Type de compte** : Standard (spreads variables, 0 commission)
**⚠️ Coûts non confirmés en démo** — à valider avant prod avec capture MT5 ou export Symbol Specifications.

## Comparaison avant/après

| Actif | spread v3 | slippage v3 | total v3 | spread v4 | slippage v4 | total v4 | Facteur correction |
|---|---|---|---|---|---|---|---|
| US30 | 3.0 | 5.0 | 8.0 | 1.5 | 0.3 | 1.8 | ÷ 4.4 |
| US500 | 1.5 | 2.0 | 3.5 | 0.5 | 0.1 | 0.6 | ÷ 5.8 |
| GER30 | 2.0 | 3.0 | 5.0 | 1.0 | 0.2 | 1.2 | ÷ 4.2 |
| XAUUSD | 25 | 10 | 35 | 0.30 | 0.05 | 0.35 | ÷ 100 |
| XAGUSD | 30 | 15 | 45 | 0.025 | 0.01 | 0.035 | ÷ 1285 |
| USOIL | 4 | 3 | 7 | 0.05 | 0.02 | 0.07 | ÷ 100 |
| EURUSD | absent | absent | — | 0.7 | 0.2 | 0.9 | nouveau |

> Les facteurs de correction extrêmes sur XAUUSD (×100) et XAGUSD (×1285) sont dus à une confusion d'unité pip_size dans v3. Voir annexe A1 du prompt A2.

## Justification par actif

### US30 (Dow Jones CFD)
- Source : XTB.com → Trading conditions → CFD → US30
- Symbole XTB : `US30` ou `USA30`
- Spread moyen heures actives : **1.5 pts**
- Spread off-hours : jusqu'à 4 pts (exclure du backtest via session filter)
- Pip : 1 pt = 1 USD
- Slippage estimé pour ordres ≤ 2 lots : 0.3 pts (majeure liquide, 0.2 × spread)
- Commission : 0 (Standard Account)

### US500 (S&P 500 CFD)
- Source : XTB.com → CFD → US500
- Symbole XTB : `US500` ou `SP500`
- Spread moyen : **0.5 pts**
- Pip : 0.1 pt (le S&P cote au dixième)
- Slippage : 0.1 pt (majeure, 0.2 × spread)
- Commission : 0 (Standard Account)

### GER30 (DAX 40 CFD)
- Source : XTB.com → CFD → DE30
- Symbole XTB : `DE30` ou `GER40`
- Spread moyen : **1.0 pt**
- Pip : 1 pt = 1 EUR
- Slippage : 0.2 pt (majeure, 0.2 × spread)
- Commission : 0 (Standard Account)

### XAUUSD (Or spot)
- Source : XTB.com → Spot Gold → `GOLD`
- Spread moyen : **0.30 USD** (parfois 0.50 en news)
- Pip : 1 USD (convention "big figure" adoptée en v4)
- Slippage : 0.05 USD (majeure, 0.2 × spread)
- Commission : 0 (Standard Account)
- ⚠️ Ancien code v3 utilisait "25 pips" à 1 USD = 25 USD → 80× le spread réel

### XAGUSD (Argent spot)
- Source : XTB.com → Spot Silver → `SILVER`
- Spread moyen : **0.025 USD** (= 2.5 cents)
- Pip : 0.001 USD (pipette, convention cohérente avec XTB)
- Slippage : 0.01 USD (mineure, 0.5 × spread)
- Commission : 0 (Standard Account)
- ⚠️ Ancien code v3 utilisait "30 pips" sans pip_size cohérent → blow-up immédiat

### USOIL (WTI Crude CFD)
- Source : XTB.com → CFD → OIL.WTI
- Spread moyen : **0.05 USD**
- Pip : 0.01 USD
- Slippage : 0.02 USD (mineure, 0.5 × spread)
- Commission : 0 (Standard Account)

### EURUSD (Forex — nouveau v4)
- Source : XTB.com → Forex → EURUSD
- Spread moyen : **0.7 pip** (heures Londres/NY)
- Pip : 0.0001 (standard forex, 4ème décimale)
- pip_value_eur : 10 EUR par lot standard (100 000) — taux EUR/USD ≈ 1.0875
- Slippage : 0.2 pip (majeure, 0.2 × spread)
- Commission : 0 (Standard Account)

## Impact attendu

Sur Donchian US30 D1 (91 trades en test H06) :
- Coût v3 : 91 × 8 pts × 0.92 € × 2.17 lots = **1 451 €** sur 10 000 € = 14.5 % du capital absorbé en frais
- Coût v4 : 91 × 1.8 pts × 0.92 € × 2.17 lots = **327 €** sur 10 000 € = 3.3 %
- Économie estimée : **−11.2 % du capital** sur 18 mois → **Sharpe brut probable +1.5**

## Règle de slippage

| Classe d'actif | Slippage (fraction du spread) | Exemples |
|---|---|---|
| Majeures liquides | 0.2 × spread | US30, US500, GER30, EURUSD, XAUUSD |
| Mineures | 0.5 × spread | XAGUSD, USOIL |
| Crypto | 1.0 × spread | BTCUSD, ETHUSD |

Positions ≤ 5 lots, hors news. Le slippage réel est stochastique ; un modèle uniform[min, max] pourra être ajouté en phase B.

## Limites
- Coûts basés sur des **moyennes** XTB Standard Account documentées publiquement.
- La réalité oscille selon volatilité, news, heure de la journée.
- Le slippage est **stochastique** (cf. prompt 09 du plan v3) — un slippage stochastique uniform[min, max] sera implémenté dans la phase B si nécessaire.
- Ne pas confondre **XTB Standard** (spreads variables, 0 commission) et **XTB Pro** (spreads serrés, commission ~3.5 USD/lot).
- **À reconfirmer 1× par trimestre** (les spreads XTB peuvent changer).

## Convention pip_size v4

Pour éviter la confusion d'unité qui a causé les surestimations massives en v3 :

| Actif | pip_size | Justification |
|---|---|---|
| US30 | 1.0 | 1 point = 1 USD |
| US500 | 0.1 | Le S&P cote au dixième de point |
| GER30 | 1.0 | 1 point DAX = 1 EUR |
| XAUUSD | 1.0 | 1 big figure = 1 USD |
| XAGUSD | 0.001 | 1 pipette SILVER = 0.001 USD |
| USOIL | 0.01 | 1 cent WTI = 0.01 USD |
| EURUSD | 0.0001 | 1 pip forex = 4ème décimale |

## Ratio coût/SL par actif

Vérification que `total_cost_pips / sl_points ≤ 0.10` (sinon stratégie impossible).

| Actif | total_cost_pips | sl_points | ratio | OK ? |
|---|---|---|---|---|
| US30 | 1.80 | 100 | 1.8 % | ✅ |
| US500 | 0.60 | 100 | 0.6 % | ✅ |
| GER30 | 1.20 | 200 | 0.6 % | ✅ |
| XAUUSD | 0.35 | 10 | 3.5 % | ✅ |
| XAGUSD | 0.035 | 150 | 0.02 % | ✅ |
| USOIL | 0.07 | 100 | 0.07 % | ✅ |
| EURUSD | 0.90 | 10 | 9.0 % | ✅ (limite) |

> EURUSD est à 9 % — acceptable pour un majeur forex avec spread 0.7 pip et SL 10 pips. Le ratio s'améliorera si le SL est élargi ou le spread resserré en Pro Account.

## Comment vérifier en compte démo

1. Ouvrir un compte démo XTB (gratuit, sans engagement).
2. MT4/MT5 → Market Watch → clic droit sur chaque symbole → "Specifications".
3. Noter `Spread` (en pips ou points selon symbole).
4. Comparer avec `ASSET_CONFIGS` v4.
5. Si écart > 30 %, corriger et noter dans ce document.
