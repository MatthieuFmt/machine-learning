# Prompt 16 — H15 : Phase décisive — choix du timeframe

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (état complet de la recherche jusqu'ici)
3. `prompts/15_h14_vol_targeting.md`

## Objectif
**Décision irréversible** : choisir le timeframe (D1 / H4 / H1) sur lequel le bot va trader. La décision se base sur les résultats des top 3 stratégies sur chaque TF, validés par `validate_edge`.

> ⚠️ Ce prompt est un point de bascule. Une fois le TF choisi, tous les prompts suivants (17-24) supposent ce TF.

## Definition of Done (testable)
- [ ] Pour chaque TF disponible (D1 obligatoire, H4 si données dispo, H1 si données dispo) :
  - Top 3 stratégies (depuis `JOURNAL.md`) sont rejouées sur le TF
  - Coûts adaptés au TF (les coûts en pips/points sont identiques mais le nombre de trades diffère donc l'impact varie)
  - `validate_edge` calculé pour chaque (strategy, TF, asset)
- [ ] Tableau récapitulatif `predictions/h15_tf_decision.json` avec ranking :
  ```json
  {
    "D1": {"best_sharpe": 1.3, "best_trades_per_year": 35, "best_dsr": 0.5, "go_count": 3},
    "H4": {"best_sharpe": 1.5, "best_trades_per_year": 80, "best_dsr": 0.8, "go_count": 2},
    "H1": {"best_sharpe": 0.7, "best_trades_per_year": 200, "best_dsr": -0.1, "go_count": 0}
  }
  ```
- [ ] Sélection du TF qui maximise le score composite : `0.4 × Sharpe + 0.3 × DSR + 0.3 × (go_count / max_go_count)`.
- [ ] **Décision validée par l'utilisateur** avant de continuer.
- [ ] Rapport `docs/v3_hypothesis_15.md` avec justification chiffrée.
- [ ] `JOURNAL.md` mis à jour avec : « TF retenu : <TF> ».
- [ ] `app/config/timeframe.py` créé : constante `PRIMARY_TF: Literal["D1", "H4", "H1"] = "<TF retenu>"`.

## NE PAS FAIRE
- Ne PAS choisir un TF où Aucun (strategy, asset) ne passe `validate_edge`.
- Ne PAS choisir H1 si trades/an > 500 (trop de bruit + coûts).
- Ne PAS décider sans accord utilisateur explicite.
- Ne PAS multiplier les essais — c'est une décision UNIQUE.

## Étapes

### Étape 1 — Inventaire des TF dispo
```python
assets = discover_assets()
tfs_available = set()
for tfs in assets.values():
    tfs_available |= set(tfs)
```

### Étape 2 — Pour chaque TF
- Charger les données train/val/test
- Pour chaque (strategy, asset) du top 3 de H07/H11/H12 :
  - Rejouer le backtest
  - `validate_edge`

### Étape 3 — Score composite
Score normalisé entre 0 et 1 par TF.

### Étape 4 — Présenter à l'utilisateur
Format clair :

```
Résultats par TF :

D1 :
  - Sharpe portfolio : 1.3
  - DSR : 0.5 (p = 0.03)
  - Trades/an : 35
  - GO count : 3/3 stratégies passent

H4 :
  - Sharpe : 1.5
  ... etc

Recommandation : H4 (meilleur score composite)
Confirmer ? (oui / non / autre TF)
```

### Étape 5 — Inscrire dans config
```python
# app/config/timeframe.py
from typing import Literal
PRIMARY_TF: Literal["D1", "H4", "H1"] = "D1"  # ou H4 selon choix
```

## Critères go/no-go
- **GO prompt 17** si : un TF est retenu, validé par utilisateur, inscrit dans `app/config/timeframe.py`.
- **NO-GO**, revenir à : prompt 09 (portfolio) si aucun TF ne donne de configuration GO. Revoir les sleeves de base.
