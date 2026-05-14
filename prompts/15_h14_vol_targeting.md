# Prompt 15 — H14 : Volatility targeting adaptatif

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/14_h13_correlation_weighting.md`
4. `docs/v3_roadmap.md` section "H14"

## Objectif
**Question H14** : Ajuster dynamiquement la taille des positions selon la volatilité réalisée 20j (cible 10 % annualisée) améliore-t-il le Sharpe portfolio par rapport à une position size fixe ?

## Definition of Done (testable)
- [ ] `app/portfolio/sizing.py` contient :
  - `vol_targeted_size(returns: pd.Series, target_vol: float = 0.10, leverage_cap: float = 2.0, lookback: int = 20) -> pd.Series`
  - `fixed_size(returns: pd.Series, capital_pct: float = 0.10) -> pd.Series` (baseline pour comparaison)
- [ ] `scripts/run_h14_vol_targeting.py` rejoue le portfolio H13 avec vol targeting vs fixed size.
- [ ] Comparer : Sharpe, vol réalisée portfolio, max DD, leverage moyen utilisé.
- [ ] Rapport `docs/v3_hypothesis_14.md`.

## NE PAS FAIRE
- Ne PAS calculer la vol sur < 10 jours (trop bruité).
- Ne PAS lever le leverage cap au-delà de 2.0 (risque de blow-up).
- Ne PAS oublier les coûts de transaction induits par le rebalance.

## Étapes

### Étape 1 — Fonction vol targeting
```python
def vol_targeted_size(
    returns: pd.Series,
    target_vol: float = 0.10,
    leverage_cap: float = 2.0,
    lookback: int = 20,
) -> pd.Series:
    realized_vol = returns.rolling(lookback).std() * np.sqrt(252)
    size = (target_vol / realized_vol).clip(upper=leverage_cap).fillna(1.0)
    return size.shift(1)  # appliquer la taille calculée à t-1 sur t (anti-look-ahead)
```

### Étape 2 — Backtest comparatif
- Portfolio H13 baseline (fixed size par sleeve)
- Portfolio H14 (vol targeted par sleeve)
- Comparer sur 2024-2025.

### Étape 3 — Tableau récapitulatif

| Métrique | H13 fixed | H14 vol-targeted | Delta |
|---|---|---|---|
| Sharpe | ... | ... | ... |
| Vol réalisée annu. | ... | ... | ... |
| Max DD | ... | ... | ... |
| Leverage moyen | 1.0 | ... | ... |

## Critères go/no-go
- **GO prompt 16** si : Sharpe H14 ≥ H13 + 0.05 OU réduction vol réalisée ≥ 10 %.
- **NO-GO partiel** : conserver H13. Documenter pourquoi vol targeting n'a pas aidé (probable : sleeves déjà peu volatiles).
