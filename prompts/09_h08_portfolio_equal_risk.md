# Prompt 09 — H08 : Portfolio equal-risk multi-actif

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (avec verdicts H06 + H07)
3. `prompts/08_h07_strategies_alt.md`
4. `docs/v3_roadmap.md` section "H08"

## Objectif
**Question H08** : La combinaison de N stratégies × M actifs décorrélés en portefeuille **equal risk weight** produit-elle un Sharpe portfolio ≥ 1.0 ?

## Definition of Done (testable)
- [ ] `app/portfolio/constructor.py` contient `build_equal_risk_portfolio(returns: dict[str, pd.Series], target_vol: float = 0.10, leverage_max: float = 2.0) -> pd.Series` :
  - Calcule la volatilité rolling 60j de chaque sleeve.
  - Position size = (target_vol × capital) / (σ × √252)
  - Allocation = capital / N_sleeves
  - Rebalance quotidien
- [ ] `app/portfolio/correlation.py` contient `filter_correlated_sleeves(returns: dict, cap: float = 0.7) -> list[str]` qui désactive la sleeve la moins bonne en Sharpe rolling 6M si ρ > cap.
- [ ] `scripts/run_h08_portfolio.py` charge tous les sleeves GO (depuis `JOURNAL.md`), construit le portfolio, calcule l'equity portfolio test ≥ 2024.
- [ ] `validate_edge(equity_portfolio, n_trials=8)` est appelé.
- [ ] Rapport `docs/v3_hypothesis_08.md` avec :
  - Sleeves inclus, sleeves exclus (et raison)
  - Sharpe portfolio train / val / test
  - DD portfolio
  - Comparaison vs Sharpe moyen des sleeves individuels
- [ ] `predictions/h08_portfolio.json` produit.
- [ ] `tests/unit/test_portfolio_constructor.py` : ≥ 5 tests (equal risk weight, cap correlation, vol targeting, leverage cap, rebalance).
- [ ] `rtk pytest tests/unit/test_portfolio_constructor.py -v` passe.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS introduire d'allocation dynamique basée sur le Sharpe (c'est H15).
- Ne PAS introduire la corrélation rolling adaptative (c'est H13).
- Ne PAS oublier les coûts réels par sleeve (déjà calculés dans `app/config/instruments.py`).
- Ne PAS utiliser un slippage constant — utiliser un slippage stochastique (cf. Étape 5).
- Ne PAS oublier le tie-breaker du filtre corrélation (cf. Étape 4).

## Étapes

### Étape 1 — Charger les retours par sleeve
Une sleeve = (stratégie, actif). Format : `returns["donchian_US30"] = pd.Series` indexée par jour.

### Étape 2 — Vol targeting par sleeve
```python
def vol_target_weights(returns: dict[str, pd.Series], target_vol: float = 0.10) -> pd.DataFrame:
    weights = {}
    for name, r in returns.items():
        sigma_daily = r.rolling(60).std()
        sigma_annual = sigma_daily * np.sqrt(252)
        weights[name] = (target_vol / sigma_annual).clip(upper=2.0)  # leverage cap
    return pd.DataFrame(weights).fillna(0)
```

### Étape 3 — Allocation equal risk
```python
def equal_risk_allocation(returns: dict, target_vol: float) -> pd.DataFrame:
    w = vol_target_weights(returns, target_vol)
    n = len(returns)
    return w / n
```

### Étape 4 — Filtre correlation
Matrice de corrélation rolling 60j. Si ρ_ij > 0.7 pour deux sleeves, garder celle avec le meilleur Sharpe rolling 6M.

### Étape 4 bis — Tie-breaker du filtre corrélation
Si plusieurs sleeves ont un Sharpe rolling 6M identique (à 1e-4 près) :
1. Garder celle avec le moins de trades (proxy : moins de coûts cumulés).
2. Si encore tie : ordre alphabétique de la clé sleeve (déterministe).

### Étape 5 — Slippage stochastique et spread asymétrique
Étendre `app/config/instruments.py` :
```python
@dataclass(frozen=True)
class AssetConfig:
    name: str
    spread_bid_ask: float       # spread moyen en points
    slippage_min: float          # min en points
    slippage_max: float          # max en points
    point_value_eur: float
    contract_size: float
    ...
```

Dans `app/backtest/simulator.py`, à chaque trade :
```python
from app.core.seeds import set_global_seeds  # déjà appelé en haut du script
rng = np.random.default_rng(42)

def apply_costs(entry_price: float, direction: int, asset_cfg: AssetConfig, rng) -> float:
    spread_cost = asset_cfg.spread_bid_ask / 2  # half-spread par côté
    slippage = rng.uniform(asset_cfg.slippage_min, asset_cfg.slippage_max)
    adjusted = entry_price + direction * (spread_cost + slippage)
    return adjusted
```

Le `rng` est **partagé** entre tous les trades du backtest pour reproductibilité. Le seed est fixé par `set_global_seeds()` en haut du script.

### Étape 6 — Equity portfolio
```python
portfolio_returns = (weights * pd.DataFrame(returns)).sum(axis=1)
equity = (1 + portfolio_returns).cumprod()
```

### Étape 7 — Validation
```python
from app.testing.snooping_guard import read_oos

report = validate_edge(equity, all_trades_concat, n_trials=8)
read_oos(prompt="09", hypothesis="H08", sharpe=report.metrics["sharpe"], n_trades=len(all_trades_concat))
```

## Critères go/no-go
- **GO prompt 10** si : `report.metrics["sharpe"] ≥ 0.5` (on continue à raffiner en Phase 2 même si on n'a pas encore Sharpe ≥ 1).
- **NO-GO**, revenir à : prompt 08 si Sharpe portfolio < 0. Probable que les stratégies sont mal sélectionnées.
