# Prompt 14 — H13 : Pondération portfolio correlation-aware

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (verdict H08)
3. `prompts/09_h08_portfolio_equal_risk.md`
4. `docs/v3_roadmap.md` section "H13"

## Objectif
**Question H13** : Pénaliser les sleeves trop corrélés (ρ rolling 60j > 0.7) améliore-t-il le Sharpe portfolio par rapport à l'equal risk weight ?

## Definition of Done (testable)
- [ ] `app/portfolio/correlation.py` étendu avec :
  - `rolling_correlation_matrix(returns: pd.DataFrame, window: int = 60) -> dict[date, pd.DataFrame]`
  - `correlation_aware_weights(returns: pd.DataFrame, cap: float = 0.7, rolling_sharpe_window: int = 126) -> pd.DataFrame`
- [ ] Logique : à chaque jour t, calculer la matrice de corrélation rolling 60j. Pour chaque paire (i, j) avec ρ > 0.7, désactiver la sleeve avec le moins bon Sharpe rolling 6M (~126 jours).
- [ ] Rebalance hebdomadaire (pour éviter le flip-flap quotidien).
- [ ] `scripts/run_h13_corr_aware.py` rejoue le portfolio H08 avec cette pondération.
- [ ] Comparer Sharpe portfolio, max DD, turnover.
- [ ] Rapport `docs/v3_hypothesis_13.md`.

## NE PAS FAIRE
- Ne PAS calculer la corrélation sur < 30 jours (instable).
- Ne PAS rebalancer quotidiennement (turnover trop élevé).
- Ne PAS désactiver une sleeve sans alternative — si toutes désactivées, garder la meilleure.

## Étapes

### Étape 1 — Matrice rolling
```python
def rolling_correlation_matrix(returns_df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Retourne une Series indexée par date, chaque valeur = matrice de corrélation."""
    return returns_df.rolling(window).corr().dropna()
```

### Étape 2 — Pondération correlation-aware
```python
def correlation_aware_weights(returns_df: pd.DataFrame, cap: float = 0.7) -> pd.DataFrame:
    weekly = returns_df.resample("W-FRI").last()  # date de rebalance
    rolling_sharpe = returns_df.rolling(126).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )

    weights = pd.DataFrame(index=weekly.index, columns=returns_df.columns, data=1.0)
    for date in weekly.index[60:]:
        corr_mat = returns_df.loc[:date].tail(60).corr()
        active = set(returns_df.columns)
        for i in returns_df.columns:
            for j in returns_df.columns:
                if i >= j:
                    continue
                if corr_mat.loc[i, j] > cap and i in active and j in active:
                    drop = i if rolling_sharpe.loc[date, i] < rolling_sharpe.loc[date, j] else j
                    active.discard(drop)
        for c in returns_df.columns:
            weights.loc[date, c] = 1.0 if c in active else 0.0

    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
    return weights.reindex(returns_df.index, method="ffill")
```

### Étape 3 — Backtest comparatif
- Portfolio H08 baseline (equal risk weight)
- Portfolio H13 (correlation-aware)
- Mêmes données, mêmes coûts.

## Critères go/no-go
- **GO prompt 15** si : Sharpe portfolio H13 ≥ Sharpe H08 + 0.1 OU réduction max DD ≥ 20 %.
- **NO-GO**, conserver H08 et continuer.
