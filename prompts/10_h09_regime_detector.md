# Prompt 10 — H09 : Détecteur de régime Trending/Ranging

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (verdicts H06–H08)
3. `prompts/09_h08_portfolio_equal_risk.md`
4. `docs/v3_roadmap.md` section "H09"

## Objectif
**Question H09** : Peut-on classifier de manière **robuste** le régime de marché (Trending vs Ranging) par actif pour conditionner l'exécution des stratégies trend-following ?

## Definition of Done (testable)
- [ ] `app/features/regime.py` contient `classify_regime(df: pd.DataFrame, period: int = 60) -> pd.Series` retournant 1 (Trending) ou 0 (Ranging). Combinaison de :
  1. `ADX(14) > 25`
  2. `|Close − SMA(200)| / ATR(14) > 2.0`
  3. `efficiency_ratio(period) > 0.3`
  - Trending si **au moins 2 sur 3** sont vrais.
- [ ] Test anti-look-ahead : `assert_no_look_ahead(classify_regime, df_close)`.
- [ ] Validation par actif GO de H06 :
  - % de temps en Trending sur train ≤ 2022 (doit être 30 % à 70 % — sinon classifieur inutile)
  - Stabilité OOS val 2023 : ratio Trending similaire à train (±20 %)
  - % de trades Donchian GAGNANTS en Trending vs Ranging (sur train) — devrait être ≥ 70 % en Trending
- [ ] Rapport `docs/v3_hypothesis_09.md` avec un tableau par actif (% Trending train, % Trending val, % gagnants en Trending, % gagnants en Ranging).
- [ ] `tests/unit/test_regime.py` : ≥ 5 tests.
- [ ] `JOURNAL.md` mis à jour avec verdict GO/NO-GO.

## NE PAS FAIRE
- Ne PAS appliquer le filtre régime aux stratégies (c'est H10).
- Ne PAS utiliser de classifieur ML pour le régime (ADX + ratio + ER est suffisant et robuste — un ML overfit beaucoup).
- Ne PAS faire de fenêtre flottante > 200 jours sans `min_periods` (sinon NaN trop nombreux).

## Étapes

### Étape 1 — Efficiency Ratio (Kaufman)
```python
def efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (direction / volatility).clip(0, 1)
```

### Étape 2 — `classify_regime`
```python
def classify_regime(df: pd.DataFrame, period: int = 60) -> pd.Series:
    adx_val = adx(df, 14)
    sma200 = df["close"].rolling(200, min_periods=50).mean()
    atr14 = atr(df, 14)
    distance = (df["close"] - sma200).abs() / atr14
    er = efficiency_ratio(df["close"], period)

    cond1 = adx_val > 25
    cond2 = distance > 2.0
    cond3 = er > 0.3

    score = cond1.astype(int) + cond2.astype(int) + cond3.astype(int)
    return (score >= 2).shift(1).fillna(False).astype(int)  # shift(1) anti-look-ahead
```

### Étape 3 — Diagnostic par actif
```python
def diagnose_regime(asset: str):
    df = load_asset(asset, "D1")
    df_train = df.loc[:"2022-12-31"]
    df_val = df.loc["2023-01-01":"2023-12-31"]

    regime_train = classify_regime(df_train)
    regime_val = classify_regime(df_val)

    pct_train = regime_train.mean()
    pct_val = regime_val.mean()

    # Trades Donchian gagnants en Trending vs Ranging (sur train)
    # ... charger les trades Donchian de H06 ...
```

### Étape 4 — Critères GO globaux
- ≥ 70 % des trades Donchian gagnants sont en Trending (sur train)
- Le filtre Trending réduit le nombre de trades de ≤ 50 % (sinon trop agressif)
- Stabilité OOS : `|pct_val - pct_train| ≤ 0.20`

Si ces 3 conditions sont remplies pour ≥ 50 % des actifs GO : régime GO global.

### Étape 5 — Tests
- Cas synthétique : trend pur (close = range(1000)) → Trending = 1 quasi tout le temps.
- Range pur (close oscille autour d'une moyenne) → Trending = 0.
- Anti-look-ahead.

## Critères go/no-go
- **GO prompt 11** si : régime GO sur ≥ 50 % des actifs.
- **NO-GO**, revenir à : ce prompt si régime ne discrimine pas mieux que le hasard. Abandonner le filtre régime → aller directement au prompt 12 (méta-labeling sans régime).
