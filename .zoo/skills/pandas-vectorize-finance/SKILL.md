---
name: pandas-vectorize-finance
description: Vectorisation pandas pour la finance — zéro boucle Python, priorité .shift()/.rolling(), astype early.
---

# Pandas Vectorize Finance — Calculs Financiers Performants

## Règle cardinale

**ZÉRO boucle `for i in range(len(df))` sur des DataFrames de prix. ZÉRO `.apply()`. ZÉRO `.iterrows()`. Tout calcul financier doit être vectorisé via les primitives pandas/NumPy.**

## Instructions

### 1. Hiérarchie des opérations vectorisées

| Priorité | Opération | Usage |
|----------|-----------|-------|
| 1 | `.shift()` | Lag features, rendements |
| 2 | `.rolling().agg()` | Moyennes mobiles, volatilité |
| 3 | `.expanding()` | Statistiques cumulatives |
| 4 | `pandas_ta` | Indicateurs techniques standards |
| 5 | `np.log()`, `np.where()` | Opérations mathématiques |
| 6 | `.groupby().transform()` | Calculs par groupe avec alignement d'index |

### 2. Patterns vectorisés obligatoires

```python
# ✅ Rendement log
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

# ✅ Lag features (N périodes)
for lag in [1, 2, 3, 5, 8, 13]:
    df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)

# ✅ Rolling stats fenêtrées
df["MA_20"] = df["Close"].rolling(20).mean()
df["Vol_20"] = df["Log_Return"].rolling(20).std()
df["MaxHigh_24"] = df["High"].rolling(24).max()
df["MinLow_24"] = df["Low"].rolling(24).min()

# ✅ Z-score fenêtré
df["Close_ZScore_24"] = (
    (df["Close"] - df["Close"].rolling(24).mean())
    / df["Close"].rolling(24).std()
)

# ✅ Variation en %
for period in [1, 2, 5]:
    df[f"Close_Pct_{period}"] = df["Close"].pct_change(period)

# ✅ Ratio rendement/volatilité
df["RSI_14"] = ta.rsi(df["Close"], length=14)
df["ATR_Norm"] = ta.atr(df["High"], df["Low"], df["Close"], length=14) / df["Close"]
```

### 3. Interdictions absolues

```python
# ❌ JAMAIS : boucle for sur les lignes
for i in range(len(df)):
    df.loc[i, "MA"] = df["Close"].iloc[max(0,i-20):i].mean()

# ❌ JAMAIS : .apply() sur un axe rows
df["MA"] = df.apply(lambda row: ..., axis=1)

# ❌ JAMAIS : .iterrows()
for _, row in df.iterrows():
    ...

# ❌ JAMAIS : .at/.iat dans une boucle
for i in range(len(df)):
    df.at[i, "col"] = ...
```

### 4. Optimisation dtype et mémoire

```python
# ✅ Astype early — immédiatement après ingestion
df = df.astype({
    "Open": "float64", "High": "float64", "Low": "float64",
    "Close": "float64", "Volume": "float64",
})

# ✅ Downcast quand possible (séries longues)
df["Volume"] = pd.to_numeric(df["Volume"], downcast="float")

# ✅ Éviter les copies inutiles
result = df[["Close"]].copy()  # copie explicite, utile SEULEMENT si modification
result["MA"] = df["Close"].rolling(20).mean()  # pas de .copy() ici
```

### 5. merge_asof pour multi-timeframe

```python
# ✅ CORRECT : merge_asof direction='backward'
h1_d1 = pd.merge_asof(
    h1.sort_index(),
    d1_features.sort_index(),
    left_index=True, right_index=True,
    direction="backward",  # la D1 est connue au close de la D1 précédente
)

# ❌ INCORRECT : merge classique = perte de données si timestamps non alignés
merged = h1.join(d1_features)  # → NaN si les timestamps ne correspondent pas
```

### 6. Calculs groupby sans boucle

```python
# ✅ CORRECT : groupby.transform pour aligner l'index
df["Daily_Mean"] = df.groupby(df.index.date)["Close"].transform("mean")

# ❌ INCORRECT : boucle sur les groupes
for date, group in df.groupby(df.index.date):
    df.loc[group.index, "Daily_Mean"] = group["Close"].mean()
```

## Checklist de performance

1. `df.info(memory_usage='deep')` — vérifier la mémoire consommée
2. Pas de `object` dtype sur les colonnes numériques
3. `len(df) > 100_000` → envisager `pd.to_numeric(downcast='float')`
4. Toute opération > 1s sur >100k lignes suspecte — probablement une boucle cachée
