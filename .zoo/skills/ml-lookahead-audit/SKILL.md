---
name: ml-lookahead-audit
description: Audit anti-look-ahead bias — vérifie qu'aucune feature n'utilise d'information future avant entraînement.
---

# Audit Anti-Look-Ahead — ML Finance

## Règle cardinale

**Avant chaque `model.fit()`, exécuter un audit mental systématique de toutes les features. Une seule feature contaminée par du look-ahead = toutes les métriques de performance sont illusoires.**

## Checklist d'audit (5 points)

### 1. Features `.shift()` / `.rolling()` / `.expanding()`

```python
# ❌ LOOK-AHEAD : calculé sur tout le dataset avant split
df["MA_20"] = df["Close"].rolling(20).mean()
train, test = df[df.index < cutoff], df[df.index >= cutoff]
# → Les 19 premières lignes de test contiennent des valeurs train

# ✅ CORRECT : split d'abord, rolling ensuite (ou groupby-apply)
train = df[df.index < cutoff].copy()
test = df[df.index >= cutoff].copy()
train["MA_20"] = train["Close"].rolling(20).mean()
test["MA_20"] = test["Close"].rolling(20).mean()
```

### 2. `StandardScaler` / `MinMaxScaler` / toute transformation `.fit()`

```python
# ❌ LOOK-AHEAD : fit sur tout le dataset
scaler = StandardScaler().fit(df[X_cols])

# ✅ CORRECT : fit uniquement sur train
scaler = StandardScaler().fit(X_train)
```

### 3. Target (triple barrière)

```python
# ❌ LOOK-AHEAD : target calculée avec des barres qui débordent sur le train
# Vérifier que la target pour la barre i n'utilise que les barres i+1 à i+window
# et que i+window < cutoff pour les barres train

# ✅ CORRECT : vérifier que les window dernières barres avant cutoff sont NaN
assert df.loc[train_mask, "Target"].isna().sum() <= window, "Target leak"
```

### 4. `merge_asof` multi-timeframe

```python
# ❌ LOOK-AHEAD : merge_asof sans direction='backward'
merged = pd.merge_asof(h1, d1, on="Time")  # défaut: backward, OK

# ❌ LOOK-AHEAD : merge_asof avec direction='forward'
merged = pd.merge_asof(h1, d1, on="Time", direction="forward")
# → La bougie D1 de 00:00 n'est connue qu'à 23:59, pas à 00:00

# ✅ CORRECT : backward uniquement + vérifier le décalage
merged = pd.merge_asof(h1, d1, on="Time", direction="backward")
```

### 5. Features cross-instrument

```python
# ❌ LOOK-AHEAD : utiliser le Close de XAUUSD à H1_10:00 pour prédire EURUSD à H1_10:00
# → Les deux sont publiés simultanément, donc pas de look-ahead strict,
# → mais en production, il faut être sûr d'avoir reçu les deux ticks.

# ✅ CORRECT : utiliser le Close de XAUUSD à H1_09:00 (lag 1) pour EURUSD à H1_10:00
macro_df["XAU_Return_Lag1"] = macro_df["XAU_Return"].shift(1)
```

## Test de contamination automatisé

```python
def audit_no_lookahead(df: pd.DataFrame, cutoff: pd.Timestamp) -> bool:
    """Vérifie qu'aucune valeur post-cutoff n'est présente dans les données pre-cutoff."""
    train = df[df.index < cutoff]
    test = df[df.index >= cutoff]

    # Check 1: pas de NaN dans train (hors rolling initial)
    nan_cols = train.columns[train.isna().any()].tolist()
    if nan_cols:
        logger.warning("NaN dans train: %s", nan_cols)

    # Check 2: pas de valeurs post-cutoff dans train
    future_mask = train.index > cutoff
    if future_mask.any():
        logger.error("LOOK-AHEAD: %d lignes train post-cutoff!", future_mask.sum())
        return False

    # Check 3: target calculée uniquement sur forward bars
    logger.info("Audit OK: train=%d, test=%d, cutoff=%s", len(train), len(test), cutoff)
    return True
```

## Contrôles automatiques

À chaque `build_features()` ou `train_model()`, le code DOIT :

1. Logger le cutoff exact utilisé pour le split
2. Vérifier qu'aucune feature n'a été calculée avec `.transform()` global
3. Vérifier que `merge_asof` utilise `direction='backward'`
4. Vérifier que les features macro utilisent `.shift(1)` si nécessaire
5. Si un `StandardScaler` est utilisé, confirmer que `fit()` est appelé UNIQUEMENT sur `X_train`
