---
name: splittemporelstrict
description: Split temporel strict pour séries financières — jamais de shuffle, toujours chronologique.
---

# Split Temporel Strict — ML Finance

## Règle cardinale

**Tout split train/test/OOS sur des séries temporelles financières DOIT être strictement chronologique. Le désordre temporel = look-ahead bias = modèle inutilisable en réel.**

## Instructions

### 1. Split manuel par date (méthode recommandée)

```python
# ✅ CORRECT : split par cutoff datetime
train_mask = df.index < pd.to_datetime("2024-01-01")
test_mask = df.index >= pd.to_datetime("2024-01-01")
X_train = df.loc[train_mask, X_cols]
y_train = df.loc[train_mask, "Target"]
X_test = df.loc[test_mask, X_cols]
y_test = df.loc[test_mask, "Target"]
```

### 2. TimeSeriesSplit (scikit-learn) — uniquement si nécessaire

```python
# ✅ CORRECT : TimeSeriesSplit avec gap
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=48)  # gap = purge_hours
for train_idx, test_idx in tscv.split(X):
    # train_idx toujours < test_idx, pas de contamination future
```

### 3. Interdictions absolues

```python
# ❌ JAMAIS : shuffle aléatoire sur des séries temporelles
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ❌ JAMAIS : KFold standard (shuffle implicite)
from sklearn.model_selection import KFold

# ❌ JAMAIS : GroupShuffleSplit sans contrainte temporelle
```

### 4. Vérification post-split obligatoire

```python
# Toujours exécuter ce check après chaque split
assert X_train.index.max() < X_test.index.min(), \
    "LOOK-AHEAD DETECTÉ : données train postérieures aux données test"
```

### 5. Features rolling/shift — calculer APRES le split

```python
# ✅ CORRECT : fit scaler uniquement sur train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)                        # fit sur train SEULEMENT
X_train_scaled = scaler.transform(X_train)  # transform train
X_test_scaled = scaler.transform(X_test)    # transform test avec params train

# ❌ INCORRECT : fit sur tout le dataset avant split
scaler = StandardScaler()
scaler.fit(X)  # fuit l'information du test dans le train
```

### 6. Purge anti-overlap (López de Prado)

```python
# Toujours ajouter une purge entre train et OOS
from datetime import timedelta
purge_hours = max(window, 48)  # window = horizon de la triple barrière
train_cutoff = pd.to_datetime(f"{train_end_year + 1}-01-01") - timedelta(hours=purge_hours)
```

## Contrôles automatiques

À chaque fin de fonction de split, vérifier systématiquement :

1. `assert df.index.is_monotonic_increasing` — l'index trié chronologiquement
2. `assert isinstance(df.index, pd.DatetimeIndex)` — index datetime obligatoire
3. `assert X_train.index.max() < X_test.index.min()` — pas de chevauchement
4. Logger : `logger.info("Split: train=%d, test=%d, cutoff=%s", len(X_train), len(X_test), cutoff)`
