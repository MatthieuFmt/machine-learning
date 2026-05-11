---
name: triple-barrier-methodology
description: Méthodologie Triple Barrière (López de Prado) — labels directionnels, pas de look-ahead, gestion des NaN.
---

# Triple Barrier Methodology — Labelling Directionnel

## Règle cardinale

**La target pour la barre `i` doit être calculée UNIQUEMENT à partir des barres `i+1` à `i+window`. Jamais d'information de la barre `i` elle-même (hormis le prix d'entrée). Le label est directionnel : 1 (LONG gagnant), -1 (SHORT gagnant), 0 (neutre/timeout/ambigu).**

## Instructions

### 1. Définition des labels

```python
# Label = 1   → LONG gagnant : TP touché AVANT SL (ou rien touché mais Close > entry)
# Label = -1  → SHORT gagnant : TP touché AVANT SL (ou rien touché mais Close < entry)
# Label = 0   → Neutre :
#                 - Rien touché → timeout (pas de direction claire)
#                 - Les deux directions gagnent → ambigu
#                 - SL touché avant TP dans les deux directions → perdant
```

### 2. Paramètres par instrument

```python
# EURUSD
apply_triple_barrier(df, tp_pips=20.0, sl_pips=10.0, window=24, pip_size=0.0001)

# BTCUSD (1$ = 1 unité, pas de pip forex)
apply_triple_barrier(df, tp_pips=400.0, sl_pips=200.0, window=24, pip_size=1.0)
```

### 3. Gestion des NaN

```python
# Les WINDOW dernières barres sont NaN (pas assez de forward bars)
# → Elles seront dropnées automatiquement par dropna(subset=["Target"])
# → Ne PAS les forward-fill, ne PAS les remplacer par 0

targets = apply_triple_barrier(df, ...)
n_before = len(targets)
targets = targets[~np.isnan(targets)]
n_after = len(targets)
logger.info("Triple barrière : %d → %d labels (%.1f%% loss)", n_before, n_after,
            (n_before - n_after) / n_before * 100)
```

### 4. Distribution cible

```python
# Distribution saine :
#   ~30-35% LONG (1)
#   ~30-35% SHORT (-1)
#   ~30-35% NEUTRE (0)
#
# Si une classe > 50% → ajuster tp_pips ou sl_pips
# Si NEUTRE > 50% → réduire window ou augmenter tp_pips
dist = label_distribution(targets)
logger.info("Distribution: -1=%.1f%%, 0=%.1f%%, 1=%.1f%%", dist["-1"], dist["0"], dist["1"])
```

### 5. Validation anti-look-ahead

```python
# Vérifier que les labels ne contiennent pas d'info post-cutoff
def validate_triple_barrier_no_leak(df: pd.DataFrame, cutoff: pd.Timestamp) -> bool:
    """Vérifie qu'aucune barre avant cutoff n'a un label basé sur des barres après cutoff."""
    train = df[df.index < cutoff]
    # Les 'window' dernières barres de train doivent être NaN
    last_train = train.tail(window)
    if last_train["Target"].notna().any():
        n_leak = int(last_train["Target"].notna().sum())
        logger.error("TRIPLE BARRIER LEAK: %d labels train basés sur barres post-cutoff", n_leak)
        return False
    return True
```

### 6. Optimisation de la boucle

```python
# ✅ CORRECT : sortie anticipée si les deux directions sont résolues
for j in range(1, window + 1):
    # ... check barriers ...
    if (long_win or long_dead) and (short_win or short_dead):
        break  # plus besoin de continuer

# ✅ CORRECT : utiliser les arrays NumPy pas les Series pandas dans la boucle
highs = df["High"].values   # .values → ndarray, plus rapide
lows = df["Low"].values
closes = df["Close"].values
```

## Checklist finale

1. `tp_pips > sl_pips` — ratio minimum 2:1 (recommandé)
2. `window` assez grand pour que TP soit atteignable
3. Distribution des 3 classes entre 25% et 40% chacune
4. Les `window` dernières lignes de tout split train sont NaN
5. `pip_size` correct pour l'instrument
