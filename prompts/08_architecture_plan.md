# Prompt 08 — Plan d'architecture H07 : Stratégies trend-following alternatives

**Date** : 2026-05-14
**Mode** : Architect → puis Code
**Statut** : 📋 Planifié

---

## 0. Contexte H06 → H07

- **H06 verdict** : 🔴 NO-GO — 0 actif GO sur 7. Univers réduit à US30.
- **H07 scope** : Tester 4 stratégies alternatives sur le(s) actif(s) GO de H06.
- **Actif cible** : US30 D1 (seul actif avec un edge Donchian historique, même si H06 l'a dégradé).
- **Note** : XAUUSD Sharpe brut +1.46 mais WR 22.5% — candidat secondaire si US30 échoue.

---

## 1. Audit des 4 stratégies existantes

Les 4 fichiers existent dans `app/strategies/` (héritage v2). Chacun a des défauts critiques.

### 1.1 `dual_ma.py` — Dual Moving Average

| Problème | Sévérité | Détail |
|----------|----------|--------|
| Colonnes PascalCase | 🔴 Bloquant | `df["Close"]` → KeyError (`loader.py` normalise en lowercase) |
| Pas de `.shift(1)` | 🔴 Bloquant | `sma_fast > sma_slow` utilise le close de la barre `t` pour décider du trade à l'ouverture de `t` → look-ahead. Les SMA sont calculées sur `close[t]` inclus, or le trade s'exécute au close de `t`. |
| `__init__` signature | 🟡 Mineur | `BaseStrategy.__init__` accepte `**params: int \| float`, `DualMovingAverage.__init__` n'est pas défini → utilise `self.params.get("fast", 10)` dans `generate_signals`. OK mais pas explicite. |

**Fix** :
```python
# Colonnes lowercase + shift(1) anti-look-ahead
sma_fast = df["close"].rolling(window=fast).mean()
sma_slow = df["close"].rolling(window=slow).mean()
signals = pd.Series(0, index=df.index, dtype=int)
signals[sma_fast > sma_slow] = 1
signals[sma_fast < sma_slow] = -1
return signals.shift(1).fillna(0).astype(int)
```

### 1.2 `keltner.py` — Keltner Channel

| Problème | Sévérité | Détail |
|----------|----------|--------|
| Colonnes PascalCase | 🔴 Bloquant | `df["Close"]`, `df["High"]`, `df["Low"]` → KeyError |
| Pas de `.shift(1)` | 🔴 Bloquant | `df["Close"] > upper` utilise le close courant |
| `_atr()` helper | 🟡 OK | Implémentation correcte, utilise `.shift(1)` sur close |

**Fix** :
```python
# Colonnes lowercase + shift(1) sur les signaux
ema = df["close"].ewm(span=period, adjust=False).mean()
atr_val = _atr(df["high"], df["low"], df["close"], period)
upper = ema + mult * atr_val
lower = ema - mult * atr_val
signals = pd.Series(0, index=df.index, dtype=int)
signals[df["close"] > upper] = 1
signals[df["close"] < lower] = -1
return signals.shift(1).fillna(0).astype(int)
```

### 1.3 `chandelier.py` — Chandelier Exit

| Problème | Sévérité | Détail |
|----------|----------|--------|
| Colonnes PascalCase | 🔴 Bloquant | `df["Close"]`, `df["High"]`, `df["Low"]` → KeyError |
| `.shift(1)` partiel | 🟡 OK | `highest.shift(1)` est correct. Mais `df["Close"] > long_trigger` n'a pas de shift supplémentaire → le close courant est comparé à un trigger basé sur `t-1`. Pas de leak car `long_trigger` utilise déjà `shift(1)`. |
| `_atr` import depuis `keltner` | 🟡 OK | Import interne, pas de duplication |

**Fix** : Uniquement les noms de colonnes → lowercase.

### 1.4 `parabolic.py` — Parabolic SAR

| Problème | Sévérité | Détail |
|----------|----------|--------|
| Colonnes PascalCase | 🔴 Bloquant | `df["High"]`, `df["Low"]`, `df["Close"]` → KeyError |
| Boucle `for` dans `_compute_psar` | 🟡 Acceptable | PSAR est stateful par nature. La boucle est inévitable sans réécriture complexe. Optimisée avec pré-allocation NumPy. |
| Anti-look-ahead | 🟡 OK | La boucle traite barre par barre de `i=1` à `n-1`, n'utilisant que `high[:i]`, `low[:i]` → pas de look-ahead. |

**Fix** : Uniquement les noms de colonnes → lowercase.

### 1.5 `base.py` — Classe abstraite

| Problème | Sévérité | Détail |
|----------|----------|--------|
| `__init__` accepte `**params` | 🟢 OK | Flexible, les sous-classes lisent via `self.params.get()` |
| `generate_signals` abstraite | 🟢 OK | Signature conforme |

---

## 2. Problème transverse : convention de colonnes

### Contexte

- `app/data/loader.py` (Prompt 03) normalise TOUTES les colonnes en **lowercase** : `open`, `high`, `low`, `close`, `volume`, `spread`.
- `app/backtest/deterministic.py` lit `df["Close"].values`, `df["High"].values`, `df["Low"].values` — mais il a un fallback :
  ```python
  if "Time" in df.columns:
      df = df.set_index("Time")
  ```
  Les colonnes `Close`, `High`, `Low` sont en PascalCase dans la v2 originale. Le loader v3 produit du lowercase → le backtest échouerait aussi.

**Wait — comment H06 a-t-il fonctionné alors ?**

Le script `run_h06_donchian_multi_asset.py` utilise `run_deterministic_backtest(df=df_train, ...)`. Le `df_train` vient de `load_asset(asset, "D1")`, qui produit des colonnes lowercase. MAIS `run_deterministic_backtest` accède `df["Close"].values` (PascalCase).

**H06 a dû planter ou le loader n'a pas été utilisé.** Vérification nécessaire en mode Code.

### Décision architecturale

**Option A** : Modifier `run_deterministic_backtest` pour accepter les deux casses (robustesse).
**Option B** : Modifier les 4 stratégies pour utiliser lowercase, et `run_deterministic_backtest` aussi.

**Décision** : Option B — lowercase partout. C'est la convention v3 (`loader.py`). On ne maintient pas deux conventions. Le backtest engine sera patché en même temps.

---

## 3. Signature de `generate_signals` et intégration backtest

Les stratégies actuelles retournent des signaux **non shiftés** (le signal à `t` dit quoi faire à `t`). Le backtest stateful (`run_deterministic_backtest`) consomme `signals[t]` pour décider d'entrer au close de la barre `t`. 

**Règle anti-look-ahead** : Le signal à `t` ne doit utiliser que l'information ≤ `t-1`. Donc :
- SMA → calculée sur `close[t]`, mais le signal pour `t` doit utiliser `SMA[t-1]`.
- Solution : `.shift(1)` en dernière ligne de `generate_signals`.

**Vérification** : `DonchianBreakout.generate_signals` utilise `high_roll.shift(1)` → déjà correct. Les 4 nouvelles stratégies doivent faire de même.

---

## 4. Grid search — paramètres exacts

D'après `docs/v3_roadmap.md` § H07 :

| Stratégie | Paramètres | Combinaisons |
|-----------|-----------|--------------|
| Dual MA | fast ∈ {5, 10, 20}, slow ∈ {50, 100, 200} | 9 |
| Keltner | period ∈ {10, 20, 50}, mult ∈ {1.5, 2.0, 2.5} | 9 |
| Chandelier | period ∈ {11, 22, 44}, atr_mult ∈ {2.0, 3.0, 4.0} | 9 |
| Parabolic SAR | step ∈ {0.01, 0.02, 0.03}, max ∈ {0.1, 0.2, 0.3} | 9 |

**Total** : 36 combinaisons × 1 actif (US30) = 36 backtests. Léger.

**Sélection** : Meilleur Sharpe train (≤ 2022). Puis évaluation val (2023) et test (≥ 2024).

---

## 5. Plan d'exécution (mode Code)

### Étape 1 — Fix des 4 stratégies (lowercase + shift)
- [ ] `app/strategies/dual_ma.py` : `df["close"]`, `.shift(1).fillna(0).astype(int)`
- [ ] `app/strategies/keltner.py` : `df["close"]`, `df["high"]`, `df["low"]`, `.shift(1).fillna(0).astype(int)`
- [ ] `app/strategies/chandelier.py` : `df["close"]`, `df["high"]`, `df["low"]`
- [ ] `app/strategies/parabolic.py` : `df["high"]`, `df["low"]`, `df["close"]`

### Étape 2 — Patch `run_deterministic_backtest` (lowercase)
- [ ] `app/backtest/deterministic.py` : `df["Close"]` → `df["close"]`, `df["High"]` → `df["high"]`, `df["Low"]` → `df["low"]`

### Étape 3 — Tests unitaires (×4)
- [ ] `tests/unit/test_strategy_dual_ma.py` : ≥ 3 tests dont anti-look-ahead
- [ ] `tests/unit/test_strategy_keltner.py` : ≥ 3 tests dont anti-look-ahead
- [ ] `tests/unit/test_strategy_chandelier.py` : ≥ 3 tests dont anti-look-ahead
- [ ] `tests/unit/test_strategy_parabolic.py` : ≥ 3 tests (anti-look-ahead via vérification que SAR[t] n'utilise que high[:t], low[:t])

### Étape 4 — Script `run_h07_strategies_alt.py`
- [ ] Créer `scripts/run_h07_strategies_alt.py` (template : `run_h06_donchian_multi_asset.py`)
- [ ] Grid search 4 stratégies × US30 D1
- [ ] Corrélation rolling 60j des retours quotidiens vs Donchian
- [ ] Sortie : `predictions/h07_strategies_alt.json`

### Étape 5 — Rapport + JOURNAL.md
- [ ] `docs/v3_hypothesis_07.md`
- [ ] Mise à jour `JOURNAL.md`

---

## 6. Points de vigilance

1. **Le Donchian lui-même a planté en H06** (US30 Sharpe −0.09 avec coûts v3). Les stratégies alternatives partent avec le même désavantage de coûts. Attentes réalistes : Sharpe probablement négatif aussi.
2. **Parabolic SAR** : la boucle `for` est inévitable mais sur US30 D1 (~2500 barres) c'est négligeable (~2 ms).
3. **Corrélation avec Donchian** : si une stratégie est trop corrélée (> 0.7), elle n'apporte pas de diversification. Le but n'est pas de remplacer Donchian mais de le compléter.
4. **Si 0 stratégie alternative ne passe** → NO-GO partiel (documenté), on continue vers H08 avec Donchian seul sur US30.

---

## 7. Dépendances de mode

Ce plan est rédigé en mode Architect. L'implémentation nécessite un passage en mode **Code** (édition de `.py`).
