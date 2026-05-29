# Prompt 08 — H07 : Stratégies trend-following alternatives

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (avec verdict H06)
3. `prompts/07_h06_us500_donchian.md`
4. `docs/v3_roadmap.md` section "H07"

## Objectif
**Question H07** : Existe-t-il d'autres stratégies trend-following déterministes qui surpassent Donchian ou sont décorrélées de Donchian sur les actifs GO de H06 ?

## Definition of Done (testable)
- [ ] Les 4 stratégies suivantes sont implémentées dans `app/strategies/` :
  - `dual_ma.py` : `DualMovingAverage(fast, slow)` — fast ∈ {5,10,20}, slow ∈ {50,100,200}
  - `keltner.py` : `KeltnerChannel(period, mult)` — period ∈ {10,20,50}, mult ∈ {1.5,2.0,2.5}
  - `chandelier.py` : `ChandelierExit(period, atr_mult)` — period ∈ {11,22,44}, atr_mult ∈ {2.0,3.0,4.0}
  - `parabolic.py` : `ParabolicSAR(step, max_)` — step ∈ {0.01,0.02,0.03}, max ∈ {0.1,0.2,0.3}
- [ ] Chaque stratégie hérite de `app/strategies/base.py` (existant v2) et expose `generate_signals(df) -> pd.Series` (valeurs -1, 0, +1).
- [ ] Tests unitaires pour chaque stratégie (`tests/unit/test_strategy_<name>.py`) : ≥ 3 tests dont 1 anti-look-ahead.
- [ ] `scripts/run_h07_strategies_alt.py` lance le grid search de chaque stratégie sur chaque actif GO de H06.
- [ ] Calcul de la corrélation des retours quotidiens entre chaque (stratégie, actif) et Donchian sur le même actif. Une nouvelle stratégie est retenue si Sharpe > 0 ET corrélation < 0.7 avec Donchian.
- [ ] Sortie : `predictions/h07_strategies_alt.json` + `docs/v3_hypothesis_07.md`.
- [ ] `JOURNAL.md` mis à jour avec la liste finale (stratégie × actif) GO.

## NE PAS FAIRE
- Ne PAS modifier `app/strategies/donchian.py`.
- Ne PAS introduire de stratégie mean-reversion à ce stade (hors scope H07, sera couvert en H10 conditionnel régime).
- Ne PAS lancer Python automatiquement.
- Ne PAS lire le test set pour sélectionner les params (sélection sur train uniquement).

## Étapes

### Étape 1 — Implémenter `BaseStrategy` (si pas déjà fait)
```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Retourne une série indexée comme df, valeurs ∈ {-1, 0, +1}."""
```

### Étape 2 — Dual MA (exemple)
```python
class DualMovingAverage(BaseStrategy):
    def __init__(self, fast: int, slow: int):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma_f = df["close"].rolling(self.fast).mean()
        sma_s = df["close"].rolling(self.slow).mean()
        sig = pd.Series(0, index=df.index)
        sig[sma_f > sma_s] = 1
        sig[sma_f < sma_s] = -1
        return sig.shift(1).fillna(0)  # shift(1) = anti-look-ahead
```

### Étape 3 — Keltner Channel
Bande Keltner :
```python
def keltner_bands(df, period=20, mult=2.0):
    ema = df["close"].ewm(span=period, adjust=False).mean()
    atr_val = atr(df, period)
    return ema, ema + mult * atr_val, ema - mult * atr_val
```
Signal : `close > upper` → LONG, `close < lower` → SHORT, sinon HOLD précédent.

### Étape 4 — Chandelier Exit
```python
def chandelier_levels(df, period=22, atr_mult=3.0):
    atr_val = atr(df, period)
    long_stop = df["high"].rolling(period).max() - atr_mult * atr_val
    short_stop = df["low"].rolling(period).min() + atr_mult * atr_val
    return long_stop, short_stop
```
Signal trend-following : positionnement long si close au-dessus du long_stop précédent, etc.

### Étape 5 — Parabolic SAR
Itératif par nature (l'état SAR évolue trade par trade). Implémenter avec un `numba.njit` si lent ; sinon vectoriser autant que possible et tester sur cas connus.

### Étape 6 — Grid search global
Pour chaque (stratégie, actif GO de H06) :
- Grid search sur params → meilleur Sharpe train
- Eval sur test → `validate_edge`
- Corrélation rolling 60j vs Donchian sur le même actif

### Étape 7 — Filtrage final
Garder uniquement les (stratégie, actif) avec :
- Sharpe test > 0
- Corrélation avec Donchian < 0.7

### Étape 8 — Rapport `docs/v3_hypothesis_07.md`

## Critères go/no-go
- **GO prompt 09** si : au moins 1 stratégie alternative passe sur au moins 1 actif (sinon, on n'a qu'une stratégie dans le portfolio, ce qui est risqué mais on continue).
- **NO-GO partiel** : aucune stratégie alt ne passe. Documenter et continuer.
