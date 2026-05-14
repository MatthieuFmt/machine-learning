# Prompt 17 — H16 : Timeframe stacking D1/H4 (optionnel)

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md` (TF retenu)
3. `prompts/16_h15_tf_decision.md`
4. `docs/v3_roadmap.md` section "H16"

## Objectif
**Optionnel selon TF retenu** : si TF = D1, tester si l'ajout d'un filtre/timing H4 améliore le Sharpe. Si TF retenu = H4 ou H1, ce prompt est SKIPPÉ (passer directement au prompt 18).

**Question H16** : Signal D1 pour la direction + H4 pour le timing d'entrée + stop ATR(14) H4 ×1.5 améliore-t-il le Sharpe vs D1-only ?

## Definition of Done (testable)
- [ ] Si TF retenu ≠ D1 : skip le prompt, logger dans JOURNAL.md « Prompt 17 skippé : TF retenu = <TF> ».
- [ ] Si TF retenu = D1 ET H4 disponible :
  - `app/strategies/stacked.py` contient `StackedDonchian(daily_strat, h4_filter, atr_period=14, atr_mult=1.5)`
  - Le signal D1 donne la direction (LONG/SHORT/FLAT)
  - L'entrée H4 = pullback dans la direction D1 (Close H4 < SMA20 H4 en D1-LONG)
  - SL = ATR(14) H4 × 1.5
  - TP = 2 × SL (ratio R:R = 2:1)
  - `scripts/run_h16_tf_stacking.py` compare Sharpe stacked vs Sharpe D1-only.
- [ ] Rapport `docs/v3_hypothesis_16.md`.

## NE PAS FAIRE
- Ne PAS skip ce prompt sans noter explicitement dans JOURNAL.md.
- Ne PAS générer des signaux H4 sans data H4 vérifiée.
- Ne PAS modifier le signal D1 (uniquement timing).

## Étapes

### Étape 1 — Vérifier TF retenu
```python
from app.config.timeframe import PRIMARY_TF
if PRIMARY_TF != "D1":
    log("Prompt 17 skippé : TF retenu = " + PRIMARY_TF)
    exit(0)
```

### Étape 2 — Charger D1 + H4
Pour chaque actif GO :
```python
df_d1 = load_asset(asset, "D1")
df_h4 = load_asset(asset, "H4")
```

### Étape 3 — Signal D1 forward-filled sur H4
```python
signal_d1 = strategy.generate_signals(df_d1)
signal_d1_h4 = signal_d1.reindex(df_h4.index, method="ffill")
```

### Étape 4 — Timing H4 (pullback)
```python
sma20_h4 = df_h4["close"].rolling(20).mean()
entry_long = (signal_d1_h4 == 1) & (df_h4["close"] < sma20_h4) & (df_h4["close"] > sma20_h4.shift(1))
```

### Étape 5 — SL/TP dynamiques
ATR H4 × 1.5 pour SL. TP = 2 × distance SL.

### Étape 6 — Backtest comparatif

## Critères go/no-go
- **GO prompt 18** systématique.
- Si Sharpe stacked > Sharpe D1-only + 0.1 : adopter stacking. Sinon, garder D1-only.
