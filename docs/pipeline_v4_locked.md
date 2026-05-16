# Pipeline ML v4 — Locked configuration

**Date du lock** : 2026-05-15
**Version** : v4.0.0-locked
**Pipeline checksums** : voir `TEST_SET_LOCK.json` → `pipeline_locked`

## Vue d'ensemble

Tout le pipeline ML est figé à partir de cette date. Toute modification des fichiers
`features_selected.py`, `model_selected.py`, `hyperparams_tuned.py`, `ml_pipeline_v4.py`
sera détectée par `tests/integration/test_pipeline_integrity.py` qui échoue automatiquement.

## Configurations par actif

### US30 D1
- **Features (15)** : `dist_sma_20`, `autocorr_returns_lag1_20`, `range_atr_ratio`, `close_zscore_20`, `dist_ema_26`, `dist_ema_12`, `dist_sma_200`, `stoch_k_14`, `cci_20`, `stoch_d_14`, `atr_14`, `rsi_21`, `dist_sma_200_abs_atr`, `slope_sma_20`, `macd`
- **Modèle** : `rf`
- **Hyperparams** : `n_estimators=100, max_depth=3, min_samples_leaf=10`
- **Threshold** : `0.55`
- **Expected Sharpe outer** : `1.913`
- **Expected WR** : `0.575`

### EURUSD H4
- **Features (15)** : `bb_width_20`, `usdchf_return_5`, `kc_width_20`, `close_zscore_20`, `lower_shadow_ratio`, `atr_pct_14`, `cci_20`, `body_to_range_ratio`, `btcusd_return_5`, `dist_ema_12`, `xauusd_return_5`, `atr_14`, `sma_50`, `range_atr_ratio`, `dist_sma_20`
- **Modèle** : `rf`
- **Hyperparams** : `n_estimators=100, max_depth=6, min_samples_leaf=10`
- **Threshold** : `0.55`
- **Expected Sharpe outer** : `0.592`
- **Expected WR** : `0.515`

### XAUUSD D1
- **Features (15)** : `ema_12`, `upper_shadow_ratio`, `gap_overnight`, `ema_26`, `btcusd_return_5`, `volume_zscore_20`, `sma_50`, `dist_sma_200_abs_atr`, `dist_sma_200`, `mfi_14`, `autocorr_returns_lag1_20`, `body_to_range_ratio`, `kc_width_20`, `range_atr_ratio`, `month_cos`
- **Modèle** : `stacking` (defaults)
- **Hyperparams** : `{}` (stacking defaults)
- **Threshold** : `0.50`
- **Expected Sharpe outer** : `0.000` (placeholder)
- **Expected WR** : `0.000` (placeholder)

## Comment utiliser

Dans B1, B2, B3 :

```python
from app.config.ml_pipeline_v4 import get_pipeline
from app.models.build import build_locked_model

cfg = get_pipeline("US30", "D1")
print(f"Features: {cfg.features}")
print(f"Threshold: {cfg.threshold}")

model = build_locked_model("US30", "D1", seed=42)
# model est prêt à fit
```

## Comment vérifier l'intégrité

```bash
rtk pytest tests/integration/test_pipeline_integrity.py -v
```

ou

```bash
rtk make verify  # appelle pipeline_check automatiquement
```

## Que faire si le test échoue

1. **Si modification accidentelle** : `git diff` les fichiers config, revert.
2. **Si modification intentionnelle** : il faut faire un nouveau pivot complet (V5).
   La modification actuelle invalide statistiquement Phase B.
3. **Ne JAMAIS** simplement re-faire `run_a9_pipeline_lock.py` avec les nouveaux checksums
   sans avoir fait un nouveau cycle A5-A8 sur de nouvelles données.

## Limites du gel

Le gel protège contre la modification accidentelle des configs. Il ne protège pas contre :
- Modification du code des indicateurs dans `app/features/indicators.py` (changerait les valeurs)
- Modification du code du simulateur (changerait les PnL train)

Pour ces cas, la vigilance manuelle reste nécessaire. Ils sont audités à chaque PR/commit.
