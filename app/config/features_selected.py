"""FROZEN après pivot v4 A6. NE PAS MODIFIER sans nouveau pivot."""
from __future__ import annotations

FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {
    ("US30", "D1"): ('dist_sma_20', 'autocorr_returns_lag1_20', 'range_atr_ratio', 'close_zscore_20', 'dist_ema_26', 'dist_ema_12', 'dist_sma_200', 'stoch_k_14', 'cci_20', 'stoch_d_14', 'atr_14', 'rsi_21', 'dist_sma_200_abs_atr', 'slope_sma_20', 'macd'),
    ("EURUSD", "H4"): ('bb_width_20', 'usdchf_return_5', 'kc_width_20', 'close_zscore_20', 'lower_shadow_ratio', 'atr_pct_14', 'cci_20', 'body_to_range_ratio', 'btcusd_return_5', 'dist_ema_12', 'xauusd_return_5', 'atr_14', 'sma_50', 'range_atr_ratio', 'dist_sma_20'),
    ("XAUUSD", "D1"): ('ema_12', 'upper_shadow_ratio', 'gap_overnight', 'ema_26', 'btcusd_return_5', 'volume_zscore_20', 'sma_50', 'dist_sma_200_abs_atr', 'dist_sma_200', 'mfi_14', 'autocorr_returns_lag1_20', 'body_to_range_ratio', 'kc_width_20', 'range_atr_ratio', 'month_cos'),
}
