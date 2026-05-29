"""FROZEN après pivot v4 A6 (3 entrées) + C2 (extension multi-actifs).

NE PAS MODIFIER MANUELLEMENT. Seules les phases A6 / C2 peuvent y ajouter.
"""
from __future__ import annotations

FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {
    ('US30', 'D1'): ('dist_sma_20', 'autocorr_returns_lag1_20', 'range_atr_ratio', 'close_zscore_20', 'dist_ema_26', 'dist_ema_12', 'dist_sma_200', 'stoch_k_14', 'cci_20', 'stoch_d_14', 'atr_14', 'rsi_21', 'dist_sma_200_abs_atr', 'slope_sma_20', 'macd'),
    ('EURUSD', 'H4'): ('bb_width_20', 'usdchf_return_5', 'kc_width_20', 'close_zscore_20', 'lower_shadow_ratio', 'atr_pct_14', 'cci_20', 'body_to_range_ratio', 'btcusd_return_5', 'dist_ema_12', 'xauusd_return_5', 'atr_14', 'sma_50', 'range_atr_ratio', 'dist_sma_20'),
    ('XAUUSD', 'D1'): ('ema_12', 'upper_shadow_ratio', 'gap_overnight', 'ema_26', 'btcusd_return_5', 'volume_zscore_20', 'sma_50', 'dist_sma_200_abs_atr', 'dist_sma_200', 'mfi_14', 'autocorr_returns_lag1_20', 'body_to_range_ratio', 'kc_width_20', 'range_atr_ratio', 'month_cos'),
    ('BTCUSD', 'D1'): ('atr_14', 'atr_pct_14', 'sma_50', 'cci_20', 'dist_ema_12', 'skew_returns_20', 'macd', 'bb_width_20', 'williams_r_14', 'slope_sma_20', 'atr_zscore_60', 'upper_shadow_ratio', 'rsi_7', 'rsi_21', 'sma_20'),
    ('ETHUSD', 'D1'): ('kc_width_20', 'upper_shadow_ratio', 'kurt_returns_20', 'stoch_d_14', 'efficiency_ratio_20', 'macd_signal', 'macd', 'stoch_k_14', 'williams_r_14', 'skew_returns_20', 'lower_shadow_ratio', 'dist_ema_12', 'close_zscore_20', 'atr_zscore_60', 'atr_pct_14'),
    ('ETHUSD', 'H4'): ('body_to_range_ratio', 'kc_width_20', 'atr_pct_14', 'lower_shadow_ratio', 'bb_width_20', 'atr_14', 'rsi_21', 'btcusd_return_5', 'range_atr_ratio', 'stoch_k_14', 'volume_zscore_20', 'upper_shadow_ratio', 'autocorr_returns_lag1_20', 'skew_returns_20', 'macd_hist'),
    ('ETHUSD', 'H1'): ('kurt_returns_20', 'atr_pct_14', 'kc_width_20', 'autocorr_returns_lag1_20', 'bb_width_20', 'upper_shadow_ratio', 'macd_signal', 'efficiency_ratio_20', 'atr_14', 'ema_26', 'range_atr_ratio', 'volume_zscore_20', 'atr_zscore_60', 'usdchf_return_5', 'mfi_14'),
    ('EURUSD', 'D1'): ('body_to_range_ratio', 'range_atr_ratio', 'upper_shadow_ratio', 'macd_signal', 'slope_sma_50', 'atr_pct_14', 'vol_percentile_60', 'skew_returns_20', 'xauusd_return_5', 'mfi_14', 'stoch_k_14', 'kc_width_20', 'close_zscore_20', 'stoch_d_14', 'atr_14'),
    ('GBPUSD', 'D1'): ('usdchf_return_5', 'range_atr_ratio', 'volume_zscore_20', 'day_cos', 'day_sin', 'xauusd_return_5', 'close_zscore_20', 'cci_20', 'efficiency_ratio_20', 'skew_returns_20', 'slope_sma_20', 'kurt_returns_20', 'autocorr_returns_lag1_20', 'return_percentile_20', 'atr_pct_14'),
    ('GBPUSD', 'H4'): ('autocorr_returns_lag1_20', 'kurt_returns_20', 'dist_ema_12', 'stoch_k_14', 'range_atr_ratio', 'cci_20', 'slope_sma_50', 'rsi_7', 'usdchf_return_5', 'dist_sma_200', 'atr_zscore_60', 'btcusd_return_5', 'session_tokyo', 'slope_sma_20', 'close_zscore_20'),
    ('USDCHF', 'D1'): ('williams_r_14', 'upper_shadow_ratio', 'btcusd_return_5', 'body_to_range_ratio', 'lower_shadow_ratio', 'slope_sma_20', 'stoch_d_14', 'dist_sma_50', 'rsi_7', 'day_sin', 'efficiency_ratio_20', 'slope_sma_50', 'ema_12', 'bb_width_20', 'skew_returns_20'),
    ('USDCHF', 'H4'): ('atr_zscore_60', 'upper_shadow_ratio', 'range_atr_ratio', 'xauusd_return_5', 'volume_zscore_20', 'kc_width_20', 'sma_200', 'macd_signal', 'rsi_14', 'close_zscore_60', 'btcusd_return_5', 'body_to_range_ratio', 'slope_sma_50', 'sma_20', 'dist_sma_200_abs_atr'),
}
