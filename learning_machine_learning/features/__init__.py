"""Feature engineering, triple barrier labelling, merge multi-timeframe."""

from learning_machine_learning.features.triple_barrier import (
    apply_triple_barrier,
    compute_target_series,
    label_distribution,
)
from learning_machine_learning.features.technical import (
    calc_base_features,
    calc_log_return,
    calc_ema_distance,
    calc_rsi,
    calc_adx,
    calc_atr_norm,
    calc_bb_width,
    calc_cyclical_time,
)
from learning_machine_learning.features.regime import (
    calc_volatilite_realisee,
    calc_range_atr_ratio,
    calc_rsi_d1_delta,
    calc_dist_sma200_d1,
)
from learning_machine_learning.features.macro import calc_macro_return
from learning_machine_learning.features.merger import merge_features, log_row_loss

__all__ = [
    "apply_triple_barrier",
    "compute_target_series",
    "label_distribution",
    "calc_base_features",
    "calc_log_return",
    "calc_ema_distance",
    "calc_rsi",
    "calc_adx",
    "calc_atr_norm",
    "calc_bb_width",
    "calc_cyclical_time",
    "calc_volatilite_realisee",
    "calc_range_atr_ratio",
    "calc_rsi_d1_delta",
    "calc_dist_sma200_d1",
    "calc_macro_return",
    "merge_features",
    "log_row_loss",
]
