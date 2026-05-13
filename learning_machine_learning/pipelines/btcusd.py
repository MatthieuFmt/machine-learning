"""Pipeline BTC/USD — assemble tous les composants pour l'actif crypto.

Différences vs EURUSD :
- Pas de macro_instruments (BTC n'a pas d'instruments corrélés au sens forex)
- tp_sl_scale_factor = 5.0 (BTC ~5× plus volatile en pips que EURUSD)
- Calendrier économique conservé (événements USD impactent BTC)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from learning_machine_learning.core.logging import get_logger
from learning_machine_learning.features.pipeline import build_ml_ready
from learning_machine_learning.pipelines.base import BasePipeline

if TYPE_CHECKING:
    from learning_machine_learning.config.backtest import BacktestConfig

logger = get_logger(__name__)


class BtcUsdPipeline(BasePipeline):
    """Pipeline complet BTC/USD : données → features → modèle → backtest."""

    def __init__(self, backtest_cfg: BacktestConfig | None = None) -> None:
        super().__init__("BTCUSD", backtest_cfg=backtest_cfg)

    def load_data(self) -> dict[str, Any]:
        """Charge les données BTC/USD depuis les fichiers CSV nettoyés.

        Pas de macro_instruments (frozenset() vide), mais le calendrier
        économique USD est chargé car les événements USD impactent BTC.
        """
        from learning_machine_learning.data.loader import load_all_timeframes

        paths = {
            "h1": self.paths.clean_file("BTCUSD", "H1"),
            "h4": self.paths.clean_file("BTCUSD", "H4"),
            "d1": self.paths.clean_file("BTCUSD", "D1"),
        }

        data = load_all_timeframes(paths)

        # Calendrier économique (les événements USD impactent BTC)
        from learning_machine_learning.data.calendar_loader import load_calendar

        h1_data = data["h1"]
        cal_start = h1_data.index.min()
        cal_end = h1_data.index.max()

        try:
            calendar_df = load_calendar(cal_start, cal_end)
            data["_calendar"] = calendar_df
            logger.info(
                "Calendrier économique chargé : %d événements",
                len(calendar_df),
            )
        except (FileNotFoundError, OSError) as e:
            logger.warning(
                "Calendrier économique indisponible : %s — ignoré.", e
            )
            data["_calendar"] = None

        return data

    def build_features(
        self, data: dict[str, Any], train_end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Construit le DataFrame ML-ready via le pipeline de features.

        Args:
            data: Dict contenant les DataFrames H1, H4, D1 et _calendar.
            train_end: Si fourni, le SessionVolatilityScaler est fit uniquement sur
                les données ≤ train_end (anti-look-ahead). None = fit sur tout l'historique.
        """
        ml = build_ml_ready(
            instrument=self.instrument,
            data={"H1": data["h1"], "H4": data["h4"], "D1": data["d1"]},
            macro_data={},  # Pas de macro pour BTC
            calendar_df=data.get("_calendar"),
            tp_pips=self.backtest_cfg.tp_pips,
            sl_pips=self.backtest_cfg.sl_pips,
            window=self.backtest_cfg.window_hours,
            features_dropped=list(self.instrument.features_dropped),
            train_end=train_end,
        )
        return ml
