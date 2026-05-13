"""Pipeline EUR/USD — assemble tous les composants pour l'actif principal."""

from __future__ import annotations

from typing import Any

import pandas as pd

from learning_machine_learning.core.logging import get_logger
from learning_machine_learning.features.pipeline import build_ml_ready
from learning_machine_learning.pipelines.base import BasePipeline

logger = get_logger(__name__)


class EurUsdPipeline(BasePipeline):
    """Pipeline complet EUR/USD : données → features → modèle → backtest."""

    def __init__(self) -> None:
        super().__init__("EURUSD")

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Charge les données EUR/USD et macro depuis les fichiers CSV nettoyés."""
        from learning_machine_learning.data.loader import load_all_timeframes

        paths = {
            "h1": self.paths.clean_file("EURUSD", "H1"),
            "h4": self.paths.clean_file("EURUSD", "H4"),
            "d1": self.paths.clean_file("EURUSD", "D1"),
        }

        # Ajouter les instruments macro
        for macro_name in self.instrument.macro_instruments:
            paths[f"macro_{macro_name}"] = self.paths.clean_file(macro_name, "H1")

        data = load_all_timeframes(paths)

        # Charger aussi les données macro dans un dict séparé
        macro_data = {}
        for macro_name in self.instrument.macro_instruments:
            key = f"macro_{macro_name}"
            if key in data:
                macro_data[macro_name] = data[key]

        data["_macro"] = macro_data

        # ★ Step 05 : Charger le calendrier économique
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
        except FileNotFoundError:
            logger.warning(
                "Dossier calendrier économique introuvable — ignoré."
            )
            data["_calendar"] = None
        except OSError as e:
            logger.error(
                "Erreur chargement calendrier : %s — ignoré.", e
            )
            data["_calendar"] = None

        return data

    def build_features(
        self, data: dict[str, Any], train_end: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Construit le DataFrame ML-ready via le pipeline de features.

        Args:
            data: Dict contenant les DataFrames H1, H4, D1 et éventuellement _macro.
            train_end: Si fourni, le SessionVolatilityScaler est fit uniquement sur
                les données ≤ train_end (anti-look-ahead). None = fit sur tout l'historique.
        """
        ml = build_ml_ready(
            instrument=self.instrument,
            data={"H1": data["h1"], "H4": data["h4"], "D1": data["d1"]},
            macro_data=data.get("_macro", {}),
            calendar_df=data.get("_calendar"),
            tp_pips=self.backtest_cfg.tp_pips,
            sl_pips=self.backtest_cfg.sl_pips,
            window=self.backtest_cfg.window_hours,
            features_dropped=list(self.instrument.features_dropped),
            train_end=train_end,
        )
        return ml
