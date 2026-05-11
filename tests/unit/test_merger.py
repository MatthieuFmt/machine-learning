"""Tests unitaires pour features.merger — log_row_loss, merge_features, _log_merge_nan."""

import pandas as pd
import numpy as np
from datetime import datetime

from learning_machine_learning.features.merger import merge_features, log_row_loss


class TestLogRowLoss:
    def test_no_loss(self) -> None:
        """0 perte = pas d'erreur."""
        log_row_loss("test", 100, 100)  # ne doit pas lever d'exception

    def test_small_loss(self) -> None:
        """Perte < 5% = info, pas d'erreur."""
        log_row_loss("test", 100, 96)  # 4% perte

    def test_large_loss(self) -> None:
        """Perte > 5% = warning, pas d'erreur."""
        log_row_loss("test", 100, 90)  # 10% perte

    def test_zero_before(self) -> None:
        """Before=0 ne cause pas de division par zero."""
        log_row_loss("test", 0, 0)


class TestMergeFeatures:
    def test_basic_merge_no_lookahead(self) -> None:
        """Merge H4 et D1 sur H1 sans decalage (offset=0 pour test deterministe)."""
        n = 10
        h1 = pd.DataFrame(
            {"Close": np.ones(n), "Target": np.zeros(n)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time"),
        )
        feat_h4 = pd.DataFrame(
            {"RSI_14_H4": np.arange(10, 10 + n, dtype=float)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="4h", name="Time"),
        )
        feat_d1 = pd.DataFrame(
            {"RSI_14_D1": np.arange(20, 20 + 3, dtype=float)},
            index=pd.date_range("2024-01-01", periods=3, freq="D", name="Time"),
        )

        result = merge_features(
            h1, feat_h4, feat_d1,
            h4_offset=pd.Timedelta(0),
            d1_offset=pd.Timedelta(0),
        )
        assert "RSI_14_H4" in result.columns
        assert "RSI_14_D1" in result.columns

    def test_merge_preserves_target(self) -> None:
        """La colonne Target survit au merge."""
        n = 5
        h1 = pd.DataFrame(
            {"Close": np.ones(n), "Target": [1, -1, 0, 1, 0]},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time"),
        )
        feat_h4 = pd.DataFrame(
            {"RSI_14_H4": np.ones(n, dtype=float)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="4h", name="Time"),
        )
        feat_d1 = pd.DataFrame(
            {"RSI_14_D1": np.ones(2, dtype=float)},
            index=pd.date_range("2024-01-01", periods=2, freq="D", name="Time"),
        )

        result = merge_features(
            h1, feat_h4, feat_d1,
            h4_offset=pd.Timedelta(0),
            d1_offset=pd.Timedelta(0),
        )
        # Target est preservee (mais des NaN peuvent apparaitre si pas de merge)
        assert "Target" in result.columns

    def test_merge_with_macro(self) -> None:
        """Merge avec DataFrame macro."""
        n = 5
        h1 = pd.DataFrame(
            {"Close": np.ones(n), "Target": np.zeros(n)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time"),
        )
        macro = pd.DataFrame(
            {"XAU_Return": np.arange(n, dtype=float) * 0.001},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time"),
        )
        feat_h4 = pd.DataFrame(
            {"RSI_14_H4": np.ones(n, dtype=float)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="4h", name="Time"),
        )
        feat_d1 = pd.DataFrame(
            {"RSI_14_D1": np.ones(2, dtype=float)},
            index=pd.date_range("2024-01-01", periods=2, freq="D", name="Time"),
        )

        result = merge_features(
            h1, feat_h4, feat_d1, macro_frames=[macro],
            h4_offset=pd.Timedelta(0),
            d1_offset=pd.Timedelta(0),
        )
        assert "XAU_Return" in result.columns

    def test_index_is_datetime(self) -> None:
        """L'index de sortie est un DatetimeIndex."""
        n = 5
        h1 = pd.DataFrame(
            {"Close": np.ones(n), "Target": np.zeros(n)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time"),
        )
        feat_h4 = pd.DataFrame(
            {"RSI_14_H4": np.ones(n, dtype=float)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="4h", name="Time"),
        )
        feat_d1 = pd.DataFrame(
            {"RSI_14_D1": np.ones(2, dtype=float)},
            index=pd.date_range("2024-01-01", periods=2, freq="D", name="Time"),
        )

        result = merge_features(
            h1, feat_h4, feat_d1,
            h4_offset=pd.Timedelta(0),
            d1_offset=pd.Timedelta(0),
        )
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_empty_macro_list(self) -> None:
        """macro_frames vide ne cause pas d'erreur."""
        n = 5
        h1 = pd.DataFrame(
            {"Close": np.ones(n), "Target": np.zeros(n)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="h", name="Time"),
        )
        feat_h4 = pd.DataFrame(
            {"RSI_14_H4": np.ones(n, dtype=float)},
            index=pd.date_range("2024-01-01 00:00", periods=n, freq="4h", name="Time"),
        )
        feat_d1 = pd.DataFrame(
            {"RSI_14_D1": np.ones(2, dtype=float)},
            index=pd.date_range("2024-01-01", periods=2, freq="D", name="Time"),
        )

        result = merge_features(
            h1, feat_h4, feat_d1, macro_frames=[],
            h4_offset=pd.Timedelta(0),
            d1_offset=pd.Timedelta(0),
        )
        assert isinstance(result, pd.DataFrame)
