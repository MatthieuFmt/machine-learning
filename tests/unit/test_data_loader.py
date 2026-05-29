"""Tests unitaires pour app.data.loader — chargement et validation CSV OHLCV.

Tous les CSV sont synthétiques, créés dans tmp_path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import DataValidationError
from app.data.loader import _find_csv, load_asset

# ── Helpers pour créer des CSV synthétiques ────────────────────────────────


def _write_csv(dir_path: Path, filename: str, content: str) -> Path:
    """Écrit un fichier CSV dans le répertoire spécifié."""
    p = dir_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _valid_csv_content() -> str:
    """Contenu CSV valide avec 6 colonnes."""
    return (
        "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
        "2024-01-02 00:00:00\t1.1000\t1.1050\t1.0990\t1.1020\t1500\n"
        "2024-01-03 00:00:00\t1.1020\t1.1080\t1.1010\t1.1060\t2000\n"
        "2024-01-04 00:00:00\t1.1060\t1.1100\t1.1040\t1.1080\t1800\n"
        "2024-01-05 00:00:00\t1.1080\t1.1120\t1.1070\t1.1110\t2200\n"
        "2024-01-08 00:00:00\t1.1110\t1.1150\t1.1090\t1.1130\t1700\n"
        "2024-01-09 00:00:00\t1.1130\t1.1180\t1.1120\t1.1160\t1900\n"
    )


def _valid_csv_7cols_content() -> str:
    """Contenu CSV avec 7 colonnes (timestamp implicite + Spread)."""
    return (
        "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
        "2024-01-02 00:00:00\t1.1000\t1.1050\t1.0990\t1.1020\t1500\t10\n"
        "2024-01-03 00:00:00\t1.1020\t1.1080\t1.1010\t1.1060\t2000\t12\n"
        "2024-01-04 00:00:00\t1.1060\t1.1100\t1.1040\t1.1080\t1800\t11\n"
        "2024-01-05 00:00:00\t1.1080\t1.1120\t1.1070\t1.1110\t2200\t10\n"
        "2024-01-08 00:00:00\t1.1110\t1.1150\t1.1090\t1.1130\t1700\t12\n"
        "2024-01-09 00:00:00\t1.1130\t1.1180\t1.1120\t1.1160\t1900\t11\n"
    )


# ── Tests ──────────────────────────────────────────────────────────────────


class TestLoadAssetValid:
    """Cas nominaux — CSV valides."""

    def test_valid_csv_6cols_returns_correct_df(self, tmp_path: Path) -> None:
        """Un CSV valide à 6 colonnes (Time, OHLCV) doit être chargé correctement."""
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", _valid_csv_content())

        df = load_asset("EURUSD", "D1", data_root=tmp_path)

        assert df.shape == (6, 5)  # 6 rows, 5 cols (OHLCV)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.name == "timestamp"
        assert df.index.is_monotonic_increasing
        assert df.iloc[0]["Open"] == 1.1000
        assert df.iloc[-1]["Close"] == 1.1160

    def test_valid_csv_7cols_with_spread(self, tmp_path: Path) -> None:
        """CSV avec 7 colonnes (6 headers + Spread implicite) doit être chargé."""
        asset_dir = tmp_path / "US30"
        asset_dir.mkdir()
        _write_csv(asset_dir, "USA30IDXUSD_D1.csv", _valid_csv_7cols_content())

        df = load_asset("US30", "D1", data_root=tmp_path)

        assert "Spread" in df.columns
        assert df.shape == (6, 6)  # OHLCV + Spread
        assert df.iloc[0]["Spread"] == 10.0

    def test_normal_weekend_gap_logged_not_error(self, tmp_path: Path) -> None:
        """Un gap weekend (vendredi→lundi) ne doit pas lever d'erreur."""
        content = (
            "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
            "2024-06-14 00:00:00\t1.1000\t1.1050\t1.0990\t1.1020\t1500\n"
            "2024-06-17 00:00:00\t1.1020\t1.1080\t1.1010\t1.1060\t2000\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        # Ne doit pas lever d'exception
        df = load_asset("EURUSD", "D1", data_root=tmp_path)
        assert len(df) == 2


class TestLoadAssetErrors:
    """Cas d'erreur — validation stricte."""

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        """Colonne 'close' manquante → DataValidationError."""
        content = (
            "Time\tOpen\tHigh\tLow\tVolume\n"
            "2024-01-02 00:00:00\t1.1000\t1.1050\t1.0990\t1500\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        with pytest.raises(DataValidationError, match="close"):
            load_asset("EURUSD", "D1", data_root=tmp_path)

    def test_duplicate_timestamps_raises(self, tmp_path: Path) -> None:
        """Timestamps dupliqués → DataValidationError."""
        content = (
            "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
            "2024-01-02 00:00:00\t1.1000\t1.1050\t1.0990\t1.1020\t1500\n"
            "2024-01-02 00:00:00\t1.1020\t1.1080\t1.1010\t1.1060\t2000\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        with pytest.raises(DataValidationError, match="dupliqués"):
            load_asset("EURUSD", "D1", data_root=tmp_path)

    def test_ohlc_incoherent_raises(self, tmp_path: Path) -> None:
        """High < Open → DataValidationError."""
        content = (
            "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
            "2024-01-02 00:00:00\t1.1050\t1.1000\t1.0990\t1.1020\t1500\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        with pytest.raises(DataValidationError, match="OHLC incohérentes"):
            load_asset("EURUSD", "D1", data_root=tmp_path)

    def test_negative_price_raises(self, tmp_path: Path) -> None:
        """Prix négatif → DataValidationError."""
        content = (
            "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
            "2024-01-02 00:00:00\t-1.1000\t1.1050\t1.0990\t1.1020\t1500\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        with pytest.raises(DataValidationError, match="négatifs"):
            load_asset("EURUSD", "D1", data_root=tmp_path)

    def test_negative_volume_raises(self, tmp_path: Path) -> None:
        """Volume négatif → DataValidationError."""
        content = (
            "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
            "2024-01-02 00:00:00\t1.1000\t1.1050\t1.0990\t1.1020\t-1500\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        with pytest.raises(DataValidationError, match="volumes négatifs"):
            load_asset("EURUSD", "D1", data_root=tmp_path)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Fichier inexistant → DataValidationError (via _find_csv)."""
        with pytest.raises(DataValidationError, match="introuvable"):
            load_asset("INEXISTANT", "D1", data_root=tmp_path)

    def test_nan_rows_dropped(self, tmp_path: Path) -> None:
        """Les lignes contenant NaN sont supprimées silencieusement."""
        content = (
            "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
            "2024-01-02 00:00:00\t1.1000\t1.1050\t1.0990\t1.1020\t1500\n"
            "2024-01-03 00:00:00\t\t1.1080\t1.1010\t1.1060\t2000\n"
            "2024-01-04 00:00:00\t1.1060\t1.1100\t1.1040\t1.1080\t1800\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        df = load_asset("EURUSD", "D1", data_root=tmp_path)
        assert len(df) == 2  # La ligne avec Open NaN est supprimée

    def test_abnormal_gap_raises(self, tmp_path: Path) -> None:
        """Un gap > 7j entre deux jours ouvrés → DataValidationError."""
        content = (
            "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
            "2024-06-18 00:00:00\t1.1000\t1.1050\t1.0990\t1.1020\t1500\n"
            "2024-06-26 00:00:00\t1.1020\t1.1080\t1.1010\t1.1060\t2000\n"
        )
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", content)

        with pytest.raises(DataValidationError, match="gaps anormaux"):
            load_asset("EURUSD", "D1", data_root=tmp_path)


class TestFindCsv:
    """Tests pour _find_csv."""

    def test_single_match_returns_path(self, tmp_path: Path) -> None:
        """Un seul CSV correspondant → retourne son Path."""
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        expected = _write_csv(asset_dir, "EURUSD_D1.csv", _valid_csv_content())
        # Créer un fichier parasite pour tester la robustesse
        _write_csv(asset_dir, "EURUSD_H4.csv", "dummy\n")

        result = _find_csv(tmp_path, "EURUSD", "D1")
        assert result == expected

    def test_no_match_raises(self, tmp_path: Path) -> None:
        """Aucun CSV ne correspond → DataValidationError."""
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_H4.csv", "dummy\n")

        with pytest.raises(DataValidationError, match="introuvable"):
            _find_csv(tmp_path, "EURUSD", "D1")

    def test_multiple_matches_raises(self, tmp_path: Path) -> None:
        """Plusieurs CSV pour le même TF → DataValidationError."""
        asset_dir = tmp_path / "EURUSD"
        asset_dir.mkdir()
        _write_csv(asset_dir, "EURUSD_D1.csv", _valid_csv_content())
        _write_csv(asset_dir, "EURUSD_COPY_D1.csv", _valid_csv_content())

        with pytest.raises(DataValidationError, match="plusieurs CSV"):
            _find_csv(tmp_path, "EURUSD", "D1")
