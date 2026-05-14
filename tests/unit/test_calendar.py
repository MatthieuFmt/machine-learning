"""Tests unitaires pour app.config.calendar — jours fériés XTB, gap normal."""
from __future__ import annotations

from datetime import UTC, datetime

from app.config.calendar import is_market_open, is_normal_gap


class TestIsMarketOpen:
    """Tests pour is_market_open()."""

    def test_weekend_closed_saturday(self) -> None:
        """Un samedi est toujours fermé."""
        ts = datetime(2024, 6, 15, 14, 0, tzinfo=UTC)  # samedi
        assert is_market_open("US30", ts) is False

    def test_weekend_closed_sunday(self) -> None:
        """Un dimanche est toujours fermé."""
        ts = datetime(2024, 6, 16, 14, 0, tzinfo=UTC)  # dimanche
        assert is_market_open("US30", ts) is False

    def test_holiday_closed(self) -> None:
        """Un jour férié connu est fermé — Noël US30."""
        ts = datetime(2024, 12, 25, 14, 0, tzinfo=UTC)
        assert is_market_open("US30", ts) is False

    def test_normal_trading_day_open(self) -> None:
        """Un mardi ordinaire est ouvert."""
        ts = datetime(2024, 6, 18, 14, 0, tzinfo=UTC)  # mardi
        assert is_market_open("US30", ts) is True

    def test_unknown_asset_defaults_no_holidays(self) -> None:
        """Un actif inconnu n'a pas de jours fériés → ouvert si pas weekend."""
        ts = datetime(2024, 12, 25, 14, 0, tzinfo=UTC)  # Noël
        assert is_market_open("INCONNU", ts) is True

    def test_btc_always_open(self) -> None:
        """BTC n'a pas de jours fériés XTB."""
        ts = datetime(2024, 12, 25, 14, 0, tzinfo=UTC)
        assert is_market_open("BTCUSD", ts) is True

    def test_us30_thanksgiving(self) -> None:
        """Thanksgiving 2024 est férié pour US30."""
        ts = datetime(2024, 11, 28, 14, 0, tzinfo=UTC)
        assert is_market_open("US30", ts) is False


class TestIsNormalGap:
    """Tests pour is_normal_gap()."""

    def test_normal_weekend_gap(self) -> None:
        """Gap du vendredi soir au lundi matin = normal (weekend)."""
        t1 = datetime(2024, 6, 14, 22, 0, tzinfo=UTC)  # vendredi
        t2 = datetime(2024, 6, 17, 8, 0, tzinfo=UTC)  # lundi
        assert is_normal_gap("US30", t1, t2) is True

    def test_consecutive_days_not_normal(self) -> None:
        """Gap traversant 2 jours ouvrés (mardi→jeudi) = anormal — mercredi ouvert."""
        t1 = datetime(2024, 6, 18, 0, 0, tzinfo=UTC)  # mardi
        t2 = datetime(2024, 6, 20, 0, 0, tzinfo=UTC)  # jeudi
        assert is_normal_gap("US30", t1, t2) is False

    def test_holiday_plus_weekend_normal(self) -> None:
        """Noël un mercredi → gap mardi→jeudi = anormal (mercredi férié)."""
        t1 = datetime(2024, 12, 24, 22, 0, tzinfo=UTC)  # mardi
        t2 = datetime(2024, 12, 26, 8, 0, tzinfo=UTC)  # jeudi
        # Le 25 est Noël (fermé) → le gap est normal
        assert is_normal_gap("US30", t1, t2) is True

    def test_t2_before_t1_returns_true(self) -> None:
        """Si t2 <= t1, c'est considéré comme normal (pas de gap)."""
        t1 = datetime(2024, 6, 19, 22, 0, tzinfo=UTC)
        t2 = datetime(2024, 6, 18, 22, 0, tzinfo=UTC)
        assert is_normal_gap("US30", t1, t2) is True

    def test_daily_bar_weekend_normal(self) -> None:
        """Barre D1 : gap vendredi→dimanche ou vendredi→lundi = normal."""
        # Ex: barre du vendredi 2024-05-17 00:00 → barre du lundi 2024-05-20 00:00
        t1 = datetime(2024, 5, 17, 0, 0, tzinfo=UTC)  # vendredi
        t2 = datetime(2024, 5, 20, 0, 0, tzinfo=UTC)  # lundi
        assert is_normal_gap("US30", t1, t2) is True


class TestAliasResolution:
    """Vérifie que les alias d'actifs sont correctement résolus."""

    def test_us30_alias_has_same_holidays_as_usa30idxusd(self) -> None:
        """L'alias 'US30' doit avoir les mêmes jours fériés que 'USA30IDXUSD'."""
        ts = datetime(2024, 7, 4, 14, 0, tzinfo=UTC)  # Independence Day
        assert is_market_open("US30", ts) is is_market_open("USA30IDXUSD", ts)
