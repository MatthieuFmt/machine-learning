"""Tests unitaires pour le calendrier économique macro (Step 05).

Couvre : calendar_loader (chargement, validation, parsing),
         calendar features (distances, surprise, merge).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from learning_machine_learning.data.calendar_loader import (
    CANONICAL_EVENT_NAMES,
    _parse_actual_value,
    _detect_timezone_utc,
    validate_calendar_schema,
    load_calendar,
)
from learning_machine_learning.features.calendar import (
    _SENTINEL_MINUTES,
    compute_minutes_to_next_event,
    compute_minutes_since_last_event,
    compute_surprise_zscore,
    merge_calendar_features,
)
from learning_machine_learning.core.exceptions import DataValidationError


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_calendar_df():
    """Calendrier synthétique avec 4 événements."""
    base = pd.Timestamp("2024-01-10", tz="UTC")
    return pd.DataFrame({
        "timestamp": [
            base + pd.Timedelta(hours=12),
            base + pd.Timedelta(hours=24),
            base + pd.Timedelta(hours=48),
            base + pd.Timedelta(hours=72),
        ],
        "currency": ["USD", "USD", "EUR", "USD"],
        "event_name": ["US_NFP", "US_CPI_YoY", "EU_ECB_Rate", "US_FOMC"],
        "impact": ["High", "High", "High", "High"],
        "actual": [216.0, 3.2, 4.5, np.nan],
        "forecast": [170.0, 3.1, 4.5, np.nan],
        "previous": [173.0, 3.2, 4.5, np.nan],
    })


@pytest.fixture
def sample_ohlc_index():
    """Index H1 synthétique sur 4 jours."""
    base = pd.Timestamp("2024-01-10", tz="UTC")
    return pd.DatetimeIndex([
        base + pd.Timedelta(hours=h) for h in range(0, 96)
    ])


@pytest.fixture
def tmp_calendar_dir(tmp_path):
    """Crée un dossier calendrier temporaire avec un CSV valide."""
    data_dir = tmp_path / "economic_calendar"
    data_dir.mkdir()

    csv_content = (
        "date,time,currency,event,impact,actual,forecast,previous\n"
        "2024-01-05,13:30,USD,Non-Farm Employment Change,High,216K,170K,173K\n"
        "2024-01-10,13:30,USD,Consumer Price Index (YoY),High,3.2%,3.1%,3.2%\n"
        "2024-01-25,13:15,EUR,Main Refinancing Rate,High,4.5%,4.5%,4.5%\n"
        "2024-01-31,19:00,USD,FOMC Statement,High,,,\n"
    )
    csv_path = data_dir / "2024.csv"
    csv_path.write_text(csv_content)
    return data_dir


# ── Tests _parse_actual_value ────────────────────────────────────────────

def test_parse_actual_K():
    assert _parse_actual_value("216K") == 216_000.0


def test_parse_actual_M():
    assert _parse_actual_value("-3.2M") == -3_200_000.0


def test_parse_actual_percent():
    assert _parse_actual_value("3.2%") == 3.2
    assert _parse_actual_value("-0.5%") == -0.5


def test_parse_actual_plain_number():
    assert _parse_actual_value("4.5") == 4.5
    assert _parse_actual_value("0") == 0.0


def test_parse_actual_empty():
    assert _parse_actual_value("") is None
    assert _parse_actual_value("-") is None


def test_parse_actual_nan():
    assert _parse_actual_value(np.nan) is None
    assert _parse_actual_value(None) is None


def test_parse_actual_with_comma():
    assert _parse_actual_value("1,234") == 1234.0


# ── Tests _detect_timezone_utc ──────────────────────────────────────────

def test_detect_utc_when_hours_around_13():
    df = pd.DataFrame({"time": ["13:30", "14:00", "13:15"]})
    assert _detect_timezone_utc(df, "time") == True


def test_detect_et_when_hours_around_8():
    df = pd.DataFrame({"time": ["08:30", "08:30", "10:00"]})
    assert _detect_timezone_utc(df, "time") == False


def test_detect_empty_returns_true():
    df = pd.DataFrame({"time": []})
    assert _detect_timezone_utc(df, "time") is True


# ── Tests CANONICAL_EVENT_NAMES ──────────────────────────────────────────

def test_canonical_nfp_mapping():
    assert CANONICAL_EVENT_NAMES["Non-Farm Employment Change"] == "US_NFP"
    assert CANONICAL_EVENT_NAMES["NFP"] == "US_NFP"


def test_canonical_fomc_mapping():
    assert CANONICAL_EVENT_NAMES["FOMC Statement"] == "US_FOMC"


def test_canonical_ecb_mapping():
    assert CANONICAL_EVENT_NAMES["Main Refinancing Rate"] == "EU_ECB_Rate"


# ── Tests validate_calendar_schema ───────────────────────────────────────

def test_validate_schema_ok(sample_calendar_df):
    validate_calendar_schema(sample_calendar_df)  # ne doit pas raise


def test_validate_schema_missing_timestamp():
    df = pd.DataFrame({"currency": ["USD"], "event_name": ["NFP"], "impact": ["High"]})
    with pytest.raises(DataValidationError, match="timestamp"):
        validate_calendar_schema(df)


def test_validate_schema_bad_type():
    df = pd.DataFrame({
        "timestamp": ["not_a_date"],
        "currency": ["USD"],
        "event_name": ["NFP"],
        "impact": ["High"],
    })
    with pytest.raises(DataValidationError, match="datetime"):
        validate_calendar_schema(df)


def test_validate_schema_bad_impact(sample_calendar_df):
    sample_calendar_df.loc[0, "impact"] = "Critical"
    with pytest.raises(DataValidationError, match="impact"):
        validate_calendar_schema(sample_calendar_df)


# ── Tests load_calendar ─────────────────────────────────────────────────

def test_load_calendar_basic(tmp_calendar_dir):
    df = load_calendar(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-12-31", tz="UTC"),
        data_dir=tmp_calendar_dir,
    )
    assert len(df) == 4
    assert set(df.columns) == {
        "timestamp", "currency", "event_name", "impact",
        "actual", "forecast", "previous",
    }
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_calendar_date_filter(tmp_calendar_dir):
    df = load_calendar(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-06", tz="UTC"),
        data_dir=tmp_calendar_dir,
    )
    assert len(df) == 1  # Seulement le NFP du 5 janvier


def test_load_calendar_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        load_calendar(
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-12-31", tz="UTC"),
            data_dir="/nonexistent/path/12345",
        )


def test_load_calendar_normalizes_event_names(tmp_calendar_dir):
    df = load_calendar(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-12-31", tz="UTC"),
        data_dir=tmp_calendar_dir,
    )
    names = df["event_name"].tolist()
    assert "US_NFP" in names
    assert "US_CPI_YoY" in names
    assert "EU_ECB_Rate" in names
    assert "US_FOMC" in names


def test_load_calendar_parses_K_actual(tmp_calendar_dir):
    df = load_calendar(
        start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2024-01-06", tz="UTC"),
        data_dir=tmp_calendar_dir,
    )
    assert df.iloc[0]["actual"] == 216_000.0
    assert df.iloc[0]["forecast"] == 170_000.0


def test_load_calendar_empty_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty_cal"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Aucun fichier CSV"):
        load_calendar(
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            end=pd.Timestamp("2024-12-31", tz="UTC"),
            data_dir=empty_dir,
        )


# ── Tests compute_minutes_to_next_event ──────────────────────────────────

def test_minutes_to_next_event_basic(sample_calendar_df, sample_ohlc_index):
    result = compute_minutes_to_next_event(sample_ohlc_index, sample_calendar_df)
    assert len(result) == len(sample_ohlc_index)
    assert result.dtype == "float64"

    # Barre avant le premier événement (à 12h) → premier event à 12h
    # Barre à t=0h → 12h d'écart = 720 minutes
    assert result.iloc[0] == pytest.approx(720.0, abs=1)


def test_minutes_to_next_event_no_future(sample_calendar_df):
    # Timestamps après le dernier événement
    late_idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-20", tz="UTC"),
        pd.Timestamp("2024-01-25", tz="UTC"),
    ])
    result = compute_minutes_to_next_event(late_idx, sample_calendar_df)
    assert (result == _SENTINEL_MINUTES).all()


def test_minutes_to_next_event_impact_filter(sample_calendar_df):
    # Marquer un événement comme Medium
    df = sample_calendar_df.copy()
    df.loc[0, "impact"] = "Medium"
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-10", tz="UTC")])
    result_high = compute_minutes_to_next_event(idx, df, impact="high")
    result_medium = compute_minutes_to_next_event(idx, df, impact="medium")
    # Avec impact=high, le 1er événement est ignoré → prochain = 24h
    # Avec impact=medium, le 1er événement est inclus → 12h
    assert result_high.iloc[0] > result_medium.iloc[0]


def test_minutes_to_next_event_empty_events():
    empty = pd.DataFrame(columns=["timestamp", "impact"])
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-10", tz="UTC")])
    result = compute_minutes_to_next_event(idx, empty, impact="high")
    assert result.iloc[0] == _SENTINEL_MINUTES


# ── Tests compute_minutes_since_last_event ───────────────────────────────

def test_minutes_since_last_event_basic(sample_calendar_df, sample_ohlc_index):
    result = compute_minutes_since_last_event(sample_ohlc_index, sample_calendar_df)
    assert len(result) == len(sample_ohlc_index)
    assert result.dtype == "float64"


def test_minutes_since_last_event_no_past(sample_calendar_df):
    # Timestamps avant le premier événement
    early_idx = pd.DatetimeIndex([
        pd.Timestamp("2024-01-09", tz="UTC"),
        pd.Timestamp("2024-01-10", tz="UTC"),
    ])
    result = compute_minutes_since_last_event(early_idx, sample_calendar_df)
    assert (result == _SENTINEL_MINUTES).all()


# ── Tests compute_surprise_zscore ────────────────────────────────────────

def test_surprise_zscore_basic():
    """Test de base avec assez d'historique."""
    base = pd.Timestamp("2024-01-01", tz="UTC")
    events = []
    # 30 événements NFP identiques pour avoir un historique
    for i in range(30):
        events.append({
            "timestamp": base + pd.Timedelta(days=i * 30),
            "currency": "USD",
            "event_name": "US_NFP",
            "impact": "High",
            "actual": 200.0 + i * 1.0,
            "forecast": 200.0 + i * 0.5,
            "previous": 199.0,
        })
    df = pd.DataFrame(events)
    result = compute_surprise_zscore(df, lookback=50, min_history=20)

    assert "surprise" in result.columns
    assert "surprise_zscore" in result.columns
    # Les 19 premiers doivent être NaN (pas assez d'historique)
    assert result["surprise_zscore"].iloc[:19].isna().all()
    # À partir du 20ème, zscore devrait être défini
    assert result["surprise_zscore"].iloc[20:].notna().any()


def test_surprise_zscore_insufficient_history():
    """Avec moins de min_history événements, tout est NaN."""
    base = pd.Timestamp("2024-01-01", tz="UTC")
    events = []
    for i in range(5):
        events.append({
            "timestamp": base + pd.Timedelta(days=i * 30),
            "currency": "USD",
            "event_name": "US_NFP",
            "impact": "High",
            "actual": 200.0,
            "forecast": 200.0,
            "previous": 200.0,
        })
    df = pd.DataFrame(events)
    result = compute_surprise_zscore(df, lookback=50, min_history=20)
    assert result["surprise_zscore"].isna().all()


# ── Tests merge_calendar_features ───────────────────────────────────────

def test_merge_calendar_features_columns(sample_calendar_df, sample_ohlc_index):
    ohlc = pd.DataFrame({"Close": np.ones(len(sample_ohlc_index))}, index=sample_ohlc_index)
    result = merge_calendar_features(ohlc, sample_calendar_df)

    expected_cols = {
        "minutes_to_next_event", "minutes_since_last_event",
        "surprise_zscore", "near_high_impact_event",
    }
    assert expected_cols.issubset(result.columns)
    assert len(result) == len(sample_ohlc_index)


def test_merge_calendar_near_high_impact_flag(sample_calendar_df):
    """Vérifie que near_high_impact_event capte correctement ±2h."""
    # Créer un index pile autour d'un événement
    event_time = pd.Timestamp("2024-01-10T12:00:00", tz="UTC")
    idx = pd.DatetimeIndex([
        event_time - pd.Timedelta(hours=3),
        event_time - pd.Timedelta(hours=1),
        event_time,
        event_time + pd.Timedelta(hours=1),
        event_time + pd.Timedelta(hours=3),
    ])
    ohlc = pd.DataFrame({"Close": np.ones(5)}, index=idx)
    result = merge_calendar_features(ohlc, sample_calendar_df)

    # Les barres à -1h, 0h, +1h doivent être flagged
    # Celles à -3h et +3h non
    expected = np.array([0, 1, 1, 1, 0], dtype="int8")
    assert (result["near_high_impact_event"].values == expected).all()


def test_merge_calendar_no_lookahead(sample_calendar_df):
    """La surprise_zscore à t ne doit pas utiliser d'info future."""
    event_time = pd.Timestamp("2024-01-10T12:00:00", tz="UTC")
    idx = pd.DatetimeIndex([
        event_time - pd.Timedelta(hours=1),
        event_time,
        event_time + pd.Timedelta(hours=1),
    ])
    ohlc = pd.DataFrame({"Close": np.ones(3)}, index=idx)
    result = merge_calendar_features(ohlc, sample_calendar_df)

    # À event_time - 1h, le zscore ne doit PAS encore refléter l'événement de 12h
    # (merge_asof backward : l'événement n'est pas encore arrivé)
    # Note: avec merge_asof backward, la barre à t_i prend le dernier zscore ≤ t_i,
    # donc la barre avant l'événement n'a pas le zscore de cet événement
    assert result["surprise_zscore"].iloc[0] != result["surprise_zscore"].iloc[2] or \
           pd.isna(result["surprise_zscore"].iloc[0])


def test_merge_calendar_empty_events(sample_ohlc_index):
    """Avec un calendrier vide, toutes les valeurs sont sentinelle/NaN/0."""
    empty = pd.DataFrame(columns=["timestamp", "currency", "event_name",
                                   "impact", "actual", "forecast", "previous"])
    ohlc = pd.DataFrame({"Close": np.ones(len(sample_ohlc_index))}, index=sample_ohlc_index)
    result = merge_calendar_features(ohlc, empty)

    assert (result["minutes_to_next_event"] == _SENTINEL_MINUTES).all()
    assert (result["minutes_since_last_event"] == _SENTINEL_MINUTES).all()
    assert result["surprise_zscore"].isna().all()
    assert (result["near_high_impact_event"] == 0).all()


def test_merge_calendar_bad_index_raises():
    df_no_dt = pd.DataFrame({"Close": [1.0]}, index=[0, 1, 2])
    empty = pd.DataFrame(columns=["timestamp", "impact"])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        merge_calendar_features(df_no_dt, empty)


def test_sentinel_value():
    """Vérifie que la valeur sentinelle est cohérente."""
    assert _SENTINEL_MINUTES >= 99999
    assert _SENTINEL_MINUTES < np.inf
    assert isinstance(_SENTINEL_MINUTES, float)


def test_minutes_both_directions_consistency(sample_calendar_df):
    """next + last devrait être cohérent sur toute la plage."""
    base = pd.Timestamp("2024-01-10", tz="UTC")
    idx = pd.DatetimeIndex([base + pd.Timedelta(hours=h) for h in range(0, 100)])
    nxt = compute_minutes_to_next_event(idx, sample_calendar_df)
    lst = compute_minutes_since_last_event(idx, sample_calendar_df)

    # Aucune valeur ne devrait être négative
    assert (nxt >= 0).all()
    assert (lst >= 0).all()
    # L'un des deux devrait toujours être ≤ 120 quand l'autre est ≤ 120
    # (sauf si aucun événement => les deux à _SENTINEL_MINUTES)
