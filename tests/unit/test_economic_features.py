"""Tests pour le module de calendrier économique (Prompt 05).

Calendrier 100% synthétique via tmp_path — pas de dépendance sur data/raw/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from app.features.economic import (
    _event_within_window,
    _parse_time_str,
    _to_ns,
    compute_event_features,
    load_calendar,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

_NANOSECONDS_PER_HOUR = int(3.6e12)


@pytest.fixture
def synthetic_calendar_csv(tmp_path):
    """Crée 2 fichiers CSV calendrier synthétiques (2024-2025)."""
    cal_dir = tmp_path / "economic_calendar"
    cal_dir.mkdir()

    # 2024 : NFP 5 jan 13:30 UTC, FOMC 31 jan 19:00 UTC, EUR High 15 mars 10:00
    csv_2024 = cal_dir / "2024.csv"
    with open(csv_2024, "w") as f:
        f.write("date,time,currency,event,impact,actual,forecast,previous\n")
        f.write("2024-01-05,1:30pm,USD,Non-Farm Employment Change,High,256K,180K,150K\n")
        f.write("2024-01-31,7:00pm,USD,FOMC Statement,High,,,\n")
        f.write("2024-03-15,10:00am,EUR,CPI Flash Estimate y/y,High,,,\n")
        f.write("2024-06-10,All Day,EUR,German Prelim CPI m/m,Medium,,,\n")

    # 2025 (vide)
    csv_2025 = cal_dir / "2025.csv"
    with open(csv_2025, "w") as f:
        f.write("date,time,currency,event,impact,actual,forecast,previous\n")

    return cal_dir


@pytest.fixture
def loaded_calendar(synthetic_calendar_csv):
    """Charge le calendrier synthétique."""
    return load_calendar([2024, 2025], root=synthetic_calendar_csv)


@pytest.fixture
def price_index():
    """Index de prix synthétique couvrant jan-fév 2024, pas de 6h, UTC."""
    return pd.date_range("2024-01-01", "2024-02-28", freq="6h", tz="UTC")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests _parse_time_str
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("time_str", "expected"),
    [
        ("1:30pm", "13:30:00"),
        ("10:00am", "10:00:00"),
        ("12:00pm", "12:00:00"),
        ("12:00am", "00:00:00"),
        ("All Day", "00:00:00"),
        ("Tentative", "00:00:00"),
        ("", "00:00:00"),
        ("   ", "00:00:00"),
    ],
)
def test_parse_time_str(time_str, expected):
    """Parse correct des formats d'heure ForexFactory."""
    assert _parse_time_str(time_str) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Tests load_calendar
# ═══════════════════════════════════════════════════════════════════════════════


def test_load_calendar_ok(loaded_calendar):
    """Chargement OK : bonnes colonnes, timestamps UTC, triés."""
    assert isinstance(loaded_calendar, pd.DataFrame)
    assert len(loaded_calendar) == 4
    assert list(loaded_calendar.columns) == [
        "timestamp", "currency", "event", "impact", "actual", "forecast", "previous",
    ]
    assert loaded_calendar["timestamp"].dtype.kind == "M"  # datetime64
    assert loaded_calendar["timestamp"].dt.tz is not None  # UTC
    assert loaded_calendar["timestamp"].is_monotonic_increasing


def test_load_calendar_timestamps(loaded_calendar):
    """Vérification des valeurs de timestamp exactes."""
    ts = loaded_calendar.set_index("currency").loc["USD"]
    nfp = ts[ts["event"].str.contains("Non-Farm", na=False)]
    fomc = ts[ts["event"].str.contains("FOMC", na=False)]
    assert nfp["timestamp"].iloc[0] == pd.Timestamp("2024-01-05 13:30", tz="UTC")
    assert fomc["timestamp"].iloc[0] == pd.Timestamp("2024-01-31 19:00", tz="UTC")


def test_load_calendar_missing_file(tmp_path):
    """DataValidationError si fichier d'année introuvable."""
    cal_dir = tmp_path / "nonexistent"
    cal_dir.mkdir()
    with pytest.raises(DataValidationError, match="2024"):
        load_calendar([2024], root=cal_dir)


def test_load_calendar_empty_csv_no_error(tmp_path):
    """CSV vide (header only) → DataFrame vide sans erreur."""
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    with open(cal_dir / "2024.csv", "w") as f:
        f.write("date,time,currency,event,impact,actual,forecast,previous\n")
    result = load_calendar([2024], root=cal_dir)
    assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests _event_within_window (helper interne)
# ═══════════════════════════════════════════════════════════════════════════════


def test_event_within_window_exact():
    """Barre à t-Xh d'un event : within_4h=1, within_1h=0."""
    price_idx = pd.date_range("2024-01-05 08:00", "2024-01-05 18:00", freq="1h", tz="UTC")
    price_ns = _to_ns(price_idx)
    event_ns = np.array([pd.Timestamp("2024-01-05 13:30", tz="UTC").value], dtype=np.int64)

    within_1h = _event_within_window(price_ns, event_ns, 1 * _NANOSECONDS_PER_HOUR)
    within_4h = _event_within_window(price_ns, event_ns, 4 * _NANOSECONDS_PER_HOUR)
    within_24h = _event_within_window(price_ns, event_ns, 24 * _NANOSECONDS_PER_HOUR)

    # Barre à 13:00 (30 min avant l'event) → dans 1h et 4h
    bar_1300 = price_idx.get_loc(pd.Timestamp("2024-01-05 13:00", tz="UTC"))
    assert within_1h[bar_1300] == 1
    assert within_4h[bar_1300] == 1

    # Barre à 09:00 (4h30 avant) → dans 4h (car window=4h avant l'event = 09:30..13:30)
    # 09:00 est avant 09:30 → pas dans 4h, mais dans 24h
    bar_0900 = price_idx.get_loc(pd.Timestamp("2024-01-05 09:00", tz="UTC"))
    assert within_1h[bar_0900] == 0
    assert within_4h[bar_0900] == 0
    assert within_24h[bar_0900] == 1

    # Barre à 08:00 (5h30 avant) → ni 1h ni 4h, mais 24h oui
    bar_0800 = price_idx.get_loc(pd.Timestamp("2024-01-05 08:00", tz="UTC"))
    assert within_1h[bar_0800] == 0
    assert within_4h[bar_0800] == 0
    assert within_24h[bar_0800] == 1

    # Barre à 14:00 (30 min après l'event) → rien
    bar_1400 = price_idx.get_loc(pd.Timestamp("2024-01-05 14:00", tz="UTC"))
    assert within_1h[bar_1400] == 0
    assert within_4h[bar_1400] == 0
    assert within_24h[bar_1400] == 0


def test_event_within_window_empty_events():
    """Aucun événement → toutes les barres à 0."""
    price_ns = np.arange(10, dtype=np.int64) * _NANOSECONDS_PER_HOUR
    result = _event_within_window(price_ns, np.array([], dtype=np.int64), _NANOSECONDS_PER_HOUR)
    assert (result == 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests compute_event_features
# ═══════════════════════════════════════════════════════════════════════════════


def test_compute_event_features_columns(price_index, loaded_calendar):
    """Toutes les colonnes attendues sont présentes."""
    features = compute_event_features(price_index, loaded_calendar)
    expected_cols = [
        "event_high_within_1h_USD", "event_high_within_4h_USD", "event_high_within_24h_USD",
        "event_high_within_1h_EUR", "event_high_within_4h_EUR", "event_high_within_24h_EUR",
        "hours_since_last_nfp", "hours_since_last_fomc", "hours_to_next_event_high",
    ]
    for col in expected_cols:
        assert col in features.columns, f"Colonne {col} manquante"
    assert len(features.columns) == 9
    assert len(features) == len(price_index)


def test_compute_event_features_usd_nfp_window(price_index, loaded_calendar):
    """USD within_24h activé avant NFP."""
    features = compute_event_features(price_index, loaded_calendar)
    # 2024-01-04 18:00 = 24h avant NFP → within_24h_USD = 1
    t = pd.Timestamp("2024-01-04 18:00", tz="UTC")
    assert t in features.index
    assert features.at[t, "event_high_within_24h_USD"] == 1
    # Mais pas dans 4h
    assert features.at[t, "event_high_within_4h_USD"] == 0


def test_compute_event_features_hours_since_nfp(price_index, loaded_calendar):
    """NFP à 2024-01-05 13:30 → vérifie hours_since sur grille 6h."""
    features = compute_event_features(price_index, loaded_calendar)
    # 2024-01-05 18:00 = 4.5h après NFP
    t = pd.Timestamp("2024-01-05 18:00", tz="UTC")
    assert t in features.index
    assert np.isclose(features.at[t, "hours_since_last_nfp"], 4.5, atol=0.1)

    # 2024-01-06 12:00 = 22.5h après NFP
    t2 = pd.Timestamp("2024-01-06 12:00", tz="UTC")
    assert t2 in features.index
    assert np.isclose(features.at[t2, "hours_since_last_nfp"], 22.5, atol=0.1)

    # 2024-01-05 12:00 = 1.5h avant NFP → NaN (pas encore d'event)
    t_before = pd.Timestamp("2024-01-05 12:00", tz="UTC")
    assert t_before in features.index
    assert np.isnan(features.at[t_before, "hours_since_last_nfp"])


def test_compute_event_features_hours_to_next(price_index, loaded_calendar):
    """7.5h avant NFP → hours_to_next ≈ 7.5."""
    features = compute_event_features(price_index, loaded_calendar)
    t = pd.Timestamp("2024-01-05 06:00", tz="UTC")
    assert t in features.index
    # Le prochain High event après 06:00 le 5 jan est le NFP à 13:30 → 7.5h
    assert np.isclose(features.at[t, "hours_to_next_event_high"], 7.5, atol=0.1)


def test_compute_event_features_no_eur_high(price_index, loaded_calendar):
    """EUR High le 15 mars — hors de la fenêtre du price_index (jan-fév)."""
    features = compute_event_features(price_index, loaded_calendar)
    # Aucune barre ne devrait avoir EUR within_Xh car l'event EUR est en mars
    assert (features["event_high_within_24h_EUR"] == 0).all()


def test_compute_event_features_dtypes(price_index, loaded_calendar):
    """booléennes en int8, numériques en float32."""
    features = compute_event_features(price_index, loaded_calendar)
    for col in features.columns:
        if col.startswith("event_"):
            assert features[col].dtype == np.int8, f"{col} dtype={features[col].dtype}"
        else:
            assert features[col].dtype == np.float32, f"{col} dtype={features[col].dtype}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test anti-look-ahead
# ═══════════════════════════════════════════════════════════════════════════════


def test_anti_look_ahead_consistency(price_index, loaded_calendar):
    """f(p[:n])[-1] == f(p)[n-1] : pas de look-ahead."""
    n = 50
    full = compute_event_features(price_index, loaded_calendar)
    partial = compute_event_features(price_index[:n], loaded_calendar)
    # Comparer sur toutes les colonnes
    for col in full.columns:
        full_val = full[col].iloc[n - 1]
        partial_val = partial[col].iloc[-1]
        if pd.isna(full_val) and pd.isna(partial_val):
            continue
        assert full_val == partial_val, (
            f"Look-ahead détecté col={col}: full[{n - 1}]={full_val}, partial[-1]={partial_val}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_empty_calendar(price_index):
    """Calendrier vide → toutes les features à sentinelle (0 ou NaN)."""
    empty_cal = pd.DataFrame(columns=["timestamp", "currency", "event", "impact"])
    features = compute_event_features(price_index, empty_cal)
    # filter(regex="^event_") pour ne pas capturer hours_to_next_event_high
    assert (features.filter(regex="^event_").values == 0).all()
    assert features["hours_since_last_nfp"].isna().all()
    assert features["hours_since_last_fomc"].isna().all()
    assert features["hours_to_next_event_high"].isna().all()


def test_no_events_for_currency(price_index, loaded_calendar):
    """Events USD uniquement → features EUR = 0, sentinelles numériques."""
    # Le calendar a 1 event EUR (mars, hors price_index)
    features = compute_event_features(price_index, loaded_calendar)
    assert (features["event_high_within_24h_EUR"] == 0).all()
    # USD a bien des features actives
    assert features["event_high_within_24h_USD"].any()


def test_hours_since_last_before_first_event(price_index, loaded_calendar):
    """Avant le premier NFP → NaN."""
    features = compute_event_features(price_index, loaded_calendar)
    t = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert t in features.index
    assert np.isnan(features.at[t, "hours_since_last_nfp"])


def test_hours_to_next_after_last_event(price_index, loaded_calendar):
    """Après le dernier High event → NaN."""
    features = compute_event_features(price_index, loaded_calendar)
    # Le dernier High event est 2024-03-15 EUR (hors price_index en jan-fév)
    # ou 2024-01-31 FOMC. Price index s'arrête le 28 fév.
    t = pd.Timestamp("2024-02-01 00:00", tz="UTC")
    assert t in features.index
    # FOMC était le 31 jan, donc le prochain High event après est le CPI EUR du 15 mars
    assert not np.isnan(features.at[t, "hours_to_next_event_high"]), (
        "Il devrait y avoir un event High futur (CPI EUR 15 mars)"
    )
    # Vérifions ~ (15 mars - 1er fév) ≈ 43 jours ≈ 1032h
    val = features.at[t, "hours_to_next_event_high"]
    assert 1000 < val < 1100, f"Attendu ~1032h, obtenu {val}"
