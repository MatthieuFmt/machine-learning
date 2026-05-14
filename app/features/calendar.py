"""Features dérivées du calendrier économique macro.

Toutes les fonctions sont anti-look-ahead : à l'instant t,
seule l'information ≤ t est utilisée.

Colonnes générées :
- minutes_to_next_event : distance au prochain événement high-impact
- minutes_since_last_event : distance depuis le dernier événement high-impact
- surprise_zscore : z-score de surprise (actual - forecast) / rolling_std
- near_high_impact_event : flag booléen (1 si dans ±2h d'un événement high)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.testing.look_ahead_validator import look_ahead_safe

logger = get_logger(__name__)

# Valeur sentinelle signifiant "pas d'événement proche" (≈ 69 jours)
_SENTINEL_MINUTES: float = 99999.0


def _filter_events_by_impact(
    events_df: pd.DataFrame,
    impact: str,
) -> pd.DataFrame:
    """Filtre les événements par niveau d'impact."""
    impact_map = {"high": ["High"], "medium": ["High", "Medium"]}
    labels = impact_map.get(impact.lower(), impact_map["high"])
    return events_df[events_df["impact"].isin(labels)]


@look_ahead_safe
def compute_minutes_to_next_event(
    timestamps: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    impact: str = "high",
) -> pd.Series:
    """Distance en minutes jusqu'au prochain événement ≥ impact.

    Algorithme O(n log m) via searchsorted :
    1. Extrait les timestamps des événements filtrés par impact.
    2. Pour chaque timestamp OHLC, searchsorted dans event_times.
    3. Δt = event_time - ohlc_time (en minutes).
    4. Si pas d'événement futur → _SENTINEL_MINUTES.

    Args:
        timestamps: Index H1 à enrichir (pd.DatetimeIndex, tz-aware UTC).
        events_df: DataFrame calendrier avec colonne 'timestamp'.
        impact: Seuil d'impact ('high' ou 'medium').

    Returns:
        pd.Series de float64, même index que timestamps.
    """
    filtered = _filter_events_by_impact(events_df, impact)
    if filtered.empty:
        logger.info(
            "Aucun événement %s-impact dans la plage — sentinelle partout.", impact
        )
        return pd.Series(_SENTINEL_MINUTES, index=timestamps, dtype="float64")

    # Timestamps des événements triés
    event_times = np.asarray(filtered["timestamp"].values, dtype="datetime64[ns]")
    # Timestamps OHLC en numpy datetime64
    ohlc_times = np.asarray(timestamps.values, dtype="datetime64[ns]")

    # searchsorted : pour chaque ohlc_time, trouve l'index d'insertion dans event_times
    event_indices = np.searchsorted(event_times, ohlc_times, side="left")

    # Créer un tableau de event_time ou NaT
    result_minutes = np.full(len(ohlc_times), _SENTINEL_MINUTES, dtype="float64")

    has_future = event_indices < len(event_times)
    if has_future.any():
        future_events = event_times[event_indices[has_future]]
        # Delta en minutes (timedelta64 → float64 minutes)
        deltas = (future_events - ohlc_times[has_future]) / np.timedelta64(1, "m")
        result_minutes[has_future] = deltas.astype("float64")

    result = pd.Series(result_minutes, index=timestamps, dtype="float64")
    n_within_2h = (result <= 120).sum()
    logger.debug(
        "minutes_to_next_event (%s): %d/%d barres dans les 2h d'un événement.",
        impact, n_within_2h, len(result),
    )
    return result


@look_ahead_safe
def compute_minutes_since_last_event(
    timestamps: pd.DatetimeIndex,
    events_df: pd.DataFrame,
    impact: str = "high",
) -> pd.Series:
    """Distance en minutes depuis le dernier événement ≥ impact.

    Symétrique de compute_minutes_to_next_event, utilise searchsorted
    pour trouver l'événement précédent le plus proche.

    Args:
        timestamps: Index H1 à enrichir.
        events_df: DataFrame calendrier avec colonne 'timestamp'.
        impact: Seuil d'impact ('high' ou 'medium').

    Returns:
        pd.Series de float64, même index que timestamps.
        _SENTINEL_MINUTES si pas d'événement passé.
    """
    filtered = _filter_events_by_impact(events_df, impact)
    if filtered.empty:
        return pd.Series(_SENTINEL_MINUTES, index=timestamps, dtype="float64")

    event_times = np.asarray(filtered["timestamp"].values, dtype="datetime64[ns]")
    ohlc_times = np.asarray(timestamps.values, dtype="datetime64[ns]")

    # searchsorted side='right' pour trouver le premier événement > ohlc_time
    event_indices = np.searchsorted(event_times, ohlc_times, side="right")

    result_minutes = np.full(len(ohlc_times), _SENTINEL_MINUTES, dtype="float64")

    # L'événement précédent est à event_indices - 1
    has_past = event_indices > 0
    if has_past.any():
        prev_idx = event_indices[has_past] - 1
        prev_events = event_times[prev_idx]
        deltas = (ohlc_times[has_past] - prev_events) / np.timedelta64(1, "m")
        result_minutes[has_past] = deltas.astype("float64")

    result = pd.Series(result_minutes, index=timestamps, dtype="float64")
    return result


@look_ahead_safe
def compute_surprise_zscore(
    events_df: pd.DataFrame,
    lookback: int = 50,
    min_history: int = 20,
) -> pd.DataFrame:
    """Calcule le z-score de surprise (actual - forecast) / rolling_std.

    Anti-look-ahead strict :
    - Pour chaque événement à t_i, calcule le rolling std des
      (actual - forecast) sur les `lookback` événements précédents
      du même canonical event_name.
    - Si < min_history occurrences historiques → surprise_zscore = NaN.
    - zscore = (actual_i - forecast_i) / rolling_std.

    Args:
        events_df: DataFrame calendrier (doit avoir 'event_name', 'actual',
                   'forecast', 'timestamp').
        lookback: Nombre d'occurrences passées pour rolling std.
        min_history: Nombre minimum d'occurrences pour un zscore fiable.

    Returns:
        DataFrame avec colonnes ajoutées :
        - surprise: float64 (actual - forecast)
        - surprise_zscore: float64 (NaN si historique insuffisant)
    """
    df = events_df.copy()
    df = df.sort_values("timestamp")

    # Calculer la surprise brute
    df["surprise"] = df["actual"] - df["forecast"]

    # Rolling backward : pour chaque event_name, rolling std des surprises passées
    df["surprise_std"] = np.nan

    for _event_name, group in df.groupby("event_name"):
        indices = group.index
        surprise_vals = df.loc[indices, "surprise"].values

        # Rolling std backward (pas de look-ahead)
        rolling_stds = np.full(len(surprise_vals), np.nan)
        for i in range(len(surprise_vals)):
            start = max(0, i - lookback)
            window = surprise_vals[start:i]  # seulement les valeurs avant i
            if len(window) >= min_history:
                rolling_stds[i] = np.nanstd(window)

        df.loc[indices, "surprise_std"] = rolling_stds

    # Z-score
    df["surprise_zscore"] = np.where(
        df["surprise_std"].notna() & (df["surprise_std"] > 0),
        df["surprise"] / df["surprise_std"],
        np.nan,
    )

    logger.info(
        "Surprise zscore : %d/%d événements avec zscore valide (min_history=%d).",
        df["surprise_zscore"].notna().sum(), len(df), min_history,
    )

    return df[["timestamp", "event_name", "surprise", "surprise_zscore"]]


@look_ahead_safe
def merge_calendar_features(
    ohlc: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute les colonnes calendrier au DataFrame OHLC H1.

    Utilise merge_asof(direction='backward') pour le surprise_zscore
    (la surprise n'est connue qu'après le release).

    Colonnes ajoutées :
    - minutes_to_next_event: float64
    - minutes_since_last_event: float64
    - surprise_zscore: float64 (NaN si pas de release récent)
    - near_high_impact_event: int8 (1 si ≤ 120 min d'un high-impact)

    Args:
        ohlc: DataFrame H1 avec DatetimeIndex.
        events_df: DataFrame calendrier validé.

    Returns:
        DataFrame avec les 4 colonnes ajoutées.
    """
    if not isinstance(ohlc.index, pd.DatetimeIndex):
        raise ValueError("L'index de ohlc doit être un DatetimeIndex.")

    timestamps = ohlc.index

    # Normaliser les timestamps : OHLC est tz-naive datetime64[ns],
    # calendrier est datetime64[us, UTC] → conversion nécessaire
    calendar_tz = events_df["timestamp"].dt.tz
    if calendar_tz is not None:
        events_df = events_df.copy()
        events_df["timestamp"] = pd.DatetimeIndex(
            events_df["timestamp"].dt.tz_convert(None)
        ).as_unit("ns")

    # 1. Distances temporelles
    minutes_next = compute_minutes_to_next_event(timestamps, events_df, impact="high")
    minutes_last = compute_minutes_since_last_event(timestamps, events_df, impact="high")

    # 2. Surprise zscore avec merge_asof backward
    # D'abord calculer les zscores sur tout le calendrier
    zscores = compute_surprise_zscore(events_df)

    # merge_asof backward : pour chaque barre H1, prend le dernier zscore connu
    if not zscores.empty:
        zscores_indexed = zscores.set_index("timestamp")[["surprise_zscore"]]
        ohlc_with_surprise = pd.merge_asof(
            ohlc[[]],  # DataFrame vide avec juste l'index
            zscores_indexed,
            left_index=True,
            right_index=True,
            direction="backward",
        )
        surprise_zscore = ohlc_with_surprise["surprise_zscore"]
    else:
        surprise_zscore = pd.Series(np.nan, index=timestamps, dtype="float64")

    # 3. Flag near_high_impact_event
    near_event = ((minutes_next <= 120) | (minutes_last <= 120)).astype("int8")

    result = pd.DataFrame(
        {
            "minutes_to_next_event": minutes_next,
            "minutes_since_last_event": minutes_last,
            "surprise_zscore": surprise_zscore,
            "near_high_impact_event": near_event,
        },
        index=timestamps,
    )

    logger.info(
        "Features calendrier mergées : %d colonnes, near_high_impact=%d barres.",
        len(result.columns), near_event.sum(),
    )

    return result
