"""Chargement et validation stricte des CSV OHLCV.

Contrat CSV — chaque fichier ``data/raw/<ASSET>/<*>_<TF>.csv`` :

Colonnes obligatoires (ordre indifférent, casse insensible) :
- une colonne temps : ``Time`` / ``Timestamp`` / ``Date`` / ``Datetime`` / ``Gmt time``
  au format ISO 8601 ('2024-01-15 13:00:00' ou '2024-01-15T13:00:00Z')
- ``open, high, low, close`` : float
- ``volume`` : float (peut être 0 pour les indices CFD)
- ``spread`` : float, OPTIONNEL. Tolère aussi le cas « export MT5 » où le header
  ne nomme que 6 colonnes (Time..Volume) mais chaque ligne contient une 7e valeur
  non nommée : celle-ci est alors interprétée comme ``Spread``.

Séparateur : détecté automatiquement (tabulation, point-virgule ou virgule).

Contraintes validées (sinon ``DataValidationError``) :
- timestamps strictement croissants, sans doublon
- ``open/high/low/close > 0`` (pas de prix négatif ou nul)
- ``volume >= 0``
- ``high >= max(open, close)`` et ``low <= min(open, close)``
- gaps : un trou > seuil (D1: 7j, H4: 3j, H1: 2j) est toléré s'il est explicable
  par un weekend ou un jour férié XTB (cf. ``app/config/calendar.py``), sinon erreur.
- les lignes contenant des NaN sont droppées silencieusement (warmup).

Le DataFrame retourné est indexé par ``timestamp`` (UTC, monotone) et expose les
colonnes capitalisées ``Open, High, Low, Close, Volume`` (+ ``Spread`` si présent).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config.calendar import is_normal_gap
from app.core.exceptions import DataValidationError
from app.core.logging import get_logger
from app.core.retry import retry_with_backoff

logger = get_logger(__name__)

# Colonnes OHLCV obligatoires (hors colonne temps), en minuscules.
_REQUIRED = ("open", "high", "low", "close", "volume")
# Alias acceptés pour la colonne temps (minuscules).
_TIME_ALIASES = ("timestamp", "time", "date", "datetime", "gmt time", "local time")
# Mapping minuscule -> nom canonique de sortie.
_CANONICAL = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "spread": "Spread",
}
# Gap maximum toléré (en heures) avant analyse weekend/holiday, par timeframe.
_MAX_GAP_HOURS = {"D1": 7 * 24, "H4": 3 * 24, "H1": 2 * 24, "M15": 6, "M5": 2}
# Timeframes reconnus dans les noms de fichiers (utilisé aussi par le registry).
KNOWN_TIMEFRAMES = ("D1", "H4", "H1", "M15", "M5", "W1")


def _find_csv(data_root: Path, asset: str, tf: str) -> Path:
    """Localise l'unique CSV ``*_<TF>.csv`` dans ``data_root/<asset>/``.

    Lève ``DataValidationError`` si aucun ou plusieurs fichiers correspondent.
    Ne lit jamais le contenu des fichiers.
    """
    asset_dir = Path(data_root) / asset
    matches = sorted(asset_dir.glob(f"*_{tf}.csv")) if asset_dir.is_dir() else []
    if not matches:
        raise DataValidationError(
            f"Aucun CSV introuvable pour {asset}/{tf} (cherché : {asset_dir}/*_{tf}.csv)"
        )
    if len(matches) > 1:
        raise DataValidationError(
            f"Ambiguïté : plusieurs CSV pour {asset}/{tf} : {[m.name for m in matches]}"
        )
    return matches[0]


def _sniff_separator(header_line: str) -> str:
    """Détecte le séparateur de colonnes à partir de la ligne d'en-tête."""
    for sep in ("\t", ";", ","):
        if sep in header_line:
            return sep
    return "\t"


@retry_with_backoff(max_attempts=3, exceptions=(OSError,))
def load_asset(asset: str, tf: str, data_root: Path = Path("data/raw")) -> pd.DataFrame:
    """Charge et valide le CSV OHLCV d'un couple (asset, timeframe).

    Retry 3× sur ``OSError`` (verrou fichier, disque transitoire).
    Retourne un DataFrame indexé par ``timestamp`` UTC, colonnes capitalisées.
    """
    path = _find_csv(data_root, asset, tf)

    header_line = path.read_text(encoding="utf-8").splitlines()[0]
    sep = _sniff_separator(header_line)
    names = [c.strip().lower() for c in header_line.split(sep)]

    # Lecture des données seules (header ignoré) pour gérer le cas « 7e colonne
    # non nommée » (Spread implicite des exports MT5).
    raw = pd.read_csv(path, sep=sep, header=None, skiprows=1)
    n_data_cols = raw.shape[1]
    if n_data_cols == len(names):
        cols = list(names)
    elif n_data_cols == len(names) + 1:
        cols = [*names, "spread"]
    else:
        raise DataValidationError(
            f"{path.name} : {n_data_cols} colonnes de données pour {len(names)} en-têtes"
        )
    raw.columns = cols

    # Colonne temps.
    time_col = next((c for c in cols if c in _TIME_ALIASES), None)
    if time_col is None:
        time_col = cols[0]  # repli : première colonne

    # Colonnes OHLCV obligatoires.
    missing = [c for c in _REQUIRED if c not in cols]
    if missing:
        raise DataValidationError(f"{path.name} : colonnes manquantes {missing}")

    # Index temporel UTC.
    df = raw.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.set_index(time_col).sort_index()
    df.index.name = "timestamp"

    if df.index.isna().any():
        raise DataValidationError(f"{path.name} : timestamps invalides (non parsables)")
    if df.index.has_duplicates:
        raise DataValidationError(f"{path.name} : timestamps dupliqués")

    # Conversion numérique + suppression des lignes NaN (warmup).
    value_cols = [c for c in cols if c in _CANONICAL]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[value_cols].dropna()

    # Prix strictement positifs.
    if (df[["open", "high", "low", "close"]] <= 0).to_numpy().any():
        raise DataValidationError(f"{path.name} : prix négatifs ou nuls détectés")

    # Volume non négatif.
    if (df["volume"] < 0).any():
        raise DataValidationError(f"{path.name} : volumes négatifs détectés")

    # Cohérence OHLC.
    hi_ok = df["high"] >= df[["open", "close"]].max(axis=1)
    lo_ok = df["low"] <= df[["open", "close"]].min(axis=1)
    n_incoherent = int((~(hi_ok & lo_ok)).sum())
    if n_incoherent:
        raise DataValidationError(f"{path.name} : {n_incoherent} barres OHLC incohérentes")

    # Analyse des gaps (normal weekend/holiday vs anormal = données manquantes).
    _check_gaps(df.index, asset, tf, path.name)

    # Renommage canonique.
    df = df.rename(columns=_CANONICAL)
    ordered = [_CANONICAL[c] for c in value_cols]
    return df[ordered]


def _check_gaps(index: pd.DatetimeIndex, asset: str, tf: str, fname: str) -> None:
    """Lève ``DataValidationError`` si un gap dépassant le seuil n'est pas explicable
    par un weekend ou un jour férié XTB."""
    if len(index) < 2:
        return
    gaps_h = index.to_series().diff().dt.total_seconds() / 3600.0
    threshold = _MAX_GAP_HOURS.get(tf, 24)
    big = gaps_h[gaps_h > threshold]
    n_normal = n_abnormal = 0
    for ts in big.index:
        loc = index.get_loc(ts)
        prev_ts = index[loc - 1]
        if is_normal_gap(asset, prev_ts.to_pydatetime(), ts.to_pydatetime()):
            n_normal += 1
        else:
            n_abnormal += 1
    if n_normal:
        logger.info("%s : %d gaps normaux (weekend/holiday)", fname, n_normal)
    if n_abnormal:
        logger.error("%s : %d gaps anormaux (données manquantes)", fname, n_abnormal)
        raise DataValidationError(f"{fname} : {n_abnormal} gaps anormaux détectés")
