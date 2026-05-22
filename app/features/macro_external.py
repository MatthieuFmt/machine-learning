"""Features macro externes (DXY, VIX, yield slope) — Phase F5.

Source : yfinance (sans clé API). Les séries sont mises en cache disque
sous `data/raw/macro/<SYMBOL>.csv` pour éviter de retoucher au réseau
à chaque appel.

Anti-look-ahead : la fonction publique `add_external_macro` shifte de
1 barre toutes les valeurs macro avant le merge_asof, de sorte qu'à
l'instant `t` on n'utilise QUE des fermetures macro déjà publiées en `t-1`.

Symboles yfinance utilisés :
    DXY  → DX-Y.NYB     (Dollar Index, quotidien)
    VIX  → ^VIX          (CBOE Volatility Index, quotidien)
    10Y  → ^TNX          (10-Year Treasury Yield)
    3M   → ^IRX          (13-Week T-Bill Yield, proxy 3M)

Features livrées :
    - dxy_zscore_60        : (Close - rolling_mean_60) / rolling_std_60 sur DXY
    - vix_level            : Close brut du VIX
    - vix_zscore_60        : zscore 60j du VIX
    - yield_slope_10y_3m   : ^TNX - ^IRX (en points de %)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.testing.look_ahead_validator import look_ahead_safe

logger = get_logger(__name__)

MACRO_CACHE_DIR = Path("data/raw/macro")

# Mapping nom logique → symbole yfinance
SYMBOLS: dict[str, str] = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
    "TNX": "^TNX",
    "IRX": "^IRX",
}

ZSCORE_WINDOW: int = 60


def _cache_path(name: str) -> Path:
    return MACRO_CACHE_DIR / f"{name}.csv"


def _load_from_cache(name: str) -> pd.Series | None:
    """Lit le CSV cache si présent ; retourne None sinon."""
    path = _cache_path(name)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df.set_index("Date")["Close"].sort_index()


def _save_to_cache(name: str, series: pd.Series) -> None:
    """Persiste la série en CSV (1 colonne Close, index Date UTC)."""
    MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = series.rename("Close").to_frame()
    out.index.name = "Date"
    out.to_csv(_cache_path(name))


def _fetch_yfinance(symbol: str, start: str = "2005-01-01") -> pd.Series:
    """Télécharge une série Close depuis yfinance et la retourne en UTC.

    Args:
        symbol: Symbole yfinance (ex: '^VIX').
        start: Date de début (ISO).

    Returns:
        pd.Series indexée par DatetimeIndex UTC, valeurs = Close ajusté.
    """
    import yfinance as yf

    raw = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance: aucune donnée pour {symbol}")
    # yfinance retourne parfois un MultiIndex (Price, Ticker) — on aplatit
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].dropna()
    close.index = pd.to_datetime(close.index, utc=True)
    return close.sort_index()


@look_ahead_safe
def load_macro_series(name: str, refresh: bool = False) -> pd.Series:
    """Charge une série macro depuis le cache, ou télécharge si absente.

    Args:
        name: Clé logique dans SYMBOLS ('DXY', 'VIX', 'TNX', 'IRX').
        refresh: Si True, force un re-téléchargement.

    Returns:
        pd.Series Close indexée UTC.
    """
    if name not in SYMBOLS:
        raise ValueError(f"Macro inconnu : {name}. Connus : {list(SYMBOLS)}")

    if not refresh:
        cached = _load_from_cache(name)
        if cached is not None and not cached.empty:
            logger.info("macro_cache_hit", extra={"context": {"name": name, "n": len(cached)}})
            return cached

    symbol = SYMBOLS[name]
    logger.info("macro_download", extra={"context": {"name": name, "symbol": symbol}})
    series = _fetch_yfinance(symbol)
    _save_to_cache(name, series)
    return series


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Z-score rolling sans look-ahead (mean/std sur fenêtre passée)."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)


@look_ahead_safe
def build_macro_dataframe(refresh: bool = False) -> pd.DataFrame:
    """Construit un DataFrame quotidien de toutes les features macro.

    L'index est la date UTC à la résolution du jour. Les NaN initiaux
    (warmup du zscore) sont conservés.

    Args:
        refresh: Si True, re-télécharge toutes les séries.

    Returns:
        DataFrame avec colonnes [dxy_zscore_60, vix_level, vix_zscore_60,
        yield_slope_10y_3m].
    """
    dxy = load_macro_series("DXY", refresh=refresh)
    vix = load_macro_series("VIX", refresh=refresh)
    tnx = load_macro_series("TNX", refresh=refresh)
    irx = load_macro_series("IRX", refresh=refresh)

    # Aligner toutes les séries sur un index union (puis ffill court terme
    # pour combler les jours fériés US où une seule des deux yields manque)
    df = pd.concat(
        {
            "DXY": dxy,
            "VIX": vix,
            "TNX": tnx,
            "IRX": irx,
        },
        axis=1,
    ).sort_index()

    out = pd.DataFrame(index=df.index)
    out["dxy_zscore_60"] = _zscore(df["DXY"], ZSCORE_WINDOW)
    out["vix_level"] = df["VIX"]
    out["vix_zscore_60"] = _zscore(df["VIX"], ZSCORE_WINDOW)
    out["yield_slope_10y_3m"] = df["TNX"] - df["IRX"]

    return out


@look_ahead_safe
def add_external_macro(
    df: pd.DataFrame,
    asset: str | None = None,  # noqa: ARG001  (param conservé pour compat spec)
    refresh: bool = False,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ajoute les features macro DXY/VIX/yield à un DataFrame OHLC indexé time.

    Les valeurs macro sont décalées de +1 jour avant le merge_asof pour
    garantir qu'à l'instant `t` on n'utilise QUE des fermetures publiées
    en `t-1` (anti-look-ahead).

    Args:
        df: DataFrame OHLCV indexé DatetimeIndex UTC trié.
        asset: Inutilisé (param conservé pour la spec — toutes les features
            sont communes à tous les actifs).
        refresh: Force le re-téléchargement des séries macro.
        macro_df: DataFrame macro pré-construit (utile pour tests, évite
            de hit le réseau / cache). Si None, appelle build_macro_dataframe.

    Returns:
        Copie de df avec colonnes supplémentaires : dxy_zscore_60,
        vix_level, vix_zscore_60, yield_slope_10y_3m.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_external_macro : df.index doit être un DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("add_external_macro : df.index doit être tz-aware (UTC)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("add_external_macro : df.index doit être trié")

    if macro_df is None:
        macro_df = build_macro_dataframe(refresh=refresh)

    # Décalage anti-look-ahead : la valeur du jour D ne devient utilisable
    # qu'à partir du jour D+1 00:00 UTC (les marchés US ferment ~21h UTC,
    # donc la prochaine barre H1 EU démarrant à 00:00 le lendemain est sûre).
    shifted = macro_df.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    shifted = shifted.sort_index()

    # merge_asof : on prépare deux frames avec un nom de timestamp explicite
    original_index_name = df.index.name
    left = df.copy()
    left.index.name = "_time"
    left = left.reset_index()

    right = shifted.copy()
    right.index.name = "_macro_time"
    right = right.reset_index()

    merged = pd.merge_asof(
        left.sort_values("_time"),
        right.sort_values("_macro_time"),
        left_on="_time",
        right_on="_macro_time",
        direction="backward",
    )

    merged = merged.drop(columns=["_macro_time"]).set_index("_time")
    merged.index.name = original_index_name
    return merged
