"""Chargement et validation du calendrier économique macro.

Source : CSV Forex Factory historique.
Format attendu par ligne :
    date,time,currency,event,impact,actual,forecast,previous
    2024-01-05,13:30,USD,Non-Farm Employment Change,High,216K,170K,173K

Les noms d'événements sont normalisés via CANONICAL_EVENT_NAMES pour
stabiliser le calcul de surprise_zscore.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from learning_machine_learning.core.exceptions import DataValidationError
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)

# ── Mapping nom Forex Factory → canonical_name ──────────────────────────
CANONICAL_EVENT_NAMES: dict[str, str] = {
    # US — Emploi
    "Non-Farm Employment Change": "US_NFP",
    "NFP": "US_NFP",
    "US Non-Farm Employment Change": "US_NFP",
    "Average Hourly Earnings (MoM)": "US_AHE_MoM",
    "Average Hourly Earnings (YoY)": "US_AHE_YoY",
    "Unemployment Rate": "US_Unemployment",
    "Initial Jobless Claims": "US_JoblessClaims",
    "Continuing Jobless Claims": "US_ContClaims",
    "JOLTS Job Openings": "US_JOLTS",
    "ADP Non-Farm Employment Change": "US_ADP",
    # US — Inflation
    "USD:Consumer Price Index (MoM)": "US_CPI_MoM",
    "USD:Consumer Price Index (YoY)": "US_CPI_YoY",
    "USD:Core Consumer Price Index (MoM)": "US_CoreCPI_MoM",
    "USD:Core Consumer Price Index (YoY)": "US_CoreCPI_YoY",
    "Producer Price Index (MoM)": "US_PPI_MoM",
    "Producer Price Index (YoY)": "US_PPI_YoY",
    "Core Producer Price Index (MoM)": "US_CorePPI_MoM",
    "Core Personal Consumption Expenditures (MoM)": "US_PCE_MoM",
    "Core Personal Consumption Expenditures (YoY)": "US_PCE_YoY",
    "Personal Consumption Expenditures (MoM)": "US_PCE_MoM",
    "Personal Consumption Expenditures (YoY)": "US_PCE_YoY",
    # US — Politique monétaire
    "FOMC Statement": "US_FOMC",
    "FOMC Press Conference": "US_FOMC_Press",
    "FOMC Minutes": "US_FOMC_Minutes",
    "Fed Interest Rate Decision": "US_Fed_Rate",
    "Fed Chair Press Conference": "US_Fed_Press",
    "Fed Chair Speech": "US_Fed_Speech",
    "FOMC Member Speech": "US_FOMC_Speech",
    # US — Activité / Consommation
    "USD:Core Retail Sales (MoM)": "US_CoreRetailSales_MoM",
    "USD:Retail Sales (MoM)": "US_RetailSales_MoM",
    "USD:Gross Domestic Product (QoQ)": "US_GDP_QoQ",
    "Gross Domestic Product Annualized": "US_GDP",
    "ISM Manufacturing PMI": "US_ISM_Mfg",
    "ISM Services PMI": "US_ISM_Services",
    "Industrial Production (MoM)": "US_IndProd_MoM",
    "Durable Goods Orders (MoM)": "US_DurableGoods_MoM",
    "New Home Sales": "US_NewHomeSales",
    "Existing Home Sales": "US_ExistHomeSales",
    "Consumer Confidence": "US_ConsumerConf",
    "Michigan Consumer Sentiment": "US_MichSentiment",
    # EU — Politique monétaire
    "Main Refinancing Rate": "EU_ECB_Rate",
    "ECB Interest Rate Decision": "EU_ECB_Rate",
    "ECB Press Conference": "EU_ECB_Press",
    "ECB Monetary Policy Statement": "EU_ECB_Statement",
    "ECB President Lagarde Speech": "EU_ECB_Speech",
    "ECB Minutes": "EU_ECB_Minutes",
    # EU — Inflation
    "EU Consumer Price Index (YoY)": "EU_CPI_YoY",
    "EU Core Consumer Price Index (YoY)": "EU_CoreCPI_YoY",
    "Consumer Price Index - Core (YoY)": "EU_CoreCPI_YoY",
    "EUR:Consumer Price Index (MoM)": "EU_CPI_MoM",
    "EUR:Consumer Price Index (YoY)": "EU_CPI_YoY",
    "EUR:Core Consumer Price Index (MoM)": "EU_CoreCPI_MoM",
    "EUR:Core Consumer Price Index (YoY)": "EU_CoreCPI_YoY",
    # EU — Activité
    "EUR:Gross Domestic Product (YoY)": "EU_GDP_YoY",
    "EUR:Gross Domestic Product (QoQ)": "EU_GDP_QoQ",
    "German GDP (QoQ)": "DE_GDP_QoQ",
    "German ZEW Economic Sentiment": "DE_ZEW",
    "German Ifo Business Climate": "DE_Ifo",
    "German CPI (MoM)": "DE_CPI_MoM",
    "German Retail Sales (MoM)": "DE_RetailSales_MoM",
    "German Industrial Production (MoM)": "DE_IndProd_MoM",
    # UK — Politique monétaire
    "BOE MPC Official Bank Rate Votes": "UK_BOE_Rate",
    "BoE Interest Rate Decision": "UK_BOE_Rate",
    "Bank of England Minutes": "UK_BOE_Minutes",
    "BOE Governor Bailey Speech": "UK_BOE_Speech",
    # UK — Inflation / Activité
    "GBP:Consumer Price Index (MoM)": "UK_CPI_MoM",
    "GBP:Consumer Price Index (YoY)": "UK_CPI_YoY",
    "GBP:Core Consumer Price Index (MoM)": "UK_CoreCPI_MoM",
    "GBP:Core Consumer Price Index (YoY)": "UK_CoreCPI_YoY",
    "GBP:Retail Sales (MoM)": "UK_RetailSales_MoM",
    "GBP:Gross Domestic Product (MoM)": "UK_GDP_MoM",
    "GBP:Gross Domestic Product (YoY)": "UK_GDP_YoY",
    "Claimant Count Change": "UK_ClaimantCount",
    "Average Earnings Index +Bonus": "UK_AHE",
    "Manufacturing Production (MoM)": "UK_MfgProd_MoM",
    # JP — Politique monétaire
    "BoJ Interest Rate Decision": "JP_BOJ_Rate",
    "BoJ Press Conference": "JP_BOJ_Press",
    "BoJ Monetary Policy Statement": "JP_BOJ_Statement",
    "BOJ Governor Kuroda Speech": "JP_BOJ_Speech",
    "BOJ Governor Ueda Speech": "JP_BOJ_Speech",
    # JP — Inflation / Activité
    "Tokyo Consumer Price Index (YoY)": "JP_TokyoCPI_YoY",
    "National Consumer Price Index (YoY)": "JP_CPI_YoY",
    "JPY:Consumer Price Index (MoM)": "JP_CPI_MoM",
    "JPY:Consumer Price Index (YoY)": "JP_CPI_YoY",
    "JPY:Gross Domestic Product (QoQ)": "JP_GDP_QoQ",
    "JPY:Gross Domestic Product (YoY)": "JP_GDP_YoY",
    "Tankan Large Manufacturers Index": "JP_Tankan",
    # CH — Politique monétaire
    "SNB Interest Rate Decision": "CH_SNB_Rate",
    "SNB Press Conference": "CH_SNB_Press",
    # CH — Inflation / Activité
    "CHF:Consumer Price Index (MoM)": "CH_CPI_MoM",
    "CHF:Consumer Price Index (YoY)": "CH_CPI_YoY",
    "CHF:Gross Domestic Product (QoQ)": "CH_GDP_QoQ",
    "CHF:Gross Domestic Product (YoY)": "CH_GDP_YoY",
    "KOF Leading Indicator": "CH_KOF",
}


def _parse_actual_value(raw: str) -> float | None:
    """Parse une valeur 'actual' avec suffixes K, M, B, T, %, <, |, votes.

    >>> _parse_actual_value("216K")
    216000.0
    >>> _parse_actual_value("-3.2M")
    -3200000.0
    >>> _parse_actual_value("0.3%")
    0.3
    >>> _parse_actual_value("2.18T")
    2180000000000.0
    >>> _parse_actual_value("<0.10%")
    0.1
    >>> _parse_actual_value("2.75|3.0")
    2.75
    >>> _parse_actual_value("")
    None
    """
    if pd.isna(raw) or str(raw).strip() in ("", "-", "N/A", "n/a"):
        return None
    s = str(raw).strip().replace(",", "")

    # Votes style "0-1-8" → pas parsable, retourner None silencieusement
    if "-" in s and s.count("-") >= 2 and all(
        part.isdigit() for part in s.replace("-", " ").split()
    ):
        return None

    # Révision "2.75|3.0" → garder la première valeur
    if "|" in s:
        s = s.split("|")[0].strip()

    # Opérateur de comparaison "<0.10%", ">0.5" → supprimer le préfixe
    if s and s[0] in ("<", ">"):
        s = s[1:]

    # Supprimer le signe % (valeur déjà en %)
    s = s.replace("%", "")
    # Extraire le suffixe multiplicateur
    multiplier = 1.0
    upper = s.upper()
    if upper.endswith("K"):
        multiplier = 1_000.0
        s = s[:-1]
    elif upper.endswith("M"):
        multiplier = 1_000_000.0
        s = s[:-1]
    elif upper.endswith("B"):
        multiplier = 1_000_000_000.0
        s = s[:-1]
    elif upper.endswith("T"):
        multiplier = 1_000_000_000_000.0
        s = s[:-1]
    if s == "" or s == "-":
        return None
    try:
        return float(s) * multiplier
    except (ValueError, TypeError):
        logger.warning("Valeur 'actual' non parsable : %r", raw)
        return None


def _detect_timezone_utc(df: pd.DataFrame, col: str = "time") -> bool:
    """Heuristique : détecte si les heures sont déjà UTC.

    Forex Factory exporte parfois en US/Eastern (ET). Si les heures typiques
    de release US (NFP=8:30 ET = 13:30 UTC) apparaissent comme 8 ou 9, on
    est en ET et il faut convertir.

    Returns:
        True si les heures semblent déjà UTC.
    """
    if col not in df.columns:
        return True  # assume UTC par défaut
    hours = pd.to_numeric(df[col].astype(str).str[:2], errors="coerce").dropna()
    if hours.empty:
        return True
    # Si on voit des heures autour de 13-14, UTC probable
    mean_hour = float(hours.mean())
    # Les releases US majeurs sont à 8:30 ou 10:00 ET = 13:30 ou 15:00 UTC
    # Si mean_hour < 10, probablement ET
    return mean_hour >= 11.0


def _normalize_time_col(time_series: pd.Series) -> pd.Series:
    """Normalise une colonne time vers le format 24h HH:MM.

    Gère : '13:30', '13:30:00', '1:30pm', 'All Day', 'Tentative'.
    """
    import re as _re

    def _convert_one(t: str) -> str:
        t = str(t).strip().lower().replace("\xa0", "")
        if not t or t in ("all day", "tentative", "none", "nan"):
            return "12:00"
        # Déjà en 24h ? "13:30" ou "13:30:00"
        if _re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", t):
            return t.split(":")[0].zfill(2) + ":" + t.split(":")[1]
        # Format 12h : "6:01pm", "2:15am", "12:00pm"
        m = _re.match(r"^(\d{1,2}):(\d{2})\s*(am|pm)$", t)
        if m:
            hour = int(m.group(1))
            minute = m.group(2)
            ampm = m.group(3)
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            return f"{hour:02d}:{minute}"
        return "12:00"

    return time_series.apply(_convert_one)


def _convert_to_utc(df: pd.DataFrame, date_col: str, time_col: str) -> "pd.DatetimeIndex":
    """Convertit date + time en datetime UTC.

    Gère formats 12h (6:01pm), 24h (13:30), All Day.
    """
    time_norm = _normalize_time_col(df[time_col])
    raw = df[date_col].astype(str) + " " + time_norm.astype(str)

    dt = pd.to_datetime(raw, format="%Y-%m-%d %H:%M", errors="coerce")

    # Fallback pour les lignes qui auraient échoué
    mask_na = dt.isna()
    if mask_na.any():
        dt_fallback = pd.to_datetime(raw[mask_na], errors="coerce")
        dt = dt.where(~mask_na, dt_fallback)

    dt = pd.DatetimeIndex(dt.dt.tz_localize("UTC"))
    return dt


def validate_calendar_schema(df: pd.DataFrame) -> None:
    """Valide le schéma du DataFrame calendrier.

    Raise DataValidationError si colonnes manquantes ou types incorrects.
    """
    required_cols = {"timestamp", "currency", "event_name", "impact"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise DataValidationError(
            f"Colonnes calendrier manquantes : {sorted(missing)}. "
            f"Colonnes présentes : {sorted(df.columns)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise DataValidationError(
            f"Colonne 'timestamp' doit être datetime64, reçu {df['timestamp'].dtype}"
        )

    valid_impacts = {"Low", "Medium", "High", "low", "medium", "high"}
    actual_impacts = set(df["impact"].dropna().unique())
    unknown = actual_impacts - valid_impacts
    if unknown:
        raise DataValidationError(
            f"Valeurs d'impact inconnues : {sorted(unknown)}. "
            f"Attendu : {sorted(valid_impacts)}"
        )

    logger.info(
        "Schéma calendrier validé : %d événements, %d devises, %d types.",
        len(df),
        df["currency"].nunique(),
        df["event_name"].nunique(),
    )


def load_calendar(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    data_dir: str | Path = "data/raw/economic_calendar",
) -> pd.DataFrame:
    """Charge tous les CSV du calendrier entre start et end.

    Args:
        start: Début de la plage de dates (inclus), str ou Timestamp.
        end: Fin de la plage de dates (inclus), str ou Timestamp.
        data_dir: Dossier contenant les CSV (ex: 2010.csv ... 2025.csv).

    Returns:
        DataFrame avec colonnes :
        - timestamp: datetime64[ns, UTC]
        - currency: str (USD, EUR, GBP, JPY, CHF)
        - event_name: str (canonical)
        - impact: str (Low, Medium, High)
        - actual: float | None
        - forecast: float | None
        - previous: float | None
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tz is None:
        start = start.tz_localize("UTC")
    if end.tz is None:
        end = end.tz_localize("UTC")
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dossier calendrier introuvable : {data_path.resolve()}"
        )

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {data_path.resolve()}"
        )

    logger.info("Chargement de %d fichiers calendrier depuis %s...", len(csv_files), data_path)

    frames: list[pd.DataFrame] = []
    is_utc: bool | None = None

    for csv_path in csv_files:
        logger.debug("  Lecture de %s", csv_path.name)
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as e:
            logger.warning("  Échec lecture %s : %s — ignoré.", csv_path.name, e)
            continue

        if df.empty:
            logger.debug("  %s est vide — ignoré.", csv_path.name)
            continue

        # Normaliser les noms de colonnes (lowercase strip)
        df.columns = [c.strip().lower() for c in df.columns]

        # Détecter le timezone une seule fois
        if is_utc is None:
            is_utc = _detect_timezone_utc(df, "time")
            logger.info("Détection fuseau horaire : %s", "UTC" if is_utc else "US/Eastern")

        # Colonnes obligatoires
        required = {"date", "time", "currency", "event", "impact"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "  %s : colonnes manquantes %s — ignoré.",
                csv_path.name, sorted(missing),
            )
            continue

        # Parse timestamp
        try:
            df["timestamp"] = _convert_to_utc(df, "date", "time")
        except Exception as e:
            logger.warning("  %s : échec parse timestamp — %s — ignoré.", csv_path.name, e)
            continue

        # Si les heures sont en ET, ajouter le décalage UTC (5h ou 4h selon DST
        # approximé : +5h standard)
        if not is_utc:
            df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=5)

        # Normaliser impact
        df["impact"] = df["impact"].str.strip().str.title()

        # Normaliser event_name — lookup à deux niveaux : currency:event puis event seul
        raw_event = df["event"].str.strip()
        raw_currency = df["currency"].str.strip().str.upper()
        two_level = (raw_currency + ":" + raw_event).map(CANONICAL_EVENT_NAMES)
        fallback = raw_event.map(CANONICAL_EVENT_NAMES)
        df["event_name"] = two_level.fillna(fallback).fillna(raw_event)

        # Parse valeurs numériques
        for col in ("actual", "forecast", "previous"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_actual_value).astype("float64")
            else:
                df[col] = None

        # Garder les colonnes utiles
        keep_cols = ["timestamp", "currency", "event_name", "impact",
                      "actual", "forecast", "previous"]
        frames.append(df[keep_cols])

    if not frames:
        raise DataValidationError(
            f"Aucun CSV calendrier valide trouvé dans {data_path.resolve()}"
        )

    combined = pd.concat(frames, ignore_index=True)

    # Filtrer par plage de dates
    combined = combined[
        (combined["timestamp"] >= start) & (combined["timestamp"] <= end)
    ]

    # Trier par timestamp
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Valider le schéma final
    validate_calendar_schema(combined)

    n_high = (combined["impact"] == "High").sum()
    n_medium = (combined["impact"] == "Medium").sum()
    logger.info(
        "Calendrier chargé : %d événements (%d High, %d Medium) sur [%s, %s].",
        len(combined), n_high, n_medium,
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
    )

    return combined
