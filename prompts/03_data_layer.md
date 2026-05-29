# Prompt 03 — Couche de données standardisée

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/02_cleanup.md`
4. `prompts/02b_quality_gates.md` (outillage qualité installé)

## Objectif
Créer une couche `app/data/` propre qui (a) définit un contrat strict pour les CSV d'entrée, (b) découvre dynamiquement les actifs disponibles dans `data/raw/`, (c) valide chaque fichier au chargement (schéma, monotonie temporelle, pas de NaN).

## Definition of Done (testable)
- [ ] `app/data/loader.py` contient `load_asset(asset: str, tf: str) -> pd.DataFrame` qui :
  - Est décoré par `@retry_with_backoff(max_attempts=3, exceptions=(OSError,))` (cf. Règle 11)
  - Lit `data/raw/<ASSET>/<TF>.csv`
  - Valide les colonnes : `timestamp, open, high, low, close, volume`
  - Convertit `timestamp` en `pd.DatetimeIndex` (UTC, monotone, sans doublon)
  - **Distingue gap normal vs anormal** : un gap > seuil pendant un weekend ou jour férié XTB → log INFO ; sinon log ERROR + `DataValidationError`
  - Lève `DataValidationError` si schéma invalide, valeurs négatives, OHLC incohérent (high < low, etc.)
- [ ] `app/config/calendar.py` contient :
  - `XTB_HOLIDAYS: dict[str, list[date]]` — jours fériés par actif (Noël, Nouvel An, Thanksgiving pour US30, etc.)
  - `is_market_open(asset: str, ts: datetime) -> bool` — exclut weekends et jours fériés
  - `is_normal_gap(asset: str, t1: datetime, t2: datetime) -> bool` — True si gap explicable par weekend/holiday
- [ ] `app/data/registry.py` contient `discover_assets() -> dict[str, list[str]]` qui retourne `{"US30": ["D1", "H4"], "XAUUSD": ["D1"], ...}`. Utilise `os.listdir(data/raw/)` — **ne lit JAMAIS le contenu des CSV**.
- [ ] `app/core/exceptions.py` contient `DataValidationError(PipelineError)`.
- [ ] `tests/unit/test_data_loader.py` : ≥ 10 tests (schéma OK, schéma manquant, gap normal weekend → INFO, gap anormal → ERROR, NaN, OHLC incohérent, monotonie, doublons, fichier inexistant, retry sur OSError). Utilise des CSV synthétiques en `tmp_path`.
- [ ] `tests/unit/test_calendar.py` : ≥ 4 tests (jour férié connu, weekend, gap normal vs anormal).
- [ ] `rtk make verify` passe (sur demande utilisateur).
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS lire `data/raw/<ASSET>/<TF>.csv` pour tester (utiliser des CSV synthétiques en `tmp_path`).
- Ne PAS implémenter d'indicateurs techniques (c'est le prompt 04).
- Ne PAS coupler le loader à pandas-ta ou autre lib externe.
- Ne PAS faire de feature engineering ici.
- Ne PAS lire `ready-data/`, `cleaned-data/`.

## Étapes

### Étape 1 — Contrat CSV
Définir clairement dans `app/data/loader.py` un docstring :

```python
"""
Contrat CSV pour chaque fichier data/raw/<ASSET>/<TF>.csv :

Colonnes obligatoires (ordre indifférent, casse insensible) :
- timestamp : ISO 8601 UTC ('2024-01-15T13:00:00Z' ou '2024-01-15 13:00:00')
- open, high, low, close : float
- volume : float (peut être 0 pour les indices CFD)

Contraintes :
- timestamp strictement monotone croissant
- pas de doublons d'index
- high >= max(open, close), low <= min(open, close)
- pas de valeurs NaN sauf en début (warmup) — drop_na obligatoire avant retour
- gap maximum :
    D1 : 7 jours (week-ends + jours fériés)
    H4 : 3 jours
    H1 : 2 jours
- Si gap dépassé : warning loggé, pas d'erreur (les marchés peuvent fermer)
"""
```

### Étape 2 — Implémentation `loader.py` + `calendar.py`

`app/config/calendar.py` :
```python
from __future__ import annotations
from datetime import date, datetime

XTB_HOLIDAYS: dict[str, list[date]] = {
    "US30": [date(2024, 1, 1), date(2024, 7, 4), date(2024, 11, 28), date(2024, 12, 25),
             date(2025, 1, 1), date(2025, 7, 4), date(2025, 11, 27), date(2025, 12, 25)],
    "US500": [...],  # idem US30
    "GER30": [date(2024, 1, 1), date(2024, 5, 1), date(2024, 12, 25), date(2024, 12, 26), ...],
    "XAUUSD": [date(2024, 1, 1), date(2024, 12, 25), date(2025, 1, 1), date(2025, 12, 25)],
}
# Compléter au fil de l'eau. Source : calendrier officiel XTB.

def is_market_open(asset: str, ts: datetime) -> bool:
    if ts.weekday() >= 5:
        return False
    if ts.date() in XTB_HOLIDAYS.get(asset, []):
        return False
    return True

def is_normal_gap(asset: str, t1: datetime, t2: datetime) -> bool:
    """True si le gap entre t1 et t2 est explicable par weekend/holiday."""
    cur = t1
    while cur < t2:
        cur = cur.replace(hour=0, minute=0, second=0)
        from datetime import timedelta
        cur += timedelta(days=1)
        if is_market_open(asset, cur):
            return False
    return True
```

`app/data/loader.py` :
```python
from __future__ import annotations

from pathlib import Path
import pandas as pd
from app.core.exceptions import DataValidationError
from app.core.logging import get_logger
from app.core.retry import retry_with_backoff
from app.config.calendar import is_normal_gap

logger = get_logger(__name__)

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
MAX_GAP_HOURS = {"D1": 7 * 24, "H4": 3 * 24, "H1": 2 * 24}


@retry_with_backoff(max_attempts=3, exceptions=(OSError,))
def load_asset(asset: str, tf: str, data_root: Path = Path("data/raw")) -> pd.DataFrame:
    """Charge et valide un CSV asset/timeframe. Retourne DataFrame indexé par timestamp UTC.
    Retry 3× sur OSError (file lock, disk transient). Distingue gap normal/anormal via calendar."""
    path = data_root / asset / f"{tf}.csv"
    if not path.exists():
        raise DataValidationError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"{path} : colonnes manquantes {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    if df.index.duplicated().any():
        raise DataValidationError(f"{path} : timestamps dupliqués")

    invalid_ohlc = (df["high"] < df[["open", "close"]].max(axis=1)) | (
        df["low"] > df[["open", "close"]].min(axis=1)
    )
    if invalid_ohlc.any():
        raise DataValidationError(f"{path} : {invalid_ohlc.sum()} barres OHLC incohérentes")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError(f"{path} : prix négatifs ou nuls")

    df = df.dropna()

    # Gap analysis : normal (weekend/holiday) vs anormal (data manquante)
    gaps_hours = df.index.to_series().diff().dt.total_seconds() / 3600
    max_gap = MAX_GAP_HOURS.get(tf, 24)
    big_gaps = gaps_hours[gaps_hours > max_gap]
    n_normal = n_abnormal = 0
    for ts in big_gaps.index:
        prev_ts = df.index[df.index.get_loc(ts) - 1]
        if is_normal_gap(asset, prev_ts.to_pydatetime(), ts.to_pydatetime()):
            n_normal += 1
        else:
            n_abnormal += 1

    if n_normal > 0:
        logger.info(f"{path} : {n_normal} gaps normaux (weekend/holiday)")
    if n_abnormal > 0:
        logger.error(f"{path} : {n_abnormal} gaps ANORMAUX (data manquante)")
        raise DataValidationError(f"{path} : {n_abnormal} gaps anormaux détectés")

    return df
```

### Étape 3 — Implémentation `registry.py`
```python
from __future__ import annotations

from pathlib import Path


def discover_assets(data_root: Path = Path("data/raw")) -> dict[str, list[str]]:
    """Retourne {asset: [tf1, tf2, ...]} en scannant data/raw/. Ne lit pas les CSV."""
    if not data_root.exists():
        return {}
    out: dict[str, list[str]] = {}
    for asset_dir in sorted(data_root.iterdir()):
        if not asset_dir.is_dir():
            continue
        tfs = sorted(
            p.stem for p in asset_dir.glob("*.csv") if p.stem in {"D1", "H4", "H1", "M15", "M5"}
        )
        if tfs:
            out[asset_dir.name] = tfs
    return out
```

### Étape 4 — `exceptions.py` (ajouter si manquante)
```python
class PipelineError(Exception): ...
class DataValidationError(PipelineError): ...
class LookAheadError(PipelineError): ...
```

### Étape 5 — Tests unitaires (`tests/unit/test_data_loader.py`)
Au moins 8 tests avec fixtures `tmp_path` :
1. CSV valide → DataFrame correct
2. Colonne manquante → `DataValidationError`
3. Timestamps dupliqués → `DataValidationError`
4. OHLC incohérent (high < open) → `DataValidationError`
5. Prix négatifs → `DataValidationError`
6. Gap > seuil → warning loggé, pas d'erreur
7. NaN → drop silencieux
8. Fichier inexistant → `DataValidationError`
9. (Bonus) test `discover_assets` sur arborescence synthétique.

### Étape 6 — Mettre à jour `INVENTORY.md`
Ajouter une section « Couche data » avec les chemins des nouveaux modules.

## Logging
```markdown
## 2026-MM-DD — Prompt 03 : Data layer
- **Statut** : ✅ Terminé
- **Fichiers créés** : app/data/loader.py, app/data/registry.py, tests/unit/test_data_loader.py
- **Fichiers modifiés** : app/core/exceptions.py (ajout DataValidationError)
- **Tests pytest** : ✅ X tests, 0 failures
- **Actifs détectés via discover_assets()** : <liste depuis Étape 5 manuelle>
- **Problèmes rencontrés** : ...
```

## Critères go/no-go
- **GO prompt 04** si : tests passent, `discover_assets()` retourne au moins US30 et XAUUSD.
- **NO-GO, revenir à** : ce prompt si validation échoue sur les fichiers réels (ajuster le contrat).
