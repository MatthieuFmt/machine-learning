# Prompt 05 — Intégration du calendrier économique

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/03_data_layer.md`

## Objectif
Intégrer le scraper `scripts/scrape_forexfactory.py` (déjà présent) comme source de features time-aware. Produire un DataFrame de features `is_within_X_of_event(impact, currency)` pour chaque barre, en respectant strictement le principe anti-look-ahead.

## Definition of Done (testable)
- [ ] `app/features/economic.py` contient :
  - `load_calendar(years: list[int]) -> pd.DataFrame` : charge les CSV scrapés (déjà présents dans `data/raw/economic_calendar/`).
  - `compute_event_features(price_index: pd.DatetimeIndex, calendar: pd.DataFrame) -> pd.DataFrame` : retourne pour chaque barre des features booléennes/numériques.
- [ ] Features produites (au minimum) :
  - `event_high_impact_within_24h_USD` : 1 si un événement High Impact USD dans les 24h à venir
  - `event_high_impact_within_4h_USD` : idem 4h
  - `event_high_impact_within_1h_USD` : idem 1h
  - Idem pour EUR
  - `hours_since_last_nfp` : heures depuis dernier NFP (release)
  - `hours_since_last_fomc` : heures depuis dernier FOMC
  - `hours_to_next_event_high_impact` : heures avant prochain événement High
- [ ] Cache local : si `data/raw/economic_calendar/<year>.csv` n'existe pas, lever `DataValidationError` (NE PAS scraper automatiquement — c'est à l'utilisateur de lancer `scripts/scrape_forexfactory.py`).
- [ ] **Test anti-look-ahead** : `compute_event_features(price_index[:n])[-1]` est identique à `compute_event_features(price_index)[n-1]`. Aucune feature ne doit utiliser une info `> t` SAUF `hours_to_next_event_*` qui est par nature forward — mais cette feature est CONNUE à `t` car les annonces sont publiées plusieurs jours à l'avance. Vérifier que les timestamps des events sont bien la date d'annonce, pas la date de release (sinon ce serait du leak).
- [ ] `tests/unit/test_economic_features.py` : ≥ 6 tests (chargement OK, event féatures correctes sur cas synthétiques, anti-look-ahead, gestion fichier manquant, fuseaux horaires UTC, robustesse aux events sans heure précise).
- [ ] `rtk pytest tests/unit/test_economic_features.py -v` passe.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS lancer le scraper automatiquement.
- Ne PAS supposer un format différent — utiliser le format produit par `scripts/scrape_forexfactory.py` tel quel.
- Ne PAS ajouter une feature qui utilise la VALEUR de l'annonce (actual, forecast, previous) sans vérification anti-leak stricte. Si une annonce est encore future, la valeur `actual` est NaN.
- Ne PAS lire `data/raw/economic_calendar/*.csv` directement dans les tests (utiliser `tmp_path`).

## Étapes

### Étape 1 — Lire le format produit par le scraper
Lire `scripts/scrape_forexfactory.py` pour comprendre le schéma exact du CSV produit (colonnes : `timestamp`, `currency`, `impact`, `event`, `actual`, `forecast`, `previous`).

### Étape 2 — `load_calendar`
```python
from __future__ import annotations

from pathlib import Path
import pandas as pd
from app.core.exceptions import DataValidationError


def load_calendar(years: list[int], root: Path = Path("data/raw/economic_calendar")) -> pd.DataFrame:
    dfs = []
    for y in years:
        path = root / f"{y}.csv"
        if not path.exists():
            raise DataValidationError(
                f"Calendrier {y} introuvable. Lance `python scripts/scrape_forexfactory.py --year {y}` d'abord."
            )
        dfs.append(pd.read_csv(path, parse_dates=["timestamp"]))
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)
```

### Étape 3 — `compute_event_features`
Vectorisé. Pour chaque timestamp du prix, calculer la fenêtre des events à venir et passés via `searchsorted` sur le calendrier trié.

```python
def compute_event_features(price_index: pd.DatetimeIndex, calendar: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=price_index)

    for currency in ["USD", "EUR"]:
        cal_c = calendar[(calendar["currency"] == currency) & (calendar["impact"] == "High")]
        event_ts = cal_c["timestamp"].values
        for hours in [1, 4, 24]:
            col = f"event_high_within_{hours}h_{currency}"
            out[col] = 0
            for ts in event_ts:
                mask = (price_index <= ts) & (price_index >= ts - pd.Timedelta(hours=hours))
                out.loc[mask, col] = 1

    # Hours since last NFP / FOMC
    for keyword, name in [("Non-Farm", "nfp"), ("FOMC", "fomc")]:
        events_k = calendar[calendar["event"].str.contains(keyword, case=False, na=False)]["timestamp"]
        if len(events_k) > 0:
            out[f"hours_since_last_{name}"] = (
                pd.Series(price_index, index=price_index)
                .apply(lambda t: (t - events_k[events_k <= t].max()).total_seconds() / 3600 if (events_k <= t).any() else float("nan"))
            )

    return out.fillna(-1).astype(float)
```

> Optimisation : remplacer le `.apply` par un `searchsorted` pour les grandes séries. À implémenter si lent.

### Étape 4 — Tests
Avec calendrier synthétique en `tmp_path` :
- 2 events : NFP 2024-01-05 13:30 UTC High USD, FOMC 2024-01-31 19:00 UTC High USD
- Vérifier `event_high_within_24h_USD` est 1 pour 2024-01-04 14:00 (dans la fenêtre 24h pré-NFP), 0 pour 2024-01-01 (avant)
- Vérifier `hours_since_last_nfp` est 0 à 2024-01-05 13:30, 24 à 2024-01-06 13:30

### Étape 5 — Lien avec `04_features_research_harness`
Étendre `compute_all_indicators` (prompt 04) pour OPTIONNELLEMENT inclure les features économiques si `data/raw/economic_calendar/` existe.

## Logging
```markdown
## 2026-MM-DD — Prompt 05 : Economic calendar
- **Statut** : ✅ Terminé
- **Fichiers créés** : app/features/economic.py, tests/unit/test_economic_features.py
- **Fichiers modifiés** : app/features/indicators.py (intégration optionnelle)
- **Tests pytest** : ✅ X tests
- **Anti-look-ahead vérifié** : oui
```

## Critères go/no-go
- **GO prompt 06** si : tests passent ET, après exécution manuelle utilisateur sur des données réelles, les features sont produites sans erreur.
- **NO-GO, revenir à** : ce prompt si format scraper différent (mettre à jour le contrat).
