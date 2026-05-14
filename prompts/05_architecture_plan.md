# Architecture — Prompt 05 : Calendrier économique

## Audit critique du pseudo-code du prompt

| # | Problème | Gravité | Correction |
|---|----------|---------|------------|
| 1 | **Schéma CSV incorrect** : le scraper produit `date` + `time` séparés, pas `timestamp` | 🔴 Bloquant | `load_calendar()` combine `date` + `time` → `pd.Timestamp` UTC |
| 2 | **Boucle `for ts in event_ts:`** O(bars × events) | 🟡 Performance | `np.searchsorted` vectorisé |
| 3 | **`.apply()` sur `hours_since_last`** — violation Règle 6 constitution + skill pandas-vectorize | 🔴 Constitution | `searchsorted` + soustraction vectorielle |
| 4 | **Calendriers inexistants** : `data/raw/economic_calendar/` vide | 🟡 Tests | Tests 100% synthétiques (`tmp_path`) ; `DataValidationError` levée |
| 5 | **`fillna(-1)` ambigu** | 🟡 Sémantique | `np.nan` + docstring : NaN = "aucun événement connu" |
| 6 | **`hours_to_next_event`** : dates rétrospectives potentiellement corrigées | 🟡 Acceptable | Documenté, non utilisé comme target |

---

## Diagramme des composants

```
┌──────────────────────────────────────────────────────────────┐
│                   app/features/economic.py                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  load_calendar(years, root) ─────► pd.DataFrame              │
│  │   • Combine date + time → timestamp UTC                   │
│  │   • DataValidationError si fichier manquant               │
│  │   • Colonnes: timestamp, currency, event, impact,         │
│  │               actual, forecast, previous                  │
│  │                                                           │
│  compute_event_features(price_index, calendar) ──► DataFrame │
│  │   • searchsorted vectorisé (zéro boucle Python)           │
│  │   • 6 features booléennes (USD/EUR × 1h/4h/24h)          │
│  │   • 3 features numériques (nfp, fomc, next_event)         │
│  │   • Anti-look-ahead vérifié mécaniquement                 │
│  │                                                           │
│  _event_within_window(price_ns, event_ns, window_ns) → array │
│      • Helper interne : searchsorted + broadcast             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│           app/features/indicators.py (MODIFIÉ)               │
│  compute_all_indicators(df, include_economic=True)            │
│      • Si calendar_dir existe → charge + merge               │
│      • Sinon → warning log, skip silencieux                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Algorithme `searchsorted` vectorisé

```
Problème: pour chaque barre i à t_i, déterminer si un événement
         existe dans (t_i, t_i + window].

Méthode (boucle sur les events uniquement, O(E × log B)):
  1. Convertir timestamps → int64 nanosecondes
  2. Pour chaque event à ts_e:
     left  = searchsorted(price_ns, ts_e - window_ns)   # 1ère barre voyant l'event
     right = searchsorted(price_ns, ts_e, side='right') # 1ère barre APRÈS l'event
     result[left:right] = 1
  3. E ~ 500/an, B ~ 250k/an → négligeable

Pour hours_since_last_X:
  idx = searchsorted(event_ns, price_ns) - 1           # index du dernier event ≤ t
  result = (price_ns - event_ns[idx]) / 3.6e12          # ns → heures
  idx == -1 → NaN

Pour hours_to_next_X:
  idx = searchsorted(event_ns, price_ns, side='right')  # index du 1er event > t
  result = (event_ns[idx] - price_ns) / 3.6e12
  idx >= len(events) → NaN
```

---

## Features produites

| Feature | Type | Description | Sentinelle |
|---------|------|-------------|------------|
| `event_high_within_1h_USD` | int8 | Event High USD dans l'heure | 0 |
| `event_high_within_4h_USD` | int8 | Event High USD dans les 4h | 0 |
| `event_high_within_24h_USD` | int8 | Event High USD dans les 24h | 0 |
| `event_high_within_1h_EUR` | int8 | Event High EUR dans l'heure | 0 |
| `event_high_within_4h_EUR` | int8 | Event High EUR dans les 4h | 0 |
| `event_high_within_24h_EUR` | int8 | Event High EUR dans les 24h | 0 |
| `hours_since_last_nfp` | float32 | Heures depuis dernier NFP | NaN |
| `hours_since_last_fomc` | float32 | Heures depuis dernier FOMC | NaN |
| `hours_to_next_event_high` | float32 | Heures avant prochain High | NaN |

---

## Matrice de tests

| # | Test | Vérifie |
|---|------|---------|
| 1 | `test_load_calendar_ok` | Chargement CSV synthétique → bonnes colonnes, timestamps UTC |
| 2 | `test_load_calendar_missing_file` | `DataValidationError` levée |
| 3 | `test_event_within_window_exact` | Barre à t-2h d'un event → `within_4h`=1, `within_1h`=0 |
| 4 | `test_hours_since_last_nfp` | 24h après NFP → `hours_since` = 24.0 |
| 5 | `test_hours_to_next_event` | 3h avant event → `hours_to_next` = 3.0 |
| 6 | `test_anti_look_ahead_consistency` | `f(p[:n])[-1] == f(p)[n-1]` |
| 7 | `test_empty_calendar` | Calendrier vide → toutes features = sentinelle |
| 8 | `test_no_events_for_currency` | Events USD seulement, features EUR = 0 |

---

## Intégration `compute_all_indicators`

- Paramètre `include_economic: bool = True`
- Si `True` et `data/raw/economic_calendar/` contient des CSV : charge + merge
- Si `True` mais répertoire vide/absent : `logger.warning()` + skip
- Tests Prompt 04 inchangés car `include_economic=False` par défaut dans le contexte de test
- `load_calendar()` utilise le répertoire `data/raw/economic_calendar` (pas `data_root`)

---

## Risques

| Risque | Probabilité | Mitigation |
|--------|-------------|------------|
| `searchsorted` overflow int64 | Nulle | Timestamps < 2100 |
| Format CSV scraper changé | Faible | Bloqué par `test_load_calendar_ok` |
| `hours_to_next_event` leak rétrospectif | Faible | Documenté, non utilisé comme target |
| `compute_all_indicators` casse tests Prompt 04 | Moyen | `include_economic=False` dans les tests existants |
| Performance sur 10+ ans | Négligeable | Events/an ~ 500, boucle events seulement |
