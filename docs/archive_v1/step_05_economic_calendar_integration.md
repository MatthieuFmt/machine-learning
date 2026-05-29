# Step 05 — Intégration du calendrier économique macro

**Catégorie** : Feature exogène
**Priorité** : 🟠 Moyenne
**Effort estimé** : 3-5 jours (lourd : sourcing + ingestion + intégration)
**Dépendances** : aucune (peut être fait en parallèle de 03/04, mais step_02 nécessaire pour valider)

---

## 1. Hypothèse mathématique

### Observation empirique

Sur EURUSD H1, ~80 % des mouvements > 30 pips dans la journée surviennent dans une fenêtre de ±2h autour d'un release macro majeur (NFP, CPI, FOMC, BCE, BoE, BoJ). Le pipeline actuel **ignore complètement** ce signal exogène :

- Aucune feature liée au calendrier dans [`features/`](../learning_machine_learning/features/).
- Aucun filtre temporel d'exclusion autour des releases.
- Les trades pris pendant les minutes pré-release sont susceptibles d'être stoppés par la volatilité du release (SL serré à 10 pips = ~1 ATR, easily traversé par un NFP surprise).

### Formalisation

Soit $\mathcal{E} = \{(t_i, \text{event}_i, \text{impact}_i, \text{actual}_i, \text{forecast}_i)\}$ l'ensemble des événements macro avec leurs timestamps UTC.

**Features dérivées** :

1. **Distance temporelle au prochain événement high-impact** :
$$\Delta t_{next}(t) = \min_{t_i \ge t, \text{impact}_i = high} (t_i - t) \quad \text{[en minutes]}$$

2. **Surprise normalisée du dernier release** :
$$\text{surprise}_i = \frac{\text{actual}_i - \text{forecast}_i}{\sigma_{\text{historical}}(\text{actual} - \text{forecast})}$$

3. **Indicator de fenêtre dangereuse** :
$$\mathbb{1}_{\text{near\_event}}(t) = \mathbb{1}\{\exists\, t_i \in \mathcal{E}_{high} : |t_i - t| \le 120 \text{ min}\}$$

**Hypothèse $H_1$** : exclure les trades dans $\mathbb{1}_{\text{near\_event}}$ réduit le drawdown sans impacter significativement le profit total (les mouvements pré/post-release sont chaotiques et non prédictibles par features techniques).

**Hypothèse $H_2$** : `surprise_zscore` post-release est directement prédictif de la direction des 24-48h suivantes — un signal exogène fort que le modèle peut apprendre à exploiter.

---

## 2. Méthodologie d'implémentation

### Sourcing des données

- **Source primaire recommandée** : Forex Factory historique (gratuit, format CSV téléchargeable mensuellement) ou scraping de [forexfactory.com/calendar](https://www.forexfactory.com/calendar).
- **Alternatives** :
  - Investing.com (scraping, à risque de blocage)
  - API payantes : [FXStreet](https://www.fxstreet.com/economic-calendar), [Trading Economics API](https://tradingeconomics.com/api/)
  - Pour back-test historique 2010-2025 : Forex Factory CSV est le standard de facto.

- **Couverture minimale** :
  - Pays : US, EUR (zone euro), UK, JP, CH
  - Events : NFP, CPI, PPI, Retail Sales, FOMC/BCE/BoE/BoJ rate decisions, GDP, Unemployment, ECB/Fed speeches majeurs
  - Champs : timestamp UTC, country, event_name, impact (low/medium/high), actual, forecast, previous

### Fichiers concernés

- **Créer** [`data/raw/economic_calendar/`](../data/) : dossier des CSV bruts par année (`2010.csv`, ..., `2025.csv`)
- **Créer** [`learning_machine_learning/data/calendar_loader.py`](../learning_machine_learning/data/) avec :
  - `load_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame` : charge, parse et concatène les CSV bruts ; normalise les timestamps UTC ; filtre par impact ≥ medium par défaut.
  - `validate_calendar_schema(df: pd.DataFrame) -> None` : raise si colonnes manquantes ou types incohérents.

- **Créer** [`learning_machine_learning/features/calendar.py`](../learning_machine_learning/features/) avec :
  - `compute_minutes_to_next_event(timestamps: pd.DatetimeIndex, events_df: pd.DataFrame, impact: str = "high") -> pd.Series` : O(n log m) avec `searchsorted`
  - `compute_minutes_since_last_event(...)` (symétrique, signal post-release)
  - `compute_surprise_zscore(events_df: pd.DataFrame, lookback: int = 50) -> pd.Series` : rolling std des (actual - forecast) pour chaque event_name, puis z-score du dernier release par event_name
  - `merge_calendar_features(ohlc: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame` : ajoute les colonnes au DataFrame H1

- **Étendre** [`features/pipeline.py`](../learning_machine_learning/features/pipeline.py) : appeler `merge_calendar_features` après le merge macro.

- **Étendre** [`backtest/filters.py`](../learning_machine_learning/backtest/filters.py) avec une nouvelle classe :
  ```
  class CalendarFilter:
      name = "calendar"
      def __init__(self, exclude_window_minutes: int = 120, impact_threshold: str = "high"): ...
      def apply(self, df, mask_long, mask_short): ...
  ```
  Raise si `minutes_to_next_event` non présente.

- **Modifier** [`config/backtest.py`](../learning_machine_learning/config/backtest.py) : ajouter
  - `use_calendar_filter: bool = True`
  - `calendar_exclude_window_minutes: int = 120`
  - `calendar_impact_threshold: Literal["medium", "high"] = "high"`

### Choix techniques

- **Format de stockage** : Parquet (`results/economic_calendar.parquet`) après parsing pour I/O rapide.
- **Granularité** : minute UTC. Les releases sont publiés à des minutes spécifiques (ex : NFP toujours à 13:30 UTC).
- **Normalisation des noms d'événements** : un mapping `event_name → canonical_name` (ex : "Non-Farm Payrolls", "NFP", "US Non-Farm Employment Change" → `US_NFP`) pour stabiliser le calcul `surprise_zscore`.
- **Fallback si données manquantes** : `minutes_to_next_event = 99999` (très grande valeur signifiant "pas d'événement proche"). Documenter pour interprétation du modèle.
- **Time zone** : tout en UTC. Source DST-aware (Forex Factory donne en UTC ou US/Eastern selon export).

### Anti-leak / précautions
- **Pas de surprise_zscore futur** : la feature à $t$ doit utiliser uniquement les surprises observées avant $t$. Implémenter via `merge_asof(direction='backward')`.
- **Pas de calibration de seuils sur futur** : le `lookback=50` pour rolling std est strict backward.
- **Cohérence horaire avec OHLC** : si OHLC est UTC, calendrier doit être UTC. Vérifier par sanity check : un NFP à 13:30 UTC un vendredi de juin 2024 doit s'aligner sur une bougie H1 EURUSD à 13:00-14:00.

---

## 3. Métriques de validation

### Métriques cibles
| Métrique | Baseline v15 | Objectif step_05 |
|---|---|---|
| Max drawdown OOS 2025 | -688 pips | **< -400 pips** (filtre réduit volatilité catastrophique) |
| Sharpe OOS 2025 | +0.04 | **> 0.40** |
| WR conditionnel "pas d'event < 2h" | NA (33 %) | **> 38 %** (filtrage améliore propreté) |
| Permutation importance `minutes_to_next_event` | NA | **> 0.003** |
| Permutation importance `surprise_zscore` | NA | **> 0.005** |
| Nombre de trades évités par filtre | NA | **~10-20 %** des signaux bruts |

### Métriques secondaires
- **Distribution des SL par fenêtre temporelle** : avant filtre, pourcentage de SL hits dans la fenêtre ±2h des high-impact events. Devrait être disproportionné (typiquement 25-40 % des SL).
- **PnL conditionnel** : analyser PnL_total par bucket de `minutes_to_next_event` ∈ {0-15, 15-60, 60-120, 120+ min}. Devrait montrer que les trades < 60min de release sont les pires.
- **Sharpe annualisé excluant les jours de FOMC/NFP** : sanity check additionnel.

### Critère d'arrêt
- Si **le filtre CalendarFilter ne change pas le Sharpe de +0.15** ET **`minutes_to_next_event` n'apparaît pas dans top 5 permutation importance**, le calendrier macro n'est pas exploitable sur EURUSD H1 (probable si les releases sont déjà capturés implicitement par ATR_Norm/Volatilite_Realisee_24h).
- Si gains observés → conserver, intégrer comme dépendance permanente (mettre à jour le calendrier mensuellement).

---

## 4. Risques & dépendances

- **R1 — Maintenance des données** : Forex Factory CSV doit être mis à jour mensuellement pour le live trading futur. Documenter une procédure dans `docs/data_update_procedure.md` (futur).
- **R2 — Données rétroactivement révisées** : les `actual` peuvent être révisés post-release (BLS révise NFP). Forex Factory historique stocke la version révisée — risque léger de look-ahead intra-jour de release (1-2h après publication). Mitigation : décaler `surprise_zscore` de 2h.
- **R3 — Surprise zscore instable si peu d'occurrences** : un release rare (ex : ECB unscheduled meeting) avec < 10 occurrences historiques donne un zscore non fiable. Mitigation : `if n_history < 20: surprise_zscore = NaN` (pas de signal).
- **R4 — Surcorrélation avec sessions (step_04)** : la majorité des releases US sont à 13:30 UTC = session NY début. Risque que `minutes_to_next_event` devienne proxy de `session_id`. Surveiller via VIF (Variance Inflation Factor) ou corrélation.
- **R5 — Délais de scraping si update auto** : prévoir retry et fallback (ex : si scraping échoue, garder l'ancien calendrier et logger un WARNING).

---

## 5. Références

- Andersen, T. G., Bollerslev, T., Diebold, F. X., & Vega, C. (2003). *Micro effects of macro announcements: Real-time price discovery in foreign exchange*. American Economic Review.
- Faust, J., Rogers, J. H., Wang, S. Y., & Wright, J. H. (2007). *The high-frequency response of exchange rates and interest rates to macroeconomic announcements*. Journal of Monetary Economics.
- Forex Factory historical CSV : https://www.forexfactory.com/calendar (export via scraping)
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 17 (Structural Breaks) — pour le test de surprise comme structural break.
