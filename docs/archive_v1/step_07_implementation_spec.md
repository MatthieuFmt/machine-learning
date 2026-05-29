# Step 07 — Spécification d'implémentation BTCUSD H1

**Date** : 2026-05-13
**Mode** : Architect → Code
**Dépendance** : CSV BTCUSD bruts dans `data/` (BTCUSD_H1.csv, BTCUSD_H4.csv, BTCUSD_D1.csv)

---

## 1. Résumé des changements

| # | Fichier | Action | Lignes |
|---|---------|--------|--------|
| 1 | `scripts/inspect_btc_csv.py` | **Créer** — diagnostic des CSV bruts | ~30 |
| 2 | `learning_machine_learning/pipelines/btcusd.py` | **Créer** — pipeline BTCUSD | ~80 |
| 3 | `run_pipeline_btcusd.py` | **Créer** — script de lancement | ~60 |
| 4 | `learning_machine_learning/config/instruments.py` | **Modifier** — `features_dropped` + `tp_sl_scale_factor` | 2 champs |
| 5 | `learning_machine_learning/pipelines/base.py` | **Modifier** — multiplier TP/SL par `scale_factor` | 2 lignes |

---

## 2. Prérequis — Données

### 2.1 Format attendu

Les CSV bruts dans `data/` doivent contenir les colonnes :

```
Time,Open,High,Low,Close,Volume[,Spread]
```

- `Time` : datetime ISO 8601 (ex: `2023-01-01 00:00:00`)
- `Spread` : optionnel. Si absent → fallback `0.0` (déjà géré par [`CPCV`](../learning_machine_learning/analysis/cpcv.py:351))

### 2.2 Script de diagnostic

[`scripts/inspect_btc_csv.py`](../scripts/inspect_btc_csv.py) :

1. Tente `pd.read_csv(f"data/BTCUSD_{tf}.csv")` pour `tf ∈ {H1, H4, D1}`
2. Vérifie colonnes obligatoires `{Time, Open, High, Low, Close}`
3. Parse `Time` et vérifie monotonie croissante
4. Affiche plage de dates, nombre de barres, taux de NaN
5. Copie les fichiers valides dans `cleaned-data/BTCUSD_{tf}_cleaned.csv`

### 2.3 Fichiers de sortie

```
cleaned-data/
├── BTCUSD_H1_cleaned.csv
├── BTCUSD_H4_cleaned.csv
├── BTCUSD_D1_cleaned.csv
├── EURUSD_H1_cleaned.csv   (existant)
├── EURUSD_H4_cleaned.csv   (existant)
└── EURUSD_D1_cleaned.csv   (existant)
```

---

## 3. Modifications de code

### 3.1 [`instruments.py`](../learning_machine_learning/config/instruments.py) — 2 changements

#### a) `InstrumentConfig` : nouveau champ `tp_sl_scale_factor`

```python
@dataclass(frozen=True)
class InstrumentConfig:
    # ... champs existants ...

    # ── Step 07 — Scaling TP/SL par instrument ───────────────────────
    tp_sl_scale_factor: float = 1.0
```

**Justification** : BTC ATR ≈ 5× EURUSD en termes de pips. Le simulateur utilise `tp_dist = tp_pips * pip_size` — correct mathématiquement, mais TP=30 pips pour BTC = $30, ce qui est 0.05% du prix (vs 0.3% pour EURUSD). Le `scale_factor` compense cette différence sans toucher au simulateur.

**Validation `__post_init__`** : `if self.tp_sl_scale_factor <= 0: raise ValueError(...)`

#### b) `BtcUsdConfig` : remplir `features_dropped` et `tp_sl_scale_factor`

```python
@dataclass(frozen=True)
class BtcUsdConfig(InstrumentConfig):
    name: str = "BTCUSD"
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    timeframes: FrozenSet[str] = frozenset({"H1", "H4", "D1"})
    primary_tf: str = "H1"
    macro_instruments: FrozenSet[str] = frozenset()
    features_dropped: tuple[str, ...] = (
        "Dist_EMA_9", "Dist_EMA_21", "Dist_EMA_20",
        "Log_Return", "CHF_Return", "Dist_EMA_50_D1",
        "BB_Width", "Hour_Cos", "Hour_Sin",
        "RSI_14_H4", "Dist_EMA_20_H4", "Dist_EMA_50_H4",
        "ATR_Norm", "Volatilite_Realisee_24h", "Range_ATR_ratio",
        "Momentum_5", "Momentum_10", "Momentum_20",
        "EMA_20_50_cross", "Volatility_Ratio",
    )
    tp_sl_scale_factor: float = 5.0  # BTC ~5× plus volatile en pips que EURUSD
```

Note : `CHF_Return` dans `features_dropped` même si BTC n'a pas de macro CHF — c'est intentionnel : la spec exige une liste identique pour éviter le tuning par actif. La colonne `CHF_Return` n'existera simplement pas dans le dataset BTC → sera ignorée par le filtrage `if c in combined.columns`.

### 3.2 [`base.py`](../learning_machine_learning/pipelines/base.py) — scaling TP/SL

Dans `run_backtest()`, lignes 180-182, remplacer :

```python
tp_pips=cfg.tp_pips,
sl_pips=cfg.sl_pips,
```

par :

```python
tp_pips=cfg.tp_pips * self.instrument.tp_sl_scale_factor,
sl_pips=cfg.sl_pips * self.instrument.tp_sl_scale_factor,
```

Même changement dans `run_walk_forward()` (pas d'appel direct à simulate, mais cohérence).

**Impact EURUSD** : `scale_factor=1.0` → aucun changement. Rétrocompatibilité totale.

### 3.3 `BacktestConfig` — commission/slippage BTC

Deux options (décision au moment du code) :

| Option | Approche |
|---|---|
| A | Passer `commission_pips=10, slippage_pips=5` dans le script `run_pipeline_btcusd.py` |
| B | Ajouter `commission_pips`/`slippage_pips` à `InstrumentConfig` |

**Recommandé : Option A** — plus simple, évite de gonfler `InstrumentConfig`. BTC spot : commission ~0.075% × $60k = $45 round-trip. Mais le pipeline trade avec `pip_size=1.0` donc `commission_pips=10` représente $10 de coût fixe par trade + $5 slippage. Sous-estimation acceptable pour un premier test.

---

## 4. Nouveau pipeline [`btcusd.py`](../learning_machine_learning/pipelines/btcusd.py)

Structure identique à [`eurusd.py`](../learning_machine_learning/pipelines/eurusd.py) avec 3 différences :

1. `super().__init__("BTCUSD")`
2. `load_data()` ne charge **pas** de macro (`macro_instruments` est vide)
3. Les chemins de fichiers pointent vers `BTCUSD` au lieu d'`EURUSD`

```python
class BtcUsdPipeline(BasePipeline):
    def __init__(self) -> None:
        super().__init__("BTCUSD")

    def load_data(self) -> dict[str, pd.DataFrame]:
        from learning_machine_learning.data.loader import load_all_timeframes

        paths = {
            "h1": self.paths.clean_file("BTCUSD", "H1"),
            "h4": self.paths.clean_file("BTCUSD", "H4"),
            "d1": self.paths.clean_file("BTCUSD", "D1"),
        }
        data = load_all_timeframes(paths)

        # ★ Calendrier économique (les événements USD impactent BTC)
        from learning_machine_learning.data.calendar_loader import load_calendar
        h1_data = data["h1"]
        cal_start = h1_data.index.min()
        cal_end = h1_data.index.max()

        try:
            calendar_df = load_calendar(cal_start, cal_end)
            data["_calendar"] = calendar_df
            logger.info("Calendrier économique chargé : %d événements", len(calendar_df))
        except (FileNotFoundError, OSError) as e:
            logger.warning("Calendrier économique indisponible : %s — ignoré.", e)
            data["_calendar"] = None

        return data

    def build_features(self, data, train_end=None):
        ml = build_ml_ready(
            instrument=self.instrument,
            data={"H1": data["h1"], "H4": data["h4"], "D1": data["d1"]},
            macro_data={},  # Pas de macro pour BTC
            calendar_df=data.get("_calendar"),
            tp_pips=self.backtest_cfg.tp_pips,
            sl_pips=self.backtest_cfg.sl_pips,
            window=self.backtest_cfg.window_hours,
            features_dropped=list(self.instrument.features_dropped),
            train_end=train_end,
        )
        return ml
```

---

## 5. Script de lancement [`run_pipeline_btcusd.py`](../run_pipeline_btcusd.py)

Version simplifiée de `run_pipeline_v1.py` :

- **Pas de méta-labeling** (le méta-modèle EURUSD n'est pas transférable à BTC sans recalibration)
- Backtest simple : RF → prédictions → simulateur → métriques
- Sauvegarde JSON dans `predictions/btcusd_metrics_{year}.json`

---

## 6. BacktestConfig pour BTC

Instanciation spécifique dans le script de run :

```python
btc_backtest = BacktestConfig(
    tp_pips=30.0,    # → effectif = 30 × 5.0 = 150 pips BTC
    sl_pips=10.0,    # → effectif = 10 × 5.0 = 50 pips BTC
    commission_pips=10.0,  # ~$10 round-trip
    slippage_pips=5.0,     # ~$5 slippage
    # Mêmes filtres que EURUSD
    use_momentum_filter=True,
    use_vol_filter=True,
    use_session_filter=True,
    use_calendar_filter=True,
)
```

---

## 7. Vérifications anti-leak

| Risque | Mitigation |
|---|---|
| `features_dropped` différent par actif | Liste **strictement identique** à EurUsdConfig |
| TP/SL optimisé sur BTC | Scale factor calculé a priori (ratio ATR), pas de sweep |
| Données BTC post-2025 | Split train≤2023, val=2024, test=2025 — identique EURUSD |
| Calendrier forward-looking | Déjà vérifié (Step 05) — `merge_asof(direction='backward')` |

---

## 8. Ordre d'exécution (mode Code)

1. `scripts/inspect_btc_csv.py` → diagnostiquer et copier les CSV
2. Modifier `instruments.py` → `features_dropped` + `tp_sl_scale_factor`
3. Modifier `base.py` → multiplier TP/SL par scale_factor
4. Créer `pipelines/btcusd.py`
5. Créer `run_pipeline_btcusd.py`
6. `python run_pipeline_btcusd.py` → premier run
7. Comparer Sharpe 2024/2025 BTC vs EURUSD
8. Générer `predictions/btcusd_cross_asset_report.md`
