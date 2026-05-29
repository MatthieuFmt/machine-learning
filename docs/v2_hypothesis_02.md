# V2 Hypothesis 02 — XAUUSD H4 : Trend-Following mono-TF, cible binaire symétrique

**Date** : 2026-05-13
**Statut** : 🔴 NO-GO — Sharpe 2024-2025 = −2.52
**Priorité** : 🔴 P0 (fallback après NO-GO US30 D1)
**n_trials cumulatif v2** : 2 (H01 + H02)
**Données** : `data/XAUUSD_H4.csv`, `data/XAUUSD_D1.csv` — 26 820 barres H4 (2009-2026)

---

## 1. Justification

### 1.1 Pourquoi XAUUSD H4 après échec US30 D1

| Leçon H01 | Application H02 |
|-----------|-----------------|
| 0 signal en validation 2023 → features D1 trop pauvres | H4 : 6 barres/jour × 10 ans = ~15k échantillons (vs 2.6k D1) |
| Seuil confiance 0.55 trop élevé pour RF sur petit dataset | Seuil 0.45 pour laisser passer plus de signaux |
| 66 trades en 17 mois trop peu pour évaluer un edge | H4 avec seuil plus bas → 150-300 trades attendus |
| Volume CFD non fiable → Volume_Ratio inutile | Exclure Volume_Ratio, garder 5-6 features |

### 1.2 Pourquoi XAUUSD spécifiquement

- **Corrélation négative USD** : l'or est un actif anti-dollar → les features de momentum capturent les retournements
- **Volatilité élevée** : ATR H4 typique ~80-120 pips-or → TP=300/SL=150 (ratio 2:1) = cibles atteignables en 1-5 jours
- **Moins d'efficience que le FX** : l'or est influencé par les flux physiques, la demande bijouterie, les banques centrales → patterns plus exploitables
- **Déjà testé cross-asset en v1** : le pipeline BTCUSD peut servir de template (pip_size=1.0, tp_sl_scale_factor)

---

## 2. Spécification

### 2.1 Cible

`directional_clean` — classification binaire (HAUSSE/BAISSE sur horizon 24 barres H4 = 4 jours). Seuil de bruit = 0.5 × ATR(14).

```python
labels = compute_directional_clean(
    h4_data,
    horizon_hours=96,      # 24 barres H4 = 4 jours
    noise_atr=0.5,
    atr_period=14,
)
```

### 2.2 Features (5 — parcimonie extrême)

| # | Feature | Timeframe | Construction |
|---|---------|-----------|--------------|
| 1 | `RSI_14` | H4 | `talib.RSI(close, 14)` |
| 2 | `ADX_14` | H4 | `talib.ADX(high, low, close, 14)` |
| 3 | `Dist_SMA50` | H4 | `(Close − SMA50) / SMA50 × 100` |
| 4 | `ATR_Norm` | H4 | `ATR(14) / Close × 100` |
| 5 | `Log_Return_24` | H4 | `log(Close / Close.shift(24))` — momentum 4 jours |

**Pas de features D1** : mono-TF strict. Si un signal existe en H4, il doit être autonome. Les features D1 seront ajoutées seulement si H4 seul montre un edge (v2-02b).

**Pas de macro, pas de calendrier** : simplification maximale. Si un edge est trouvé, on enrichira.

### 2.3 Modèle

```python
RF_PARAMS = {
    "n_estimators": 300,       # plus que D1 (200), dataset H4 ~15k barres
    "max_depth": 5,            # légèrement plus profond que D1 (4)
    "min_samples_leaf": 50,    # ~0.3% de l'échantillon
    "class_weight": "balanced_subsample",
    "random_state": 42,
}
```

### 2.4 Split temporel

Split neuf (aucun overlap avec v1) :
- **Train** : ≤ 2022-12-31
- **Val** : 2023-01-01 → 2023-12-31
- **Test** : 2024-01-01 → 2025-05-13

### 2.5 Backtest

```python
XAUUSD_CONFIG = BacktestConfig(
    tp_pips=300.0,      # 300 pips-or (≈ $3.00)
    sl_pips=150.0,      # 150 pips-or (ratio 2:1)
    window_hours=96,    # 4 jours = 24 barres H4
    commission_pips=25.0,    # spread XAUUSD typique ~25 pips-or
    slippage_pips=10.0,
    confidence_threshold=0.45,  # moins restrictif que US30 (0.55)
    use_momentum_filter=False,
    use_vol_filter=True,        # VolFilter utile pour l'or (régimes de stress)
    use_session_filter=False,   # pas pertinent en H4
    use_calendar_filter=False,
)
```

### 2.6 InstrumentConfig

```python
@dataclass(frozen=True)
class XauUsdConfig(InstrumentConfig):
    name: str = "XAUUSD"
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    timeframes: FrozenSet[str] = frozenset({"H4", "D1"})
    primary_tf: str = "H4"
    macro_instruments: FrozenSet[str] = frozenset()  # pas de macro
    features_dropped: tuple[str, ...] = ()           # construction explicite, pas de drop
    tp_sl_scale_factor: float = 1.0
```

Note : `pip_size=1.0` (1 pip-or = 1 cent = $0.01), donc `tp_pips=300` = $3.00 par once. Pour un contrat standard (100 oz), TP = $300. Cohérent.

---

## 3. Protocole anti-snooping (rappel)

| Règle | Application H02 |
|-------|-----------------|
| Split fixé ex ante | ≤2022/2023/2024-2025, **gelé** |
| Un seul regard | `python run_pipeline_xauusd.py` exécuté une fois |
| DSR | n_trials = 2 (H01 + H02) |
| Instrument hold-out | USDCHF D1 (jamais touché) |

---

## 4. Fichiers à créer

| # | Fichier | Description |
|---|---------|-------------|
| 1 | `scripts/inspect_xauusd_csv.py` | Diagnostic CSV XAUUSD H4/D1 |
| 2 | `learning_machine_learning_v2/pipelines/xauusd.py` | Pipeline XAUUSD H4 (~80 lignes, template us30.py) |
| 3 | `run_pipeline_xauusd.py` | Script de lancement |

### 5. Fichier à modifier

| # | Fichier | Changement |
|---|---------|------------|
| 1 | `learning_machine_learning_v2/config/instruments.py` | Ajouter `XauUsdConfig` |

---

## 6. Résultats

| Split | Barres | Signaux | Trades | Sharpe | WR | PnL (pips) |
|-------|--------|---------|--------|--------|-----|------------|
| Train (≤2022) | 16 454 | — | — | — | — | — |
| Val (2023) | 1 460 | 12 | 12 | **−1.526** | 8.3% | −417.5 |
| Test (2024) | 1 452 | 29 | 18 | **−2.277** | 22.2% | −1 441.2 |
| Test (2025) | 1 465 | 34 | 24 | **−2.832** | 12.5% | −3 035.5 |
| **Test 2024-2025** | **2 917** | **63** | **42** | **−2.516** | **16.7%** | **−4 476.7** |

**Verdict** : ❌ NO-GO. Sharpe agrégé = −2.52, WR = 16.7%. Le biais de prédiction (+1 : 48% train, 43% test) et la domination des features OHLC brutes dans les feature importances indiquent que le signal technique est noyé. La VolFilter H4 ne bloque presque rien (1 trade filtré sur 2024).

---

## 7. Métriques GO / NO-GO

### GO si :
- **Sharpe test (2024-2025) > 0** ET **WR > breakeven WR** (~37% avec TP=300/SL=150/friction=35)
- → CPCV avec n_trials=2

### NO-GO si :
- Sharpe ≤ 0 OU WR ≤ breakeven
- → **Atteint** — passage à H03
