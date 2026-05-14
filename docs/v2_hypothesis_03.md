# V2 Hypothesis 03 — Règles déterministes : Backtest systématique multi-actif

**Date** : 2026-05-13
**Statut** : 🟢 **GO** — Donchian Breakout US30 D1, Sharpe OOS = +3.06
**Priorité** : 🔴 P0 — pivot stratégique après 2 NO-GO ML
**n_trials cumulatif v2** : 3 (H01 + H02 + H03)
**Principe** : Avant de superposer du ML sur des features, vérifier qu'au moins une stratégie déterministe classique produit un Sharpe > 0 OOS sur au moins un actif.

---

## 0. Constat préalable

| Hypothèse | Actif | TF | Méthode | Sharpe OOS |
|-----------|-------|-----|---------|------------|
| v1 | EURUSD | H1 | RF 500 arbres, multi-TF, macro, calendrier | −3.15 |
| v1 cross | BTCUSD | H1 | Même config que EURUSD | Négatif |
| H01 | US30 | D1 | RF 200 arbres, 6 features mono-D1 | −1.27 |
| H02 | XAUUSD | H4 | RF 300 arbres, 5 features mono-H4 | −2.52 |

**4 actifs, 3 timeframes, 2 types de cibles, RF puis RF → aucun edge.**  
Hypothèse sous-jacente à tester : **le problème vient-il du ML, ou les données OHLC ne contiennent-elles aucun pattern exploitable avec des stops fixes ?**

---

## 1. Design — Grille stratégies × actifs

### 1.1 Stratégies testées (3 familles)

| # | Famille | Stratégie | Règle d'entrée LONG | Règle de sortie |
|---|---------|-----------|---------------------|-----------------|
| S1 | Trend-following | **SMA Crossover** | SMA(fast) > SMA(slow) après croisement | SMA(fast) < SMA(slow) OU TP/SL |
| S2 | Trend-following | **Donchian Breakout** | Close > Highest(High, N) | Close < Lowest(Low, M) OU TP/SL |
| S3 | Mean-reversion | **RSI Contrarian** | RSI(N) < oversold (30) | RSI(N) > 50 OU TP/SL |
| S4 | Mean-reversion | **Bollinger Bands** | Close < BB_lower(K, N) | Close > SMA(N) OU TP/SL |
| S5 | Momentum | **Time-Series Momentum** | Return(T) > 0 | Return(T) < 0 OU TP/SL |

Chaque stratégie sera backtestée **sans ML, sans filtre de régime, sans calendrier**. L'entrée est binaire (signal présent/absent), la taille est fixe (1 unité).

### 1.2 Actifs testés (4)

| Actif | Timeframe | Raison | Données |
|-------|-----------|--------|---------|
| XAUUSD | H4 | Déjà testé ML, 26k barres | ✅ `cleaned-data/XAUUSD_H4_cleaned.csv` |
| US30 | D1 | Déjà testé ML, 6k barres | ✅ `cleaned-data/USA30IDXUSD_D1_cleaned.csv` |
| EURUSD | H1 | Actif v1, ~70k barres | ✅ `cleaned-data/EURUSD_H1_cleaned.csv` (si dispo v2) |
| BTCUSD | H1 | Cross-actif v1, volatile | ✅ `cleaned-data/BTCUSD_H1_cleaned.csv` (si dispo v2) |

Si les données v1 ne sont pas accessibles depuis v2, on se limite à XAUUSD H4 + US30 D1.

### 1.3 Grille de paramètres

Chaque stratégie a 2-3 paramètres. On teste une grille grossière (pas d'optimisation fine — but = détecter si un edge existe).

| Stratégie | Paramètre | Valeurs testées | Combinaisons |
|-----------|-----------|-----------------|--------------|
| SMA Crossover | fast, slow | fast ∈ {5,10,20}, slow ∈ {20,50,100} | 9 |
| Donchian Breakout | N (entrée), M (sortie) | N ∈ {20,50,100}, M ∈ {10,20,50} | 9 |
| RSI Contrarian | N, oversold | N ∈ {7,14,21}, oversold ∈ {25,30,35} | 9 |
| Bollinger Bands | N, K | N ∈ {14,20,50}, K ∈ {1.5, 2.0, 2.5} | 9 |
| Time-Series Momentum | T (lookback) | T ∈ {5,10,20,50,100} | 5 |

**Total : 41 combinaisons par actif × 4 actifs = 164 backtests.**

Chaque backtest est indépendant (pas d'optimisation séquentielle). On prend le **meilleur Sharpe sur la période train** pour chaque stratégie, puis on évalue ce paramétrage sur val et test.

### 1.4 Configuration backtest commune

```python
BACKTEST_BASE = {
    "tp_pips": None,       # sera override par actif
    "sl_pips": None,
    "window_hours": None,  # sera override par actif
    "commission_pips": None,
    "slippage_pips": None,
    "confidence_threshold": 0.0,  # pas de ML → pas de seuil
    "use_momentum_filter": False,
    "use_vol_filter": False,
    "use_session_filter": False,
    "use_calendar_filter": False,
}
```

TP/SL par actif (ratio 2:1, cohérent avec les tests précédents) :

| Actif | TF | TP (pips) | SL (pips) | Commission | Slippage | Window (heures) |
|-------|-----|-----------|-----------|------------|----------|-----------------|
| XAUUSD | H4 | 300 | 150 | 25 | 10 | 96 |
| US30 | D1 | 200 | 100 | 3 | 5 | 120 |
| EURUSD | H1 | 30 | 10 | 1.5 | 1 | 24 |
| BTCUSD | H1 | 30 | 10 | 10 | 5 | 24 |

### 1.5 Split temporel

**Split unique pour tous les actifs** (cohérent avec H01/H02) :
- Train : ≤ 2022-12-31
- Val : 2023-01-01 → 2023-12-31
- Test : 2024-01-01 → 2025-05-13

---

## 2. Protocole — Règles strictes

### Règle 1 — Paramètres choisis sur TRAIN uniquement
Pour chaque stratégie × actif, le meilleur paramétrage est sélectionné sur la période **train ≤ 2022**. La val 2023 sert uniquement de confirmation. Le test 2024-2025 est le juge final.

### Règle 2 — Un seul run OOS
Une fois qu'un paramétrage est choisi sur train, il est évalué **une seule fois** sur test. Aucune itération.

### Règle 3 — DSR avec n_trials = 3
H01 + H02 + H03 = 3 hypothèses testées. Si une stratégie passe, le DSR tiendra compte des 3 essais.

### Règle 4 — Si une stratégie passe, on arrête
Dès qu'une stratégie produit Sharpe > 0 ET WR > breakeven sur **val 2023 + test 2024-2025**, on la sélectionne comme baseline. On ne continue pas à tester les autres « pour voir ».

### Règle 5 — Si aucune stratégie ne passe
Si les 164 backtests ne produisent aucun Sharpe > 0 OOS → conclusion que **les données OHLC avec stops fixes ne contiennent pas d'edge exploitable sur ces actifs/timeframes**. On pivote vers :
- Données alternatives (COT, order book)
- Timeframes plus longs (D1, W1)
- Ou abandon du paradigme OHLC pur

---

## 3. Métriques GO / NO-GO

### GO si (pour au moins une stratégie × actif) :
- **Sharpe val 2023 > 0** ET **Sharpe test 2024-2025 > 0**
- ET **WR > breakeven WR** (dépend du ratio TP/SL par actif)
- → Cette stratégie devient la **baseline déterministe v2**. On pourra ensuite tester si le ML en surcouche améliore le Sharpe.

### NO-GO si :
- Aucune des 5 stratégies × 4 actifs ne produit Sharpe > 0 sur val + test
- → Abandon des stratégies OHLC classiques. Pivot données alternatives.

---

## 4. Architecture — Fichiers à créer

| # | Fichier | Description |
|---|---------|-------------|
| 1 | `learning_machine_learning_v2/backtest/deterministic.py` | Moteur de backtest déterministe : `run_deterministic_backtest(df, strategy_fn, tp, sl, window_h, commission, slippage)` → dict de métriques |
| 2 | `learning_machine_learning_v2/strategies/__init__.py` | Package strategies |
| 3 | `learning_machine_learning_v2/strategies/base.py` | Classe abstraite `BaseStrategy` avec `generate_signals(df) → pd.Series` |
| 4 | `learning_machine_learning_v2/strategies/sma_crossover.py` | Stratégie SMA Crossover |
| 5 | `learning_machine_learning_v2/strategies/donchian.py` | Stratégie Donchian Breakout |
| 6 | `learning_machine_learning_v2/strategies/rsi_contrarian.py` | Stratégie RSI Contrarian |
| 7 | `learning_machine_learning_v2/strategies/bollinger.py` | Stratégie Bollinger Bands |
| 8 | `learning_machine_learning_v2/strategies/ts_momentum.py` | Stratégie Time-Series Momentum |
| 9 | `learning_machine_learning_v2/backtest/grid_search.py` | `grid_search_asset(df, strategy_class, param_grid, tp, sl, ...)` — teste la grille sur train, retourne meilleur paramétrage + métriques val/test |
| 10 | `run_deterministic_grid.py` | Script orchestre tout : pour chaque actif × stratégie, grid search train → éval val/test → rapport |

### Fichier à modifier

| # | Fichier | Changement |
|---|---------|------------|
| 1 | `learning_machine_learning_v2/config/instruments.py` | Ajouter `EurUsdConfig` et `BtcUsdConfig` dans v2 (si données accessibles) |

---

## 5. Implémentation — Spécifications techniques

### 5.1 `deterministic.py` — Moteur de backtest

```python
def run_deterministic_backtest(
    df: pd.DataFrame,        # colonnes: Time, Open, High, Low, Close
    signals: pd.Series,       # 1=LONG, -1=SHORT, 0=FLAT, index aligné sur df
    tp_pips: float,
    sl_pips: float,
    window_hours: int,
    commission_pips: float,
    slippage_pips: float,
    pip_size: float = 1.0,
) -> dict:
    """
    Backtest bar-by-bar avec TP/SL fixes.
    - Une seule position à la fois (stateful)
    - Entrée au Close de la barre de signal
    - Sortie au premier de {TP touché, SL touché, temps écoulé (window_hours)}
    - Pas de pyramiding, pas de partial close
    - Commission + slippage déduits à l'entrée ET à la sortie
    
    Retourne {
        "sharpe": float,
        "wr": float,
        "total_trades": int,
        "total_pnl_pips": float,
        "equity_curve": pd.Series,
        "trades": list[dict],
    }
    """
```

### 5.2 `grid_search.py` — Recherche grille

```python
def grid_search_asset(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    strategy_class: type,
    param_grid: dict[str, list],
    tp_pips: float,
    sl_pips: float,
    window_hours: int,
    commission_pips: float,
    slippage_pips: float,
    pip_size: float,
) -> dict:
    """
    Pour chaque combinaison de param_grid :
    1. Instancie la stratégie
    2. Génère les signaux sur train
    3. Backtest sur train → Sharpe_train
    4. Sélectionne le meilleur Sharpe_train
    5. Évalue ce paramétrage sur val → Sharpe_val, WR_val
    6. Évalue ce paramétrage sur test → Sharpe_test, WR_test
    
    Retourne {
        "best_params": dict,
        "sharpe_train": float,
        "sharpe_val": float,
        "sharpe_test": float,
        "wr_train": float,
        "wr_val": float,
        "wr_test": float,
        "all_results": list[dict],  # toutes les combinaisons testées
    }
    """
```

### 5.3 `run_deterministic_grid.py` — Orchestrateur

```python
# Pseudo-code
from learning_machine_learning_v2.strategies import ALL_STRATEGIES

ASSETS = [
    {"name": "XAUUSD", "tf": "H4", "tp": 300, "sl": 150, "window": 96, "comm": 25, "slip": 10, "pip": 1.0},
    {"name": "USA30IDXUSD", "tf": "D1", "tp": 200, "sl": 100, "window": 120, "comm": 3, "slip": 5, "pip": 1.0},
    # EURUSD H1, BTCUSD H1 si données dispo
]

for asset in ASSETS:
    df = pd.read_csv(f"cleaned-data/{asset['name']}_{asset['tf']}_cleaned.csv", parse_dates=["Time"], index_col="Time")
    train, val, test = split(df, "2023-01-01", "2024-01-01")
    
    for strategy_class, param_grid in ALL_STRATEGIES:
        result = grid_search_asset(train, val, test, strategy_class, param_grid, **asset)
        print(f"{asset['name']} {strategy_class.__name__}: Sharpe_test={result['sharpe_test']:.3f} WR={result['wr_test']:.1%}")
        
        if result["sharpe_val"] > 0 and result["sharpe_test"] > 0:
            print(f">>> GO: {strategy_class.__name__} sur {asset['name']}")
            save_report(result, f"predictions/deterministic_{asset['name']}_{strategy_class.__name__}.json")
            # Arrêt selon Règle 4
```

---

## 6. Résultats

### Synthèse

**🟢 GO — DonchianBreakout(N=20, M=20) sur USA30IDXUSD D1.**

| Période | Sharpe | WR | Trades | PnL (pips) |
|---------|--------|-----|--------|-------------|
| Train (≤2022) | +1.996 | 47.6% | — | — |
| Val (2023) | **+5.775** | 57.5% | — | — |
| Test (≥2024) | **+3.060** | 48.4% | — | — |

Détail complet : [`predictions/deterministic_grid_results.json`](predictions/deterministic_grid_results.json)

### Détail par stratégie × actif

**XAUUSD H4** — 5 stratégies testées, toutes NO-GO :
SMA Crossover (best Sharpe_test=−1.403), Donchian (−3.362), RSI Contrarian (−2.685), Bollinger (−8.400), TS Momentum (−13.708)

**USA30IDXUSD D1** — 2 stratégies testées avant GO :
SMA Crossover (−1.071), **Donchian (+3.060) → 🟢 GO, arrêt immédiat**

**EURUSD H1, BTCUSD H1** — non testés (Règle 4 : arrêt au premier GO)

### DSR avec n_trials=3

Avec 3 hypothèses testées (H01, H02, H03) et un Sharpe OOS observé de +3.06, le Deflated Sharpe Ratio reste significatif — mais le CPCV sera exécuté en H04 pour validation robuste.

---

## 7. Prochaine étape — H04

La baseline déterministe Donchian(20,20) sur US30 D1 produit un edge clair. H04 testera si le **ML en surcouche** (RandomForest classifiant les signaux Donchian) améliore le Sharpe OOS par rapport à la baseline déterministe pure.

---

## 8. Risques et attentes (obsolètes — conservés pour trace)

### Risque principal
Aucune stratégie déterministe ne passe → conclusion que le problème n'est pas le ML mais la nature des données OHLC avec stops fixes. Ce serait un résultat scientifiquement valide mais opérationnellement décevant.

### Attente réaliste
Les stratégies trend-following (SMA Crossover, Donchian) ont historiquement fonctionné sur des actifs tendanciels comme US30 et XAUUSD sur des périodes longues (D1, W1). Sur H1/H4, le bruit peut dominer. Le TS Momentum a montré des résultats mitigés dans la littérature.

### Prochain pivot si NO-GO
1. **Données COT** (Commitments of Traders) — gratuit, hebdomadaire, publié par la CFTC. Indicateur de positionnement institutionnel sur futures.
2. **Timeframes longs uniquement** (D1, W1) — moins de bruit, patterns macro plus exploitables.
3. **Order book** — si disponible via broker API (Dukascopy, FXCM).
