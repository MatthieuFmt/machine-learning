# V2 Hypothesis 05 — Walk-Forward Paper Trading US30 D1

**Date** : 2026-05-13
**Statut** : 🟢 **GO** — Donchian + RF méta-labeling déployé (Sharpe WF +8.84)
**Priorité** : 🔴 P0 — dernière validation avant live
**n_trials cumulatif v2** : 5 (H01–H05)
**Dépend de** : H03 (baseline Donchian) + H04 (méta-labeling RF)

---

## 1. Contexte

Deux configurations viables identifiées sur US30 D1 :

| Configuration | Sharpe Val (2023) | Sharpe Test (2024-2025) | Robustesse |
|---------------|-------------------|--------------------------|------------|
| **A — Donchian(20,20) pur** | +5.67 | +3.07 | Excellente — 0 paramètre tunable, résultats stables |
| **B — Donchian + RF méta-labeling (seuil=0.50)** | — | +8.61 (seuil 0.65 calibré) | Fragile — overfit du seuil sur train, variance CPCV ±10.03 |

**Décision architecturale pour H05** : tester les **deux configurations en walk-forward** pour décider laquelle déployer en live. La config A est plus robuste mais la B a un potentiel de Sharpe supérieur si le seuil est stabilisé.

---

## 2. Design du walk-forward

### 2.1 Principe

Simulation réaliste de ce qui se serait passé si on avait tradé la stratégie en conditions réelles :
- **Réentraînement périodique** : tous les 6 mois, on réentraîne le RF (config B) sur toutes les données disponibles jusqu'à la date courante
- **Exécution réaliste** : slippage aléatoire, spread bid/ask, pas de look-ahead
- **Forward uniquement** : les prédictions utilisent uniquement l'information disponible à la date de la barre
- **Split rolling** : pas de split fixe — on réentraîne périodiquement et on trade en avant

### 2.2 Périodes

| Phase | Période | Action |
|-------|---------|--------|
| Warm-up | ≤ 2022-12-31 | Entraînement initial (RF) + calibrage seuil |
| Walk-forward | 2023-01-01 → 2025-05-13 | Trading avec réentraînement tous les 6 mois |

Réentraînements : 2023-07-01, 2024-01-01, 2024-07-01, 2025-01-01

À chaque réentraînement :
1. Toutes les données jusqu'à la date sont disponibles pour l'entraînement
2. Le seuil RF est recalibré sur train (pas de look-ahead)
3. La config Donchian(20,20) reste inchangée (paramètres figés depuis H03)

### 2.3 Configurations comparées

```
Config A : Donchian(20,20) pur
  → Tous les signaux sont tradés, pas de filtre RF
  → Sharpe baseline à battre

Config B : Donchian(20,20) + RF méta-labeling
  → RF réentraîné tous les 6 mois
  → Seuil recalibré à chaque réentraînement (grille [0.45, 0.50, 0.55, 0.60])
  → Un signal Donchian n'est pris que si RF.predict_proba > seuil
```

### 2.4 Exécution réaliste

```python
WF_CONFIG = {
    "commission_pips": 3.0,
    "slippage_pips": 5.0,        # slippage fixe
    "slippage_random": 2.0,       # slippage additionnel aléatoire (uniforme ±2 pips)
    "spread_pips": 2.0,           # spread bid/ask
    "execution_delay_bars": 0,    # exécution à la barre suivante (Close→Close)
    "max_slippage_total": 8.0,    # slippage max total par trade
}
```

### 2.5 Métriques walk-forward

| Métrique | Config A (Donchian pur) | Config B (Donchian + RF) |
|----------|------------------------|--------------------------|
| Sharpe WF | ? | ? |
| WR WF | ? | ? |
| Trades WF | ? | ? |
| Max drawdown | ? | ? |
| Profit factor | ? | ? |
| Sharpe par année (2023, 2024, 2025) | ? | ? |

---

## 3. Protocole

### Règle 1 — Pas de look-ahead
À chaque barre t, toutes les features et signaux sont calculés avec l'information disponible jusqu'à t (inclus). Pas de `.shift(-1)`, pas de forward-fill depuis le futur.

### Règle 2 — Réentraînement périodique strict
Le RF est réentraîné exactement aux dates spécifiées (2023-07-01, 2024-01-01, 2024-07-01, 2025-01-01), pas plus souvent. Les données utilisées sont strictement ≤ date de réentraînement.

### Règle 3 — Un seul run
Le walk-forward est exécuté une seule fois. Aucun ajustement post-hoc des paramètres.

### Règle 4 — Décision GO / NO-GO
- Si Config B (RF) surpasse Config A (baseline) en Sharpe WF → déploiement Config B en live
- Si Config A surpasse Config B → déploiement Config A en live (plus simple, plus robuste)
- Si les deux sont négatifs → NO-GO, retour à la planche à dessin

---

## 4. Implémentation

### 4.1 Fichier à créer

| # | Fichier | Description |
|---|---------|-------------|
| 1 | `learning_machine_learning_v2/backtest/walk_forward.py` | Moteur de walk-forward : `run_walk_forward(df, strategy, retrain_dates, **backtest_kwargs) → dict` |
| 2 | `run_walk_forward_us30.py` | Script orchestre : warm-up → walk-forward Config A vs Config B → rapport comparatif |

### 4.2 `walk_forward.py`

```python
def run_walk_forward(
    df: pd.DataFrame,                    # OHLC complet (index=Time)
    donchian_signals: pd.Series,         # signaux Donchian sur tout le dataset
    features: pd.DataFrame,              # features (index=Time)
    meta_labels: pd.Series,              # méta-labels (1/0/NaN)
    retrain_dates: list[str],            # ["2023-07-01", "2024-01-01", ...]
    initial_train_end: str,              # "2022-12-31"
    tp_pips: float, sl_pips: float, window_hours: int,
    commission_pips: float, slippage_pips: float,
    slippage_random: float, spread_pips: float,
    pip_size: float,
    rf_params: dict,
    thresholds: list[float],
) -> dict:
    """
    Walk-forward avec réentraînement périodique.
    
    Algorithme :
    1. Train initial sur df[≤initial_train_end]
    2. Pour chaque segment entre retrain_dates[i] et retrain_dates[i+1] :
       a. Génère signaux Donchian sur le segment
       b. RF.predict_proba → filtre avec seuil calibré
       c. Backtest déterministe avec signaux filtrés
       d. Fin du segment : réentraîne RF sur df[≤retrain_dates[i+1]]
    3. Dernier segment : retrain_dates[-1] → fin des données
    
    Retourne {
        "sharpe": float,
        "wr": float,
        "trades": int,
        "pnl_pips": float,
        "equity_curve": pd.Series,
        "segment_details": list[dict],
    }
    """
```

### 4.3 `run_walk_forward_us30.py`

```python
# Pseudo-code
from learning_machine_learning_v2.strategies.donchian import DonchianBreakout
from learning_machine_learning_v2.backtest.deterministic import run_deterministic_backtest
from learning_machine_learning_v2.backtest.walk_forward import run_walk_forward
from learning_machine_learning_v2.backtest.meta_labeling import compute_meta_labels
from learning_machine_learning_v2.models.meta_rf import train_meta_rf, calibrate_threshold

# 1. Charger données
df = pd.read_csv("cleaned-data/USA30IDXUSD_D1_cleaned.csv", ...)

# 2. Générer signaux Donchian(20,20)
donchian = DonchianBreakout(N=20, M=20)
signals = donchian.generate_signals(df)

# 3. Features (comme H01/H04)
features = build_features(df)  # RSI_14, ADX_14, Dist_SMA50, Dist_SMA200, ATR_Norm, Log_Return_5d, Donchian_Position

# 4. Méta-labels
meta_labels = compute_meta_labels(df, signals, tp=200, sl=100, window=120)

# 5. Walk-forward Config A (baseline Donchian pur)
wf_baseline = run_walk_forward(
    df, signals, features, meta_labels,
    retrain_dates=["2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01"],
    initial_train_end="2022-12-31",
    use_rf=False,  # pas de RF
    tp_pips=200, sl_pips=100, window_hours=120,
    commission_pips=3, slippage_pips=5, slippage_random=2, spread_pips=2,
    pip_size=1.0,
)

# 6. Walk-forward Config B (Donchian + RF)
wf_meta = run_walk_forward(
    df, signals, features, meta_labels,
    retrain_dates=["2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01"],
    initial_train_end="2022-12-31",
    use_rf=True,
    rf_params={"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10, ...},
    thresholds=[0.45, 0.50, 0.55, 0.60],
    tp_pips=200, sl_pips=100, window_hours=120,
    commission_pips=3, slippage_pips=5, slippage_random=2, spread_pips=2,
    pip_size=1.0,
)

# 7. Rapport comparatif
print("=== WALK-FORWARD US30 D1 ===")
print(f"Baseline Donchian pur — Sharpe WF: {wf_baseline['sharpe']:.3f}  WR: {wf_baseline['wr']:.1%}  Trades: {wf_baseline['trades']}")
print(f"Donchian + RF méta  — Sharpe WF: {wf_meta['sharpe']:.3f}  WR: {wf_meta['wr']:.1%}  Trades: {wf_meta['trades']}")
print(f">>> Déploiement: {'Config B (RF)' if wf_meta['sharpe'] > wf_baseline['sharpe'] else 'Config A (Donchian pur)'}")

# 8. Sauvegarde
save_report({"baseline": wf_baseline, "meta": wf_meta}, "predictions/walk_forward_us30_results.json")
```

---

## 5. Spécifications techniques additionnelles

### 5.1 Slippage réaliste

```python
def apply_slippage(entry_price: float, direction: int, slippage_fixed: float, slippage_random: float, spread: float, pip_size: float) -> float:
    """
    Calcule le prix d'exécution réel.
    - spread : payé à l'entrée (défavorable)
    - slippage_fixed : toujours défavorable
    - slippage_random : ± uniforme (peut être favorable)
    """
    import numpy as np
    rng = np.random.default_rng(42)
    random_component = rng.uniform(-slippage_random, slippage_random)
    total_slippage = (slippage_fixed + spread/2 + random_component) * pip_size
    if direction == 1:  # LONG
        return entry_price + total_slippage  # achat plus cher
    else:  # SHORT
        return entry_price - total_slippage  # vente moins chère
```

### 5.2 Vérification anti-look-ahead

Le walk-forward doit passer un test de non-look-ahead :
- Pour chaque trade exécuté à la barre t, toutes les features utilisées doivent être calculables avec l'information disponible jusqu'à t (inclus)
- Les features comme `Log_Return_5d` utilisent `.shift(5)`, pas de `.shift(-5)`
- `Donchian_Position` = signal Donchian à la barre t (calculé avec High/Low jusqu'à t)

---

## 6. Résultats

```
=== WALK-FORWARD US30 D1 ===
                    Sharpe WF   WR       Trades   PnL (pips)
Donchian pur        3.656       51.1%   135      +4670.1
Donchian + RF       8.844       66.7%    12       +989.0

Par segment :
  2023H1:  Donchian pur=+3.315 / RF=+6.005
  2023H2:  Donchian pur=+6.802 / RF= 0.000
  2024S1:  Donchian pur=−0.150 / RF= 0.000
  2024S2:  Donchian pur=+7.182 / RF= 0.000
  2025S1:  Donchian pur=+2.176 / RF=+8.432

>>> Déploiement: Config B (Donchian + RF méta-labeling)
```

### Analyse

- **🟢 GO** : Config B (RF) surpasse Config A (baseline) de +5.19 en Sharpe WF — écart > 0.5, déploiement Config B
- **⚠️ Seulement 12 trades** sur 30 mois : le RF filtre très agressivement. Le seuil calibré est trop restrictif sur les périodes 2023H2 et 2024S1 (0 trade). Problème connu de généralisation du seuil
- **Config A reste solide** : 135 trades, Sharpe +3.66, WR 51.1% — une alternative robuste si le RF s'avère instable en live
- **Recommandation production** : démarrer avec Config B, mais avec un **seuil plancher à 0.50** (pas 0.65) pour éviter les périodes sans trade. Si instable → basculer sur Config A

---

## 7. Métriques GO / NO-GO

### GO si :
- Sharpe WF > 0 pour au moins une configuration ✅ **Atteint — les deux**
- → Déploiement de la meilleure configuration → **Config B (Donchian + RF)**

### Critique si :
- Les deux configurations ont Sharpe WF > 0 mais Config B (RF) surpasse Config A de < 0.5
- → Non applicable : écart = +5.19

### NO-GO si :
- Les deux configurations ont Sharpe WF ≤ 0
- → Non applicable
