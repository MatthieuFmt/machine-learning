# Pivot v4 — B1 : H_new1 — Méta-labeling RF sur Donchian US30 D1

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. [04_replay_h06_h07.md](04_replay_h06_h07.md) — **DOIT ÊTRE TERMINÉ** + Phase A complète ✅
3. [../00_constitution.md](../00_constitution.md) — toutes les règles 1-15
4. [JOURNAL.md](../../JOURNAL.md) — vérifier que n_trials_cumul = 22 et que A1-A4 sont ✅
5. **Code v2 H05** (référence historique, à porter dans `app/`) :
   - `learning_machine_learning_v2/models/meta_labeling.py` (si encore présent, sinon recréer)
   - `learning_machine_learning_v2/backtest/walk_forward.py`
   - `docs/archive_v1/` pour les rapports v2 H04 et H05

## Objectif
**Question H_new1** : Le méta-labeling RF qui a donné Sharpe walk-forward **+8.84** en v2 H05 résiste-t-il aux coûts XTB réels (Phase A2) avec sizing au risque 2 % (Phase A1) et Sharpe routing correct (Phase A3) ?

**C'est l'hypothèse #1 absolue.** C'est la seule configuration v2 GO jamais correctement retestée. Si elle est NO-GO ici, alors l'edge v2 était illusoire, et le projet doit pivoter (H_new3 ou abandon).

## Type d'opération
🟢 **Nouvelle hypothèse OOS** — **+1 n_trial** (cumul 23).

> ⚠️ **C'est une vraie hypothèse**, qui consomme un regard test set. `read_oos()` obligatoire.

## Definition of Done (testable)

- [ ] `app/models/meta_labeling.py` (NOUVEAU ou réécrit) contient :
  - `class MetaLabelingRF` avec `fit(X_train, y_train)`, `predict_proba(X)`, `calibrate_threshold(X_train, y_train, candidates=[0.45, 0.50, 0.55, 0.60])`
  - Fallback : si tous seuils éliminent > 80 % des trades train → seuil = 0.50 + warning loggé
  - RF : `n_estimators=200, max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=42`
- [ ] `app/pipelines/walk_forward.py` (NOUVEAU ou réécrit) contient :
  - `class WalkForwardMetaLabeling` avec `run(df, strat, cfg, retrain_months=6)` qui :
    - Splite en segments de retrain_months
    - Entraîne le méta-modèle sur la fenêtre expansive ≤ retrain_date
    - Applique sur le segment suivant
    - Retourne equity continue + trades continus
- [ ] `scripts/run_h_new1_meta_us30.py` (NOUVEAU) :
  - Charge US30 D1
  - Donchian baseline N=20, M=20 (params figés depuis v2 H03, **pas re-grid searché**)
  - Features RF : `["RSI_14", "ADX_14", "Dist_SMA_50", "Dist_SMA_200", "ATR_Norm_14", "Log_Return_5", "Signal_Donchian"]`
  - Walk-forward retrain 6M, expanding window depuis 2018 (warmup ≤ 2022)
  - Test set OOS ≥ 2024 lu **UNE SEULE FOIS**
  - `read_oos(prompt="pivot_v4_B1", hypothesis="H_new1", sharpe=..., n_trades=...)` appelé
- [ ] `predictions/h_new1_meta_us30.json` : metrics complets train / val / test, walk-forward segments.
- [ ] `docs/h_new1_meta_us30.md` : rapport détaillé avec :
  - Comparaison avec v2 H05 (Sharpe +8.84) — pourquoi différent
  - Métriques 5 critères constitution
  - Décision GO/NO-GO chiffrée
- [ ] Tableau `n_trials` dans `JOURNAL.md` mis à jour : ligne H_new1 avec n_trials_cumul = 23.
- [ ] CPCV obligatoire : `app/backtest/cpcv.py` exécuté sur la config retenue, std/mean documenté.
- [ ] `rtk make verify` → 0 erreur.
- [ ] `JOURNAL.md` mis à jour.

## Critères GO/NO-GO chiffrés

| Critère | Cible | Notes |
|---|---|---|
| Sharpe walk-forward test | ≥ 1.0 | Calculé via `sharpe_annualized` routing (probablement weekly) |
| DSR | > 0 ET p < 0.05 | Avec n_trials_cumul = 23 |
| Max DD | < 15 % | Borné par sizing 2 % |
| WR | > 35 % | Méta-labeling doit améliorer vs baseline |
| Trades/an | ≥ 30 | Critère 5 constitution |
| **GO si** | TOUS passent | — |
| **NO-GO si** | UN SEUL échoue | Bascule H_new3 ou H_new2 |

## NE PAS FAIRE

- ❌ **Ne PAS re-grid searcher Donchian (N, M)** — params figés (20, 20) depuis v2 H03. Modifier = data snooping + n_trial supplémentaire non justifié.
- ❌ Ne PAS lire le test set 2024+ plus d'une fois.
- ❌ Ne PAS optimiser le seuil méta sur val (overfit). Calibration train uniquement.
- ❌ Ne PAS ajouter de features hors de la liste v2 H05 sans documenter pourquoi + +1 n_trial.
- ❌ Ne PAS modifier le retrain_months en réaction au résultat (6 mois = standard v2 H05).
- ❌ Ne PAS oublier `read_oos()`.
- ❌ Ne PAS commit sans accord utilisateur.
- ❌ Ne PAS oublier d'incrémenter `n_trials_cumul` dans `JOURNAL.md` AVANT l'exécution.

## Étapes détaillées

### Étape 1 — Porter/écrire `app/models/meta_labeling.py`

```python
"""Méta-labeling RF — filtre binaire en surcouche d'un signal déterministe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetaLabelingConfig:
    n_estimators: int = 200
    max_depth: int = 4
    min_samples_leaf: int = 10
    class_weight: str = "balanced"
    random_state: int = 42
    threshold_candidates: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60)
    min_trade_retention: float = 0.20  # plancher : garder ≥ 20% des trades train


class MetaLabelingRF:
    def __init__(self, config: MetaLabelingConfig | None = None):
        self.config = config or MetaLabelingConfig()
        self.model: RandomForestClassifier | None = None
        self.threshold: float = 0.50
        self.disabled: bool = False  # True si fallback : pas d'amélioration possible

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train.values, y_train.values)

    def calibrate_threshold(
        self,
        X_train: pd.DataFrame,
        backtest_fn,
    ) -> None:
        """Calibre le seuil sur train uniquement. Sélectionne le Sharpe train max
        sous contrainte de rétention de trades ≥ min_trade_retention.
        Fallback à 0.50 si aucun seuil ne tient."""
        proba = self.model.predict_proba(X_train.values)[:, 1]
        best_t, best_sharpe = None, -np.inf
        for t in self.config.threshold_candidates:
            mask = proba > t
            if mask.sum() < len(X_train) * self.config.min_trade_retention:
                continue
            sharpe = backtest_fn(mask)
            if sharpe > best_sharpe:
                best_sharpe, best_t = sharpe, t
        if best_t is None:
            logger.warning(
                "Aucun seuil ne retient ≥ 20%% des trades train. "
                "Désactivation du méta-labeling (fallback baseline)."
            )
            self.disabled = True
            self.threshold = 0.0  # accepte tous les trades = baseline
        else:
            self.threshold = max(best_t, 0.50)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.disabled:
            return np.ones(len(X), dtype=bool)
        proba = self.model.predict_proba(X.values)[:, 1]
        return proba > self.threshold
```

### Étape 2 — `app/pipelines/walk_forward.py`

```python
"""Walk-forward avec méta-labeling et retrain 6 mois."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from app.backtest.deterministic import run_deterministic_backtest
from app.config.instruments import AssetConfig
from app.models.meta_labeling import MetaLabelingRF
from app.testing.snooping_guard import read_oos


@dataclass
class WalkForwardSegment:
    start: pd.Timestamp
    end: pd.Timestamp
    n_train: int
    n_oos_trades: int
    sharpe_oos: float
    wr_oos: float


def walk_forward_meta(
    df: pd.DataFrame,
    strat,
    cfg: AssetConfig,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    target_builder: Callable[[pd.DataFrame, pd.Series], pd.Series],
    retrain_months: int = 6,
    test_start: str = "2024-01-01",
    capital_eur: float = 10_000.0,
) -> tuple[pd.DataFrame, list[WalkForwardSegment]]:
    """Walk-forward continu sur df.loc[test_start:].

    À chaque date de retrain (1er janvier et 1er juillet à partir de test_start) :
      1. Train RF sur toutes les données train+val ≤ retrain_date - embargo.
      2. Calibre le seuil sur train.
      3. Applique sur le segment suivant (jusqu'au prochain retrain).
    Retourne (all_trades_oos, segments).
    """
    df = df.sort_index()
    retrain_dates = pd.date_range(start=test_start, end=df.index[-1], freq="6MS")  # 6-month-start

    all_oos_trades: list[pd.DataFrame] = []
    segments: list[WalkForwardSegment] = []

    for i, retrain_dt in enumerate(retrain_dates):
        segment_end = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else df.index[-1]
        df_train = df.loc[: retrain_dt - pd.Timedelta(days=2)]  # embargo 2 jours
        df_oos = df.loc[retrain_dt:segment_end]
        if df_train.empty or df_oos.empty:
            continue

        # Backtest baseline sur train pour générer les signaux et labels
        trades_train = run_deterministic_backtest(df_train, strat, cfg)
        X_train = feature_builder(df_train).loc[trades_train.index]
        y_train = target_builder(df_train, trades_train["Pips_Bruts"])

        meta = MetaLabelingRF()
        meta.fit(X_train, y_train)
        # calibrate sur train : re-backtest avec différents seuils
        def _sharpe_for_mask(mask):
            filtered_trades = trades_train.loc[mask.values]
            if len(filtered_trades) < 5:
                return -np.inf
            from app.backtest.metrics import compute_metrics
            return compute_metrics(filtered_trades, asset_cfg=cfg, capital_eur=capital_eur)["sharpe"]
        meta.calibrate_threshold(X_train, _sharpe_for_mask)

        # Backtest OOS + filter par méta
        trades_oos = run_deterministic_backtest(df_oos, strat, cfg)
        if trades_oos.empty:
            continue
        X_oos = feature_builder(df_oos).loc[trades_oos.index]
        keep_mask = meta.predict(X_oos)
        trades_oos_filtered = trades_oos.loc[keep_mask]

        all_oos_trades.append(trades_oos_filtered)
        from app.backtest.metrics import compute_metrics
        m = compute_metrics(trades_oos_filtered, asset_cfg=cfg, capital_eur=capital_eur)
        segments.append(WalkForwardSegment(
            start=retrain_dt,
            end=segment_end,
            n_train=len(trades_train),
            n_oos_trades=int(m["trades"]),
            sharpe_oos=float(m["sharpe"]),
            wr_oos=float(m["win_rate"]),
        ))

    all_trades = pd.concat(all_oos_trades) if all_oos_trades else pd.DataFrame()
    return all_trades, segments
```

### Étape 3 — `scripts/run_h_new1_meta_us30.py`

```python
"""Pivot v4 B1 — H_new1 : méta-labeling RF sur Donchian US30 D1.

⚠️ Consomme 1 n_trial. Lecture OOS test set ≥ 2024 = unique.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.config.instruments import ASSET_CONFIGS
from app.strategies.donchian import DonchianBreakout
from app.features.indicators import rsi, adx, atr
from app.pipelines.walk_forward import walk_forward_meta
from app.backtest.metrics import compute_metrics
from app.backtest.cpcv import purged_kfold_cv
from app.analysis.edge_validation import validate_edge
from app.testing.snooping_guard import read_oos


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14 = atr(df, 14)
    out = pd.DataFrame({
        "RSI_14": rsi(close, 14),
        "ADX_14": adx(df, 14),
        "Dist_SMA_50": (close - sma50) / atr14,
        "Dist_SMA_200": (close - sma200) / atr14,
        "ATR_Norm_14": atr14 / close,
        "Log_Return_5": np.log(close / close.shift(5)),
    }, index=df.index)
    return out.dropna()


def build_target(df: pd.DataFrame, pnl_brut: pd.Series) -> pd.Series:
    """Cible binaire : 1 si trade gagnant (pnl > 0)."""
    return (pnl_brut > 0).astype(int)


def main() -> int:
    set_global_seeds()
    df = load_asset("US30", "D1")
    cfg = ASSET_CONFIGS["US30"]

    strat = DonchianBreakout(N=20, M=20)

    # Walk-forward méta-labeling sur test set ≥ 2024
    all_trades_oos, segments = walk_forward_meta(
        df=df,
        strat=strat,
        cfg=cfg,
        feature_builder=build_features,
        target_builder=build_target,
        retrain_months=6,
        test_start="2024-01-01",
        capital_eur=10_000.0,
    )

    metrics = compute_metrics(all_trades_oos, asset_cfg=cfg, capital_eur=10_000.0)

    # CPCV sur train+val pour évaluer stabilité (pas sur test)
    df_train_val = df.loc[:"2023-12-31"]
    cpcv_results = purged_kfold_cv(df_train_val, strat, cfg, k=5, embargo_pct=0.01)

    # validate_edge sur le walk-forward OOS
    equity = 10_000 + (
        all_trades_oos["Pips_Nets"] * all_trades_oos["position_size_lots"] * cfg.pip_value_eur
    ).cumsum()
    report = validate_edge(
        equity=equity,
        trades=all_trades_oos,
        n_trials=23,  # 22 + H_new1
    )

    # READ_OOS — UNIQUE
    read_oos(
        prompt="pivot_v4_B1",
        hypothesis="H_new1_meta_us30_d1",
        sharpe=metrics["sharpe"],
        n_trades=int(metrics["trades"]),
    )

    out = {
        "config": {
            "strat": "DonchianBreakout(N=20, M=20)",
            "features": list(build_features(df.head(300)).columns),
            "asset": "US30",
            "tf": "D1",
            "retrain_months": 6,
            "capital_eur": 10_000.0,
            "risk_per_trade": 0.02,
        },
        "metrics_walk_forward_oos": metrics,
        "segments": [s.__dict__ for s in segments],
        "cpcv_train_val": cpcv_results,
        "validate_edge": {
            "go": report.go,
            "reasons": report.reasons,
            "metrics": report.metrics,
        },
    }
    out_path = Path("predictions/h_new1_meta_us30.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(f"H_new1 terminé. Verdict : {'GO' if report.go else 'NO-GO'}")
    print(f"Raisons : {report.reasons}")
    print(f"Résultats : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 4 — Mise à jour `JOURNAL.md` AVANT exécution

Insérer dans le tableau n_trials :

```markdown
| pivot_v4 | H_new1 (méta-labeling RF Donchian US30 D1) | 1 | 23 | EN COURS | — |
```

Cette ligne est ajoutée AVANT le run, pour engagement formel.

### Étape 5 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_h_new1_meta_us30.py
```

**⚠️ Cette exécution lit le test set 2024+ pour la première et dernière fois sur cette configuration.**

### Étape 6 — Lecture des résultats & rapport

`docs/h_new1_meta_us30.md` :

```markdown
# H_new1 — Méta-labeling RF Donchian US30 D1 (pivot v4 B1)

**Date** : YYYY-MM-DD
**n_trials** : 23 (22 hérités + 1 H_new1)
**Verdict** : [GO / NO-GO]

## Question
Le méta-labeling v2 H05 (Sharpe +8.84) résiste-t-il aux coûts XTB réels ?

## Méthode
- Donchian Breakout (N=20, M=20) US30 D1 baseline
- Méta-labeling RF (n_estimators=200, max_depth=4, class_weight=balanced, threshold plancher 0.50)
- Walk-forward retrain 6M depuis 2024-01-01
- Coûts v4 (spread=1.5, slippage=0.3)
- Sizing au risque 2 %

## Résultats walk-forward OOS

| Métrique | Cible | Observé | Status |
|---|---|---|---|
| Sharpe walk-forward | ≥ 1.0 | ? | ✓/✗ |
| DSR (n=23) | > 0 (p<0.05) | ? | ✓/✗ |
| Max DD | < 15 % | ? | ✓/✗ |
| WR | > 35 % | ? | ✓/✗ |
| Trades/an | ≥ 30 | ? | ✓/✗ |

## Segments walk-forward

| Segment | Période | n_trades | Sharpe | WR |
|---|---|---|---|---|
| 2024 H1 | 2024-01 → 2024-06 | ? | ? | ? |
| 2024 H2 | 2024-07 → 2024-12 | ? | ? | ? |
| 2025 H1 | 2025-01 → 2025-06 | ? | ? | ? |
| ... | ... | ... | ... | ... |

## Comparaison vs v2 H05

| | v2 H05 | Pivot v4 B1 |
|---|---|---|
| Sharpe WF | +8.84 | ? |
| n_trades | 12 (sur 30 mois) | ? |
| Coûts US30 | sous-estimés | XTB réels (1.8 pts) |
| Méthode Sharpe | — | weekly / per-trade |

## Décision

[GO] → Passer en Phase 4 production (prompts 20-24).
[NO-GO] → Bascule H_new2 (walk-forward rolling adaptatif) ou H_new3 (EURUSD H4 mean-rev).

## Erreurs de v2 H05 corrigées
- Sizing au risque 2 % implémenté (v2 H05 utilisait 1 lot fixe)
- Coûts XTB réels (v2 H05 utilisait 0.5 pip de coût total)
- Sharpe routing weekly (v2 H05 utilisait daily biaisé)
```

### Étape 7 — `JOURNAL.md` entrée finale

```markdown
## YYYY-MM-DD — Pivot v4 B1 : H_new1 méta-labeling US30 D1

- **Statut** : [✅ GO / ❌ NO-GO]
- **n_trials_cumul** : 23
- **Sharpe walk-forward OOS** : ?
- **Comparaison v2 H05** : Sharpe +8.84 (v2) vs ? (v4)
- **Fichiers créés** : app/models/meta_labeling.py, app/pipelines/walk_forward.py, scripts/run_h_new1_meta_us30.py, predictions/h_new1_meta_us30.json, docs/h_new1_meta_us30.md
- **Décision** : [GO Phase production / NO-GO → bascule H_new3]
- **Notes** : test set 2024+ lu UNE seule fois, read_oos() appelé.
```

## Tests unitaires associés

`tests/unit/test_meta_labeling_rf.py` :

```python
import numpy as np
import pandas as pd
from app.models.meta_labeling import MetaLabelingRF, MetaLabelingConfig


def test_fit_predict_proba():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=list("ABCDE"))
    y = pd.Series((X.iloc[:, 0] > 0).astype(int))
    meta = MetaLabelingRF()
    meta.fit(X, y)
    pred = meta.predict(X)
    assert pred.dtype == bool
    assert len(pred) == 200


def test_calibrate_threshold_falls_back_if_all_strict():
    X = pd.DataFrame(np.zeros((100, 5)), columns=list("ABCDE"))
    y = pd.Series([0] * 99 + [1])
    meta = MetaLabelingRF()
    meta.fit(X, y)
    # backtest_fn renvoie toujours -inf si on filtre tout
    meta.calibrate_threshold(X, lambda mask: -1e9 if mask.sum() < 50 else 0.0)
    # Soit fallback (disabled), soit threshold ≥ 0.50
    assert meta.disabled or meta.threshold >= 0.50


def test_class_weight_balanced():
    X = pd.DataFrame(np.random.RandomState(0).normal(0, 1, (100, 5)))
    y = pd.Series([0] * 80 + [1] * 20)
    meta = MetaLabelingRF()
    meta.fit(X, y)
    # Avec class_weight="balanced", le modèle ne doit pas tout prédire à 0
    proba = meta.model.predict_proba(X.values)[:, 1]
    assert proba.max() > 0.4
```

## Logging obligatoire

Cf. Étape 7. Plus l'entrée n_trials_cumul mise à jour AVANT le run (Étape 4) puis confirmée APRÈS.

## Critères go/no-go

- **GO Phase 4 production** (prompts 20-24) si :
  - Sharpe walk-forward ≥ 1.0
  - DSR > 0 avec p < 0.05 (n_trials = 23)
  - DD max < 15 %
  - WR > 35 %
  - Trades/an ≥ 30
  - **TOUS les 5 critères**.
- **NO-GO, bascule** :
  - Si Sharpe < 0.5 → directement vers H_new3 (EURUSD H4 mean-rev)
  - Si Sharpe 0.5-1.0 OU 1+ critère seulement échoue → bascule H_new2 (walk-forward rolling adaptatif)

## Annexes

### A1 — Pourquoi figer Donchian (20, 20) sans re-grid search

En v2 H03, le grid search 9 combos a sélectionné (20, 20) comme baseline best train. Re-tester (50, 10) qui était best train en v3 H06 = essayer un autre point d'hyperparamètres → +1 n_trial supplémentaire.

Pour économiser les n_trials, on **fige** les params Donchian au valeur historique v2 H03/H04/H05. Si NO-GO, on ne saura pas si c'est à cause des params ou du méta-labeling — mais on aura économisé un n_trial.

### A2 — Pourquoi class_weight="balanced"

Donchian US30 D1 sur train ≤ 2022 produit en moyenne ~45 % WR (selon v3 H06 train). Donc 55 % de la cible y_train = 0 (loser). Sans class_weight, le RF tend à toujours prédire 0 (rejeter tous les signaux) → 0 trade en OOS, comme le H1 XAUUSD du pivot v3.

Avec `class_weight="balanced"`, le RF pénalise les erreurs sur la classe minoritaire (1=winner) proportionnellement à son inverse de fréquence → équilibre les prédictions.

### A3 — Embargo 2 jours dans le walk-forward

Entre la fin du train et le début de l'OOS, on saute 2 jours pour éviter qu'un trade ouvert en train se prolonge dans l'OOS. Le timeout Donchian = 120h ≈ 5 jours, donc 2 jours d'embargo est plutôt court. Si signal de leakage détecté en tests → passer à 7 jours.

### A4 — Si le méta-labeling est désactivé (fallback)

Si `MetaLabelingRF.disabled == True` à un retrain → ce segment OOS tourne en baseline Donchian pur (sans filtre). C'est documenté dans `segments` du JSON. Si > 50 % des segments sont en fallback → le méta-labeling n'apporte rien, NO-GO.

### A5 — Pourquoi pas tester aussi sur XAUUSD ici

XAUUSD était NO-GO en H06 v3 (WR 22.5 % + trades/an 18) et a échoué en H1 pivot v3 (split mono-classe). On le testera en H_new2 (walk-forward rolling adaptatif) qui résout le problème du split figé. Ici on se concentre sur US30 = la seule baseline v2 GO.

## Fin du prompt B1.
**Suivant si GO** : `prompts/20_signal_engine.md` (Phase 4 production)
**Suivant si NO-GO** : [06_h_new3_eurusd_h4_meanrev.md](06_h_new3_eurusd_h4_meanrev.md) ou [07_h_new2_walk_forward_rolling.md](07_h_new2_walk_forward_rolling.md)
