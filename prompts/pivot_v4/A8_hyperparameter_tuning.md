# Pivot v4 — A8 : Tuning hyperparams + seuil via nested CPCV

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. [A5_feature_generation.md](A5_feature_generation.md), [A6_feature_ranking.md](A6_feature_ranking.md), [A7_model_selection.md](A7_model_selection.md) — **✅ Terminés**
3. [app/config/features_selected.py](../../app/config/features_selected.py) — top 15 figé
4. [app/config/model_selected.py](../../app/config/model_selected.py) — modèle retenu par actif
5. [app/models/cpcv_evaluation.py](../../app/models/cpcv_evaluation.py) — CPCV existant
6. [../00_constitution.md](../00_constitution.md)

## Objectif
Tuner les hyperparams du modèle retenu en A7 + calibrer le **seuil de décision** via **nested CPCV** (CPCV imbriqué : outer pour évaluation honnête, inner pour grid search). Toujours sur train ≤ 2022 uniquement.

> **Principe critique** : la **nested CV** est indispensable. Si on tune sur les mêmes folds qu'on évalue, on overfit sur le validation set. Outer CV évalue la performance attendue ; inner CV choisit les hyperparams sur train de chaque outer fold.

## Type d'opération
🔧 **Tuning train-only** — **0 n_trial consommé**. Aucune lecture du test set ≥ 2024.

## Definition of Done (testable)

- [ ] `app/models/nested_tuning.py` (NOUVEAU) :
  - `nested_cpcv_tuning(model_builder, param_grid, X, y, pnl, threshold_grid, outer_k=5, inner_k=3) -> TuningResult`
  - Retourne meilleurs hyperparams + seuil + Sharpe outer (honnête)
- [ ] Grids hyperparams **petits** (3 valeurs max par axe) pour limiter le coût combinatoire :
  - **RF** : `n_estimators ∈ {100, 200, 400}`, `max_depth ∈ {3, 4, 6}`, `min_samples_leaf ∈ {5, 10, 20}` → 27 combos
  - **HGBM** : `max_iter ∈ {100, 200, 400}`, `max_depth ∈ {4, 5, 7}`, `learning_rate ∈ {0.02, 0.05, 0.1}` → 27 combos
  - **Stacking** : pas tuné (trop lent), garder defaults A7
- [ ] **Threshold grid** : `[0.50, 0.55, 0.60]` (jamais < 0.50 — leçon H04 v2)
- [ ] `scripts/run_a8_hyperparam_tuning.py` :
  - Pour chaque actif/TF retenu en A7
  - Charge `MODEL_SELECTED[(asset, tf)]`
  - Lance nested CPCV
  - Sauvegarde meilleurs hyperparams dans `app/config/hyperparams_tuned.py` (frozen)
- [ ] `app/config/hyperparams_tuned.py` (NOUVEAU, FROZEN) :
  ```python
  HYPERPARAMS_TUNED: dict[tuple[str, str], dict] = {
      ("US30", "D1"): {
          "model": "hgbm",
          "params": {"max_iter": 200, "max_depth": 5, "learning_rate": 0.05},
          "threshold": 0.55,
          "expected_sharpe_outer": 1.65,
          "expected_wr": 0.54,
      },
      ...
  }
  ```
- [ ] `docs/hyperparam_tuning_v4.md` : rapport détaillé
- [ ] `tests/unit/test_nested_tuning.py` : ≥ 5 tests :
  - Nested CPCV ne fuit pas d'info outer → inner
  - Threshold plancher 0.50 respecté
  - Reproductibilité (même seed → même résultat)
  - Grid produit param_grid puissance des axes
  - Sharpe outer cohérent avec attendu sur synthétique
- [ ] `rtk make verify` → 0 erreur
- [ ] `JOURNAL.md` mis à jour

## NE PAS FAIRE

- ❌ Ne PAS utiliser le test set ≥ 2024.
- ❌ Ne PAS tuner sur outer CV (= overfit garanti). Inner CV uniquement.
- ❌ Ne PAS tuner > 3 valeurs par axe (combinatoire 4^3 = 64 trop lent, 5^3 = 125 trop coûteux).
- ❌ Ne PAS choisir un seuil < 0.50 (leçon H04 v2).
- ❌ Ne PAS modifier `features_selected.py` ou `model_selected.py` ici.
- ❌ Ne PAS introduire de nouveaux hyperparams hors du grid déclaré (RF/HGBM uniquement).
- ❌ Ne PAS incrémenter `n_trials`.

## Étapes détaillées

### Étape 1 — `app/models/nested_tuning.py`

```python
"""Tuning hyperparams + seuil via nested CPCV (pivot v4 A8)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from app.models.cpcv_evaluation import _purged_kfold_indices, _compute_sharpe_per_trade


@dataclass
class TuningResult:
    best_params: dict
    best_threshold: float
    sharpe_outer_mean: float
    sharpe_outer_std: float
    wr_outer_mean: float
    n_kept_outer_mean: float
    outer_fold_results: list[dict]
    n_combos_evaluated: int


def _expand_grid(param_grid: dict) -> list[dict]:
    """Expand un grid {k: [v1, v2], k2: [w1]} en liste de dicts."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def nested_cpcv_tuning(
    model_factory: Callable[[dict, int], object],
    param_grid: dict,
    threshold_grid: list[float],
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    outer_k: int = 5,
    inner_k: int = 3,
    embargo_pct: float = 0.01,
    seed: int = 42,
) -> TuningResult:
    """Nested CPCV : outer pour évaluation honnête, inner pour sélection hyperparams.

    Args:
        model_factory: fonction (params: dict, seed: int) -> modèle sklearn-like.
        param_grid: dict de listes pour chaque hyperparam.
        threshold_grid: liste de seuils candidats (plancher 0.50).
        X, y, pnl: données train.

    Returns:
        TuningResult avec best_params, best_threshold, Sharpe outer non biaisé.
    """
    assert all(t >= 0.50 for t in threshold_grid), "Seuil < 0.50 interdit (leçon H04)"
    n = len(X)
    outer_splits = _purged_kfold_indices(n, k=outer_k, embargo_pct=embargo_pct)
    combos = _expand_grid(param_grid)
    n_combos = len(combos)

    # Pour chaque outer fold, sélectionner les meilleurs hyperparams via inner CV sur le train outer
    outer_results = []
    best_combo_votes: dict[tuple, int] = {}
    best_threshold_votes: dict[float, int] = {}

    for outer_train_idx, outer_test_idx in outer_splits:
        if len(outer_train_idx) < 60:
            continue
        X_outer_tr = X.iloc[outer_train_idx]
        y_outer_tr = y.iloc[outer_train_idx]
        pnl_outer_tr = pnl.iloc[outer_train_idx]
        X_outer_te = X.iloc[outer_test_idx]
        y_outer_te = y.iloc[outer_test_idx]
        pnl_outer_te = pnl.iloc[outer_test_idx]

        # Inner CV sur outer_train
        inner_splits = _purged_kfold_indices(len(X_outer_tr), k=inner_k, embargo_pct=embargo_pct)
        best_inner_score = -np.inf
        best_inner_params = combos[0]
        best_inner_threshold = 0.50

        for combo in combos:
            for threshold in threshold_grid:
                inner_sharpes = []
                for in_tr, in_te in inner_splits:
                    if len(in_tr) < 30 or len(in_te) < 10:
                        continue
                    model = model_factory(combo, seed)
                    model.fit(
                        X_outer_tr.iloc[in_tr].values,
                        y_outer_tr.iloc[in_tr].values,
                    )
                    proba = model.predict_proba(X_outer_tr.iloc[in_te].values)[:, 1]
                    sr, _, _ = _compute_sharpe_per_trade(
                        proba, pnl_outer_tr.iloc[in_te].values, threshold,
                    )
                    inner_sharpes.append(sr)
                if not inner_sharpes:
                    continue
                inner_mean = float(np.mean(inner_sharpes))
                if inner_mean > best_inner_score:
                    best_inner_score = inner_mean
                    best_inner_params = combo
                    best_inner_threshold = threshold

        # Évalue ces hyperparams sur outer_test (honnête : jamais vu)
        model = model_factory(best_inner_params, seed)
        model.fit(X_outer_tr.values, y_outer_tr.values)
        proba_te = model.predict_proba(X_outer_te.values)[:, 1]
        outer_sr, outer_wr, outer_n_kept = _compute_sharpe_per_trade(
            proba_te, pnl_outer_te.values, best_inner_threshold,
        )
        outer_results.append({
            "best_params": best_inner_params,
            "best_threshold": best_inner_threshold,
            "outer_sharpe": outer_sr,
            "outer_wr": outer_wr,
            "outer_n_kept": outer_n_kept,
            "inner_best_score": best_inner_score,
        })
        # Vote
        combo_key = tuple(sorted(best_inner_params.items()))
        best_combo_votes[combo_key] = best_combo_votes.get(combo_key, 0) + 1
        best_threshold_votes[best_inner_threshold] = (
            best_threshold_votes.get(best_inner_threshold, 0) + 1
        )

    if not outer_results:
        raise RuntimeError("Aucun outer fold n'a produit de résultat")

    # Sélection finale : params le plus voté + threshold le plus voté
    final_combo = max(best_combo_votes.items(), key=lambda kv: kv[1])[0]
    final_params = dict(final_combo)
    final_threshold = max(best_threshold_votes.items(), key=lambda kv: kv[1])[0]

    outer_sharpes = [r["outer_sharpe"] for r in outer_results]
    outer_wrs = [r["outer_wr"] for r in outer_results]
    outer_n_kepts = [r["outer_n_kept"] for r in outer_results]

    return TuningResult(
        best_params=final_params,
        best_threshold=final_threshold,
        sharpe_outer_mean=float(np.mean(outer_sharpes)),
        sharpe_outer_std=float(np.std(outer_sharpes)),
        wr_outer_mean=float(np.mean(outer_wrs)),
        n_kept_outer_mean=float(np.mean(outer_n_kepts)),
        outer_fold_results=outer_results,
        n_combos_evaluated=n_combos * len(threshold_grid),
    )
```

### Étape 2 — `scripts/run_a8_hyperparam_tuning.py`

```python
"""Pivot v4 A8 — Tuning hyperparams + seuil via nested CPCV.

⚠️ Aucune lecture du test set ≥ 2024.
Coût : ~30-60 min sur CPU 8-core selon le nombre d'actifs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.config.instruments import ASSET_CONFIGS
from app.config.features_selected import FEATURES_SELECTED
from app.config.model_selected import MODEL_SELECTED
from app.features.superset import build_superset
from app.models.nested_tuning import nested_cpcv_tuning
from app.strategies.donchian import DonchianBreakout
from app.backtest.deterministic import run_deterministic_backtest

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42


STRAT_MAP = {
    ("US30", "D1"): (DonchianBreakout, {"N": 20, "M": 20}),
    ("XAUUSD", "D1"): (DonchianBreakout, {"N": 100, "M": 20}),
}


def rf_factory(params: dict, seed: int):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 4),
        min_samples_leaf=params.get("min_samples_leaf", 10),
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )


def hgbm_factory(params: dict, seed: int):
    return HistGradientBoostingClassifier(
        max_iter=params.get("max_iter", 200),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.05),
        l2_regularization=1.0,
        class_weight="balanced", random_state=seed, early_stopping=False,
    )


PARAM_GRIDS = {
    "rf": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 4, 6],
        "min_samples_leaf": [5, 10, 20],
    },
    "hgbm": {
        "max_iter": [100, 200, 400],
        "max_depth": [4, 5, 7],
        "learning_rate": [0.02, 0.05, 0.10],
    },
    # stacking : pas tuné (trop lent)
}


THRESHOLD_GRID = [0.50, 0.55, 0.60]


def tune_one_asset(asset: str, tf: str) -> dict:
    key = (asset, tf)
    model_name = MODEL_SELECTED.get(key)
    if not model_name:
        return {"error": f"No model selected for {asset} {tf}"}
    if model_name == "stacking":
        return {
            "asset": asset, "tf": tf, "model": "stacking",
            "best_params": {}, "best_threshold": 0.50,
            "note": "Stacking pas tuné (trop lent). Defaults A7 conservés.",
        }

    factory = {"rf": rf_factory, "hgbm": hgbm_factory}[model_name]
    grid = PARAM_GRIDS[model_name]

    cfg = ASSET_CONFIGS[asset]
    df = load_asset(asset, tf).loc[lambda d: d.index <= CUTOFF]
    strat_cls, strat_kwargs = STRAT_MAP[(asset, tf)]
    strat = strat_cls(**strat_kwargs)
    trades = run_deterministic_backtest(df, strat, cfg)
    if trades.empty or len(trades) < 80:
        return {"error": f"Too few trades: {len(trades)}"}

    X_full = build_superset(df, asset=asset)
    selected = list(FEATURES_SELECTED[key])
    X = X_full.loc[trades.index, selected].dropna()
    y = (trades["Pips_Bruts"] > 0).astype(int).loc[X.index]
    pnl = trades["Pips_Bruts"].loc[X.index]

    if len(X) < 80:
        return {"error": f"Too few samples after align: {len(X)}"}

    print(f"  Tuning {asset} {tf} ({model_name}) on {len(X)} samples, "
          f"{len(grid)} axes, {len(THRESHOLD_GRID)} thresholds...")
    r = nested_cpcv_tuning(
        model_factory=factory,
        param_grid=grid,
        threshold_grid=THRESHOLD_GRID,
        X=X, y=y, pnl=pnl,
        outer_k=5, inner_k=3, embargo_pct=0.01, seed=SEED,
    )
    return {
        "asset": asset,
        "tf": tf,
        "model": model_name,
        "best_params": r.best_params,
        "best_threshold": r.best_threshold,
        "expected_sharpe_outer": r.sharpe_outer_mean,
        "expected_wr": r.wr_outer_mean,
        "expected_n_kept": r.n_kept_outer_mean,
        "sharpe_outer_std": r.sharpe_outer_std,
        "n_combos_evaluated": r.n_combos_evaluated,
        "outer_folds": r.outer_fold_results,
    }


def main() -> int:
    set_global_seeds()
    out = {}
    for (asset, tf) in FEATURES_SELECTED.keys():
        key = f"{asset}_{tf}"
        print(f"Tuning {key}...")
        out[key] = tune_one_asset(asset, tf)

    out_path = Path("predictions/hyperparam_tuning_v4.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    # Génère app/config/hyperparams_tuned.py
    lines = []
    for (asset, tf), entry_key in [((a, t), f"{a}_{t}") for (a, t) in FEATURES_SELECTED]:
        if entry_key not in out or "error" in out[entry_key]:
            continue
        e = out[entry_key]
        lines.append(
            f'    ("{asset}", "{tf}"): {{\n'
            f'        "model": "{e["model"]}",\n'
            f'        "params": {e["best_params"]!r},\n'
            f'        "threshold": {e["best_threshold"]},\n'
            f'        "expected_sharpe_outer": {e["expected_sharpe_outer"]:.3f},\n'
            f'        "expected_wr": {e["expected_wr"]:.3f},\n'
            f'    }},'
        )
    content = '"""FROZEN après pivot v4 A8. NE PAS MODIFIER sans nouveau pivot."""\n' \
              'from __future__ import annotations\n\n' \
              'HYPERPARAMS_TUNED: dict[tuple[str, str], dict] = {\n'
    content += "\n".join(lines) + "\n}\n"
    Path("app/config/hyperparams_tuned.py").write_text(content, encoding="utf-8")

    print(f"Hyperparams sauvegardés dans {out_path} et app/config/hyperparams_tuned.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — Tests `tests/unit/test_nested_tuning.py`

```python
"""Tests nested CPCV tuning pivot v4 A8."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier
from app.models.nested_tuning import (
    nested_cpcv_tuning, _expand_grid, TuningResult,
)


def _rf_factory(params, seed):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 50),
        max_depth=params.get("max_depth", 3),
        class_weight="balanced", random_state=seed, n_jobs=1,
    )


def _make_xy_pnl(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, 10)), columns=[f"f{i}" for i in range(10)])
    logit = X.iloc[:, :3].sum(axis=1) + rng.normal(0, 0.3, n)
    y = pd.Series((logit > 0).astype(int))
    pnl = pd.Series(np.where(y == 1, rng.uniform(50, 200, n), rng.uniform(-150, -50, n)))
    return X, y, pnl


def test_expand_grid_produces_cartesian():
    g = {"a": [1, 2], "b": [3, 4, 5]}
    combos = _expand_grid(g)
    assert len(combos) == 6
    assert all("a" in c and "b" in c for c in combos)


def test_nested_tuning_runs():
    X, y, pnl = _make_xy_pnl(n=300)
    r = nested_cpcv_tuning(
        model_factory=_rf_factory,
        param_grid={"n_estimators": [50, 100], "max_depth": [3, 4]},
        threshold_grid=[0.50, 0.55],
        X=X, y=y, pnl=pnl,
        outer_k=3, inner_k=2, embargo_pct=0.01, seed=42,
    )
    assert isinstance(r, TuningResult)
    assert r.best_threshold >= 0.50


def test_threshold_below_05_raises():
    X, y, pnl = _make_xy_pnl()
    with pytest.raises(AssertionError, match="Seuil < 0.50 interdit"):
        nested_cpcv_tuning(
            model_factory=_rf_factory,
            param_grid={"n_estimators": [50]},
            threshold_grid=[0.40],
            X=X, y=y, pnl=pnl,
            outer_k=3, inner_k=2,
        )


def test_reproducible():
    X, y, pnl = _make_xy_pnl()
    r1 = nested_cpcv_tuning(
        _rf_factory, {"n_estimators": [50, 100]}, [0.50, 0.55],
        X, y, pnl, outer_k=3, inner_k=2, seed=42,
    )
    r2 = nested_cpcv_tuning(
        _rf_factory, {"n_estimators": [50, 100]}, [0.50, 0.55],
        X, y, pnl, outer_k=3, inner_k=2, seed=42,
    )
    assert r1.best_params == r2.best_params
    assert r1.best_threshold == r2.best_threshold


def test_n_combos_correct():
    X, y, pnl = _make_xy_pnl()
    grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 4]}
    r = nested_cpcv_tuning(
        _rf_factory, grid, [0.50, 0.55, 0.60],
        X, y, pnl, outer_k=3, inner_k=2,
    )
    assert r.n_combos_evaluated == 3 * 2 * 3  # n_estim × max_depth × thresholds
```

### Étape 4 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_a8_hyperparam_tuning.py
```

Coût : 30 min – 2h selon nombre d'actifs et grids. Patient.

### Étape 5 — Rapport `docs/hyperparam_tuning_v4.md`

```markdown
# Hyperparam Tuning v4 (pivot v4 A8)

**Date** : YYYY-MM-DD
**Méthode** : Nested CPCV (outer=5, inner=3, embargo=1%)
**Périmètre** : train ≤ 2022-12-31 UNIQUEMENT
**n_trials** : inchangé

## Résultats par actif

### US30 D1 (HGBM)

| Outer fold | best_params | best_threshold | Sharpe outer | WR outer | n_kept |
|---|---|---|---|---|---|
| 1 | max_iter=200, depth=5, lr=0.05 | 0.55 | 1.68 | 0.54 | 22 |
| 2 | max_iter=200, depth=5, lr=0.05 | 0.55 | 1.42 | 0.52 | 19 |
| 3 | ... | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... | ... |

**Choix final (vote)** :
- params : `{max_iter=200, max_depth=5, learning_rate=0.05}` (4/5 folds)
- threshold : 0.55 (3/5 folds)
- **Sharpe outer attendu : 1.52 ± 0.18** (sans biais sélection)

### EURUSD H4
...

### XAUUSD D1
...

## Sharpe "outer" vs Sharpe "inner"

| Actif | Sharpe inner (biaisé) | Sharpe outer (honnête) | Écart |
|---|---|---|---|
| US30 D1 | 1.78 (A7) | 1.52 | -0.26 |
| EURUSD H4 | ... | ... | ... |

Le Sharpe outer < Sharpe inner = signe normal d'**overfit hyperparams**. L'écart < 0.5 = OK. Si écart > 1.0 → grid trop large ou inner CV trop petit.

## Décision de gel

Hyperparams + seuil FIGÉS dans `app/config/hyperparams_tuned.py`.
**Aucune modification post-A8.**

## Limites
- Grid de 27 combos × 3 thresholds = 81 essais par outer fold = 405 essais total/actif. Compromis coût/exhaustivité.
- inner_k=3 plutôt petit. Pour plus de stabilité, 5 serait mieux mais 2× plus lent.
- Le vote majoritaire peut masquer un disagreement entre folds → loggé dans `outer_folds`.
```

## Tests unitaires associés

5 tests dans `tests/unit/test_nested_tuning.py` (Étape 3).

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A8 : Tuning hyperparams + seuil

- **Statut** : ✅ Terminé
- **Type** : Tuning train (0 n_trial)
- **Fichiers créés** : `app/models/nested_tuning.py`, `scripts/run_a8_hyperparam_tuning.py`, `app/config/hyperparams_tuned.py`, `tests/unit/test_nested_tuning.py`, `docs/hyperparam_tuning_v4.md`, `predictions/hyperparam_tuning_v4.json`
- **Hyperparams retenus** :
  - US30 D1 : ?
  - EURUSD H4 : ?
  - XAUUSD D1 : ?
- **Sharpe outer (estimation honnête)** :
  - US30 D1 : ?
  - EURUSD H4 : ?
- **Tests** : 5/5 passed
- **Coût compute** : ~? min
- **make verify** : ✅ passé
- **Prochaine étape** : A9 — Pipeline lock (geler tout)
```

## Critères go/no-go

- **GO Phase A9** si :
  - `app/config/hyperparams_tuned.py` rempli
  - Sharpe outer ≥ 0.5 sur ≥ 1 actif (sinon l'edge train n'existe pas)
  - Écart Sharpe inner-outer < 1.0 (sinon overfit grid)
- **NO-GO, revenir à** :
  - Sharpe outer ≤ 0 partout → revoir features (A6) ou modèle (A7)
  - Écart inner-outer > 1.5 → grid trop large, réduire à 9 combos

## Annexes

### A1 — Pourquoi nested CV (et pas simple CV)

CV simple sur hyperparams + même CV pour évaluation = **overfit sur validation set**. Le Sharpe estimé sera biaisé optimiste de 0.3-1.0.

Nested CV :
- Inner CV sur train_outer → choisit hyperparams (peut overfit le train_outer, c'est OK)
- Outer CV évalue les hyperparams choisis sur test_outer (jamais vu)
- Sharpe outer = estimation **honnête** de la performance attendue OOS

C'est la méthode de référence pour rapporter une performance non biaisée.

### A2 — Pourquoi 27 combos et pas 1000

- 27 = 3^3 = chaque axe testé à 3 valeurs (min/middle/max)
- Coût : 27 × 3 thresholds × 5 outer × 3 inner = 1215 fits par actif. ~30 min sur CPU 8-core pour HGBM.
- 1000 combos (grid search dense) = 25× plus lent et marginal en gain.

Pour V5, on pourrait remplacer par Bayesian Optimization (optuna) → 50 itérations adaptatives.

### A3 — Pourquoi seuil ∈ {0.50, 0.55, 0.60}

- 0.50 = neutre, garde tous les signaux > moyenne
- 0.55 = filtre modérément agressif (~30 % des signaux éliminés)
- 0.60 = filtre fort (~50 % éliminés)

> 0.60 = trop agressif, risque "H1 XAUUSD-like" (0 trade test).
< 0.50 = interdit (constitution règle implicite).

### A4 — Vote majoritaire pour les hyperparams finaux

Sur 5 outer folds, chaque fold choisit ses meilleurs hyperparams. On prend ceux qui apparaissent **le plus souvent**. Si tie 2-2-1 → choix par ordre alphabétique des keys (déterministe).

Alternative : moyenne pondérée des Sharpe par combo. Plus complexe, gain marginal.

### A5 — Que faire si Stacking est choisi en A7 (non tuné en A8)

Stacking est trop lent à tuner (chaque fit = 5 cv interne). On garde les defaults A7 et on saute le tuning. C'est documenté dans `hyperparams_tuned.py` :

```python
("US30", "D1"): {
    "model": "stacking",
    "params": {},  # vide = defaults A7
    "threshold": 0.50,  # non tuné
    ...
}
```

### A6 — Pourquoi early_stopping=False pour HGBM

`early_stopping=True` utilise un validation interne aléatoire → résultats non reproductibles entre folds. Pour comparaison équitable entre folds CPCV, on fixe `early_stopping=False` et on contrôle via `max_iter`.

### A7 — Impact d'un Sharpe outer faible (< 0.5)

Si Sharpe outer < 0.5 sur tous les actifs après A8, c'est le signal que **l'edge train n'existe tout simplement pas** (avec ce modèle + ces features + cette stratégie sous-jacente).

Options :
1. Revenir A6 et essayer un superset différent (mais c'est du data snooping caché).
2. Revenir A5 et inclure des features alternatives jamais testées (COT, term structure).
3. Accepter et passer en B1 quand même (le Sharpe outer est une estimation conservative, le walk-forward réel peut être un peu meilleur).
4. Abandonner l'actif et se concentrer sur ceux où Sharpe outer ≥ 0.5.

## Fin du prompt A8.
**Suivant** : [A9_pipeline_lock.md](A9_pipeline_lock.md)
