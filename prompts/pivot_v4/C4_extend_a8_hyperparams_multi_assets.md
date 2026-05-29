# Pivot v4 — C4 : Extension A8 hyperparam tuning multi-actifs

> 📍 **Quatrième étape de la Phase C — Extension multi-actifs**
> Tuning des hyperparams (nested CPCV) + seuil de calibration pour chaque couple en shortlist C3.
> **Aucune lecture du test set ≥ 2024**. 0 n_trial consommé.

## Préalable obligatoire (à lire dans l'ordre)

1. [00_README.md](00_README.md) — vue d'ensemble pivot v4
2. [../00_constitution.md](../00_constitution.md) — règles 6, 7, 9
3. [C2_extend_a6_ranking_multi_assets.md](C2_extend_a6_ranking_multi_assets.md) — ✅ Terminé
4. [C3_extend_a7_model_selection_multi_assets.md](C3_extend_a7_model_selection_multi_assets.md) — **✅ Terminé obligatoire**
5. [A8_hyperparameter_tuning.md](A8_hyperparameter_tuning.md) — A8 d'origine (référence méthodo)
6. [`docs/hyperparam_tuning_v4.md`](../../docs/hyperparam_tuning_v4.md) — résultats A8 d'origine
7. [`predictions/c3_model_selection_multi_assets.json`](../../predictions/c3_model_selection_multi_assets.json) — shortlist C4
8. [`app/models/nested_tuning.py`](../../app/models/nested_tuning.py) — nested CPCV utilisée
9. [`scripts/run_a8_hyperparam_tuning.py`](../../scripts/run_a8_hyperparam_tuning.py) — script d'origine
10. [`app/config/hyperparams_tuned.py`](../../app/config/hyperparams_tuned.py) — 3 entrées actuelles

## Objectif

Pour chaque couple en shortlist C3 (`pass_c4_threshold=true`, modèle ∈ {`rf`, `hgbm`}) :
1. Tuning hyperparams via **nested CPCV** (outer 5-fold, inner 3-fold, embargo 1 %)
2. Calibration du **seuil méta** par fold (vote majoritaire ∈ {0.50, 0.55, 0.60})
3. Reporter Sharpe outer + écart inner-outer (proxy de l'overfit)

Figer les résultats dans [`app/config/hyperparams_tuned.py`](../../app/config/hyperparams_tuned.py) (sans toucher aux 3 entrées d'origine).

> ⚠️ **Stacking est exclu du tuning** (trop lent en nested CV — décision A8). Pour les couples shortlistés en `stacking`, on garde les defaults dans hyperparams_tuned.py avec `threshold=0.50` (équivalent au comportement XAUUSD D1 en A8).

## Type d'opération

🔧 **Infrastructure ML — 0 n_trial consommé. Aucune lecture du test set 2024+.**

## Definition of Done (testable)

- [ ] `scripts/run_c4_hyperparam_tuning_multi_assets.py` (NOUVEAU) lit `c3_model_selection_multi_assets.json`, filtre `pass_c4_threshold=true`, et tune chaque couple via `nested_tuning`.
- [ ] Chaque couple tuné se voit attribuer **params + threshold figés** dans `app/config/hyperparams_tuned.py` (sans toucher aux 3 entrées d'origine).
- [ ] `predictions/c4_hyperparam_tuning_multi_assets.json` contient pour chaque couple : params retenus, threshold retenu, Sharpe outer moyen + std, Sharpe inner moyen, écart inner-outer, n_kept moyen, WR outer.
- [ ] `docs/hyperparam_tuning_v4_extended.md` (NOUVEAU) avec tableau récapitulatif + analyse.
- [ ] `tests/unit/test_c4_hyperparams_tuned_extended.py` vérifie : 3 entrées d'origine intactes, chaque nouvelle entrée a les clés requises (`model`, `params`, `threshold`, `expected_sharpe_outer`, `expected_wr`).
- [ ] **Shortlist finale pour C5 / Phase B** : couples avec Sharpe outer ≥ 0.5 ET écart inner-outer < 1.0.
- [ ] `rtk make verify` → 0 erreur.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE

- ❌ **Ne PAS modifier les 3 entrées existantes** de `HYPERPARAMS_TUNED` (US30 D1, EURUSD H4, XAUUSD D1).
- ❌ **Ne PAS lire le test set ≥ 2024.** Cutoff strict `2022-12-31`.
- ❌ **Ne PAS modifier `app/models/nested_tuning.py`** (méthodo figée).
- ❌ **Ne PAS étendre la grille** au-delà de celle d'A8. Garder la même grille pour comparabilité.
- ❌ **Ne PAS tuner `stacking`** : trop lent. Garder defaults.
- ❌ **Ne PAS choisir manuellement** params/threshold : le vote majoritaire des outer folds décide.
- ❌ **Ne PAS incrémenter `n_trials`.**
- ❌ **Ne PAS exécuter Phase B (test set)** dans ce prompt — C4 reste strictement train.

## Étapes détaillées

### Étape 1 — Grille d'hyperparams (identique A8)

**RF (RandomForestClassifier)** :
```python
RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "min_samples_leaf": [5, 10, 20],
}
```

**HGBM (HistGradientBoostingClassifier)** :
```python
HGBM_GRID = {
    "max_depth": [3, 6, None],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}
```

**Seuils méta candidats** : `[0.50, 0.55, 0.60]`

### Étape 2 — Créer le script

Créer [`scripts/run_c4_hyperparam_tuning_multi_assets.py`](../../scripts/run_c4_hyperparam_tuning_multi_assets.py) :

```python
"""Pivot v4 C4 — Hyperparam tuning multi-actifs (nested CPCV train uniquement).

⚠️ Aucune lecture du test set ≥ 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.features_selected import FEATURES_SELECTED
from app.config.instruments import ASSET_CONFIGS
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.superset import build_superset
from app.models.nested_tuning import run_nested_cpcv_tuning
from app.strategies.donchian import DonchianBreakout

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42
SHORTLIST_C5_SHARPE = 0.5
MAX_INNER_OUTER_GAP = 1.0

RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 10],
    "min_samples_leaf": [5, 10, 20],
}
HGBM_GRID = {
    "max_depth": [3, 6, None],
    "learning_rate": [0.05, 0.1],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}
THRESHOLD_CANDIDATES = [0.50, 0.55, 0.60]


def _build_X_y_for_couple(asset: str, tf: str, donchian: dict) -> tuple[pd.DataFrame, pd.Series]:
    from app.backtest.deterministic import run_deterministic_backtest
    cfg = ASSET_CONFIGS[asset]
    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF]
    feat_train = build_superset(df_train, asset=asset)

    strat = DonchianBreakout(N=donchian["N"], M=donchian["M"])
    signals = strat.generate_signals(df_train)
    result = run_deterministic_backtest(
        df=df_train, signals=signals,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=cfg.slippage_pips,
        pip_size=cfg.pip_size,
    )
    trades = pd.DataFrame(result.get("trades", []))
    if trades.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.set_index("entry_time").sort_index()
    y = (trades["pnl_pips"] > 0).astype(int)
    selected_features = list(FEATURES_SELECTED[(asset, tf)])
    common_idx = feat_train.index.intersection(y.index)
    X = feat_train.loc[common_idx, selected_features]
    y = y.loc[common_idx]
    return X, y


def _process_couple(asset: str, tf: str, donchian: dict, model_name: str) -> dict:
    set_global_seeds(SEED)
    X, y = _build_X_y_for_couple(asset, tf, donchian)
    if len(y) < 100:
        return {"asset": asset, "tf": tf, "status": "insufficient_trades", "n_trades": int(len(y))}

    if model_name == "stacking":
        return {
            "asset": asset, "tf": tf,
            "status": "stacking_excluded_from_tuning",
            "model": "stacking",
            "params": {},
            "threshold": 0.50,
            "expected_sharpe_outer": 0.0,
            "expected_wr": 0.0,
            "n_trades_train": int(len(y)),
        }

    grid = RF_GRID if model_name == "rf" else HGBM_GRID
    result = run_nested_cpcv_tuning(
        X=X, y=y,
        model_name=model_name,
        param_grid=grid,
        threshold_candidates=THRESHOLD_CANDIDATES,
        outer_n_folds=5, inner_n_folds=3,
        embargo_pct=0.01,
        seed=SEED,
    )

    return {
        "asset": asset, "tf": tf,
        "status": "ok",
        "model": model_name,
        "params": result["best_params"],
        "threshold": result["best_threshold"],
        "expected_sharpe_outer": float(result["outer_sharpe_mean"]),
        "outer_sharpe_std": float(result["outer_sharpe_std"]),
        "sharpes_outer_per_fold": [float(s) for s in result["sharpes_outer_per_fold"]],
        "expected_wr": float(result["outer_wr_mean"]),
        "expected_n_kept": float(result.get("outer_n_kept_mean", 0)),
        "inner_outer_gap": float(result.get("inner_outer_gap", np.nan)),
        "pass_c5": (
            result["outer_sharpe_mean"] >= SHORTLIST_C5_SHARPE
            and result.get("inner_outer_gap", np.inf) < MAX_INNER_OUTER_GAP
        ),
        "n_trades_train": int(len(y)),
    }


def main() -> int:
    sel_path = _PROJECT_ROOT / "predictions" / "c3_model_selection_multi_assets.json"
    selections = json.loads(sel_path.read_text(encoding="utf-8"))

    # Récupérer les Donchian (N, M) depuis C2
    rank_path = _PROJECT_ROOT / "predictions" / "c2_ranking_multi_assets.json"
    rankings = json.loads(rank_path.read_text(encoding="utf-8"))
    donchian_by_couple = {(r["asset"], r["tf"]): r["donchian"] for r in rankings if r["status"] == "ok"}

    shortlist = [r for r in selections if r["status"] == "ok" and r.get("pass_c4_threshold", False)]
    print(f"{len(shortlist)} couples en shortlist C4 (Sharpe CPCV ≥ 0.5).")

    results: list[dict] = []
    for r in shortlist:
        asset, tf = r["asset"], r["tf"]
        model_name = r["selected_model"]
        donchian = donchian_by_couple[(asset, tf)]
        print(f"  → tuning {asset}/{tf} ({model_name}) ...")
        res = _process_couple(asset, tf, donchian, model_name)
        results.append(res)
        if res["status"] == "ok":
            print(
                f"    ✓ params={res['params']}, threshold={res['threshold']}, "
                f"Sharpe outer={res['expected_sharpe_outer']:.2f}, pass_c5={res['pass_c5']}"
            )
        elif res["status"] == "stacking_excluded_from_tuning":
            print(f"    ⚠ stacking (defaults conservés)")
        else:
            print(f"    ✗ {res['status']}")

    out_path = _PROJECT_ROOT / "predictions" / "c4_hyperparam_tuning_multi_assets.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    cfg_path = _PROJECT_ROOT / "app" / "config" / "hyperparams_tuned.py"
    _update_hyperparams_tuned(cfg_path, results)

    ok_results = [r for r in results if r["status"] in ("ok", "stacking_excluded_from_tuning")]
    final_shortlist = [r for r in results if r["status"] == "ok" and r.get("pass_c5", False)]
    print()
    print(f"Couples tunés : {len(ok_results)}")
    print(f"Shortlist finale (Sharpe outer ≥ 0.5, gap < 1.0) : {len(final_shortlist)}")
    for r in final_shortlist:
        print(f"  {r['asset']}/{r['tf']} : {r['model']} Sharpe={r['expected_sharpe_outer']:.2f}")
    return 0


def _update_hyperparams_tuned(path: Path, results: list[dict]) -> None:
    from app.config.hyperparams_tuned import HYPERPARAMS_TUNED as existing
    new_entries: dict[tuple[str, str], dict] = {}
    for r in results:
        if r["status"] in ("ok", "stacking_excluded_from_tuning"):
            new_entries[(r["asset"], r["tf"])] = {
                "model": r["model"],
                "params": r["params"],
                "threshold": r["threshold"],
                "expected_sharpe_outer": r["expected_sharpe_outer"],
                "expected_wr": r["expected_wr"],
            }
    merged = {**existing, **new_entries}

    lines = [
        '"""FROZEN après pivot v4 A8 (3 entrées) + C4 (extension multi-actifs).',
        '',
        'NE PAS MODIFIER MANUELLEMENT. Seules les phases A8 / C4 peuvent y ajouter.',
        '"""',
        "from __future__ import annotations",
        "",
        "HYPERPARAMS_TUNED: dict[tuple[str, str], dict] = {",
    ]
    for (asset, tf), entry in merged.items():
        lines.append(f"    ({asset!r}, {tf!r}): {{")
        lines.append(f"        'model': {entry['model']!r},")
        lines.append(f"        'params': {entry['params']!r},")
        lines.append(f"        'threshold': {entry['threshold']!r},")
        lines.append(f"        'expected_sharpe_outer': {entry['expected_sharpe_outer']!r},")
        lines.append(f"        'expected_wr': {entry['expected_wr']!r},")
        lines.append(f"    }},")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — Tests

Créer [`tests/unit/test_c4_hyperparams_tuned_extended.py`](../../tests/unit/test_c4_hyperparams_tuned_extended.py) :

```python
"""Vérifie l'extension de HYPERPARAMS_TUNED en C4 sans régression A8."""
from __future__ import annotations

import pytest

from app.config.hyperparams_tuned import HYPERPARAMS_TUNED

_A8_ORIGINAL = {
    ("US30", "D1"): {"model": "rf", "threshold": 0.55},
    ("EURUSD", "H4"): {"model": "rf", "threshold": 0.55},
    ("XAUUSD", "D1"): {"model": "stacking", "threshold": 0.5},
}

_REQUIRED_KEYS = {"model", "params", "threshold", "expected_sharpe_outer", "expected_wr"}
_VALID_THRESHOLDS = {0.50, 0.55, 0.60}


def test_a8_original_entries_preserved() -> None:
    for key, expected in _A8_ORIGINAL.items():
        assert key in HYPERPARAMS_TUNED, f"{key} doit rester dans HYPERPARAMS_TUNED"
        entry = HYPERPARAMS_TUNED[key]
        assert entry["model"] == expected["model"], f"{key} : modèle changé"
        assert entry["threshold"] == expected["threshold"], f"{key} : threshold changé"


@pytest.mark.parametrize("key", list(HYPERPARAMS_TUNED.keys()))
def test_entry_has_required_keys(key: tuple[str, str]) -> None:
    entry = HYPERPARAMS_TUNED[key]
    missing = _REQUIRED_KEYS - set(entry.keys())
    assert not missing, f"{key} : clés manquantes {missing}"


@pytest.mark.parametrize("key", list(HYPERPARAMS_TUNED.keys()))
def test_threshold_valid(key: tuple[str, str]) -> None:
    threshold = HYPERPARAMS_TUNED[key]["threshold"]
    assert threshold in _VALID_THRESHOLDS, f"{key} : threshold {threshold} hors {_VALID_THRESHOLDS}"
```

### Étape 4 — Documentation

Créer [`docs/hyperparam_tuning_v4_extended.md`](../../docs/hyperparam_tuning_v4_extended.md) :

```markdown
# Hyperparam tuning v4 — Extension multi-actifs (Phase C4)

**Date** : YYYY-MM-DD
**Périmètre** : K couples shortlist C3 (Sharpe CPCV ≥ 0.5)
**Train cutoff** : 2022-12-31
**Méthode** : Nested CPCV (outer 5 × inner 3), embargo 1 %

## Tableau récapitulatif

| Actif | TF | Modèle | Params retenus | Threshold | Sharpe outer | Std outer | Gap I-O | n trades | Pass C5 ? |
|---|---|---|---|---|---|---|---|---|---|
| US30 | D1 | rf | n=100, d=3, leaf=10 | 0.55 | +1.913 | 2.005 | 0.16 | 338 | A8 (original) |
| EURUSD | H4 | rf | n=100, d=6, leaf=10 | 0.55 | +0.592 | 0.713 | 0.31 | 506 | A8 (original) |
| XAUUSD | D1 | stacking | (defaults) | 0.50 | 0.000 | — | — | 85 | A8 (original) |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | C4 |

## Shortlist finale (Phase B candidats)

Couples avec Sharpe outer ≥ 0.5 ET gap inner-outer < 1.0 :

(à remplir)

## Couples écartés

(raison : Sharpe outer < 0.5, gap trop élevé, stacking → defaults)
```

### Étape 5 — Exécution (sur demande utilisateur)

```bash
# Le tuning peut prendre 30-120 min selon nombre de couples × taille grille
rtk python scripts/run_c4_hyperparam_tuning_multi_assets.py

rtk pytest tests/unit/test_c4_hyperparams_tuned_extended.py -v
rtk pytest tests/unit/test_nested_tuning.py -v  # non-régression A8
rtk make verify
```

## Tests unitaires associés

`tests/unit/test_c4_hyperparams_tuned_extended.py` : 1 + N×2 tests selon couples ajoutés.

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 C4 : Extension A8 hyperparam tuning multi-actifs

- **Statut** : ✅ Terminé
- **Type** : Infrastructure ML (0 n_trial consommé)
- **Fichiers modifiés** : `app/config/hyperparams_tuned.py` (+ N entrées)
- **Fichiers créés** : `scripts/run_c4_hyperparam_tuning_multi_assets.py`, `tests/unit/test_c4_hyperparams_tuned_extended.py`, `docs/hyperparam_tuning_v4_extended.md`, `predictions/c4_hyperparam_tuning_multi_assets.json`
- **Couples tunés (RF/HGBM)** : N
- **Couples stacking (defaults conservés)** : M
- **Shortlist finale (Sharpe outer ≥ 0.5 ET gap < 1.0)** : K couples
- **Quality gates** : ruff ✅, mypy ✅, pytest ✅
- **Prochaine étape** : C5 — Pipeline lock + bilan Phase A étendue + décision Phase B
```

## Critères go/no-go

| Critère | Cible | Action si non atteint |
|---|---|---|
| Tous les couples shortlist C3 tunés (sauf stacking) | obligatoire | Investiguer les erreurs (probable timeout sur HGBM ?) |
| 3 entrées A8 originales intactes | obligatoire | STOP : régression A8 |
| ≥ 1 couple en shortlist finale | ≥ 1 | Si 0 : Phase C produit 0 nouveau sleeve viable. Documenter et arrêter avant C5. |

**GO C5** si ≥ 1 nouveau couple shortlist finale OU si l'utilisateur valide explicitement la poursuite (C5 reste utile pour le bilan documentaire).

## Annexes

### A1 — Pourquoi nested CPCV et pas grid search simple ?

GridSearchCV classique = data leakage si on regarde le Sharpe outer pour choisir les hyperparams. Nested CPCV sépare strictement :
- **Inner loop** : choisit les hyperparams sur un sous-ensemble du train.
- **Outer loop** : évalue le Sharpe avec ces hyperparams sur le reste.

L'écart **inner – outer** est un proxy de l'overfit : grand écart = overfit, petit écart = robuste. A8 d'origine acceptait gap < 1.0.

### A2 — Pourquoi exclure stacking du tuning ?

Stacking = RF + HGBM + LR meta. Le tuner devrait explorer la grille produit cartésien des 3 → des heures de compute. A8 a décidé de garder defaults pour stacking. On respecte cette décision.

### A3 — Pourquoi les 3 thresholds candidats {0.50, 0.55, 0.60} ?

- 0.50 = neutre (proba > 50 %)
- 0.55 = légèrement conservateur (utilisé par US30/EURUSD en A8)
- 0.60 = très conservateur (rejette beaucoup de trades, qualité supérieure)

A8 a observé que le vote majoritaire convergeait vers 0.55 sur RF. On garde la même grille pour comparabilité.

### A4 — Pourquoi `min_trades_train >= 100` ici ?

Nested CPCV outer 5-fold × inner 3-fold = 15 splits. Avec 100 trades, chaque inner fold a ~7 trades de validation → c'est limite. Sous 100, la méthode devient instable. En A8, EURUSD avait 506 trades (très bien), US30 338, XAUUSD 85 (limite → stacking defaults).

## Fin du prompt C4.
**Suivant** : [C5_extend_a9_pipeline_lock_multi_assets.md](C5_extend_a9_pipeline_lock_multi_assets.md)
