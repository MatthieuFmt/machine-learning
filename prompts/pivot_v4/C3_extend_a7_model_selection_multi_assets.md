# Pivot v4 — C3 : Extension A7 model selection multi-actifs

> 📍 **Troisième étape de la Phase C — Extension multi-actifs**
> Sélection du modèle ML (RF vs HistGBM vs Stacking) sur la shortlist issue de C2.
> **Aucune lecture du test set ≥ 2024**. 0 n_trial consommé.

## Préalable obligatoire (à lire dans l'ordre)

1. [00_README.md](00_README.md) — vue d'ensemble pivot v4
2. [../00_constitution.md](../00_constitution.md) — règles 6, 7, 9
3. [C1_extend_a5_multi_assets.md](C1_extend_a5_multi_assets.md) — ✅ Terminé
4. [C2_extend_a6_ranking_multi_assets.md](C2_extend_a6_ranking_multi_assets.md) — **✅ Terminé obligatoire**
5. [A7_model_selection.md](A7_model_selection.md) — A7 d'origine (référence méthodo)
6. [`docs/model_selection_v4.md`](../../docs/model_selection_v4.md) — résultats A7 d'origine
7. [`predictions/c2_ranking_multi_assets.json`](../../predictions/c2_ranking_multi_assets.json) — shortlist
8. [`app/models/candidates.py`](../../app/models/candidates.py) — 3 modèles candidats
9. [`app/models/cpcv_evaluation.py`](../../app/models/cpcv_evaluation.py) — CPCV utilisée
10. [`scripts/run_a7_model_selection.py`](../../scripts/run_a7_model_selection.py) — script d'origine
11. [`app/config/model_selected.py`](../../app/config/model_selected.py) — 3 entrées actuelles

## Objectif

Pour chaque couple en shortlist C2 (stability moyenne top 15 ≥ 0.5), évaluer 3 candidats ML via CPCV 5-fold × embargo 1 % :
- `rf` : RandomForestClassifier
- `hgbm` : HistGradientBoostingClassifier
- `stacking` : RF + HGBM + LogisticRegression meta

Sélectionner le meilleur sur **Sharpe CPCV moyen** (avec stability inter-fold comme tie-breaker) et figer dans [`app/config/model_selected.py`](../../app/config/model_selected.py).

## Type d'opération

🔧 **Infrastructure ML — 0 n_trial consommé. Aucune lecture du test set 2024+.**

## Definition of Done (testable)

- [ ] `scripts/run_c3_model_selection_multi_assets.py` (NOUVEAU) lit `c2_ranking_multi_assets.json`, filtre les couples avec `stability_mean ≥ 0.5`, et boucle sur chacun pour évaluer les 3 candidats en CPCV.
- [ ] Chaque couple en shortlist se voit attribuer **un modèle figé** dans `app/config/model_selected.py` (sans toucher aux 3 entrées d'origine).
- [ ] `predictions/c3_model_selection_multi_assets.json` contient pour chaque couple : modèle retenu, Sharpe moyen + std par modèle, stability inter-fold, n_kept moyen, WR méta moyen.
- [ ] `docs/model_selection_v4_extended.md` (NOUVEAU) avec tableau récapitulatif + analyse par classe d'actif.
- [ ] `tests/unit/test_c3_model_selected_extended.py` vérifie : 3 entrées d'origine intactes, modèle retenu ∈ {`rf`, `hgbm`, `stacking`}.
- [ ] **Shortlist filtrée pour C4** : couples avec Sharpe CPCV moyen ≥ 0.5 (relâché vs A7 qui exigeait stability < 1.0 strict).
- [ ] `rtk make verify` → 0 erreur.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE

- ❌ **Ne PAS modifier les 3 entrées existantes** de `MODEL_SELECTED` (US30 D1, EURUSD H4, XAUUSD D1).
- ❌ **Ne PAS lire le test set ≥ 2024.** Cutoff strict `2022-12-31`.
- ❌ **Ne PAS modifier `app/models/candidates.py` ni `app/models/cpcv_evaluation.py`** (gelés en A7).
- ❌ **Ne PAS ajouter un 4ème candidat** (XGBoost, LightGBM, NN) — A7 a déjà tranché : 3 candidats suffisent.
- ❌ **Ne PAS tuner les hyperparams** — c'est l'objet de C4. Utiliser les defaults A7.
- ❌ **Ne PAS sélectionner manuellement** un modèle par actif : la décision vient des Sharpe CPCV.
- ❌ **Ne PAS incrémenter `n_trials`.**

## Étapes détaillées

### Étape 1 — Créer le script

Créer [`scripts/run_c3_model_selection_multi_assets.py`](../../scripts/run_c3_model_selection_multi_assets.py) :

```python
"""Pivot v4 C3 — Sélection de modèle multi-actifs (CPCV train uniquement).

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
from app.models.candidates import build_candidate_model
from app.models.cpcv_evaluation import evaluate_with_cpcv
from app.strategies.donchian import DonchianBreakout

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42
CANDIDATES = ["rf", "hgbm", "stacking"]
THRESHOLD = 0.50  # seuil méta par défaut A7
SHORTLIST_THRESHOLD = 0.5  # stab moyenne C2 minimale
SHORTLIST_C4_SHARPE = 0.5  # Sharpe CPCV moyen minimum pour passer en C4


def _build_X_y_for_couple(asset: str, tf: str, donchian: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Reconstruit X, y comme en C2 mais en filtrant par FEATURES_SELECTED."""
    from app.backtest.deterministic import run_deterministic_backtest

    cfg = ASSET_CONFIGS[asset]
    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF]
    feat_train = build_superset(df_train, asset=asset)

    strat = DonchianBreakout(N=donchian["N"], M=donchian["M"])
    signals = strat.generate_signals(df_train)
    result = run_deterministic_backtest(
        df=df_train,
        signals=signals,
        tp_pips=cfg.tp_points,
        sl_pips=cfg.sl_points,
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

    # Filtrer aux 15 features figées en C2
    selected_features = list(FEATURES_SELECTED[(asset, tf)])
    common_idx = feat_train.index.intersection(y.index)
    X = feat_train.loc[common_idx, selected_features]
    y = y.loc[common_idx]
    return X, y


def _process_couple(asset: str, tf: str, donchian: dict) -> dict:
    set_global_seeds(SEED)
    X, y = _build_X_y_for_couple(asset, tf, donchian)
    if len(y) < 50:
        return {"asset": asset, "tf": tf, "status": "insufficient_trades", "n_trades": int(len(y))}

    per_candidate: dict[str, dict] = {}
    for cand_name in CANDIDATES:
        model = build_candidate_model(cand_name, seed=SEED)
        cpcv_result = evaluate_with_cpcv(
            X=X, y=y, model=model,
            n_folds=5, embargo_pct=0.01,
            threshold=THRESHOLD,
            seed=SEED,
        )
        per_candidate[cand_name] = {
            "sharpe_mean": float(np.mean(cpcv_result["sharpes_per_fold"])),
            "sharpe_std": float(np.std(cpcv_result["sharpes_per_fold"])),
            "sharpes_per_fold": [float(s) for s in cpcv_result["sharpes_per_fold"]],
            "stability_inter_fold": float(cpcv_result.get("stability", np.nan)),
            "n_kept_mean": float(np.mean(cpcv_result.get("n_kept_per_fold", [0]))),
            "wr_meta_mean": float(np.mean(cpcv_result.get("wr_per_fold", [0]))),
        }

    # Sélection : argmax Sharpe moyen
    best = max(per_candidate.items(), key=lambda kv: kv[1]["sharpe_mean"])
    best_name = best[0]
    best_metrics = best[1]

    return {
        "asset": asset, "tf": tf,
        "status": "ok",
        "n_trades_train": int(len(y)),
        "wr_train": float(y.mean()),
        "candidates": per_candidate,
        "selected_model": best_name,
        "selected_sharpe_mean": best_metrics["sharpe_mean"],
        "selected_stability": best_metrics["stability_inter_fold"],
        "pass_c4_threshold": best_metrics["sharpe_mean"] >= SHORTLIST_C4_SHARPE,
    }


def main() -> int:
    rank_path = _PROJECT_ROOT / "predictions" / "c2_ranking_multi_assets.json"
    rankings = json.loads(rank_path.read_text(encoding="utf-8"))
    shortlist = [r for r in rankings if r["status"] == "ok" and r["stability_mean"] >= SHORTLIST_THRESHOLD]
    print(f"{len(shortlist)} couples en shortlist C2 (stab ≥ {SHORTLIST_THRESHOLD}).")

    results: list[dict] = []
    for r in shortlist:
        asset, tf = r["asset"], r["tf"]
        donchian = r["donchian"]
        print(f"  → model selection {asset}/{tf} ...")
        res = _process_couple(asset, tf, donchian)
        results.append(res)
        if res["status"] == "ok":
            print(f"    ✓ {res['selected_model']} Sharpe={res['selected_sharpe_mean']:.2f}, pass_c4={res['pass_c4_threshold']}")
        else:
            print(f"    ✗ {res['status']}")

    # Sauvegarde JSON
    out_path = _PROJECT_ROOT / "predictions" / "c3_model_selection_multi_assets.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Mise à jour model_selected.py
    cfg_path = _PROJECT_ROOT / "app" / "config" / "model_selected.py"
    _update_model_selected(cfg_path, results)

    # Bilan
    ok_results = [r for r in results if r["status"] == "ok"]
    c4_shortlist = [r for r in ok_results if r["pass_c4_threshold"]]
    print()
    print(f"Couples évalués OK : {len(ok_results)}")
    print(f"Shortlist C4 (Sharpe CPCV ≥ {SHORTLIST_C4_SHARPE}) : {len(c4_shortlist)}")
    for r in c4_shortlist:
        print(f"  {r['asset']}/{r['tf']} : {r['selected_model']} Sharpe={r['selected_sharpe_mean']:.2f}")
    return 0


def _update_model_selected(path: Path, results: list[dict]) -> None:
    """Ajoute les nouvelles entrées tout en préservant les 3 originales."""
    from app.config.model_selected import MODEL_SELECTED as existing
    new_entries: dict[tuple[str, str], str] = {}
    for r in results:
        if r["status"] == "ok":
            new_entries[(r["asset"], r["tf"])] = r["selected_model"]
    merged = {**existing, **new_entries}

    lines = [
        '"""FROZEN après pivot v4 A7 (3 entrées) + C3 (extension multi-actifs).',
        '',
        'NE PAS MODIFIER MANUELLEMENT. Seules les phases A7 / C3 peuvent y ajouter.',
        '"""',
        "from __future__ import annotations",
        "",
        "MODEL_SELECTED: dict[tuple[str, str], str] = {",
    ]
    for (asset, tf), model_name in merged.items():
        lines.append(f"    ({asset!r}, {tf!r}): {model_name!r},")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 2 — Tests

Créer [`tests/unit/test_c3_model_selected_extended.py`](../../tests/unit/test_c3_model_selected_extended.py) :

```python
"""Vérifie l'extension de MODEL_SELECTED en C3 sans régression A7."""
from __future__ import annotations

import pytest

from app.config.model_selected import MODEL_SELECTED

_VALID_MODELS = {"rf", "hgbm", "stacking"}
_A7_ORIGINAL = {
    ("US30", "D1"): "rf",
    ("EURUSD", "H4"): "rf",
    ("XAUUSD", "D1"): "stacking",
}


def test_a7_original_entries_preserved() -> None:
    for key, expected in _A7_ORIGINAL.items():
        assert key in MODEL_SELECTED, f"{key} doit rester dans MODEL_SELECTED"
        assert MODEL_SELECTED[key] == expected, f"{key} : modèle changé ! attendu {expected}, vu {MODEL_SELECTED[key]}"


@pytest.mark.parametrize("key", list(MODEL_SELECTED.keys()))
def test_model_in_valid_set(key: tuple[str, str]) -> None:
    assert MODEL_SELECTED[key] in _VALID_MODELS, (
        f"{key} : modèle {MODEL_SELECTED[key]} non reconnu (attendu : {_VALID_MODELS})"
    )
```

### Étape 3 — Documentation

Créer [`docs/model_selection_v4_extended.md`](../../docs/model_selection_v4_extended.md) avec un tableau du type :

```markdown
# Model selection v4 — Extension multi-actifs (Phase C3)

**Date** : YYYY-MM-DD
**Périmètre** : K couples shortlist C2 (stab ≥ 0.5)
**Train cutoff** : 2022-12-31
**Méthode** : CPCV 5-fold × embargo 1%, seuil méta 0.50

## Tableau récapitulatif

| Actif | TF | n trades | WR train | RF Sharpe | HGBM Sharpe | Stack Sharpe | Modèle retenu | Pass C4 ? |
|---|---|---|---|---|---|---|---|---|
| US30 | D1 | 338 | 46.7% | +1.75 | ? | ? | rf | A7 (original) |
| EURUSD | H4 | 506 | 38.7% | +0.90 | ? | ? | rf | A7 (original) |
| XAUUSD | D1 | 85 | 11.8% | ? | ? | -1.05 | stacking | A7 (original) |
| ... | ... | ... | ... | ... | ... | ... | ... | C3 |

## Patterns par classe d'actif

(à remplir après exécution)

## Shortlist C4 (Sharpe CPCV moyen ≥ 0.5)

(liste des couples qui passent en C4 — hyperparam tuning)

## Couples exclus

(raison : insufficient_trades, Sharpe < 0.5)
```

### Étape 4 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_c3_model_selection_multi_assets.py
rtk pytest tests/unit/test_c3_model_selected_extended.py -v
rtk pytest tests/unit/test_model_selection.py -v  # non-régression A7
rtk make verify
```

## Tests unitaires associés

`tests/unit/test_c3_model_selected_extended.py` : 1 + N tests selon couples ajoutés.

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 C3 : Extension A7 model selection multi-actifs

- **Statut** : ✅ Terminé
- **Type** : Infrastructure ML (0 n_trial consommé)
- **Fichiers modifiés** : `app/config/model_selected.py` (+ N entrées)
- **Fichiers créés** : `scripts/run_c3_model_selection_multi_assets.py`, `tests/unit/test_c3_model_selected_extended.py`, `docs/model_selection_v4_extended.md`, `predictions/c3_model_selection_multi_assets.json`
- **Couples évalués** : N (shortlist C2)
- **Modèles retenus** : X RF, Y HGBM, Z Stacking
- **Shortlist C4 (Sharpe CPCV ≥ 0.5)** : K couples
- **Quality gates** : ruff ✅, mypy ✅, pytest ✅
- **Prochaine étape** : C4 — Hyperparam tuning sur les K couples shortlist
```

## Critères go/no-go

| Critère | Cible | Action si non atteint |
|---|---|---|
| Tous les couples shortlist C2 évalués sur 3 candidats | obligatoire | Investiguer les erreurs |
| 3 entrées A7 originales intactes | obligatoire | STOP : régression A7 |
| ≥ 2 couples en shortlist C4 (Sharpe CPCV ≥ 0.5) | ≥ 2 | Si 0 ou 1 : la Phase C ne produira pas d'edge ML sur ces actifs. Documenter et discuter avec utilisateur avant C4. |

**GO C4** si ≥ 2 couples passent le filtre Sharpe ≥ 0.5.

## Annexes

### A1 — Pourquoi le seuil C4 à 0.5 et pas 1.0 ?

A7 d'origine n'imposait pas de Sharpe minimal pour la sélection — il sélectionnait l'argmax. Le seuil 0.5 ici n'est pas un critère de "GO production" mais un **filtre de pertinence** : sous 0.5, le tuning hyperparams (C4) a très peu de chance de monter à 1.0. On évite de gaspiller du compute sur des cas perdus d'avance.

### A2 — Pourquoi CPCV 5-fold et pas 10 ?

Identique à A7. 5-fold sur 200-500 trades donne ~50-100 trades par fold de test. 10-fold descendrait à 25-50, trop peu pour un Sharpe stable. C'est un compromis empirique.

### A3 — Pourquoi `stability_inter_fold < 1.0` n'est plus un critère hard ?

A7 d'origine avait ce critère mais les 3 couples l'ont raté (US30 1.16, EURUSD 1.23, XAUUSD 2.00). En pratique, l'utilisateur a procédé à A8 malgré tout. On garde l'info dans les outputs mais on ne bloque plus dessus.

### A4 — Pourquoi le seuil méta reste 0.50 et pas 0.55 ?

Le seuil 0.55 est figé en A8 (après tuning). En C3, on est encore en phase de sélection de modèle ; on garde le défaut 0.50 pour permettre une comparaison équitable entre RF/HGBM/Stacking, sachant que le seuil sera tuné en C4.

## Fin du prompt C3.
**Suivant** : [C4_extend_a8_hyperparams_multi_assets.md](C4_extend_a8_hyperparams_multi_assets.md)
