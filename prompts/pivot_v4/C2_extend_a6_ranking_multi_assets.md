# Pivot v4 — C2 : Extension A6 ranking multi-actifs

> 📍 **Deuxième étape de la Phase C — Extension multi-actifs**
> Ranking des features (top 15 par couple) sur tous les nouveaux couples identifiés en C1.
> **Aucune lecture du test set ≥ 2024**. 0 n_trial consommé.

## Préalable obligatoire (à lire dans l'ordre)

1. [00_README.md](00_README.md) — vue d'ensemble pivot v4
2. [../00_constitution.md](../00_constitution.md) — règles 6, 7, 9
3. [C1_extend_a5_multi_assets.md](C1_extend_a5_multi_assets.md) — **C1 ✅ Terminé obligatoire**
4. [A6_feature_ranking.md](A6_feature_ranking.md) — A6 d'origine (référence méthodo)
5. [`predictions/c1_couples_inventory.json`](../../predictions/c1_couples_inventory.json) — liste des couples à traiter
6. [`app/features/ranking.py`](../../app/features/ranking.py) — `rank_features_bootstrap()` existant
7. [`scripts/run_a6_feature_ranking.py`](../../scripts/run_a6_feature_ranking.py) — script d'origine (à étendre)
8. [`app/config/features_selected.py`](../../app/config/features_selected.py) — 3 entrées actuelles
9. [`docs/feature_ranking_v4.md`](../../docs/feature_ranking_v4.md) — résultats A6 d'origine

## Objectif

Étendre [`app/config/features_selected.py`](../../app/config/features_selected.py) avec le top 15 figé pour chacun des nouveaux couples identifiés en C1 (`status="new_phase_c"`), via la même méthodo bootstrap stability que A6.

Produire un **bilan de stabilité par couple** pour identifier les meilleurs candidats en vue de C3.

## Type d'opération

🔧 **Infrastructure ML — 0 n_trial consommé. Aucune lecture du test set 2024+.**

## Definition of Done (testable)

- [ ] `scripts/run_c2_ranking_multi_assets.py` (NOUVEAU) lit `c1_couples_inventory.json`, filtre les couples `status="new_phase_c"`, et boucle sur chacun pour produire le top 15.
- [ ] Pour chaque nouveau couple : top 15 ajouté à `app/config/features_selected.py` (sans toucher aux 3 entrées existantes).
- [ ] `predictions/c2_ranking_multi_assets.json` contient pour chaque couple : top 15, stability moyenne, stability #1, n trades train, WR train.
- [ ] `docs/feature_ranking_v4_extended.md` (NOUVEAU) avec un **tableau récapitulatif** des 18 couples + analyse qualitative (patterns dominants par classe d'actif).
- [ ] `app/config/features_selected.py` contient désormais **jusqu'à 21 entrées** (3 originales + jusqu'à 18 nouvelles).
- [ ] `tests/unit/test_c2_features_selected_extended.py` vérifie : 3 entrées d'origine intactes, toutes les nouvelles entrées ont exactement 15 features uniques, présentes dans le superset.
- [ ] **Shortlist explicite pour C3** : couples avec stability moyenne top 15 **≥ 0.5** documentés dans le bilan markdown sous une section "Shortlist C3".
- [ ] `rtk make verify` → 0 erreur.
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE

- ❌ **Ne PAS modifier les 3 entrées existantes** de `FEATURES_SELECTED` (US30 D1, EURUSD H4, XAUUSD D1).
- ❌ **Ne PAS lire le test set ≥ 2024.** Cutoff strict `2022-12-31 23:59:59`.
- ❌ **Ne PAS modifier `app/features/ranking.py`** (méthodo figée en A6).
- ❌ **Ne PAS ré-tester des couples déjà figés** (`existing_pipeline`).
- ❌ **Ne PAS choisir manuellement les features** — laisser le bootstrap décider, sinon = data snooping.
- ❌ **Ne PAS exclure de couple sur la base de la stability** : tous les couples nouveaux sont ajoutés à `features_selected.py`. L'exclusion / shortlist se fait dans le bilan markdown (information pour C3).
- ❌ **Ne PAS incrémenter `n_trials`.**

## Étapes détaillées

### Étape 1 — Choisir un Donchian (N, M) initial par couple

La cible binaire « winner Donchian » exige des paramètres `(N, M)` initiaux pour générer suffisamment de trades sur train. Heuristique :

| TF | (N, M) par défaut | Justification |
|---|---|---|
| D1 | (20, 20) | Standard utilisé en A6 pour US30 D1 ; donne ~200-300 trades sur 8 ans train |
| H4 | (20, 20) | Standard A6 pour EURUSD H4 ; ~500-700 trades sur 8 ans |
| H1 | (50, 20) | TF plus court → on doit allonger N pour éviter du bruit excessif |

> Si un couple produit < 50 trades train avec ces params, le script doit **descendre automatiquement à `(N=10, M=10)`** une fois, puis abandonner le couple avec le statut `insufficient_trades` dans le JSON.

### Étape 2 — Créer le script

Créer [`scripts/run_c2_ranking_multi_assets.py`](../../scripts/run_c2_ranking_multi_assets.py) :

```python
"""Pivot v4 C2 — Ranking robuste multi-actifs train uniquement.

⚠️ Aucune lecture du test set ≥ 2024.
Hard filter: toutes les données postérieures à 2022-12-31 sont EXCLUES.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.instruments import ASSET_CONFIGS, AssetConfig
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.ranking import rank_features_bootstrap
from app.features.superset import build_superset
from app.strategies.donchian import DonchianBreakout

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")
SEED = 42

DEFAULT_DONCHIAN: dict[str, tuple[int, int]] = {
    "D1": (20, 20),
    "H4": (20, 20),
    "H1": (50, 20),
}

FALLBACK_DONCHIAN = (10, 10)  # si trop peu de trades avec le défaut

# Top-N à figer par couple
TOP_K = 15
MIN_TRADES_TRAIN = 50  # seuil minimum pour ranking fiable


def _backtest_target(
    df_train: pd.DataFrame,
    strat: DonchianBreakout,
    cfg: AssetConfig,
) -> pd.DataFrame:
    """Génère les trades (target = winner) sur le train."""
    from app.backtest.deterministic import run_deterministic_backtest

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
    trades_list = result.get("trades", [])
    if not trades_list:
        return pd.DataFrame()
    trades = pd.DataFrame(trades_list)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.set_index("entry_time").sort_index()
    return trades


def _build_target_X_y(
    df_train: pd.DataFrame,
    trades: pd.DataFrame,
    feat_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aligne features et label binaire 'winner' (pnl > 0)."""
    if trades.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    y = (trades["pnl_pips"] > 0).astype(int)
    y.index = trades.index
    # Aligner sur l'index features (entry_time)
    common_idx = feat_train.index.intersection(y.index)
    X = feat_train.loc[common_idx]
    y = y.loc[common_idx]
    return X, y


def _process_couple(asset: str, tf: str, inventory_entry: dict) -> dict:
    """Ranking pour un couple (asset, tf)."""
    set_global_seeds(SEED)
    cfg = ASSET_CONFIGS.get(asset)
    if cfg is None:
        return {"asset": asset, "tf": tf, "status": "no_asset_config"}

    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF]
    if len(df_train) < 300:
        return {"asset": asset, "tf": tf, "status": "train_too_short", "n_bars_train": len(df_train)}

    feat_train = build_superset(df_train, asset=asset)

    # Tentative 1 : Donchian par défaut
    N, M = DEFAULT_DONCHIAN[tf]
    strat = DonchianBreakout(N=N, M=M)
    trades = _backtest_target(df_train, strat, cfg)
    X, y = _build_target_X_y(df_train, trades, feat_train)

    fallback_used = False
    if len(y) < MIN_TRADES_TRAIN:
        N, M = FALLBACK_DONCHIAN
        strat = DonchianBreakout(N=N, M=M)
        trades = _backtest_target(df_train, strat, cfg)
        X, y = _build_target_X_y(df_train, trades, feat_train)
        fallback_used = True

    if len(y) < MIN_TRADES_TRAIN:
        return {
            "asset": asset, "tf": tf,
            "status": "insufficient_trades",
            "n_trades": int(len(y)),
            "donchian": {"N": N, "M": M, "fallback_used": fallback_used},
        }

    wr_train = float(y.mean())

    # Bootstrap stability
    ranking = rank_features_bootstrap(
        X=X, y=y,
        n_bootstrap=5,
        seed=SEED,
        top_k=TOP_K,
    )
    top_features = list(ranking["top_features"])
    stability_mean = float(np.mean([ranking["stability"].get(f, 0.0) for f in top_features]))
    stability_top1 = float(ranking["stability"].get(top_features[0], 0.0))

    return {
        "asset": asset, "tf": tf,
        "status": "ok",
        "n_trades_train": int(len(y)),
        "wr_train": wr_train,
        "donchian": {"N": N, "M": M, "fallback_used": fallback_used},
        "top_features": top_features,
        "stability_mean": stability_mean,
        "stability_top1": stability_top1,
        "stability_per_feature": {f: float(ranking["stability"].get(f, 0.0)) for f in top_features},
    }


def main() -> int:
    inv_path = _PROJECT_ROOT / "predictions" / "c1_couples_inventory.json"
    inventory = json.loads(inv_path.read_text(encoding="utf-8"))
    new_couples = [e for e in inventory if e["status"] == "new_phase_c"]
    print(f"{len(new_couples)} couples nouveaux à traiter.")

    results: list[dict] = []
    for entry in new_couples:
        asset, tf = entry["asset"], entry["tf"]
        print(f"  → ranking {asset}/{tf} ...")
        res = _process_couple(asset, tf, entry)
        results.append(res)
        if res["status"] == "ok":
            print(f"    ✓ {len(res['top_features'])} features, stability moy={res['stability_mean']:.2f}, n_trades={res['n_trades_train']}")
        else:
            print(f"    ✗ {res['status']}")

    # Sauvegarde JSON
    out_path = _PROJECT_ROOT / "predictions" / "c2_ranking_multi_assets.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Mise à jour features_selected.py
    cfg_path = _PROJECT_ROOT / "app" / "config" / "features_selected.py"
    _update_features_selected(cfg_path, results)

    # Bilan
    ok = [r for r in results if r["status"] == "ok"]
    shortlist = [r for r in ok if r["stability_mean"] >= 0.5]
    print()
    print(f"Couples ranked OK : {len(ok)} / {len(new_couples)}")
    print(f"Shortlist (stab moyenne ≥ 0.5) pour C3 : {len(shortlist)}")
    for r in shortlist:
        print(f"  {r['asset']}/{r['tf']} : stab={r['stability_mean']:.2f}, n_trades={r['n_trades_train']}")
    return 0


def _update_features_selected(path: Path, results: list[dict]) -> None:
    """Ajoute les nouvelles entrées tout en préservant les 3 originales."""
    from app.config.features_selected import FEATURES_SELECTED as existing

    new_entries: dict[tuple[str, str], tuple[str, ...]] = {}
    for r in results:
        if r["status"] == "ok":
            new_entries[(r["asset"], r["tf"])] = tuple(r["top_features"])

    merged = {**existing, **new_entries}

    lines = [
        '"""FROZEN après pivot v4 A6 (3 entrées) + C2 (extension multi-actifs).',
        '',
        'NE PAS MODIFIER MANUELLEMENT. Seules les phases A6 / C2 peuvent y ajouter.',
        '"""',
        "from __future__ import annotations",
        "",
        "FEATURES_SELECTED: dict[tuple[str, str], tuple[str, ...]] = {",
    ]
    for (asset, tf), feats in merged.items():
        feat_repr = "(" + ", ".join(f"'{f}'" for f in feats) + ")"
        lines.append(f"    ({asset!r}, {tf!r}): {feat_repr},")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — Tests

Créer [`tests/unit/test_c2_features_selected_extended.py`](../../tests/unit/test_c2_features_selected_extended.py) :

```python
"""Vérifie l'extension de FEATURES_SELECTED en C2 sans régression A6."""
from __future__ import annotations

import pytest

from app.config.features_selected import FEATURES_SELECTED
from app.features.superset import build_superset
from app.data.loader import load_asset
import pandas as pd

CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# Entrées d'origine de A6 — doivent rester intactes
_A6_ORIGINAL = {
    ("US30", "D1"),
    ("EURUSD", "H4"),
    ("XAUUSD", "D1"),
}


def test_a6_original_entries_preserved() -> None:
    for key in _A6_ORIGINAL:
        assert key in FEATURES_SELECTED, f"{key} (A6) doit rester dans FEATURES_SELECTED"
        assert len(FEATURES_SELECTED[key]) == 15


@pytest.mark.parametrize("key", list(FEATURES_SELECTED.keys()))
def test_each_entry_has_15_unique_features(key: tuple[str, str]) -> None:
    feats = FEATURES_SELECTED[key]
    assert len(feats) == 15, f"{key} : {len(feats)} features ≠ 15"
    assert len(set(feats)) == 15, f"{key} : doublons détectés"


@pytest.mark.parametrize("key", list(FEATURES_SELECTED.keys()))
def test_features_exist_in_superset(key: tuple[str, str]) -> None:
    asset, tf = key
    try:
        df = load_asset(asset, tf)
    except Exception:
        pytest.skip(f"{asset}/{tf} : données indisponibles")
    df_train = df.loc[:CUTOFF]
    if len(df_train) < 300:
        pytest.skip(f"{asset}/{tf} : train trop court")
    feat = build_superset(df_train, asset=asset)
    available = set(feat.columns)
    missing = [f for f in FEATURES_SELECTED[key] if f not in available]
    assert not missing, f"{asset}/{tf} : features absentes du superset : {missing}"
```

### Étape 4 — Documentation

Créer [`docs/feature_ranking_v4_extended.md`](../../docs/feature_ranking_v4_extended.md) :

```markdown
# Feature ranking v4 — Extension multi-actifs (Phase C2)

**Date** : YYYY-MM-DD
**Périmètre** : 18 couples nouveaux + 3 couples d'origine A6
**Train cutoff** : 2022-12-31

## Tableau récapitulatif

| Actif | TF | Donchian (N,M) | Trades train | WR | Stab top1 | Stab moy top 15 | Statut |
|---|---|---|---|---|---|---|---|
| US30 | D1 | (20, 20) | 232 | 48.3% | 1.00 | 0.72 | A6 (original) |
| EURUSD | H4 | (20, 20) | 506 | 38.7% | 0.80 | 0.59 | A6 (original) |
| XAUUSD | D1 | (100, 20) | 85 | 11.8% | 0.80 | 0.56 | A6 (original) |
| ... | ... | ... | ... | ... | ... | ... | C2 |

## Patterns dominants par classe d'actif

(à remplir après exécution)

### Indices (US30, GER30, US500)
### Forex majeures (EURUSD, GBPUSD, USDCHF)
### Métaux (XAUUSD, XAGUSD)
### Crypto (BTCUSD, ETHUSD)

## Shortlist C3 (stability moyenne ≥ 0.5)

(à remplir : liste des couples qui passent en C3)

## Couples exclus (raison)

- couple X : insufficient_trades (Y trades sur seuil 50)
- couple Z : train_too_short
- ...
```

### Étape 5 — Exécution (sur demande utilisateur)

```bash
# 1. Lancer le ranking
rtk python scripts/run_c2_ranking_multi_assets.py

# 2. Tests
rtk pytest tests/unit/test_c2_features_selected_extended.py -v

# 3. Vérifier que A6 original n'a pas régressé
rtk pytest tests/unit/test_feature_ranking.py -v

# 4. Quality gates
rtk make verify
```

## Tests unitaires associés

- `tests/unit/test_c2_features_selected_extended.py` : 1 + N×2 = ~21+ tests selon le nombre de couples ajoutés.

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 C2 : Extension A6 ranking multi-actifs

- **Statut** : ✅ Terminé
- **Type** : Infrastructure ML (0 n_trial consommé)
- **Fichiers modifiés** : `app/config/features_selected.py` (+ N entrées)
- **Fichiers créés** : `scripts/run_c2_ranking_multi_assets.py`, `tests/unit/test_c2_features_selected_extended.py`, `docs/feature_ranking_v4_extended.md`, `predictions/c2_ranking_multi_assets.json`
- **Couples ranked OK** : N/18
- **Couples exclus** : M (raisons : insufficient_trades, train_too_short, no_asset_config)
- **Shortlist C3 (stab ≥ 0.5)** : K couples → liste
- **Quality gates** : ruff ✅, mypy ✅, pytest XX/XX ✅
- **Prochaine étape** : C3 — Model selection sur les K couples shortlist
```

## Critères go/no-go

| Critère | Cible | Action si non atteint |
|---|---|---|
| ≥ 10 couples ranked OK (sur 18) | ≥ 10 | Investiguer les exclusions massives (probable problème de cutoff ou Donchian inadapté) |
| `features_selected.py` contient ≥ 13 entrées (3 + ≥ 10) | obligatoire | Vérifier l'écriture du fichier |
| 3 entrées originales intactes | obligatoire | STOP : régression A6 inacceptable |
| ≥ 3 couples en shortlist C3 (stab ≥ 0.5) | ≥ 3 | Si < 3, soit on baisse le seuil à 0.4 (avec accord utilisateur), soit on accepte que la Phase C produit peu de candidats |

**GO C3** si tous les critères passent.

## Annexes

### A1 — Pourquoi le seuil shortlist à 0.5 et pas 0.6 (comme A6) ?

A6 original avait fixé 0.6 mais sur 3 couples très différents (US30 obtient 0.72, EURUSD 0.59, XAUUSD 0.56). En pratique, A6 a accepté EURUSD et XAUUSD malgré stability < 0.6. Pour la Phase C, le seuil 0.5 reflète la pratique réelle. Si trop peu de couples passent, on rediscutera.

### A2 — Pourquoi ne pas re-ranker les 3 couples existants ?

Ils ont déjà été validés en A6 et figés dans `features_selected.py`. Les re-ranker = data snooping (on lit des résultats déjà connus). On préserve l'invariance pour la cohérence avec A9 et B1-B4.

### A3 — Pourquoi `min_trades_train=50` au lieu de 100 ?

Le bootstrap stability sur 5 itérations × 3 métriques sur 50 échantillons est limite mais reste informatif. En dessous, la stability devient bruitée. C'est la même heuristique qu'A6 (qui a accepté XAUUSD avec 85 trades).

### A4 — Les crypto (BTCUSD, ETHUSD) auront des `pip_value_eur` provisoires

Pour C2, ça n'a pas d'importance : la target est binaire (winner=1 si `pnl_pips > 0`), insensible à la valeur monétaire EUR. C3 et C4 idem (métriques de classification). Seul C5 + Phase B (si pratiqué) seront sensibles au pip_value_eur réel.

## Fin du prompt C2.
**Suivant** : [C3_extend_a7_model_selection_multi_assets.md](C3_extend_a7_model_selection_multi_assets.md)
