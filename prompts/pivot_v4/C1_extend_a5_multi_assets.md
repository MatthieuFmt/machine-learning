# Pivot v4 — C1 : Extension A5 multi-actifs (ASSET_CONFIGS + superset)

> 📍 **Première étape de la Phase C — Extension multi-actifs**
> Cette phase étend la Phase A (A5→A9) aux 18 couples (actif, TF) jamais testés en pipeline ML.
> **Aucune lecture du test set ≥ 2024**. 0 n_trial consommé sur toute la Phase C.

## Préalable obligatoire (à lire dans l'ordre)

1. [00_README.md](00_README.md) — vue d'ensemble pivot v4, section "Ordre d'exécution strict"
2. [../00_constitution.md](../00_constitution.md) — règles 6, 7, 9 (test set sanctity, dossiers interdits)
3. [A5_feature_generation.md](A5_feature_generation.md) — A5 d'origine sur US30/EURUSD/XAUUSD (référence)
4. [`JOURNAL.md`](../../JOURNAL.md) — vérifier que A5 (US30/EURUSD/XAUUSD) est ✅ Terminé
5. [`app/config/instruments.py`](../../app/config/instruments.py) — `ASSET_CONFIGS` actuel (7 actifs)
6. [`app/features/superset.py`](../../app/features/superset.py) — `build_superset()` existant
7. [`docs/cost_audit_v2.md`](../../docs/cost_audit_v2.md) — convention pip_size, justification coûts

## Contexte de la Phase C

| Aspect | Valeur |
|---|---|
| Périmètre | 18 couples (actif, TF) jamais testés en pipeline ML |
| Test set ≥ 2024 | **JAMAIS LU** — Phase C entière sur train ≤ 2022 |
| n_trials consommés | **0** sur les 5 prompts C1-C5 |
| Décision Phase B | Repoussée après C5 (bilan), sleeve par sleeve |

### Les 18 couples à couvrir

| Actif | D1 | H4 | H1 |
|---|---|---|---|
| BTCUSD | ⬜ nouveau | ⬜ nouveau | ⬜ nouveau |
| ETHUSD | ⬜ nouveau | ⬜ nouveau | ⬜ nouveau |
| EURUSD | ⬜ nouveau | ✅ A5-A9 | ⬜ nouveau |
| GBPUSD | ⬜ nouveau | ⬜ nouveau | ⬜ nouveau |
| US30 | ✅ A5-A9 | ⬜ nouveau | ⬜ nouveau |
| USDCHF | ⬜ nouveau | ⬜ nouveau | ⬜ nouveau |
| XAUUSD | ✅ A5-A9 | ⬜ nouveau | ⬜ nouveau |

> **4 actifs sont absents de `ASSET_CONFIGS`** : BTCUSD, ETHUSD, GBPUSD, USDCHF.
> Ce prompt C1 les ajoute avec des **coûts provisoires basés sur la doc publique XTB** (à valider en démo ultérieurement — c'est l'étape suivant la Phase C).

## Objectif

Préparer l'infrastructure de la Phase C :

1. Ajouter `BTCUSD`, `ETHUSD`, `GBPUSD`, `USDCHF` à `ASSET_CONFIGS` avec coûts XTB provisoires
2. Vérifier que `build_superset()` produit ≥ 60 features sur les 18 couples (smoke test)
3. Documenter l'inventaire des 21 couples (statut, dates, nombre de barres train)

## Type d'opération

🔧 **Infrastructure ML — 0 n_trial consommé. Aucune lecture du test set 2024+.**

## Definition of Done (testable)

- [ ] `app/config/instruments.py` contient les entrées `ASSET_CONFIGS["BTCUSD"]`, `ASSET_CONFIGS["ETHUSD"]`, `ASSET_CONFIGS["GBPUSD"]`, `ASSET_CONFIGS["USDCHF"]` avec coûts XTB provisoires + commentaire `# PROVISOIRE — à valider en démo`.
- [ ] `scripts/run_c1_inventory.py` produit `predictions/c1_couples_inventory.json` listant les 21 couples avec :
  - `available` (bool), `first_date`, `last_date`, `n_bars_total`, `n_bars_train` (≤ 2022-12-31), `n_features_superset`, `status` (`existing_pipeline` / `new_phase_c` / `data_missing`).
- [ ] Le script imprime un tableau lisible et un résumé : "X/18 nouveaux couples prêts pour C2".
- [ ] `tests/unit/test_c1_asset_configs_extended.py` (NOUVEAU) — 4 tests vérifiant que les 4 nouveaux `AssetConfig` sont bien construits (spread > 0, pip_size > 0, total_cost_pips cohérent).
- [ ] `tests/unit/test_c1_superset_multi_assets.py` (NOUVEAU) — paramétré sur 18 couples : `build_superset(df_train, asset=name)` retourne ≥ 60 colonnes pour chaque couple **où des données existent**.
- [ ] `rtk make verify` → 0 erreur sur le périmètre C1.
- [ ] `JOURNAL.md` mis à jour avec la section "Pivot v4 C1".

## NE PAS FAIRE

- ❌ **Ne PAS lire `data/`, `ready-data/`, `cleaned-data/`** directement. Uniquement via `app/data/loader.py`.
- ❌ **Ne PAS toucher au test set ≥ 2024.** Toutes les vérifications se font sur train ≤ 2022.
- ❌ **Ne PAS modifier les 7 entrées existantes** de `ASSET_CONFIGS` (US30, US500, GER30, XAUUSD, XAGUSD, USOIL, EURUSD).
- ❌ **Ne PAS modifier `app/features/superset.py`** — la fonction `build_superset()` est figée depuis A5.
- ❌ **Ne PAS modifier `app/config/features_selected.py`** — c'est l'objet du prompt C2.
- ❌ **Ne PAS incrémenter `n_trials`** — Phase C entière à 0 trial.
- ❌ **Ne PAS lancer A6/A7/A8/A9 dans ce prompt** — C1 est strictement préparatoire.
- ❌ **Ne PAS deviner les coûts** : utiliser uniquement les valeurs publiques documentées XTB.

## Étapes détaillées

### Étape 1 — Ajouter les 4 nouveaux `AssetConfig`

Dans [`app/config/instruments.py`](../../app/config/instruments.py), ajouter ces entrées dans `ASSET_CONFIGS` (après `EURUSD`, juste avant les commentaires `BUND`) :

```python
    # ── BTCUSD (Bitcoin spot) — NOUVEAU C1, PROVISOIRE ───────────────────
    # Source : XTB.com → Crypto → BITCOIN — spread variable selon marché
    # ⚠️ PROVISOIRE — à valider en démo MT5 (Symbol Specifications)
    "BTCUSD": AssetConfig(
        spread_pips=30.0,      # ≈ 30 USD spread typique heures actives
        slippage_pips=30.0,    # crypto : 1.0× spread (forte volatilité)
        commission_pips=0.0,
        pip_size=1.0,          # 1 pip BTC = 1 USD (big figure)
        pip_value_eur=0.92,    # 1 USD ≈ 0.92 EUR
        tp_points=2000,        # 2000 USD soit ~3-5% du prix BTC typique
        sl_points=1000,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
    ),
    # ── ETHUSD (Ethereum spot) — NOUVEAU C1, PROVISOIRE ──────────────────
    # ⚠️ PROVISOIRE — à valider en démo
    "ETHUSD": AssetConfig(
        spread_pips=3.0,       # ≈ 3 USD spread typique
        slippage_pips=3.0,     # crypto : 1.0× spread
        commission_pips=0.0,
        pip_size=0.01,         # 1 pip ETH = 0.01 USD (cotation au centime)
        pip_value_eur=0.92,
        tp_points=10000,       # 100 USD soit ~3-5% du prix ETH typique
        sl_points=5000,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
    ),
    # ── GBPUSD (Forex) — NOUVEAU C1, PROVISOIRE ──────────────────────────
    # ⚠️ PROVISOIRE — à valider en démo
    "GBPUSD": AssetConfig(
        spread_pips=0.9,       # ≈ 0.9 pip XTB Standard
        slippage_pips=0.2,     # majeure : 0.2× spread
        commission_pips=0.0,
        pip_size=0.0001,       # 1 pip forex = 4ème décimale
        pip_value_eur=9.2,     # 1 pip × 1 lot standard ≈ 10 USD ≈ 9.2 EUR
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
    ),
    # ── USDCHF (Forex) — NOUVEAU C1, PROVISOIRE ──────────────────────────
    # ⚠️ PROVISOIRE — à valider en démo
    "USDCHF": AssetConfig(
        spread_pips=1.0,
        slippage_pips=0.2,
        commission_pips=0.0,
        pip_size=0.0001,
        pip_value_eur=10.5,    # CHF base, valeur EUR variable selon taux
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
    ),
```

> **Note** : les valeurs `pip_value_eur` sont approximatives. L'utilisateur les corrigera lors de l'étape "vérification spreads démo" (post-Phase C).

### Étape 2 — Créer le script d'inventaire

Créer [`scripts/run_c1_inventory.py`](../../scripts/run_c1_inventory.py) :

```python
"""Pivot v4 C1 — Inventaire des 21 couples (actif, TF) du projet.

⚠️ Train ≤ 2022-12-31 uniquement. Test set 2024+ JAMAIS lu.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.features_selected import FEATURES_SELECTED
from app.data.loader import load_asset
from app.data.registry import discover_assets
from app.features.superset import build_superset

CUTOFF_TRAIN = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")


def main() -> int:
    available = discover_assets()  # {asset: [tf, ...]}
    target_assets = ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "US30", "USDCHF", "XAUUSD"]
    target_tfs = ["D1", "H4", "H1"]

    inventory: list[dict] = []
    for asset in target_assets:
        for tf in target_tfs:
            entry: dict = {"asset": asset, "tf": tf}
            if asset not in available or tf not in available.get(asset, []):
                entry.update({
                    "available": False,
                    "status": "data_missing",
                    "n_bars_total": 0,
                    "n_bars_train": 0,
                    "n_features_superset": 0,
                })
                inventory.append(entry)
                continue

            try:
                df = load_asset(asset, tf)
                df_train = df.loc[:CUTOFF_TRAIN]
                feat = build_superset(df_train, asset=asset)
                entry.update({
                    "available": True,
                    "first_date": str(df.index[0]),
                    "last_date": str(df.index[-1]),
                    "n_bars_total": int(len(df)),
                    "n_bars_train": int(len(df_train)),
                    "n_features_superset": int(feat.shape[1]),
                    "status": (
                        "existing_pipeline"
                        if (asset, tf) in FEATURES_SELECTED
                        else "new_phase_c"
                    ),
                })
            except Exception as exc:
                entry.update({
                    "available": False,
                    "status": "load_error",
                    "error": str(exc)[:200],
                })
            inventory.append(entry)

    out_path = _PROJECT_ROOT / "predictions" / "c1_couples_inventory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    # Console output
    print(f"{'Asset':<8} {'TF':<4} {'Avail':<6} {'NBars':<8} {'NTrain':<8} {'NFeat':<6} {'Status':<20}")
    print("-" * 70)
    for e in inventory:
        print(
            f"{e['asset']:<8} {e['tf']:<4} {str(e['available']):<6} "
            f"{e.get('n_bars_total', 0):<8} {e.get('n_bars_train', 0):<8} "
            f"{e.get('n_features_superset', 0):<6} {e['status']:<20}"
        )

    n_new = sum(1 for e in inventory if e["status"] == "new_phase_c")
    n_existing = sum(1 for e in inventory if e["status"] == "existing_pipeline")
    n_missing = sum(1 for e in inventory if e["status"] in ("data_missing", "load_error"))
    print()
    print(f"Résumé : {n_existing} déjà dans pipeline, {n_new} nouveaux pour Phase C, {n_missing} indisponibles.")
    print(f"→ {n_new} couples à traiter en C2 (ranking).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — Tests unitaires AssetConfig

Créer [`tests/unit/test_c1_asset_configs_extended.py`](../../tests/unit/test_c1_asset_configs_extended.py) :

```python
"""Tests des 4 nouveaux AssetConfig (BTCUSD, ETHUSD, GBPUSD, USDCHF)."""
from __future__ import annotations

import pytest

from app.config.instruments import ASSET_CONFIGS


@pytest.mark.parametrize("name", ["BTCUSD", "ETHUSD", "GBPUSD", "USDCHF"])
def test_asset_config_present(name: str) -> None:
    assert name in ASSET_CONFIGS, f"{name} absent de ASSET_CONFIGS"


@pytest.mark.parametrize("name", ["BTCUSD", "ETHUSD", "GBPUSD", "USDCHF"])
def test_asset_config_valid(name: str) -> None:
    cfg = ASSET_CONFIGS[name]
    assert cfg.spread_pips > 0
    assert cfg.slippage_pips >= 0
    assert cfg.pip_size > 0
    assert cfg.pip_value_eur > 0
    assert cfg.tp_points > 0
    assert cfg.sl_points > 0
    assert cfg.total_cost_pips > 0


@pytest.mark.parametrize("name,expected_pip_size", [
    ("BTCUSD", 1.0),
    ("ETHUSD", 0.01),
    ("GBPUSD", 0.0001),
    ("USDCHF", 0.0001),
])
def test_asset_config_pip_size(name: str, expected_pip_size: float) -> None:
    assert ASSET_CONFIGS[name].pip_size == expected_pip_size


def test_no_existing_asset_modified() -> None:
    """Garde-fou : les 7 entrées d'origine ne doivent pas être modifiées par C1."""
    for name in ["US30", "US500", "GER30", "XAUUSD", "XAGUSD", "USOIL", "EURUSD"]:
        assert name in ASSET_CONFIGS, f"{name} (existant) ne doit pas être supprimé"
```

### Étape 4 — Test paramétré du superset sur tous les couples

Créer [`tests/unit/test_c1_superset_multi_assets.py`](../../tests/unit/test_c1_superset_multi_assets.py) :

```python
"""Smoke test : build_superset() retourne ≥ 60 features sur tous les couples disponibles."""
from __future__ import annotations

import pandas as pd
import pytest

from app.data.loader import load_asset
from app.data.registry import discover_assets
from app.features.superset import build_superset

CUTOFF_TRAIN = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")

# Paramètres : 21 couples cibles (7 actifs × 3 TF)
_AVAILABLE = discover_assets()
_COUPLES = [
    (asset, tf)
    for asset in ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD", "US30", "USDCHF", "XAUUSD"]
    for tf in ["D1", "H4", "H1"]
    if asset in _AVAILABLE and tf in _AVAILABLE.get(asset, [])
]


@pytest.mark.parametrize("asset,tf", _COUPLES)
def test_superset_min_60_features(asset: str, tf: str) -> None:
    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF_TRAIN]
    if len(df_train) < 250:
        pytest.skip(f"{asset}/{tf} : train trop court ({len(df_train)} barres)")
    feat = build_superset(df_train, asset=asset)
    assert feat.shape[1] >= 60, f"{asset}/{tf} : {feat.shape[1]} features < 60"


@pytest.mark.parametrize("asset,tf", _COUPLES)
def test_superset_no_nan_after_warmup(asset: str, tf: str) -> None:
    df = load_asset(asset, tf)
    df_train = df.loc[:CUTOFF_TRAIN]
    if len(df_train) < 300:
        pytest.skip(f"{asset}/{tf} : train trop court")
    feat = build_superset(df_train, asset=asset)
    after_warmup = feat.iloc[250:]
    nan_cols = after_warmup.columns[after_warmup.isna().any()].tolist()
    allowed_prefixes = ("usdchf_", "xauusd_", "btcusd_")
    forbidden = [c for c in nan_cols if not c.startswith(allowed_prefixes)]
    assert not forbidden, f"{asset}/{tf} : NaN après warmup sur {forbidden}"
```

### Étape 5 — Vérification manuelle (sur demande utilisateur)

Avant d'exécuter, l'utilisateur valide par :

```bash
# 1. Lancer l'inventaire
rtk python scripts/run_c1_inventory.py

# 2. Tests
rtk pytest tests/unit/test_c1_asset_configs_extended.py tests/unit/test_c1_superset_multi_assets.py -v

# 3. Vérifier que rien n'a cassé existant
rtk pytest tests/unit/test_superset_features.py -v

# 4. Quality gates
rtk make verify
```

## Tests unitaires associés

- `tests/unit/test_c1_asset_configs_extended.py` : 4 + 4 + 4 + 1 = **13 tests**
- `tests/unit/test_c1_superset_multi_assets.py` : 2 × N_couples (≈ 36 tests si tous disponibles)

## Logging obligatoire

Ajouter dans `JOURNAL.md` à la fin :

```markdown
## YYYY-MM-DD — Pivot v4 C1 : Extension A5 multi-actifs

- **Statut** : ✅ Terminé
- **Type** : Infrastructure ML (0 n_trial consommé)
- **Fichiers modifiés** : `app/config/instruments.py` (+ 4 entrées AssetConfig provisoires)
- **Fichiers créés** : `scripts/run_c1_inventory.py`, `tests/unit/test_c1_asset_configs_extended.py`, `tests/unit/test_c1_superset_multi_assets.py`, `predictions/c1_couples_inventory.json`
- **Couples disponibles** : X/21 (dont 3 déjà testés, Y nouveaux à traiter)
- **Couples indisponibles** : Z (données absentes ou erreur load)
- **Features superset moyennes par couple** : ~XX
- **Quality gates** : ruff ✅, mypy ✅, pytest XX/XX ✅
- **⚠️ Coûts XTB BTCUSD/ETHUSD/GBPUSD/USDCHF** : PROVISOIRES, à valider en démo après Phase C
- **Prochaine étape** : C2 — Feature ranking sur les Y nouveaux couples
```

## Critères go/no-go

| Critère | Cible | Action si non atteint |
|---|---|---|
| 4 nouveaux `AssetConfig` ajoutés et tests OK | obligatoire | Corriger pip_size / pip_value_eur |
| `build_superset()` retourne ≥ 60 features sur ≥ 12 couples sur 18 nouveaux | obligatoire | Investiguer le couple défaillant (probable problème de Volume manquant ou de longueur train) |
| Tous les tests existants A5 toujours verts | obligatoire | STOP : régression non acceptable |
| Inventaire JSON généré et tableau imprimé | obligatoire | Corriger le script |

**GO C2** si tous les critères passent. **NO-GO C2** sinon : investiguer le blocage avant d'aller plus loin.

## Annexes

### A1 — Pourquoi des coûts provisoires ?

L'utilisateur a explicitement demandé : « on reparlera des spreads juste après cette tâche ». Les coûts XTB des 4 nouveaux actifs ne sont pas confirmés en démo. On utilise des valeurs publiques plausibles, **marquées `PROVISOIRE`**, pour ne pas bloquer la Phase C. À la fin de C5, l'utilisateur fera une vérification démo XTB et corrigera dans `ASSET_CONFIGS`.

Cela n'invalide pas la Phase C : A5-A9 utilisent les coûts uniquement pour générer la **cible** (winner Donchian) qui devient le label binaire. Une légère erreur sur les coûts décale le label de quelques trades à la marge, mais ne change pas l'ordre de stabilité des features. Une vraie validation OOS exigera des coûts confirmés.

### A2 — Pourquoi inclure les TF déjà testés (US30 D1, EURUSD H4, XAUUSD D1) dans l'inventaire ?

Pour vérification de non-régression. Le script doit montrer que ces 3 couples sortent avec `status="existing_pipeline"` et que leur `n_features_superset` est identique à celui figé en A5 (~71). Toute différence = régression silencieuse à investiguer.

### A3 — Pourquoi cross-asset peut être NaN sur certains couples ?

`cross_asset_features()` dans `app/features/superset.py` ajoute `usdchf_return_5`, `xauusd_return_5`, `btcusd_return_5` reindexés depuis le TF D1. Sur un actif H1, le merge ffill peut produire des NaN sur la fin si les dates ne s'alignent pas. Les tests skippent ces colonnes du critère "no NaN after warmup".

### A4 — Pourquoi pas d'incrément `n_trials` ?

La Phase C entière (C1-C5) **ne lit jamais le test set ≥ 2024**. C'est de l'ingénierie de pipeline pure sur train. Aucun verdict GO/NO-GO d'hypothèse OOS, donc 0 n_trial selon la convention de la constitution (section 6bis).

## Fin du prompt C1.
**Suivant** : [C2_extend_a6_ranking_multi_assets.md](C2_extend_a6_ranking_multi_assets.md)
