# Pivot v4 — A9 : Pipeline lock (gel immutable)

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. [A5](A5_feature_generation.md), [A6](A6_feature_ranking.md), [A7](A7_model_selection.md), [A8](A8_hyperparameter_tuning.md) — **TOUS ✅ Terminés**
3. [app/config/features_selected.py](../../app/config/features_selected.py)
4. [app/config/model_selected.py](../../app/config/model_selected.py)
5. [app/config/hyperparams_tuned.py](../../app/config/hyperparams_tuned.py)
6. [app/testing/snooping_guard.py](../../app/testing/snooping_guard.py)
7. [../00_constitution.md](../00_constitution.md) — règles 9, 14

## Objectif
**Geler définitivement** tout le pipeline ML décidé en A5-A8 dans un module unique `app/config/ml_pipeline_v4.py`. Calculer un **checksum SHA256** des 3 fichiers de config (features, model, hyperparams) et l'enregistrer dans `TEST_SET_LOCK.json` comme événement `pipeline_locked`. Toute modification ultérieure de ces fichiers sera détectée par un test `test_pipeline_integrity` qui fail-fast.

> **Principe critique** : c'est la dernière étape avant B1 (lecture OOS). Une fois le pipeline gelé, l'IA ne peut plus toucher aux features/modèle/hyperparams sans déclencher un échec.

## Type d'opération
🔒 **Verrouillage** — **0 n_trial consommé**.

## Definition of Done (testable)

- [ ] `app/config/ml_pipeline_v4.py` (NOUVEAU, FROZEN) contient :
  - Import des 3 configs (features, model, hyperparams)
  - `MLPipelineConfig` frozen dataclass agrégateur
  - `get_pipeline(asset: str, tf: str) -> MLPipelineConfig` lookup
  - Constante `PIPELINE_VERSION = "v4.0.0-locked"`
- [ ] `scripts/run_a9_pipeline_lock.py` :
  - Calcule SHA256 de chaque fichier `features_selected.py`, `model_selected.py`, `hyperparams_tuned.py`
  - Écrit ces checksums dans `TEST_SET_LOCK.json` sous clé `pipeline_locked`
  - Vérifie qu'aucun champ obligatoire ne manque
- [ ] `tests/integration/test_pipeline_integrity.py` (NOUVEAU) :
  - Vérifie que les SHA256 actuels correspondent aux SHA256 dans `TEST_SET_LOCK.json`
  - Doit être appelé par `make verify` pour bloquer toute modification post-A9
- [ ] `app/models/build.py` (NOUVEAU) : `build_locked_model(asset, tf, seed=42) -> sklearn_estimator` :
  - Utilise `HYPERPARAMS_TUNED` pour instancier le modèle exact
  - Retourne un modèle prêt à `.fit()` avec exactly les bons hyperparams
- [ ] `docs/pipeline_v4_locked.md` : récapitulatif complet de tout ce qui est figé
- [ ] `Makefile` enrichi : `make verify` doit appeler `pytest tests/integration/test_pipeline_integrity.py`
- [ ] `rtk make verify` → 0 erreur
- [ ] `JOURNAL.md` mis à jour avec la note "Phase A complète. Pipeline gelé pour B1."

## NE PAS FAIRE

- ❌ Ne PAS modifier `features_selected.py`, `model_selected.py`, `hyperparams_tuned.py` après ce prompt.
- ❌ Ne PAS toucher au test set ≥ 2024 (toujours interdit).
- ❌ Ne PAS supprimer ou modifier `TEST_SET_LOCK.json` manuellement.
- ❌ Ne PAS court-circuiter `test_pipeline_integrity` (= cheat anti-snooping).
- ❌ Ne PAS incrémenter `n_trials`.

## Étapes détaillées

### Étape 1 — `app/config/ml_pipeline_v4.py`

```python
"""Pipeline ML v4 — FROZEN après A9. Tout pipeline = lookup ici.

CE FICHIER NE DOIT PAS ÊTRE MODIFIÉ APRÈS A9.
Toute modification = data snooping → invalide la statistique Phase B.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config.features_selected import FEATURES_SELECTED
from app.config.model_selected import MODEL_SELECTED
from app.config.hyperparams_tuned import HYPERPARAMS_TUNED

PIPELINE_VERSION = "v4.0.0-locked"


@dataclass(frozen=True)
class MLPipelineConfig:
    """Configuration complète du pipeline ML pour un (asset, tf) donné."""
    asset: str
    tf: str
    features: tuple[str, ...]
    model_name: str
    model_params: dict
    threshold: float
    expected_sharpe_outer: float
    expected_wr: float
    version: str = PIPELINE_VERSION

    def __post_init__(self):
        if not 0.50 <= self.threshold <= 0.80:
            raise ValueError(f"Seuil hors plage [0.50, 0.80]: {self.threshold}")
        if not self.features:
            raise ValueError("Aucune feature sélectionnée")
        if self.model_name not in ("rf", "hgbm", "stacking"):
            raise ValueError(f"Modèle inconnu : {self.model_name}")


def get_pipeline(asset: str, tf: str) -> MLPipelineConfig:
    """Récupère le pipeline gelé pour (asset, tf). Raise KeyError si non configuré."""
    key = (asset, tf)
    if key not in FEATURES_SELECTED:
        raise KeyError(f"Pas de features sélectionnées pour {asset} {tf}")
    if key not in MODEL_SELECTED:
        raise KeyError(f"Pas de modèle sélectionné pour {asset} {tf}")
    if key not in HYPERPARAMS_TUNED:
        raise KeyError(f"Pas d'hyperparams tunés pour {asset} {tf}")

    h = HYPERPARAMS_TUNED[key]
    return MLPipelineConfig(
        asset=asset,
        tf=tf,
        features=FEATURES_SELECTED[key],
        model_name=h["model"],
        model_params=h["params"],
        threshold=h["threshold"],
        expected_sharpe_outer=h.get("expected_sharpe_outer", 0.0),
        expected_wr=h.get("expected_wr", 0.0),
    )


def all_configured_pairs() -> list[tuple[str, str]]:
    """Retourne tous les (asset, tf) avec pipeline complet."""
    return [
        k for k in FEATURES_SELECTED
        if k in MODEL_SELECTED and k in HYPERPARAMS_TUNED
    ]
```

### Étape 2 — `app/models/build.py`

```python
"""Constructeur de modèle à partir du pipeline gelé."""
from __future__ import annotations

from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from app.config.ml_pipeline_v4 import get_pipeline, MLPipelineConfig


def build_model(cfg: MLPipelineConfig, seed: int = 42):
    """Construit un modèle sklearn-like à partir d'un MLPipelineConfig gelé."""
    if cfg.model_name == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.model_params.get("n_estimators", 200),
            max_depth=cfg.model_params.get("max_depth", 4),
            min_samples_leaf=cfg.model_params.get("min_samples_leaf", 10),
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    if cfg.model_name == "hgbm":
        return HistGradientBoostingClassifier(
            max_iter=cfg.model_params.get("max_iter", 200),
            max_depth=cfg.model_params.get("max_depth", 5),
            learning_rate=cfg.model_params.get("learning_rate", 0.05),
            l2_regularization=1.0,
            class_weight="balanced", random_state=seed, early_stopping=False,
        )
    if cfg.model_name == "stacking":
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=10,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        hgbm = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=seed, early_stopping=False,
        )
        meta = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
        stacking = StackingClassifier(
            estimators=[("rf", rf), ("hgbm", hgbm)],
            final_estimator=meta, cv=5, stack_method="predict_proba", n_jobs=-1,
        )
        return CalibratedClassifierCV(stacking, method="isotonic", cv=3)
    raise ValueError(f"Modèle inconnu : {cfg.model_name}")


def build_locked_model(asset: str, tf: str, seed: int = 42):
    """Shortcut : récupère le pipeline gelé et construit le modèle."""
    cfg = get_pipeline(asset, tf)
    return build_model(cfg, seed)
```

### Étape 3 — `scripts/run_a9_pipeline_lock.py`

```python
"""Pivot v4 A9 — Pipeline lock + checksum dans TEST_SET_LOCK.json."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.config.ml_pipeline_v4 import all_configured_pairs, get_pipeline, PIPELINE_VERSION
from app.testing.snooping_guard import LOCK_PATH


CONFIG_FILES = [
    Path("app/config/features_selected.py"),
    Path("app/config/model_selected.py"),
    Path("app/config/hyperparams_tuned.py"),
    Path("app/config/ml_pipeline_v4.py"),
]


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    # Vérifier que tous les configs existent et sont cohérents
    pairs = all_configured_pairs()
    if not pairs:
        print("ERREUR : aucun (asset, tf) configuré. Re-exécuter A6/A7/A8.")
        return 1

    print(f"Pipelines configurés : {pairs}")
    for asset, tf in pairs:
        cfg = get_pipeline(asset, tf)
        print(f"  {asset} {tf}: {cfg.model_name} | "
              f"{len(cfg.features)} features | "
              f"threshold={cfg.threshold} | "
              f"expected_sharpe_outer={cfg.expected_sharpe_outer:.3f}")

    # Calculer SHA256 de chaque fichier config
    checksums = {}
    for p in CONFIG_FILES:
        if not p.exists():
            print(f"ERREUR : fichier config manquant : {p}")
            return 1
        checksums[str(p)] = _sha256_of_file(p)
        print(f"  SHA256 {p} : {checksums[str(p)][:16]}...")

    # Écrire dans TEST_SET_LOCK.json
    if LOCK_PATH.exists():
        state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    else:
        state = {"locked": False, "n_reads": 0, "read_history": []}

    state["pipeline_locked"] = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "configured_pairs": [{"asset": a, "tf": t} for (a, t) in pairs],
        "config_checksums": checksums,
    }
    LOCK_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Pipeline gelé. {len(pairs)} configurations enregistrées dans {LOCK_PATH}.")
    print("⚠️ Toute modification de config = échec test_pipeline_integrity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 4 — `tests/integration/test_pipeline_integrity.py`

```python
"""Test d'intégrité du pipeline gelé (pivot v4 A9).

Si ce test échoue : un fichier config a été modifié après A9 = data snooping potentiel.
Pour résoudre : soit revert les modifications, soit re-faire un nouveau pivot complet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.testing.snooping_guard import LOCK_PATH


CONFIG_FILES = [
    Path("app/config/features_selected.py"),
    Path("app/config/model_selected.py"),
    Path("app/config/hyperparams_tuned.py"),
    Path("app/config/ml_pipeline_v4.py"),
]


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_locked_section_present():
    """TEST_SET_LOCK.json doit contenir une section pipeline_locked après A9."""
    assert LOCK_PATH.exists(), "TEST_SET_LOCK.json absent — A9 pas exécuté"
    state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert "pipeline_locked" in state, "Pas de section pipeline_locked — A9 pas exécuté"


@pytest.mark.parametrize("config_path", CONFIG_FILES)
def test_config_checksum_unchanged(config_path):
    """Chaque config doit avoir le même SHA256 qu'au moment du lock."""
    state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if "pipeline_locked" not in state:
        pytest.skip("Pipeline pas encore locked (A9 non exécuté)")

    expected = state["pipeline_locked"]["config_checksums"].get(str(config_path))
    if expected is None:
        pytest.skip(f"{config_path} pas dans les checksums lockés")
    actual = _sha256_of_file(config_path)
    assert actual == expected, (
        f"⚠️ {config_path} a été MODIFIÉ depuis le pipeline lock.\n"
        f"Expected SHA256 : {expected}\n"
        f"Actual SHA256   : {actual}\n"
        f"= data snooping potentiel. Revert le fichier ou refaire un nouveau pivot."
    )


def test_pipeline_version_locked():
    state = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if "pipeline_locked" not in state:
        pytest.skip("Pipeline pas encore locked")
    from app.config.ml_pipeline_v4 import PIPELINE_VERSION
    assert state["pipeline_locked"]["pipeline_version"] == PIPELINE_VERSION
```

### Étape 5 — Mise à jour `Makefile`

Ajouter `pipeline_check` dans la cible `verify` :

```makefile
.PHONY: install test lint typecheck verify backtest snooping_check pipeline_check

# ... existant ...

pipeline_check:
	rtk pytest tests/integration/test_pipeline_integrity.py -v

verify: lint typecheck test snooping_check pipeline_check
	@echo "✅ All quality gates passed (including pipeline integrity)."
```

### Étape 6 — Documentation `docs/pipeline_v4_locked.md`

```markdown
# Pipeline ML v4 — Locked configuration

**Date du lock** : YYYY-MM-DD
**Version** : v4.0.0-locked
**Pipeline checksums** : voir `TEST_SET_LOCK.json` → `pipeline_locked`

## Vue d'ensemble

Tout le pipeline ML est figé à partir de cette date. Toute modification des fichiers
`features_selected.py`, `model_selected.py`, `hyperparams_tuned.py`, `ml_pipeline_v4.py`
sera détectée par `tests/integration/test_pipeline_integrity.py` qui échoue automatiquement.

## Configurations par actif

### US30 D1
- **Features (15)** : voir `FEATURES_SELECTED[("US30", "D1")]`
- **Modèle** : `MODEL_SELECTED[("US30", "D1")]`
- **Hyperparams** : `HYPERPARAMS_TUNED[("US30", "D1")]["params"]`
- **Threshold** : `HYPERPARAMS_TUNED[("US30", "D1")]["threshold"]`
- **Expected Sharpe outer** : ?

### EURUSD H4
- (idem)

### XAUUSD D1
- (idem)

## Comment utiliser

Dans B1, B2, B3 :

```python
from app.config.ml_pipeline_v4 import get_pipeline
from app.models.build import build_locked_model

cfg = get_pipeline("US30", "D1")
print(f"Features: {cfg.features}")
print(f"Threshold: {cfg.threshold}")

model = build_locked_model("US30", "D1", seed=42)
# model est prêt à fit
```

## Comment vérifier l'intégrité

```bash
rtk pytest tests/integration/test_pipeline_integrity.py -v
```

ou

```bash
rtk make verify  # appelle pipeline_check automatiquement
```

## Que faire si le test échoue

1. **Si modification accidentelle** : `git diff` les fichiers config, revert.
2. **Si modification intentionnelle** : il faut faire un nouveau pivot complet (V5).
   La modification actuelle invalide statistiquement Phase B.
3. **Ne JAMAIS** simplement re-faire `run_a9_pipeline_lock.py` avec les nouveaux checksums
   sans avoir fait un nouveau cycle A5-A8 sur de nouvelles données.

## Limites du gel

Le gel protège contre la modification accidentelle des configs. Il ne protège pas contre :
- Modification du code des indicateurs dans `app/features/indicators.py` (changerait les valeurs)
- Modification du code du simulateur (changerait les PnL train)

Pour ces cas, la vigilance manuelle reste nécessaire. Ils sont audités à chaque PR/commit.
```

### Étape 7 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_a9_pipeline_lock.py
rtk make verify
```

Sorties attendues :
```
Pipelines configurés : [('US30', 'D1'), ('EURUSD', 'H4'), ('XAUUSD', 'D1')]
  US30 D1: hgbm | 15 features | threshold=0.55 | expected_sharpe_outer=1.520
  EURUSD H4: rf | 15 features | threshold=0.55 | expected_sharpe_outer=0.890
  XAUUSD D1: stacking | 15 features | threshold=0.50 | expected_sharpe_outer=0.660
  SHA256 app/config/features_selected.py : a1b2c3d4...
  ...
✅ Pipeline gelé. 3 configurations enregistrées dans TEST_SET_LOCK.json.
⚠️ Toute modification de config = échec test_pipeline_integrity.

✅ All quality gates passed (including pipeline integrity).
```

## Tests unitaires associés

`tests/integration/test_pipeline_integrity.py` avec :
- 1 test "section pipeline_locked présente"
- 4 tests parametrisés (1 par fichier config)
- 1 test "version cohérente"

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A9 : Pipeline lock + checksums

- **Statut** : ✅ Terminé — Phase A complète (A1-A9)
- **Type** : Verrouillage (0 n_trial)
- **Fichiers créés** : `app/config/ml_pipeline_v4.py`, `app/models/build.py`, `scripts/run_a9_pipeline_lock.py`, `tests/integration/test_pipeline_integrity.py`, `docs/pipeline_v4_locked.md`
- **Fichiers modifiés** : `Makefile` (ajout pipeline_check), `TEST_SET_LOCK.json` (ajout section pipeline_locked)
- **Pipeline version** : v4.0.0-locked
- **Configurations gelées** :
  - US30 D1 : modèle ?, threshold ?, expected sharpe outer ?
  - EURUSD H4 : modèle ?, threshold ?, expected sharpe outer ?
  - XAUUSD D1 : modèle ?, threshold ?, expected sharpe outer ?
- **Checksums enregistrés** : 4 fichiers (features, model, hyperparams, ml_pipeline_v4)
- **Tests** : 6 tests pipeline_integrity passing
- **make verify** : ✅ passé (incluant pipeline_check)
- **Notes** : Phase A complète. Aucune lecture du test set ≥ 2024.
- **Prochaine étape** : B1 — H_new1 méta-labeling US30 D1 (vraie hypothèse OOS, n_trial=23).
```

## Critères go/no-go

- **GO Phase B1** si :
  - Pipeline lock exécuté avec succès
  - `TEST_SET_LOCK.json` contient la section `pipeline_locked`
  - `make verify` passe avec pipeline_check ✅
  - Au moins 1 (asset, tf) avec expected_sharpe_outer ≥ 0.5
- **NO-GO, revenir à** :
  - Si tous expected_sharpe_outer < 0.3 → le pipeline ne tient pas la promesse, refaire A6-A8 avec contraintes ajustées (ou abandonner)
  - Si test_pipeline_integrity échoue immédiatement → bug script A9, debug

## Annexes

### A1 — Pourquoi un checksum SHA256

- Détection **bit-exact** des modifications. Un seul caractère changé → SHA256 différent.
- Léger (256 bits = 64 chars hex).
- Standard, vérifiable manuellement avec `sha256sum app/config/*.py`.

### A2 — Pourquoi 4 fichiers checksumés (pas 3)

- `features_selected.py` : top features par actif
- `model_selected.py` : type de modèle par actif
- `hyperparams_tuned.py` : hyperparams + seuil par actif
- `ml_pipeline_v4.py` : agrégateur qui peut introduire des bugs s'il est modifié

Les 4 ensemble = "ML pipeline" complet. Modifier l'un = invalider Phase B.

### A3 — Et si on veut ajouter un nouvel actif après A9 ?

Cas légitime : utilisateur fournit nouveau CSV USD/JPY après A9. Comment l'intégrer ?

**Option 1 (clean)** : Lancer un nouveau pivot V5 complet (A1-A9 sur USD/JPY). Coûteux mais propre.

**Option 2 (pragmatique)** :
1. Préserver les configurations US30/EURUSD/XAUUSD existantes (ne pas les modifier).
2. Lancer A5-A8 sur USD/JPY uniquement (les autres restent figés).
3. Ajouter USD/JPY dans `features_selected.py`, `model_selected.py`, `hyperparams_tuned.py`.
4. **Modifie les SHA256** → `test_pipeline_integrity` va FAIL.
5. Re-run `scripts/run_a9_pipeline_lock.py` avec note explicite "ajout USD/JPY post-A9, configurations existantes inchangées".

Cette option ouvre une porte au snooping. À utiliser avec rigueur.

### A4 — Différence avec le snooping guard existant

`snooping_guard.py` empêche la **lecture multiple du test set OOS**.
`test_pipeline_integrity.py` empêche la **modification du pipeline après lock**.

Les deux sont complémentaires. Le snooping guard sait quels tests sont consommés. Le pipeline integrity sait que la config n'a pas changé entre A9 et B1.

### A5 — Pourquoi `make verify` inclut pipeline_check

Si Deepseek modifie un fichier config en faisant B1 ("oh, ce hyperparam serait mieux"), `make verify` échoue immédiatement, bloquant le commit. C'est le **dernier filet** anti-snooping mécanique.

### A6 — Récap final Phase A

| Phase | Action | n_trial | Sorti dans |
|---|---|---|---|
| A1 | Audit simulateur + sizing 2% | 0 | code + test |
| A2 | Coûts XTB réels | 0 | `instruments.py` + `cost_audit_v2.md` |
| A3 | Sharpe routing | 0 | `metrics.py` |
| A4 | Replay H06/H07 train+val (audit) | 0 | `_replay.md` |
| A5 | Superset 70 features | 0 | `superset.py` |
| A6 | Top 15 features par actif | 0 | `features_selected.py` |
| A7 | Modèle retenu par actif | 0 | `model_selected.py` |
| A8 | Hyperparams + seuil tunés | 0 | `hyperparams_tuned.py` |
| A9 | Pipeline lock + checksums | 0 | `ml_pipeline_v4.py` + `TEST_SET_LOCK.json` |
| **Total Phase A** | | **0 n_trials** | Pipeline complet figé |

À ce stade : `n_trials_cumul = 22`. Phase B va consommer 1 trial par hypothèse OOS.

## Fin du prompt A9.
**Suivant (ordre révisé)** : [A2_calibration_costs.md](A2_calibration_costs.md) (finition du simulateur AVANT B1)

> ⚠️ Le pipeline ML est désormais **gelé** (features + modèle + hyperparams + seuil). Il faut maintenant calibrer le simulateur (A2 coûts + A3 Sharpe routing) avant de lancer le premier test OOS B1. Voir [00_README.md](00_README.md) section "Ordre d'exécution strict — RÉVISÉ".
>
> Ordre restant : A9 ✅ → **A2 → A3** → [A4 optionnel] → B1 → B2 → [B3] → [B4]
