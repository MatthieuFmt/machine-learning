# Pivot v4 — C5 : Extension A9 pipeline lock + bilan Phase A étendue

> 📍 **Cinquième et dernière étape de la Phase C — Extension multi-actifs**
> Mise à jour du pipeline lock (`ml_pipeline_v4.py` + SHA256 + `TEST_SET_LOCK.json`) avec les nouveaux couples.
> Bilan global de la Phase A étendue et **décision conditionnelle** sur la suite (Phase B sélective, prompt 18, ou autre).
> **Aucune lecture du test set ≥ 2024**. 0 n_trial consommé.

## Préalable obligatoire (à lire dans l'ordre)

1. [00_README.md](00_README.md) — vue d'ensemble pivot v4
2. [../00_constitution.md](../00_constitution.md) — règles 6, 7, 9 + **section 6bis n_trials**
3. [C4_extend_a8_hyperparams_multi_assets.md](C4_extend_a8_hyperparams_multi_assets.md) — **✅ Terminé obligatoire**
4. [A9_pipeline_lock.md](A9_pipeline_lock.md) — A9 d'origine (référence méthodo)
5. [`docs/pipeline_v4_locked.md`](../../docs/pipeline_v4_locked.md) — documentation A9
6. [`predictions/c4_hyperparam_tuning_multi_assets.json`](../../predictions/c4_hyperparam_tuning_multi_assets.json) — couples tunés
7. [`app/config/ml_pipeline_v4.py`](../../app/config/ml_pipeline_v4.py) — pipeline actuellement gelé
8. [`TEST_SET_LOCK.json`](../../TEST_SET_LOCK.json) — section `pipeline_locked` à mettre à jour

## Objectif

1. Étendre [`app/config/ml_pipeline_v4.py`](../../app/config/ml_pipeline_v4.py) avec les nouveaux couples (registre `LOCKED_COUPLES` ou équivalent).
2. Recalculer les **SHA256** des 4 fichiers de config (`features_selected.py`, `model_selected.py`, `hyperparams_tuned.py`, `ml_pipeline_v4.py`) et les écrire dans `TEST_SET_LOCK.json`.
3. Bumper la version du pipeline de `v4.0.0-locked` à **`v4.1.0-extended`**.
4. Documenter le **bilan complet** de la Phase A étendue dans `docs/phase_a_extended_summary.md` :
   - Tableau global (21 couples : 3 originaux + 18 nouveaux)
   - Quels couples sont prêts pour Phase B (test set 2024+) ?
   - Quel ordre de priorité Phase B (par espérance Sharpe outer décroissante) ?
5. **Aucune exécution Phase B dans ce prompt** — la décision finale revient à l'utilisateur après lecture du bilan.

## Type d'opération

🔧 **Verrouillage + bilan documentaire — 0 n_trial consommé. Aucune lecture du test set 2024+.**

## Definition of Done (testable)

- [ ] `app/config/ml_pipeline_v4.py` :
  - `PIPELINE_VERSION = "v4.1.0-extended"`
  - Une constante `LOCKED_COUPLES: frozenset[tuple[str, str]]` listant tous les couples du pipeline.
  - `get_pipeline(asset, tf)` continue de fonctionner pour les anciens et nouveaux couples.
- [ ] `scripts/run_c5_pipeline_lock_extended.py` (NOUVEAU) recalcule les SHA256 et met à jour `TEST_SET_LOCK.json["pipeline_locked"]` avec :
  - `locked_at` mis à jour
  - `pipeline_version = "v4.1.0-extended"`
  - `configured_pairs` complète (21 entrées max)
  - `config_checksums` recalculés
  - Une nouvelle clé `phase_a_extended_completed_at` (timestamp).
- [ ] `docs/phase_a_extended_summary.md` (NOUVEAU) — bilan complet (cf. template Étape 4).
- [ ] `tests/integration/test_pipeline_integrity_extended.py` (NOUVEAU) — adapte les tests A9 pour couvrir tous les couples figés.
- [ ] `rtk make verify` → 0 erreur.
- [ ] `JOURNAL.md` mis à jour avec la section "Pivot v4 C5".
- [ ] **Question explicite à l'utilisateur** à la fin du prompt : « Voici les K couples nouveaux prêts pour Phase B. Lesquels veux-tu tester en priorité (chaque test = +1 n_trial) ? ». **Ne PAS lancer Phase B sans réponse.**

## NE PAS FAIRE

- ❌ **Ne PAS modifier la valeur de `pipeline_locked.locked_at` pour les 3 couples d'origine** — utiliser un champ séparé `phase_a_extended_completed_at`.
- ❌ **Ne PAS lire le test set ≥ 2024.** C5 reste 100 % train + métadonnées.
- ❌ **Ne PAS lancer Phase B (B1-B4) ni nouvelle hypothèse** dans ce prompt.
- ❌ **Ne PAS incrémenter `n_trials`.**
- ❌ **Ne PAS supprimer le verrou `pipeline_locked` existant** — l'étendre, pas le remplacer.
- ❌ **Ne PAS modifier la fonction `get_pipeline()` pour qu'elle accepte des paramètres non gelés** — le lock reste strict.

## Étapes détaillées

### Étape 1 — Étendre `app/config/ml_pipeline_v4.py`

Modifier le header et ajouter `LOCKED_COUPLES` :

```python
"""Pipeline ML v4 — FROZEN après A9 (3 couples) + C5 (extension multi-actifs).

CE FICHIER NE DOIT PAS ÊTRE MODIFIÉ APRÈS C5.
Toute modification = data snooping → invalide la statistique Phase B.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config.features_selected import FEATURES_SELECTED
from app.config.hyperparams_tuned import HYPERPARAMS_TUNED
from app.config.model_selected import MODEL_SELECTED

PIPELINE_VERSION: str = "v4.1.0-extended"

# Couples figés dans le pipeline. À jour après chaque verrouillage A9/C5.
# Pour ajouter un couple : seules les phases A6→A8 (ou C2→C4) le permettent.
LOCKED_COUPLES: frozenset[tuple[str, str]] = frozenset(
    set(FEATURES_SELECTED.keys())
    & set(MODEL_SELECTED.keys())
    & set(HYPERPARAMS_TUNED.keys())
)


@dataclass(frozen=True)
class MLPipelineConfig:
    asset: str
    tf: str
    features: tuple[str, ...]
    model_name: str
    model_params: dict
    threshold: float
    expected_sharpe_outer: float
    expected_wr: float
    version: str = PIPELINE_VERSION

    def __post_init__(self) -> None:
        if not 0.50 <= self.threshold <= 0.80:
            raise ValueError(f"Seuil hors plage [0.50, 0.80]: {self.threshold}")
        if not self.features:
            raise ValueError("Aucune feature sélectionnée")
        if self.model_name not in ("rf", "hgbm", "stacking"):
            raise ValueError(f"Modèle inconnu : {self.model_name}")


def get_pipeline(asset: str, tf: str) -> MLPipelineConfig:
    key = (asset, tf)
    if key not in LOCKED_COUPLES:
        raise KeyError(f"Couple ({asset}, {tf}) non figé dans le pipeline")
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


def list_locked_couples() -> list[tuple[str, str]]:
    """Renvoie la liste triée des couples figés."""
    return sorted(LOCKED_COUPLES)
```

### Étape 2 — Créer le script de verrouillage

Créer [`scripts/run_c5_pipeline_lock_extended.py`](../../scripts/run_c5_pipeline_lock_extended.py) :

```python
"""Pivot v4 C5 — Pipeline lock étendu (extension A9 multi-actifs).

⚠️ 0 n_trial consommé. Aucune lecture du test set 2024+.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.ml_pipeline_v4 import (
    LOCKED_COUPLES,
    PIPELINE_VERSION,
    get_pipeline,
    list_locked_couples,
)

CONFIG_FILES = [
    "app/config/features_selected.py",
    "app/config/model_selected.py",
    "app/config/hyperparams_tuned.py",
    "app/config/ml_pipeline_v4.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock_path = _PROJECT_ROOT / "TEST_SET_LOCK.json"
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))

    now_iso = datetime.now(timezone.utc).isoformat()
    couples = list_locked_couples()

    checksums: dict[str, str] = {}
    for rel in CONFIG_FILES:
        full = _PROJECT_ROOT / rel
        checksums[rel.replace("/", "\\")] = _sha256(full)

    pl = lock_data.setdefault("pipeline_locked", {})
    pl["pipeline_version"] = PIPELINE_VERSION
    pl["configured_pairs"] = [{"asset": a, "tf": tf} for (a, tf) in couples]
    pl["config_checksums"] = checksums
    pl.setdefault("locked_at", now_iso)  # ne pas écraser l'horodatage A9
    pl["phase_a_extended_completed_at"] = now_iso

    lock_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

    # Sanity check : tous les couples doivent être loadables
    failures: list[str] = []
    for (asset, tf) in couples:
        try:
            cfg = get_pipeline(asset, tf)
            assert cfg.version == PIPELINE_VERSION
        except Exception as exc:
            failures.append(f"{asset}/{tf} : {exc}")

    if failures:
        print("ÉCHEC : couples non chargeables")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"Pipeline lock étendu OK (version {PIPELINE_VERSION})")
    print(f"  {len(couples)} couples figés :")
    for (a, tf) in couples:
        cfg = get_pipeline(a, tf)
        print(f"    {a}/{tf}: {cfg.model_name} threshold={cfg.threshold} sharpe_outer={cfg.expected_sharpe_outer:.2f}")
    print(f"  Checksums :")
    for f, h in checksums.items():
        print(f"    {f} : {h[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 3 — Tests d'intégrité étendus

Créer [`tests/integration/test_pipeline_integrity_extended.py`](../../tests/integration/test_pipeline_integrity_extended.py) :

```python
"""Tests d'intégrité du pipeline lock étendu (C5)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.config.ml_pipeline_v4 import (
    LOCKED_COUPLES,
    PIPELINE_VERSION,
    get_pipeline,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCK_PATH = _PROJECT_ROOT / "TEST_SET_LOCK.json"


def test_pipeline_version_extended() -> None:
    assert PIPELINE_VERSION == "v4.1.0-extended"


def test_a9_original_couples_present() -> None:
    """Les 3 couples A9 doivent rester dans LOCKED_COUPLES."""
    for key in [("US30", "D1"), ("EURUSD", "H4"), ("XAUUSD", "D1")]:
        assert key in LOCKED_COUPLES


@pytest.mark.parametrize("key", sorted(LOCKED_COUPLES))
def test_each_couple_loadable(key: tuple[str, str]) -> None:
    asset, tf = key
    cfg = get_pipeline(asset, tf)
    assert cfg.asset == asset
    assert cfg.tf == tf
    assert cfg.version == PIPELINE_VERSION
    assert 0.50 <= cfg.threshold <= 0.80
    assert len(cfg.features) == 15
    assert cfg.model_name in ("rf", "hgbm", "stacking")


def test_test_set_lock_has_pipeline_section() -> None:
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    pl = data["pipeline_locked"]
    assert pl["pipeline_version"] == "v4.1.0-extended"
    assert "phase_a_extended_completed_at" in pl


def test_checksums_match_current_files() -> None:
    data = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    stored = data["pipeline_locked"]["config_checksums"]
    for stored_path, expected_hash in stored.items():
        rel = stored_path.replace("\\", "/")
        full = _PROJECT_ROOT / rel
        actual = hashlib.sha256(full.read_bytes()).hexdigest()
        assert actual == expected_hash, f"Drift sur {rel}"


def test_unknown_couple_raises() -> None:
    with pytest.raises(KeyError):
        get_pipeline("INEXISTANT", "D1")
```

### Étape 4 — Bilan documentaire

Créer [`docs/phase_a_extended_summary.md`](../../docs/phase_a_extended_summary.md) :

```markdown
# Phase A étendue — Bilan global (Pivot v4 + C1-C5)

**Date** : YYYY-MM-DD
**Pipeline version** : v4.1.0-extended
**n_trials consommés en Phase C** : 0
**Test set ≥ 2024 lu** : NON (préservé)

## 1. Couverture

| Catégorie | Compte |
|---|---|
| Couples cibles | 21 (7 actifs × 3 TF) |
| Couples avec données | (cf. C1) |
| Couples figés en pipeline (A9 + C5) | (cf. LOCKED_COUPLES) |
| Couples shortlist finale (Sharpe outer ≥ 0.5, gap < 1.0) | (cf. C4) |
| Couples exclus (insufficient_trades / stab / Sharpe < 0.5) | (cf. C2/C3/C4) |

## 2. Tableau global des couples figés

| Actif | TF | Modèle | Params | Threshold | Sharpe outer | WR outer | Stab top 15 | Statut shortlist | Source |
|---|---|---|---|---|---|---|---|---|---|
| US30 | D1 | rf | n=100, d=3, leaf=10 | 0.55 | +1.913 | 57.5% | 0.72 | ✅ | A9 |
| EURUSD | H4 | rf | n=100, d=6, leaf=10 | 0.55 | +0.592 | 51.5% | 0.59 | ✅ | A9 |
| XAUUSD | D1 | stacking | (defaults) | 0.50 | 0.000 | — | 0.56 | ❌ | A9 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | C5 |

(remplir à partir des JSON C2/C3/C4)

## 3. Patterns par classe d'actif

### Indices
### Forex majeures
### Métaux
### Crypto

## 4. Décision Phase B — couples candidats prioritaires

Couples shortlist finale triés par Sharpe outer décroissant :

| Rang | Couple | Modèle | Sharpe outer | Coût n_trial | Recommandation |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | +1 | ★ tester en priorité |
| 2 | ... | ... | ... | +1 | second |
| ... | | | | | |

**Cadre méthodologique** : chaque test Phase B = lecture définitive du test set 2024+ pour ce couple = +1 n_trial cumul.

- n_trials_cumul actuel : 28
- Si on teste les K candidats : n_trials_cumul → 28 + K
- DSR pénalise par √(log(n_trials)) → impact sur le seuil de significativité

## 5. Recommandations utilisateur

Trois options possibles :

### Option A — Aller en Phase B sélective (1 à 3 couples max)
- Pertinent si le bilan révèle un couple avec Sharpe outer ≥ 1.5 (très prometteur)
- Cible : compléter le portefeuille (actuellement single-sleeve EURUSD H4)
- Coût méthodologique acceptable : 28 → 29-31 n_trials

### Option B — Faire la vérification spreads démo d'abord
- L'utilisateur a explicitement demandé cette vérification après Phase C
- Mise à jour de `ASSET_CONFIGS` avec les vrais coûts XTB capturés en démo
- Si gros écart → retour en C2-C4 pour quelques couples (les coûts impactent la cible binaire winner)
- Bénéfice : Phase B avec coûts réels, résultats plus crédibles

### Option C — Aller en prompt 18 (validation finale) sur l'existant
- Le portfolio single-sleeve (EURUSD H4) doit être confronté à Buy-and-Hold + Monte Carlo
- Si ça passe → production (prompt 20). Sinon → retour Phase C avec stratégies alternatives.
- N'invalide pas la Phase C : les nouveaux couples restent dans le pipeline gelé pour usage futur.

**Recommandation par défaut** : Option B (spreads démo) puis Option A si gros candidat émerge, sinon Option C.

## 6. Annexes techniques

### Fichiers figés (SHA256)
- `app/config/features_selected.py` : ...
- `app/config/model_selected.py` : ...
- `app/config/hyperparams_tuned.py` : ...
- `app/config/ml_pipeline_v4.py` : ...

### Couples écartés (raison détaillée)
- ...
```

### Étape 5 — Exécution (sur demande utilisateur)

```bash
# 1. Verrouiller
rtk python scripts/run_c5_pipeline_lock_extended.py

# 2. Tests intégrité
rtk pytest tests/integration/test_pipeline_integrity_extended.py -v
rtk pytest tests/integration/test_pipeline_integrity.py -v  # non-régression A9

# 3. Quality gates complets
rtk make verify
```

## Tests unitaires associés

`tests/integration/test_pipeline_integrity_extended.py` : 6 + N tests (paramétrés sur tous les couples).

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 C5 : Pipeline lock étendu + bilan Phase A

- **Statut** : ✅ Terminé — Phase A étendue complète (A1-A9 + C1-C5)
- **Type** : Verrouillage + documentation (0 n_trial)
- **Fichiers modifiés** : `app/config/ml_pipeline_v4.py` (version v4.1.0-extended + `LOCKED_COUPLES`), `TEST_SET_LOCK.json` (section pipeline_locked étendue)
- **Fichiers créés** : `scripts/run_c5_pipeline_lock_extended.py`, `tests/integration/test_pipeline_integrity_extended.py`, `docs/phase_a_extended_summary.md`
- **Pipeline version** : v4.1.0-extended
- **Couples figés (LOCKED_COUPLES)** : N (3 originaux A9 + (N-3) nouveaux C5)
- **Shortlist Phase B (Sharpe outer ≥ 0.5, gap < 1.0)** : K candidats
- **Tests** : X passing
- **Quality gates** : ruff ✅, mypy ✅, pytest ✅
- **n_trials cumul** : 28 (inchangé — Phase C entière à 0 trial)
- **Prochaine étape (décision utilisateur)** :
  - (A) Phase B sélective sur 1-3 couples shortlist → +1 à +3 n_trials
  - (B) Vérification spreads démo XTB + correction ASSET_CONFIGS → 0 n_trial
  - (C) Prompt 18 validation finale sur portfolio existant → +1 n_trial
```

## Critères go/no-go

| Critère | Cible | Action si non atteint |
|---|---|---|
| `PIPELINE_VERSION = "v4.1.0-extended"` | obligatoire | Corriger |
| 3 couples A9 originaux toujours présents dans `LOCKED_COUPLES` | obligatoire | STOP : régression |
| Checksums dans `TEST_SET_LOCK.json` correspondent aux fichiers actuels | obligatoire | Recalculer |
| Bilan markdown produit avec tableau global et 3 options recommandées | obligatoire | Compléter |
| Tests intégrité étendus passent | obligatoire | Corriger les drifts |

**Phase C TERMINÉE** si tous les critères sont verts. **Décision Phase B = utilisateur** (ne pas l'exécuter dans ce prompt).

## Annexes

### A1 — Pourquoi bumper la version pipeline ?

Les SHA256 des fichiers de config ont changé (nouvelles entrées). Garder la même version (`v4.0.0-locked`) induirait en erreur tout consommateur du pipeline qui se fie au numéro. La pratique habituelle (semver minor) est de passer en `v4.1.x` pour signaler une extension non-breakings.

### A2 — Pourquoi conserver `locked_at` d'origine ?

`locked_at` documente le moment où la Phase A originale a été figée (15 mai 2026). Ce timestamp doit rester pour la traçabilité méthodologique (n_trials consommés en Phase B après ce verrou). On ajoute `phase_a_extended_completed_at` séparément pour la Phase C.

### A3 — Pourquoi ne pas lancer Phase B automatiquement ?

Chaque test Phase B brûle 1 n_trial = lecture définitive du test set pour ce couple. C'est une décision **stratégique** qui dépend de :
- Coûts réels en démo (l'utilisateur veut les vérifier)
- Priorisation par classe d'actif (forex vs crypto vs indices)
- Tolérance au DSR penalty croissant

Le bilan C5 fournit les éléments. La décision = utilisateur.

### A4 — Que se passe-t-il si C2/C3/C4 produisent 0 candidat shortlist ?

C'est possible : si toutes les stratégies Donchian (N,M) initiales produisent < 50 trades sur les nouveaux couples, ou si aucune ne dépasse Sharpe outer 0.5, alors la Phase C aura été **un grand audit qui conclut "rien de nouveau"**. Ce n'est pas un échec : c'est une information valide (l'edge ML méta-labeling Donchian n'apparaît que sur quelques marchés spécifiques).

Dans ce cas, le bilan C5 documente l'absence de candidat et recommande **Option B (vérif spreads) puis Option C (prompt 18)**.

### A5 — Évolutions futures du pipeline (post-C5)

Toute extension ultérieure devra :
1. Repartir de C2-C4 pour les nouveaux couples (jamais modifier les couples figés).
2. Bumper la version (v4.2.x, v4.3.x...).
3. Documenter dans une nouvelle section de `phase_a_extended_summary.md`.

Tant qu'**aucun couple existant n'est modifié**, ces extensions consomment 0 n_trial.

## Fin du prompt C5.
**Phase C complète.** Décision Phase B / prompt 18 / vérif coûts = utilisateur.
