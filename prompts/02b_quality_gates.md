# Prompt 02b — Quality Gates (outillage qualité, anti-erreur IA)

## Préalable obligatoire
1. `prompts/00_constitution.md`
2. `JOURNAL.md`
3. `prompts/02_cleanup.md`

## Objectif
**Installer tout l'outillage qualité AVANT toute logique métier.** Ce prompt produit :
1. Les hooks anti-bugs systémiques (anti-look-ahead, anti-data-snooping, retry I/O, seeds déterministes).
2. Le tooling Python (ruff, black, mypy, pytest, pre-commit, CI).
3. Les configs typées frozen pour éviter les bugs silencieux.

> **Sans ces garde-fous, Deepseek peut produire du code qui passe pytest mais cache des leaks, du data snooping, ou des erreurs runtime.** Ce prompt est non négociable.

## Definition of Done (testable)
- [ ] `pyproject.toml` créé avec config ruff + black + mypy + pytest.
- [ ] `.pre-commit-config.yaml` créé, `pre-commit install` exécuté.
- [ ] `requirements-dev.txt` créé.
- [ ] `Makefile` avec cibles : `install`, `test`, `lint`, `typecheck`, `verify`, `backtest`.
- [ ] `.github/workflows/ci.yml` créé (run `make verify` sur push/PR).
- [ ] `.gitignore` enrichi : `.env`, `models/snapshots/`, `logs/`, `predictions/`, `.mypy_cache/`, `.ruff_cache/`, `TEST_SET_LOCK.json`.
- [ ] Modules créés et testés :
  - `app/testing/look_ahead_validator.py` (+ `tests/unit/test_look_ahead_validator.py`)
  - `app/testing/snooping_guard.py` (+ `tests/unit/test_snooping_guard.py`)
  - `app/core/retry.py` (+ `tests/unit/test_retry.py`)
  - `app/core/seeds.py` (+ `tests/unit/test_seeds.py`)
  - `app/config/models.py` (+ `tests/unit/test_production_config.py`)
  - `scripts/verify_no_snooping.py`
- [ ] `rtk make verify` passe (sur demande utilisateur).
- [ ] `JOURNAL.md` mis à jour.

## NE PAS FAIRE
- Ne PAS installer Poetry / uv / pipenv — on reste sur `pip + requirements*.txt`.
- Ne PAS activer mypy `--strict` complet (trop de friction pour Deepseek) — strict_optional + warn_unused_ignores suffisent.
- Ne PAS coupler les modules `app/testing/` à des frameworks externes (hormis pytest pour les tests).
- Ne PAS commit `.env`, `TEST_SET_LOCK.json`, `models/snapshots/`.
- Ne PAS exécuter `git commit` automatiquement.

## Étapes

### Étape 1 — `pyproject.toml`
```toml
[project]
name = "trading-bot-ml"
version = "0.1.0"
requires-python = ">=3.12"

[tool.ruff]
line-length = 100
target-version = "py312"
extend-exclude = ["data", "ready-data", "cleaned-data", "models/snapshots"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "W", "UP", "N", "SIM"]
ignore = ["E501"]  # géré par black

[tool.black]
line-length = 100
target-version = ["py312"]

[tool.mypy]
python_version = "3.12"
strict_optional = true
warn_unused_ignores = true
warn_redundant_casts = true
warn_unreachable = true
disallow_untyped_defs = false  # progressive
ignore_missing_imports = true
exclude = ["data/", "ready-data/", "cleaned-data/", "models/snapshots/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: tests unitaires rapides (<100ms)",
    "integration: tests d'intégration (I/O)",
    "acceptance: tests bout-en-bout (lents)",
]
addopts = "-ra --strict-markers"
```

### Étape 2 — `.pre-commit-config.yaml`
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-merge-conflict
      - id: check-added-large-files
        args: ['--maxkb=500']
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: ['--fix']
      - id: ruff-format
```

### Étape 3 — `requirements-dev.txt`
```
pytest>=8.0
pytest-cov>=5.0
mypy>=1.10
ruff>=0.6
black>=24.0
pre-commit>=3.7
hypothesis>=6.100
```

### Étape 4 — `Makefile`
```makefile
.PHONY: install test lint typecheck verify backtest snooping_check

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

test:
	rtk pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/ scripts/

typecheck:
	mypy app/

snooping_check:
	python scripts/verify_no_snooping.py

verify: lint typecheck test snooping_check
	@echo "✅ All quality gates passed."

backtest:
	@echo "Lance manuellement un script run_*.py spécifique."
```

### Étape 5 — `.github/workflows/ci.yml`
```yaml
name: CI
on:
  push:
    branches: [main, test-deepseek]
  pull_request:
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: make lint
      - run: make typecheck
      - run: make test
      - run: make snooping_check
```

### Étape 6 — `.gitignore` (à enrichir)
Ajouter (sans dupliquer si déjà présent) :
```
.env
.env.local
models/snapshots/
logs/
predictions/*.json
predictions/*.csv
.mypy_cache/
.ruff_cache/
.pytest_cache/
.coverage
htmlcov/
TEST_SET_LOCK.json
```

### Étape 7 — `app/testing/look_ahead_validator.py`
```python
"""Hooks anti-look-ahead à utiliser dans tous les modules de features."""
from __future__ import annotations

from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd


def assert_no_look_ahead(
    feature_fn: Callable,
    series: pd.Series | pd.DataFrame,
    n_samples: int = 100,
    seed: int = 42,
) -> None:
    """Vérifie que feature_fn(series[:n])[-1] == feature_fn(series)[n-1]
    pour 100 indices aléatoires."""
    rng = np.random.default_rng(seed)
    full = feature_fn(series)
    start = len(series) // 2
    if len(series) - 1 - start < n_samples:
        n_samples = max(1, len(series) - 1 - start)
    indices = rng.choice(range(start, len(series) - 1), n_samples, replace=False)

    for n in indices:
        truncated = feature_fn(series.iloc[: n + 1] if hasattr(series, "iloc") else series[: n + 1])
        full_val = _at(full, n)
        trunc_val = _at(truncated, n)
        if pd.isna(full_val) and pd.isna(trunc_val):
            continue
        if not np.isclose(full_val, trunc_val, rtol=1e-9, equal_nan=True):
            raise AssertionError(
                f"Look-ahead at idx {n}: full={full_val} vs truncated={trunc_val} "
                f"(feature_fn={feature_fn.__name__})"
            )


def _at(out, n: int):
    """Récupère la valeur scalaire à l'index positionnel n (Series ou DataFrame)."""
    if isinstance(out, pd.DataFrame):
        return out.iloc[n].sum()  # signature stable même multi-colonnes
    return out.iloc[n] if hasattr(out, "iloc") else out[n]


def look_ahead_safe(fn: Callable) -> Callable:
    """Décorateur de marquage. Toutes les fonctions de features doivent l'utiliser.
    Le test pytest `test_indicators_are_marked_safe` vérifie la présence du marqueur."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._look_ahead_safe = True  # type: ignore[attr-defined]
    return wrapper
```

### Étape 8 — `app/testing/snooping_guard.py`
```python
"""Anti-data-snooping mécanique. Étape critique pour la validité statistique."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOCK_PATH = Path("TEST_SET_LOCK.json")
TEST_START = "2024-01-01"


class TestSetSnoopingError(Exception):
    """Levée si une modification post-lock est tentée."""


def _load() -> dict:
    if LOCK_PATH.exists():
        return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return {"locked": False, "n_reads": 0, "read_history": []}


def _save(state: dict) -> None:
    LOCK_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def is_locked() -> bool:
    return _load().get("locked", False)


def read_oos(
    prompt: str,
    hypothesis: str,
    sharpe: float,
    n_trades: int | None = None,
) -> None:
    """À appeler à CHAQUE lecture du test set OOS (≥ 2024)."""
    state = _load()
    state["n_reads"] += 1
    state["read_history"].append({
        "prompt": prompt,
        "hypothesis": hypothesis,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sharpe": sharpe,
        "n_trades": n_trades,
    })
    _save(state)


def lock(prompt: str) -> None:
    """À appeler une seule fois, au prompt 18 (validation finale GO)."""
    state = _load()
    state["locked"] = True
    state["locked_at"] = datetime.now(timezone.utc).isoformat()
    state["locked_by_prompt"] = prompt
    _save(state)


def check_unlocked() -> None:
    """À appeler en haut de tout script qui modifie une stratégie/feature/seuil."""
    if is_locked():
        raise TestSetSnoopingError(
            "TEST_SET_LOCK.json est verrouillé. Modifier la stratégie après lecture finale = "
            "data snooping. Pour itérer, il faut un nouveau split temporel (split ≥ 2026)."
        )


def get_history() -> list[dict]:
    return _load().get("read_history", [])
```

### Étape 9 — `scripts/verify_no_snooping.py`
```python
"""Scanne le code source pour détecter des accès au test set après lock.

À ajouter dans `make verify` et la CI."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LOCK_PATH = Path("TEST_SET_LOCK.json")
SUSPICIOUS_PATTERNS = [
    re.compile(r'\.loc\[\s*["\']?(2024|2025|2026)'),
    re.compile(r'df\[df\.index\s*>=?\s*["\']?(2024|2025|2026)'),
    re.compile(r'index\.year\s*[><=!]+\s*202[4-9]'),
]
SCAN_DIRS = ["app", "scripts"]
ALLOWED_FILES = {"app/testing/snooping_guard.py"}


def main() -> int:
    if not LOCK_PATH.exists():
        print("TEST_SET_LOCK.json absent : pas de scan nécessaire.")
        return 0
    state = json.loads(LOCK_PATH.read_text())
    if not state.get("locked"):
        print("Test set non verrouillé : pas de scan nécessaire.")
        return 0

    offenders: list[tuple[Path, int, str]] = []
    for dir_name in SCAN_DIRS:
        for py in Path(dir_name).rglob("*.py"):
            if str(py).replace("\\", "/") in ALLOWED_FILES:
                continue
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                for pat in SUSPICIOUS_PATTERNS:
                    if pat.search(line):
                        offenders.append((py, i, line.strip()))

    if offenders:
        print("❌ Accès suspects au test set détectés après verrouillage :")
        for p, i, line in offenders:
            print(f"  {p}:{i}  {line}")
        return 1
    print("✅ Aucun accès suspect au test set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 10 — `app/core/retry.py`
```python
"""Retry avec backoff exponentiel. Obligatoire pour toute I/O (cf. Règle 11)."""
from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.warning(
                            "retry",
                            extra={"context": {
                                "fn": fn.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "error": str(e),
                                "next_delay_s": delay,
                            }},
                        )
                        time.sleep(delay)
            assert last_exc is not None
            raise last_exc
        return wrapper
    return decorator
```

### Étape 11 — `app/core/seeds.py`
```python
"""Reproductibilité. À appeler en haut de chaque script run_*.py (cf. Règle 12)."""
from __future__ import annotations

import os
import random

import numpy as np

GLOBAL_SEED = 42


def set_global_seeds(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```

### Étape 12 — `app/config/models.py` (configs typées frozen)
```python
"""Configs typées immutables. Évite les bugs silencieux."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

TFLiteral = Literal["D1", "H4", "H1"]


@dataclass(frozen=True)
class Sleeve:
    asset: str
    tf: TFLiteral
    strategy_name: str
    strategy_params: Mapping[str, int | float | str]
    regime_filter: bool = False
    meta_labeling: bool = False
    meta_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0.40 <= self.meta_threshold <= 0.80:
            raise ValueError(f"meta_threshold hors plage : {self.meta_threshold}")


@dataclass(frozen=True)
class ProductionConfig:
    version: str
    sleeves: tuple[Sleeve, ...]
    portfolio_weighting: Literal["equal_risk", "correlation_aware"]
    vol_targeting: bool
    target_vol_annual: float
    leverage_cap: float
    retrain_months: int

    def __post_init__(self) -> None:
        if not 0 < self.target_vol_annual < 1.0:
            raise ValueError(f"target_vol_annual hors (0,1) : {self.target_vol_annual}")
        if not 1.0 <= self.leverage_cap <= 3.0:
            raise ValueError(f"leverage_cap hors [1,3] : {self.leverage_cap}")
        if len(self.sleeves) == 0:
            raise ValueError("ProductionConfig sans sleeve")
        if self.retrain_months not in (3, 6, 12):
            raise ValueError(f"retrain_months ∉ {{3,6,12}} : {self.retrain_months}")
```

### Étape 13 — Tests unitaires

`tests/unit/test_look_ahead_validator.py` :
```python
import numpy as np
import pandas as pd
import pytest
from app.testing.look_ahead_validator import assert_no_look_ahead


def test_safe_function_passes():
    series = pd.Series(np.arange(200, dtype=float))
    def safe(s): return s.rolling(5).mean()
    assert_no_look_ahead(safe, series, n_samples=20)


def test_leaky_function_fails():
    series = pd.Series(np.arange(200, dtype=float))
    def leaky(s): return s.shift(-1)  # utilise le futur
    with pytest.raises(AssertionError, match="Look-ahead"):
        assert_no_look_ahead(leaky, series, n_samples=20)


def test_nan_handling():
    series = pd.Series([np.nan] * 10 + list(range(190)), dtype=float)
    def safe(s): return s.rolling(3).mean()
    assert_no_look_ahead(safe, series, n_samples=20)
```

`tests/unit/test_snooping_guard.py` :
```python
import json
from app.testing.snooping_guard import (
    is_locked, read_oos, lock, check_unlocked, TestSetSnoopingError, LOCK_PATH
)


def test_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not is_locked()
    check_unlocked()  # no-op
    read_oos("07", "H06", sharpe=1.2, n_trades=40)
    assert LOCK_PATH.exists()
    state = json.loads(LOCK_PATH.read_text())
    assert state["n_reads"] == 1
    lock("18")
    assert is_locked()
    import pytest
    with pytest.raises(TestSetSnoopingError):
        check_unlocked()
```

`tests/unit/test_retry.py` :
```python
import pytest
from app.core.retry import retry_with_backoff


def test_success_first_try():
    calls = []
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def f():
        calls.append(1)
        return "ok"
    assert f() == "ok"
    assert len(calls) == 1


def test_retry_then_success():
    calls = []
    @retry_with_backoff(max_attempts=3, base_delay=0.01)
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("flaky")
        return "ok"
    assert f() == "ok"
    assert len(calls) == 3


def test_max_attempts_then_raise():
    @retry_with_backoff(max_attempts=2, base_delay=0.01)
    def f():
        raise RuntimeError("nope")
    with pytest.raises(RuntimeError, match="nope"):
        f()
```

`tests/unit/test_seeds.py` :
```python
import numpy as np
from app.core.seeds import set_global_seeds


def test_reproducible():
    set_global_seeds(123)
    a = np.random.rand(10)
    set_global_seeds(123)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)
```

`tests/unit/test_production_config.py` :
```python
import pytest
from app.config.models import Sleeve, ProductionConfig


def test_sleeve_ok():
    Sleeve(asset="US30", tf="D1", strategy_name="donchian", strategy_params={"N": 20, "M": 20})


def test_sleeve_bad_threshold():
    with pytest.raises(ValueError):
        Sleeve(asset="US30", tf="D1", strategy_name="d", strategy_params={}, meta_threshold=0.9)


def test_config_validation():
    s = Sleeve(asset="US30", tf="D1", strategy_name="d", strategy_params={"N": 20})
    with pytest.raises(ValueError, match="target_vol_annual"):
        ProductionConfig(
            version="v3.0", sleeves=(s,), portfolio_weighting="equal_risk",
            vol_targeting=True, target_vol_annual=2.0, leverage_cap=2.0, retrain_months=6,
        )
```

### Étape 14 — Vérification
```bash
rtk make install
rtk make verify
```
Tous les checks doivent passer (sur demande utilisateur uniquement).

## Logging
```markdown
## 2026-MM-DD — Prompt 02b : Quality Gates
- **Statut** : ✅ Terminé
- **Fichiers créés** : pyproject.toml, .pre-commit-config.yaml, requirements-dev.txt, Makefile, .github/workflows/ci.yml
- **Modules créés** : app/testing/{look_ahead_validator,snooping_guard}.py, app/core/{retry,seeds}.py, app/config/models.py, scripts/verify_no_snooping.py
- **Tests** : X tests, 0 failures
- **make verify** : ✅ passé
- **Notes** : pre-commit installé, hooks actifs sur les futurs commits
```

## Critères go/no-go
- **GO prompt 03** si : `make verify` passe ET `pre-commit run --all-files` ne signale aucun blocage.
- **NO-GO** : si un module manque ou si un test échoue. Ne PAS poursuivre Phase 1 tant que l'outillage n'est pas en place.
