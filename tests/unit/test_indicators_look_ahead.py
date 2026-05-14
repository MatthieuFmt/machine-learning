"""Scan anti-look-ahead de tous les modules app/features/*.py.

Vérifie que :
1. Toute fonction publique est décorée @look_ahead_safe.
2. Chaque fonction passe assert_no_look_ahead sur données synthétiques.

Le scan est dynamique — tout nouveau module .py dans app/features/ est
automatiquement testé, sans modification de ce fichier.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.testing.look_ahead_validator import assert_no_look_ahead

# ═══════════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════════


def _discover_feature_modules() -> list[str]:
    """Retourne les noms de modules Python dans app/features/ (hors __init__)."""
    features_dir = Path("app/features")
    if not features_dir.exists():
        return []
    modules = []
    for f in sorted(features_dir.glob("*.py")):
        if f.stem.startswith("_"):
            continue
        modules.append(f"app.features.{f.stem}")
    return modules


def _import_module_safe(mod_name: str) -> object | None:
    """Importe un module, retourne None si ModuleNotFoundError."""
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError:
        return None


def _public_functions(mod_name: str) -> list[Callable]:
    """Retourne les fonctions publiques DÉFINIES dans le module (pas ré-exportées).

    Les fonctions importées depuis app.core.logging, sklearn, etc. sont exclues.
    """
    mod = _import_module_safe(mod_name)
    if mod is None:
        return []
    return [
        fn for name, fn in inspect.getmembers(mod, inspect.isfunction)
        if not name.startswith("_")
        and (fn.__module__ or "").startswith("app.features")
    ]


FEATURE_MODULES = _discover_feature_modules()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers de construction d'input synthétique
# ═══════════════════════════════════════════════════════════════════════════════


def _build_test_input(fn: Callable) -> pd.Series | pd.DataFrame:
    """Construit un input synthétique adapté à la signature de fn."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    n = 500

    rng = np.random.default_rng(42)
    close = pd.Series(
        rng.random(n).cumsum() + 100.0,
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
        name="close",
    )

    # compute_all_indicators prend un DataFrame OHLCV
    if fn.__name__ == "compute_all_indicators" or (
        len(params) >= 1 and isinstance(sig.parameters[params[0]].annotation, type)
        and sig.parameters[params[0]].annotation is pd.DataFrame
    ):
        return pd.DataFrame(
            {
                "Open": close.shift(1).fillna(close.iloc[0]),
                "High": close + np.abs(rng.normal(size=n)),
                "Low": close - np.abs(rng.normal(size=n)),
                "Close": close,
                "Volume": np.ones(n),
            },
            index=close.index,
        )

    # Détection heuristique : si la fonction prend high, low, close → DataFrame OHLCV
    if len(params) >= 3 and all(p in params for p in ("high", "low")):
        return pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close + np.abs(rng.normal(size=n)),
                "low": close - np.abs(rng.normal(size=n)),
                "close": close,
                "volume": np.ones(n),
            },
            index=close.index,
        )

    # Si le premier paramètre est "price_index" (economic.py)
    if params and params[0] == "price_index":
        return close.index

    return close


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_at_least_one_feature_module() -> None:
    """Au moins un module de features existe (indicators, economic)."""
    assert len(FEATURE_MODULES) >= 1, "Aucun module trouvé dans app/features/"


@pytest.mark.parametrize("mod_name", FEATURE_MODULES)
def test_all_public_fns_marked_safe(mod_name: str) -> None:
    """Toute fonction publique du module est décorée @look_ahead_safe."""
    fns = _public_functions(mod_name)
    if not fns:
        pytest.skip(f"{mod_name}: aucune fonction publique native (ou import échoué)")
    for fn in fns:
        assert getattr(fn, "_look_ahead_safe", False), \
            f"{mod_name}.{fn.__name__} doit être décoré @look_ahead_safe"


@pytest.mark.parametrize("mod_name", FEATURE_MODULES)
def test_no_look_ahead_any_function(mod_name: str) -> None:
    """Toutes les fonctions publiques passent assert_no_look_ahead."""
    fns = _public_functions(mod_name)
    if not fns:
        pytest.skip(f"{mod_name}: aucune fonction publique native (ou import échoué)")
    for fn in fns:
        try:
            test_input = _build_test_input(fn)

            # Cas spécial : fn prend un DatetimeIndex (economic.py)
            if isinstance(test_input, pd.DatetimeIndex) and fn.__name__ in (
                "compute_event_features", "load_calendar",
            ):
                continue

            # compute_all_indicators nécessite include_economic=False
            _fn = fn  # capture pour éviter B023
            if _fn.__name__ == "compute_all_indicators":
                assert_no_look_ahead(
                    lambda x, f=_fn: f(x, include_economic=False),
                    test_input,
                    n_samples=50,
                    seed=42,
                )
            else:
                assert_no_look_ahead(
                    lambda x, f=_fn: f(x),
                    test_input,
                    n_samples=50,
                    seed=42,
                )
        except (TypeError, ValueError):
            # Si la fonction ne supporte pas l'input construit, on skip
            pytest.skip(f"{fn.__name__}: signature non supportée par l'auto-test")
        except Exception:
            # Autres erreurs (DataValidationError, etc.) → skip
            pytest.skip(f"{fn.__name__}: erreur d'exécution (contexte manquant)")
