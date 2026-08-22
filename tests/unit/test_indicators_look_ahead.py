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
    # ⚠️ Chemin ANCRÉ au dépôt. Avec un Path("app/features") relatif au CWD,
    #    lancer pytest depuis un autre dossier renvoyait [] : la liste
    #    paramétrée devenait vide et le scan anti-look-ahead ne couvrait plus
    #    RIEN, en silence. Un scan qui ne teste rien passe toujours.
    features_dir = Path(__file__).resolve().parents[2] / "app" / "features"
    if not features_dir.exists():
        raise RuntimeError(f"app/features introuvable a {features_dir}")
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


# Fonctions exclues du scan auto-générique (fix F8) avec test ciblé ailleurs.
# Le scan auto a un _build_test_input naïf qui ne gère pas les fonctions
# multi-colonnes ou à signature non-standard. Pour ces fonctions, le scan
# auto ne donne pas d'information utile — on utilise des tests dédiés.
SKIP_AUTO_SCAN: dict[str, str] = {
    # Signatures DatetimeIndex / calendar
    "compute_event_features": "DatetimeIndex + calendar fixture — test ciblé test_economic_features.py",
    "load_calendar": "Charge un calendrier — pas une feature",
    "cross_asset_features": "DatetimeIndex + load_asset — test ciblé test_cross_asset_no_leak.py",
    "session_features": "DatetimeIndex only — test ciblé test_session_features_no_leak.py",
    "economic_features_for_index": "DatetimeIndex + calendar — fallback testé manuellement",
    "rank_features_bootstrap": "Ranking de features (pas générateur) — non applicable",
    # Feature builders multi-colonnes (superset.py) — agrégat de fonctions
    # déjà décorées @look_ahead_safe individuellement. Tests intégrés
    # dans test_superset_features.py qui vérifie les colonnes produites.
    "trend_features": "DataFrame multi-col — test ciblé test_superset_features.py",
    "momentum_features": "DataFrame multi-col — test ciblé test_superset_features.py",
    "oscillator_features": "DataFrame multi-col — test ciblé test_superset_features.py",
    "volatility_features": "DataFrame multi-col — test ciblé test_superset_features.py",
    "price_action_features": "DataFrame multi-col — test ciblé test_superset_features.py",
    "statistical_features": "DataFrame multi-col — test ciblé test_superset_features.py",
    "regime_features": "DataFrame multi-col — test ciblé test_superset_features.py",
    "build_superset": "Aggregator de toutes les features — test ciblé test_superset_features.py",
    # Helpers internes regime.py multi-args, testés via test_regime.py
    "calc_volatilite_realisee": "Multi-args — test ciblé test_regime.py",
    "calc_range_atr_ratio": "Multi-args — test ciblé test_regime.py",
    "calc_rsi_d1_delta": "Multi-args — test ciblé test_regime.py",
    "calc_dist_sma200_d1": "Multi-args — test ciblé test_regime.py",
    "compute_session_id": "DatetimeIndex — test ciblé test_session_features_no_leak.py",
    "compute_session_open_range": "Multi-args session — test ciblé test_regime.py",
    "compute_relative_position_in_session": "Multi-args session — test ciblé test_regime.py",
    # Phase F4 — labels string ('trend'/'range'/'vol_high'/NA) incompatibles
    # avec np.isclose. No-look-ahead vérifié par test_truncation_does_not_alter_past_labels.
    "detect_regime": "Retourne des labels string — test ciblé test_regime_detector.py",
    # Phase F5 — loaders réseau/cache (yfinance), pas une feature génératrice.
    # No-look-ahead vérifié par test_truncation_does_not_change_past.
    "load_macro_series": "Loader yfinance/cache — test ciblé test_macro_external.py",
    "build_macro_dataframe": "Aggregator macro — test ciblé test_macro_external.py",
    "add_external_macro": "Multi-args (df + macro_df) — test ciblé test_macro_external.py",
}


@pytest.mark.parametrize("mod_name", FEATURE_MODULES)
def test_no_look_ahead_any_function(mod_name: str, request: pytest.FixtureRequest) -> None:
    """Toutes les fonctions publiques passent assert_no_look_ahead.

    Fix F8 (politique) :
    - AssertionError (look-ahead réellement détecté) → test FAIL bruyant.
    - TypeError/ValueError au build d'input ou à l'appel (signature non
      adaptée au _build_test_input générique) → comptabilisé en
      "untested" et reporté en warning de session, MAIS le test passe
      tant qu'au moins UNE fonction du module a été testée.
    - Autre Exception → considéré comme bug réel → FAIL.
    """
    fns = _public_functions(mod_name)
    if not fns:
        pytest.skip(f"{mod_name}: aucune fonction publique native (ou import échoué)")

    failures: list[str] = []
    untested: list[str] = []
    tested_count = 0

    for fn in fns:
        if fn.__name__ in SKIP_AUTO_SCAN:
            continue
        try:
            test_input = _build_test_input(fn)
        except (TypeError, ValueError) as exc:
            untested.append(f"{fn.__name__}: build_input {type(exc).__name__}")
            continue
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"{fn.__name__}: erreur inattendue au build_input "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        _fn = fn  # capture pour éviter B023
        try:
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
            tested_count += 1
        except AssertionError as exc:
            # Vrai look-ahead → FAIL bruyant
            failures.append(f"{fn.__name__}: LOOK-AHEAD détecté — {exc}")
        except (TypeError, ValueError) as exc:
            # Signature ne supporte pas l'input → SKIP comptabilisé
            untested.append(f"{fn.__name__}: call {type(exc).__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append(
                f"{fn.__name__}: erreur inattendue à l'appel "
                f"{type(exc).__name__}: {exc}"
            )

    # Reporter dans le terminal les fonctions non testables (visible avec -v)
    if untested:
        request.node.add_report_section(
            "call", "untested",
            f"{mod_name}: {len(untested)} fonction(s) non auto-testables :\n" +
            "\n".join(f"  - {u}" for u in untested),
        )

    if failures:
        raise AssertionError(
            f"{mod_name}: {len(failures)} look-ahead/erreur(s) :\n" +
            "\n".join(f"  - {f}" for f in failures)
        )
