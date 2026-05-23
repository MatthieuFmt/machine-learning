"""Phase H3 — Étape 2 cascade : NR7 + ML méta-labeling sur US500 D1.

Compare la performance de :
    - Baseline : Étape 1 (NR7 déterministe, tous les setups)
    - ML méta-labeling : filtre take/skip basé sur features macro + US500

Critère GO Étape 3 (LLM) : Sharpe_ML_OOS > Sharpe_baseline_OOS et n_trades >= 30.
Si validate_edge formel passe sur ML : sleeve documentable dans strategies-doc/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.metrics import sharpe_daily_from_trades  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.macro_external import build_macro_dataframe  # noqa: E402
from app.strategies.nr7_meta import (  # noqa: E402
    FEATURE_NAMES,
    build_features_at_entry,
    cv_select_threshold,
    filter_trades,
    train_meta_model,
)
from app.strategies.volatility_breakout import simulate_nr_breakout_trades  # noqa: E402

from scripts.run_validation_finale import TEST_START, TRAIN_CUTOFF  # noqa: E402

logger = get_logger(__name__)

ASSET = "US500"
TF = "D1"
LOOKBACK = 7
TP_MULT = 2.0
SL_MULT = 1.0


def _bootstrap_pvalue(pnls: np.ndarray, n_iter: int = 10_000, seed: int = 42) -> float:
    if len(pnls) < 5:
        return 1.0
    rng = np.random.default_rng(seed)
    centered = pnls - pnls.mean()
    observed = pnls.mean()
    if observed <= 0:
        return 1.0
    n = len(pnls)
    boots = np.array([
        rng.choice(centered, size=n, replace=True).mean() for _ in range(n_iter)
    ])
    return float((boots >= observed).mean())


def _analyze(trades: list[dict], label: str) -> dict[str, Any]:
    if not trades:
        return {
            "label": label, "n_trades": 0,
            "sharpe": 0.0, "mean_pnl": 0.0, "wr": 0.0,
            "total_pnl": 0.0, "max_dd_pips": 0.0, "p_value_bootstrap": 1.0,
        }
    pnls = np.array([t["pips_net"] for t in trades])
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    return {
        "label": label,
        "n_trades": len(trades),
        "sharpe": float(sharpe_daily_from_trades(trades)),
        "mean_pnl": float(pnls.mean()),
        "wr": float((pnls > 0).mean()),
        "total_pnl": float(pnls.sum()),
        "max_dd_pips": float((equity - peak).min()),
        "p_value_bootstrap": _bootstrap_pvalue(pnls),
    }


def _print(m: dict[str, Any]) -> None:
    print(f"  {m['label']}: n={m['n_trades']}, "
          f"Sharpe={m['sharpe']:+.2f}, WR={m['wr']:.1%}, "
          f"mean={m['mean_pnl']:+.1f} pips, total={m['total_pnl']:+.0f} pips, "
          f"max_dd={m['max_dd_pips']:.0f}, p={m['p_value_bootstrap']:.3f}")


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print(f"PHASE H3 — Étape 2 cascade : NR7 + ML méta-labeling ({ASSET} {TF})")
    print("=" * 70)

    cfg = ASSET_CONFIGS[ASSET]
    df = load_asset(ASSET, TF)
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]
    df_macro = build_macro_dataframe(refresh=False)

    print(f"\nData {ASSET} {TF} : {len(df)} bars total")
    print(f"  Train : {len(df_train)} bars (≤ {TRAIN_CUTOFF.date()})")
    print(f"  Test  : {len(df_test)} bars (≥ {TEST_START.date()})")
    print(f"  Macro : {len(df_macro)} bars ({df_macro.index.min().date()} → "
          f"{df_macro.index.max().date()})")

    # ── Baseline NR7 (Étape 1) ───────────────────────────────────────
    print("\n── Baseline NR7 (rappel Étape 1) ──")
    trades_train_base = simulate_nr_breakout_trades(
        df_train, cfg, lookback=LOOKBACK, tp_mult=TP_MULT, sl_mult=SL_MULT,
    )
    trades_test_base = simulate_nr_breakout_trades(
        df_test, cfg, lookback=LOOKBACK, tp_mult=TP_MULT, sl_mult=SL_MULT,
    )
    m_base_train = _analyze(trades_train_base, "Train baseline")
    m_base_test = _analyze(trades_test_base, "Test  baseline")
    _print(m_base_train)
    _print(m_base_test)

    # ── Features ────────────────────────────────────────────────────
    print("\n── Construction des features ──")
    X_train = build_features_at_entry(df_train, df_macro, trades_train_base)
    X_test = build_features_at_entry(df_test, df_macro, trades_test_base)
    print(f"  X_train : {X_train.shape}, NaN total = {X_train.isna().sum().sum()}")
    print(f"  X_test  : {X_test.shape}, NaN total = {X_test.isna().sum().sum()}")
    if X_train.isna().any().any():
        print("  NaN par colonne (train) :")
        for c, n in X_train.isna().sum().items():
            if n > 0:
                print(f"    {c}: {n}")

    # ── Labels train ─────────────────────────────────────────────────
    train_df = pd.DataFrame(trades_train_base)
    train_df["setup_date_ts"] = pd.to_datetime(train_df["setup_date"]).dt.tz_localize("UTC").dt.normalize()
    train_df = train_df.set_index("setup_date_ts")

    common_train = X_train.index.intersection(train_df.index)
    X_train_aligned = X_train.loc[common_train]
    pnls_train = train_df.loc[common_train, "pips_net"]
    y_train = (pnls_train > 0).astype(int)

    valid_mask = X_train_aligned.notna().all(axis=1)
    X_train_clean = X_train_aligned[valid_mask]
    y_train_clean = y_train[valid_mask]
    pnls_train_clean = pnls_train[valid_mask]
    print(f"  Train ML utilisable : {len(X_train_clean)} (sur {len(X_train_aligned)})")
    if len(X_train_clean) > 0:
        print(f"  Class balance       : {y_train_clean.mean():.1%} winners")

    if len(X_train_clean) < 50:
        print("  ⚠️ Sample train ML trop petit (<50). Abandon ML.")
        return 1

    # ── CV pour choisir le threshold ─────────────────────────────────
    print("\n── CV pour sélection du threshold (5-fold StratifiedKFold) ──")
    best_thresh, thresh_stats = cv_select_threshold(
        X_train_clean, y_train_clean, pnls_train_clean,
        n_splits=5,
    )
    print(f"  Sharpe CV moyen par threshold :")
    for t, s in thresh_stats["per_threshold"].items():
        marker = " ←" if t == best_thresh else ""
        print(f"    P ≥ {t:.2f} → Sharpe CV {s:+.2f}{marker}")
    print(f"  Best threshold : {best_thresh:.2f}")

    # ── Train modèle final ──────────────────────────────────────────
    print("\n── Train modèle ML final sur l'ensemble du train clean ──")
    model = train_meta_model(X_train_clean, y_train_clean)
    print(f"  Modèle entraîné sur n={len(X_train_clean)} setups")

    # ── Re-prédiction train (sanity check) ──────────────────────────
    trades_train_ml, probas_train = filter_trades(
        trades_train_base, X_train_aligned, model, best_thresh,
    )
    m_ml_train = _analyze(trades_train_ml, "Train ML")
    _print(m_ml_train)

    # ── Prédiction OOS ──────────────────────────────────────────────
    print("\n── Prédiction OOS ──")
    trades_test_ml, probas_test = filter_trades(
        trades_test_base, X_test, model, best_thresh,
    )
    m_ml_test = _analyze(trades_test_ml, "Test  ML")
    _print(m_ml_test)

    # ── Verdict ─────────────────────────────────────────────────────
    delta_sharpe = m_ml_test["sharpe"] - m_base_test["sharpe"]
    delta_mean = m_ml_test["mean_pnl"] - m_base_test["mean_pnl"]
    filter_ratio = (m_ml_test["n_trades"] / m_base_test["n_trades"]
                    if m_base_test["n_trades"] > 0 else 0.0)

    print("\n" + "=" * 70)
    print("VERDICT Étape 2 — apport ML méta-labeling")
    print("=" * 70)
    print(f"  Baseline Sharpe OOS : {m_base_test['sharpe']:+.2f} ({m_base_test['n_trades']} trades)")
    print(f"  ML      Sharpe OOS : {m_ml_test['sharpe']:+.2f} ({m_ml_test['n_trades']} trades)")
    print(f"  Δ Sharpe            : {delta_sharpe:+.2f}")
    print(f"  Δ Mean PnL          : {delta_mean:+.1f} pips")
    print(f"  Ratio filtre OOS    : {filter_ratio:.1%} des trades baseline retenus")

    go_apport = delta_sharpe > 0
    go_n_trades = m_ml_test["n_trades"] >= 30
    go_noise = m_ml_test["sharpe"] > 1.53  # noise floor pour n_trials=62
    go = go_apport and go_n_trades

    print(f"\n  Δ Sharpe > 0         : {'OK' if go_apport else 'FAIL'}")
    print(f"  n trades ≥ 30        : {'OK' if go_n_trades else 'FAIL'} ({m_ml_test['n_trades']})")
    print(f"  Sharpe > noise floor : {'OK' if go_noise else 'WATCH'} (vs SR₀=1.53)")
    print(f"\n  ==> {'GO Étape 3 (LLM) / validate_edge' if go else 'NO-GO ML — baseline reste candidate'}")

    # ── Save ─────────────────────────────────────────────────────────
    out = {
        "strategy": "nr7_us500_ml",
        "asset": ASSET, "tf": TF,
        "lookback": LOOKBACK, "tp_mult": TP_MULT, "sl_mult": SL_MULT,
        "best_threshold": best_thresh,
        "threshold_stats": thresh_stats["per_threshold"],
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "class_balance_train": float(y_train_clean.mean()),
        "n_train_clean": int(len(X_train_clean)),
        "n_test": int(len(X_test)),
        "metrics_train_baseline": m_base_train,
        "metrics_test_baseline": m_base_test,
        "metrics_train_ml": m_ml_train,
        "metrics_test_ml": m_ml_test,
        "delta_sharpe": delta_sharpe,
        "delta_mean_pnl": delta_mean,
        "filter_ratio": filter_ratio,
        "go": go,
        "go_breakdown": {
            "delta_sharpe_positive": go_apport,
            "n_trades_sufficient": go_n_trades,
            "above_noise_floor": go_noise,
        },
        "probas_test": [float(p) for p in probas_test.tolist()],
    }
    out_json = Path("predictions/h3_nr7_us500_ml.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardé : {out_json}")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
