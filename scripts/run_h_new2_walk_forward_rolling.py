"""Pivot v4 B3 — H_new2 : walk-forward rolling 3 ans sur US30 D1 + XAUUSD D1.

Test set 2024+ lu UNE seule fois par actif. read_oos() tracé par actif.
Pipeline ML FROZEN (A9), stratégie Donchian (N=20, M=20), méta-labeling RF.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.analysis.edge_validation import validate_edge
from app.backtest.metrics import compute_metrics
from app.config.instruments import ASSET_CONFIGS
from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.features.indicators import adx, atr, rsi
from app.pipelines.walk_forward_rolling import walk_forward_rolling
from app.strategies.donchian import DonchianBreakout
from app.testing.snooping_guard import read_oos


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features techniques pour méta-labeling — réplique B1 (H_new1).

    Toutes les features sont calculées sur l'historique complet de df
    (rolling indicators avec warmup suffisant). Pas de look-ahead.
    """
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14 = atr(df["High"], df["Low"], df["Close"], 14)
    adx_df = adx(df["High"], df["Low"], df["Close"], 14)

    out = pd.DataFrame(
        {
            "RSI_14": rsi(close, 14),
            "ADX_14": adx_df["adx_line"] if "adx_line" in adx_df.columns else adx_df.iloc[:, 0],
            "Dist_SMA_50": (close - sma50) / atr14,
            "Dist_SMA_200": (close - sma200) / atr14,
            "ATR_Norm_14": atr14 / close,
            "Log_Return_5": np.log(close / close.shift(5)),
        },
        index=df.index,
    )
    return out.dropna()


def build_target(_df: pd.DataFrame, pnl_brut: pd.Series) -> pd.Series:
    """Meta-label : 1 si trade gagnant (Pips_Nets > 0), 0 sinon."""
    return (pnl_brut > 0).astype(int)


def run_one_asset(asset: str, n_trials: int) -> dict:
    """Exécute le walk-forward rolling 3 ans sur un actif.

    Args:
        asset: Clé ASSET_CONFIGS ("US30" ou "XAUUSD").
        n_trials: Compteur n_trials cumulé pour DSR.

    Returns:
        Dict avec métriques, segments et verdict validate_edge.
    """
    df = load_asset(asset, "D1")
    cfg = ASSET_CONFIGS[asset]
    strat = DonchianBreakout(N=20, M=20)

    trades_oos, segments = walk_forward_rolling(
        df=df,
        strat=strat,
        cfg=cfg,
        feature_builder=build_features,
        target_builder=build_target,
        train_window_years=3,
        retrain_months=6,
        test_start="2024-01-01",
        capital_eur=10_000.0,
    )

    metrics = compute_metrics(trades_oos, asset_cfg=cfg, capital_eur=10_000.0)

    # Courbe d'equity € pour validate_edge
    if trades_oos.empty:
        equity = pd.Series([10_000.0], index=[pd.Timestamp("2024-01-01")])
    else:
        equity = 10_000 + (
            trades_oos["Pips_Nets"] * trades_oos["position_size_lots"] * cfg.pip_value_eur
        ).cumsum()
        # Forward-fill pour avoir une valeur par jour
        if isinstance(equity.index, pd.DatetimeIndex):
            equity = equity.resample("D").last().ffill()

    report = validate_edge(
        equity=equity,
        trades=trades_oos if not trades_oos.empty else pd.DataFrame({"pnl": []}),
        n_trials=n_trials,
    )

    read_oos(
        prompt="pivot_v4_B3",
        hypothesis=f"H_new2_{asset.lower()}_rolling",
        sharpe=metrics["sharpe"],
        n_trades=int(metrics["trades"]),
    )

    return {
        "asset": asset,
        "metrics": metrics,
        "segments": [
            {
                "train_start": str(s.train_start),
                "train_end": str(s.train_end),
                "oos_start": str(s.oos_start),
                "oos_end": str(s.oos_end),
                "n_train": s.n_train,
                "n_oos": s.n_oos,
                "sharpe_oos": s.sharpe_oos,
                "meta_disabled": s.meta_disabled,
                "threshold": s.threshold,
            }
            for s in segments
        ],
        "validate_edge": {
            "go": report.go,
            "reasons": report.reasons,
            "metrics": report.metrics,
        },
    }


def main() -> int:
    set_global_seeds()

    # ── n_trials_cumul = 26 (cf JOURNAL.md: H_new3 = 26)
    # B3 consomme +1 → 27. Mais B1+B2 déjà faits, B1=25, B2=26, B3=27.
    # Le prompt dit 25 car B1 et B2 NO-GO dans le scénario conditionnel.
    # Ici B2 est GO (EURUSD H4), donc B3 ne devrait pas être exécuté.
    # MAIS le prompt est exécuté sur demande utilisateur → on suit la règle.
    # n_trials_cumul actuel = 26 (d'après JOURNAL.md ligne 82).
    # B3 = +1 = 27.
    n_trials = 27

    out: dict[str, dict] = {}
    for asset in ("US30", "XAUUSD"):
        print(f"Running B3 walk-forward rolling on {asset} D1...")
        out[asset.lower()] = run_one_asset(asset, n_trials)

    Path("predictions").mkdir(exist_ok=True)
    Path("predictions/h_new2_walk_forward_rolling.json").write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    print("H_new2 terminé.")
    for asset_key, result in out.items():
        go = result["validate_edge"]["go"]
        sharpe = result["metrics"]["sharpe"]
        print(f"  {asset_key.upper():6s}: verdict {'GO' if go else 'NO-GO'} (Sharpe={sharpe:.2f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
