"""Pivot v4 B2 — H_new3 : EURUSD H4 mean-reversion + meta-labeling.

⚠️ Consomme 1 n_trial. Lecture OOS test set ≥ 2024 = unique.
Pipeline ML FROZEN (A9) — RF(n=100, d=6, leaf=10, seuil=0.55).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.edge_validation import validate_edge  # noqa: E402
from app.backtest.metrics import compute_metrics  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.indicators import adx, atr, rsi  # noqa: E402
from app.pipelines.walk_forward import walk_forward_meta  # noqa: E402
from app.strategies.mean_reversion import MeanReversionRSIBB  # noqa: E402
from app.testing.snooping_guard import read_oos  # noqa: E402


def build_features_h4(df: pd.DataFrame) -> pd.DataFrame:
    """Features méta-labeling pour mean-reversion H4.

    Les signatures réelles des indicateurs sont utilisées :
    - atr(high, low, close, period) → Series
    - adx(high, low, close, period) → DataFrame["adx_line"]
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    atr14 = atr(high, low, close, 14)
    adx14 = adx(high, low, close, 14)["adx_line"]
    sma50 = close.rolling(50).mean()

    out = pd.DataFrame(
        {
            "RSI_14": rsi(close, 14),
            "ADX_14": adx14,
            "Dist_SMA_50": (close - sma50) / atr14,
            "ATR_Norm_14": atr14 / close,
            "Log_Return_5": np.log(close / close.shift(5)),
            "BB_Width": (close.rolling(20).std() * 4) / close,
            # Features session H4
            "Hour_UTC": pd.Series(
                df.index.hour, index=df.index, dtype=float
            ),
            "Is_London_NY_Overlap": pd.Series(
                ((df.index.hour >= 13) & (df.index.hour < 17)).astype(int),
                index=df.index,
                dtype=float,
            ),
        },
        index=df.index,
    )
    return out.dropna()


def build_target_winner(
    _df: pd.DataFrame, pnl_brut: pd.Series
) -> pd.Series:
    """Meta-label binaire : 1 = winner (Pips_Nets > 0), 0 = loser."""
    return (pnl_brut > 0).astype(int)


def main() -> int:
    set_global_seeds()
    df = load_asset("EURUSD", "H4")
    cfg = ASSET_CONFIGS["EURUSD"]

    strat = MeanReversionRSIBB(
        rsi_period=14,
        rsi_long=30,
        rsi_short=70,
        bb_period=20,
        bb_mult=2.0,
    )

    all_trades_oos, segments = walk_forward_meta(
        df=df,
        strat=strat,
        cfg=cfg,
        feature_builder=build_features_h4,
        target_builder=build_target_winner,
        retrain_months=6,
        test_start="2024-01-01",
        capital_eur=10_000.0,
    )

    metrics = compute_metrics(
        all_trades_oos, asset_cfg=cfg, capital_eur=10_000.0
    )

    equity = (
        10_000
        + (
            all_trades_oos["Pips_Nets"]
            * all_trades_oos["position_size_lots"]
            * cfg.pip_value_eur
        ).cumsum()
    )

    # n_trials_cumul = 26 (25 hérités v1-JOURNAL + 1 B2)
    report = validate_edge(
        equity=equity,
        trades=all_trades_oos,
        n_trials=26,
    )

    read_oos(
        prompt="pivot_v4_B2",
        hypothesis="H_new3_eurusd_h4_meanrev",
        sharpe=metrics["sharpe"],
        n_trades=int(metrics["trades"]),
    )

    out = {
        "config": {
            "strat": str(strat),
            "asset": "EURUSD",
            "tf": "H4",
            "retrain_months": 6,
            "capital_eur": 10_000.0,
            "risk_per_trade": 0.02,
            "features": list(build_features_h4(df.head(300)).columns),
        },
        "metrics_walk_forward_oos": metrics,
        "segments": [
            {
                "start": str(s.start.date()),
                "end": str(s.end.date()),
                "n_train": s.n_train,
                "n_oos_trades": s.n_oos_trades,
                "sharpe_oos": s.sharpe_oos,
                "wr_oos": s.wr_oos,
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
    out_path = Path("predictions/h_new3_eurusd_h4.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"H_new3 terminé. Verdict : {'GO' if report.go else 'NO-GO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
