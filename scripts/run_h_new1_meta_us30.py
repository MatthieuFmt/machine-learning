"""Pivot v4 B1 — H_new1 : méta-labeling RF sur Donchian US30 D1.

⚠️ Consomme 1 n_trial. Lecture OOS test set ≥ 2024 = unique.
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

from app.analysis.edge_validation import validate_edge  # noqa: E402
from app.backtest.metrics import compute_metrics  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.indicators import adx, atr, rsi  # noqa: E402
from app.pipelines.walk_forward import walk_forward_meta  # noqa: E402
from app.strategies.donchian import DonchianBreakout  # noqa: E402
from app.testing.snooping_guard import read_oos  # noqa: E402


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les 6 features RF pour méta-labeling (spécification B1).

    Features : RSI_14, ADX_14, Dist_SMA_50, Dist_SMA_200, ATR_Norm_14, Log_Return_5.
    Plus Signal_Donchian ajouté par le script principal.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    atr14 = atr(high, low, close, 14)

    # adx() retourne un DataFrame, on extrait la ligne ADX
    adx_df = adx(high, low, close, 14)
    adx_line = adx_df["adx_line"]

    out = pd.DataFrame({
        "RSI_14": rsi(close, 14),
        "ADX_14": adx_line,
        "Dist_SMA_50": (close - sma50) / atr14.replace(0, np.nan),
        "Dist_SMA_200": (close - sma200) / atr14.replace(0, np.nan),
        "ATR_Norm_14": atr14 / close.replace(0, np.nan),
        "Log_Return_5": np.log(close / close.shift(5)),
    }, index=df.index)
    return out.dropna()


def build_target(df: pd.DataFrame, pnl_brut: pd.Series) -> pd.Series:
    """Cible binaire : 1 si trade gagnant (pnl > 0), 0 sinon."""
    return (pnl_brut > 0).astype(int)


def main() -> int:
    set_global_seeds()

    # ── 1. Chargement des données ─────────────────────────────────────
    print("Chargement US30 D1...")
    df = load_asset("US30", "D1")
    cfg = ASSET_CONFIGS["US30"]
    print(f"  {len(df)} barres, {df.index.min().date()} -> {df.index.max().date()}")

    # ── 2. Stratégie baseline Donchian(20,20) ─────────────────────────
    strat = DonchianBreakout(N=20, M=20)
    all_signals = strat.generate_signals(df)
    n_long = int((all_signals == 1).sum())
    n_short = int((all_signals == -1).sum())
    print(f"Signaux Donchian(20,20) : {n_long} LONG, {n_short} SHORT, {n_long + n_short} total")

    # ── 3. Features + Signal_Donchian ─────────────────────────────────
    features_base = build_features(df)
    # Aligner signals avec features (features a moins de lignes à cause de dropna)
    common_idx = features_base.index.intersection(all_signals.index)
    features_base = features_base.loc[common_idx]
    all_signals_aligned = all_signals.loc[common_idx]

    features_base["Signal_Donchian"] = all_signals_aligned.astype(int)
    features_base = features_base.dropna()

    print(f"Features : {list(features_base.columns)}")
    print(f"  {len(features_base)} lignes après alignement + dropna")

    # ── 4. Feature builder pour walk_forward (recalcule sur chaque split) ─
    def feature_builder(df_split: pd.DataFrame) -> pd.DataFrame:
        """Recalcule les features sur un split train/OOS."""
        feats = build_features(df_split)
        sigs = strat.generate_signals(df_split)
        common = feats.index.intersection(sigs.index)
        feats = feats.loc[common]
        sigs_aligned = sigs.loc[common]
        feats["Signal_Donchian"] = sigs_aligned.astype(int)
        return feats.dropna()

    # ── 5. Walk-forward méta-labeling sur test set ≥ 2024 ─────────────
    print("\nWalk-forward méta-labeling (retrain 6M, test ≥ 2024)...")
    all_trades_oos, segments = walk_forward_meta(
        df=df,
        strat=strat,
        cfg=cfg,
        feature_builder=feature_builder,
        target_builder=build_target,
        retrain_months=6,
        test_start="2024-01-01",
        capital_eur=10_000.0,
    )

    # ── 6. Métriques OOS ──────────────────────────────────────────────
    metrics: dict[str, Any] = {}
    if all_trades_oos.empty:
        print("⚠️ Aucun trade OOS généré.")
        metrics = {
            "sharpe": 0.0, "trades": 0, "win_rate": 0.0,
            "max_dd_pct": 0.0, "profit_net_eur": 0.0,
            "total_return_pct": 0.0, "sharpe_method": "daily",
        }
    else:
        metrics = compute_metrics(
            all_trades_oos, asset_cfg=cfg, capital_eur=10_000.0, df=df,
        )

    # ── 7. validate_edge ──────────────────────────────────────────────
    equity: pd.Series
    if not all_trades_oos.empty and "pnl" in all_trades_oos.columns:
        equity_eur = all_trades_oos["pnl"].cumsum() + 10_000.0
        equity = equity_eur
    else:
        equity = pd.Series([10_000.0], index=[df.index[0]])

    n_trials_cumul = 23  # 22 hérités + H_new1
    report = validate_edge(
        equity=equity,
        trades=all_trades_oos if not all_trades_oos.empty else pd.DataFrame(columns=["pnl"]),
        n_trials=n_trials_cumul,
    )

    # ── 8. READ_OOS — UNIQUE ──────────────────────────────────────────
    read_oos(
        prompt="pivot_v4_B1",
        hypothesis="H_new1_meta_us30_d1",
        sharpe=float(metrics.get("sharpe", 0.0)),
        n_trades=int(metrics.get("trades", 0)),
    )

    # ── 9. Sauvegarde ─────────────────────────────────────────────────
    out: dict[str, Any] = {
        "hypothesis": "H_new1",
        "instrument": "US30",
        "tf": "D1",
        "strategy": "DonchianBreakout(N=20, M=20) + RF meta-labeling",
        "n_trials_cumul": n_trials_cumul,
        "config": {
            "asset": "US30",
            "tf": "D1",
            "spread_pips": cfg.spread_pips,
            "slippage_pips": cfg.slippage_pips,
            "tp_points": cfg.tp_points,
            "sl_points": cfg.sl_points,
            "retrain_months": 6,
            "capital_eur": 10_000.0,
            "risk_per_trade": 0.02,
            "features": [
                "RSI_14", "ADX_14", "Dist_SMA_50", "Dist_SMA_200",
                "ATR_Norm_14", "Log_Return_5", "Signal_Donchian",
            ],
        },
        "metrics_walk_forward_oos": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, type(None)))},
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
    out_path = Path("predictions/h_new1_meta_us30.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nH_new1 terminé. Verdict : {'GO ✅' if report.go else 'NO-GO ❌'}")
    print(f"Raisons : {report.reasons}")
    print(f"Métriques : {report.metrics}")
    print(f"Résultats sauvegardés : {out_path}")
    return 0 if report.go else 1


if __name__ == "__main__":
    sys.exit(main())
