"""Diagnostic Donchian TP/SL — pourquoi WR=9% sur GBPUSD D1 ?

Hypothèse : SL=10 pips est trop serré pour D1 forex (range moyen 60-80 pips).
Le bruit intra-day touche le SL juste après l'entrée breakout.

Ce script lit UNIQUEMENT train ≤ 2022 → ZÉRO n_trial consommé. C'est de
l'analyse structurelle, pas une nouvelle hypothèse OOS.

Pour chaque couple :
1. WR Donchian sur train (12 ans) — était-il déjà mauvais ?
2. Distribution win / loss_sl / loss_timeout — quel type de perte domine ?
3. ATR moyen et range D1 moyen — quelle taille de stop serait réaliste ?
4. Ratio SL/ATR — un SL < 0.5×ATR est mécaniquement touché ~80% du temps.
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

from app.backtest.deterministic import run_deterministic_backtest  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.indicators import atr  # noqa: E402

from scripts.run_validation_finale import (  # noqa: E402
    TRAIN_CUTOFF,
    _generate_donchian_signals,
)

COUPLES: list[dict[str, Any]] = [
    # Forex (les 3 couples du portfolio "GO")
    {"asset": "GBPUSD", "tf": "D1"},
    {"asset": "EURUSD", "tf": "D1"},
    {"asset": "USDCHF", "tf": "D1"},
    # Crypto (le seul SL/ATR sain initialement)
    {"asset": "ETHUSD", "tf": "D1"},
    {"asset": "BTCUSD", "tf": "D1"},
    # Indices — VÉRIFICATION DU RÉSULTAT FONDATEUR H03 (US30 Donchian D1)
    {"asset": "US30", "tf": "D1"},
    {"asset": "US500", "tf": "D1"},
    {"asset": "GER30", "tf": "D1"},
    # Métaux
    {"asset": "XAUUSD", "tf": "D1"},
]


def diagnose_couple(asset: str, tf: str) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[TP/SL] {asset} {tf}")
    print(f"{'='*60}")

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # 1. Donchian sur train (12 ans, gratuit en n_trials)
    donchian_train = _generate_donchian_signals(df_train)
    bt = run_deterministic_backtest(
        df=df_train, signals=donchian_train,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size,
    )
    trades = bt.get("trades", [])
    if not trades:
        return {"asset": asset, "tf": tf, "skipped": "0 trades"}

    pnls = np.array([t["pips_net"] for t in trades])
    results = [t["result"] for t in trades]
    n_win = int(sum(1 for r in results if r == "win"))
    n_sl = int(sum(1 for r in results if r == "loss_sl"))
    n_timeout = int(sum(1 for r in results if r == "loss_timeout"))
    wr_train = float((pnls > 0).mean())

    print(f"  Trades train: {len(trades)} (12 ans Donchian D1)")
    print(f"  WR train: {wr_train:.1%}")
    print(f"  Distribution résultats:")
    print(f"    win        : {n_win:4d} ({n_win/len(trades):.1%})")
    print(f"    loss_sl    : {n_sl:4d} ({n_sl/len(trades):.1%})")
    print(f"    loss_timeout: {n_timeout:4d} ({n_timeout/len(trades):.1%})")

    # 2. ATR et range moyen sur train
    atr14 = atr(df_train["High"], df_train["Low"], df_train["Close"], 14)
    atr_pips = atr14 / cfg.pip_size  # en pips
    range_pips = (df_train["High"] - df_train["Low"]) / cfg.pip_size

    atr_mean = float(atr_pips.dropna().mean())
    atr_median = float(atr_pips.dropna().median())
    range_mean = float(range_pips.mean())

    sl_to_atr = cfg.sl_points / atr_mean if atr_mean > 0 else float("inf")

    print(f"  ATR moyen (14, pips): {atr_mean:.1f} (median {atr_median:.1f})")
    print(f"  Range D1 moyen (pips): {range_mean:.1f}")
    print(f"  SL config: {cfg.sl_points} pips, TP: {cfg.tp_points} pips")
    print(f"  Ratio SL/ATR: {sl_to_atr:.2f} (sain ≥ 0.5, dangereux < 0.3)")

    if sl_to_atr < 0.3:
        print(f"  🔴 SL TROP SERRÉ — touché trivialement par le bruit intra-day")

    # 3. Suggestions
    suggested_sl_atr = max(int(atr_mean * 0.7), 10)
    suggested_tp_atr = suggested_sl_atr * 2  # ratio 2:1 conservé
    print(f"  💡 Suggestion : SL≈{suggested_sl_atr}, TP≈{suggested_tp_atr} pips (0.7×ATR / 2×SL)")

    return {
        "asset": asset, "tf": tf,
        "n_trades_train": len(trades),
        "wr_train": wr_train,
        "win": n_win, "loss_sl": n_sl, "loss_timeout": n_timeout,
        "atr_mean_pips": atr_mean,
        "atr_median_pips": atr_median,
        "range_mean_pips": range_mean,
        "sl_config_pips": cfg.sl_points,
        "tp_config_pips": cfg.tp_points,
        "sl_to_atr_ratio": sl_to_atr,
        "suggested_sl_pips": suggested_sl_atr,
        "suggested_tp_pips": suggested_tp_atr,
        "verdict_sl_too_tight": sl_to_atr < 0.3,
    }


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print("DIAGNOSTIC DONCHIAN TP/SL — train ≤ 2022 (0 n_trial consommé)")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    for c in COUPLES:
        try:
            r = diagnose_couple(c["asset"], c["tf"])
            results.append(r)
        except Exception as exc:
            print(f"  ❌ {c['asset']} {c['tf']}: {exc}")
            results.append({"asset": c["asset"], "tf": c["tf"], "error": str(exc)})

    # Récap
    print("\n" + "=" * 70)
    print("RÉCAP — WR train Donchian + ratio SL/ATR")
    print("=" * 70)
    print(f"{'Couple':<12} {'WR train':>10} {'%SL':>6} {'%TO':>6} {'ATR':>8} {'SL':>6} {'SL/ATR':>8} {'verdict':>10}")
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        couple = f"{r['asset']}_{r['tf']}"
        wr_str = f"{r['wr_train']:.1%}"
        sl_pct = f"{r['loss_sl']/r['n_trades_train']:.0%}"
        to_pct = f"{r['loss_timeout']/r['n_trades_train']:.0%}"
        verdict = "🔴 trop serré" if r["verdict_sl_too_tight"] else "ok"
        print(f"{couple:<12} {wr_str:>10} {sl_pct:>6} {to_pct:>6} "
              f"{r['atr_mean_pips']:>8.1f} {r['sl_config_pips']:>6.0f} "
              f"{r['sl_to_atr_ratio']:>8.2f} {verdict:>10}")

    out_json = Path("predictions/diagnose_donchian_tp_sl.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardé : {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
