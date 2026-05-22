"""Balayage TP/SL ATR-based sur train â‰¤ 2022 â€” 0 n_trial consommÃ©.

Pour chaque couple, teste 4 ratios SL/ATR âˆˆ {0.5, 0.7, 1.0, 1.5} :
1. Calcule ATR moyen sur train.
2. SL = ratio Ã— ATR, TP = 2 Ã— SL (ratio TP/SL = 2:1 conservÃ©).
3. Backtest Donchian avec ces TP/SL ajustÃ©s.
4. MÃ©triques : Sharpe linÃ©aire, WR, n_trades, mean_pnl, max DD pips.

Question centrale : est-ce qu'un SL/ATR sain (â‰¥0.5) restaure l'edge Donchian
sur train ? Si oui â†’ la stratÃ©gie peut Ãªtre viable avec re-calibration.
Si non â†’ Donchian D1 ne marche pas, mÃªme avec stop adaptÃ©.

Aucune lecture OOS â‰¥ 2024. Aucun n_trial consommÃ©.
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
from app.backtest.metrics import sharpe_daily_from_trades  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.indicators import atr  # noqa: E402

from scripts.run_validation_finale import (  # noqa: E402
    TRAIN_CUTOFF,
    _generate_donchian_signals,
)

COUPLES: list[str] = [
    "GBPUSD", "EURUSD", "USDCHF", "ETHUSD", "BTCUSD", "US30", "US500", "GER30", "XAUUSD",
]
TF = "D1"
RATIOS = [0.5, 0.7, 1.0, 1.5]  # SL = ratio Ã— ATR
TP_OVER_SL = 2.0  # ratio TP/SL conservÃ© Ã  2:1


def _analyze_trades(trades: list[dict], capital_pips: float = 10_000.0) -> dict[str, float]:
    if not trades:
        return {"sharpe": 0.0, "wr": 0.0, "n_trades": 0, "mean_pnl": 0.0, "max_dd_pips": 0.0}
    pnls = np.array([t["pips_net"] for t in trades])
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    return {
        "sharpe": float(sharpe_daily_from_trades(trades, initial_capital_pips=capital_pips)),
        "wr": float((pnls > 0).mean()),
        "n_trades": int(len(trades)),
        "mean_pnl": float(pnls.mean()),
        "max_dd_pips": float((equity - peak).min()),
    }


def grid_one_couple(asset: str) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Grid] {asset} {TF}")
    print(f"{'='*60}")

    try:
        df = load_asset(asset, TF)
    except Exception as exc:
        print(f"  âŒ skip : {exc}")
        return {"asset": asset, "tf": TF, "error": str(exc)}

    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # ATR moyen en pips
    atr14 = atr(df_train["High"], df_train["Low"], df_train["Close"], 14)
    atr_pips = atr14 / cfg.pip_size
    atr_mean = float(atr_pips.dropna().mean())

    print(f"  ATR moyen train : {atr_mean:.1f} pips")
    print(f"  Baseline (config actuelle) : TP={cfg.tp_points}, SL={cfg.sl_points}, SL/ATR={cfg.sl_points/atr_mean:.2f}")

    donchian_train = _generate_donchian_signals(df_train)

    # Baseline (config actuelle)
    bt_base = run_deterministic_backtest(
        df=df_train, signals=donchian_train,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    m_base = _analyze_trades(bt_base.get("trades", []))
    m_base["sl_pips"] = cfg.sl_points
    m_base["tp_pips"] = cfg.tp_points
    m_base["sl_atr_ratio"] = cfg.sl_points / atr_mean if atr_mean > 0 else 0.0

    print(f"  Baseline : n={m_base['n_trades']}, WR={m_base['wr']:.1%}, Sharpe={m_base['sharpe']:.2f}")

    # Grid sur ratios
    grid: list[dict[str, Any]] = []
    for ratio in RATIOS:
        sl_pips = max(round(ratio * atr_mean), 1)
        tp_pips = max(round(sl_pips * TP_OVER_SL), 1)
        bt = run_deterministic_backtest(
            df=df_train, signals=donchian_train,
            tp_pips=tp_pips, sl_pips=sl_pips,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
        )
        m = _analyze_trades(bt.get("trades", []))
        m["sl_atr_ratio"] = ratio
        m["sl_pips"] = sl_pips
        m["tp_pips"] = tp_pips
        grid.append(m)
        print(f"  SL={ratio:.2f}Ã—ATR ({sl_pips} pips), TP={tp_pips} pips : "
              f"n={m['n_trades']}, WR={m['wr']:.1%}, Sharpe={m['sharpe']:.2f}, "
              f"mean_pnl={m['mean_pnl']:.1f}")

    return {
        "asset": asset, "tf": TF,
        "atr_mean_pips": atr_mean,
        "baseline": m_base,
        "grid": grid,
    }


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print("GRID TP/SL ATR-BASED sur train â‰¤ 2022 (0 n_trial consommÃ©)")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    for asset in COUPLES:
        try:
            r = grid_one_couple(asset)
            results.append(r)
        except Exception as exc:
            print(f"  âŒ {asset} : {exc}")
            results.append({"asset": asset, "error": str(exc)})

    # RÃ©cap : pour chaque couple, le meilleur ratio
    print("\n" + "=" * 90)
    print("RÃ‰CAP â€” Donchian D1 train : WR/Sharpe en fonction du ratio SL/ATR")
    print("=" * 90)
    header = f"{'Couple':<10} | {'ATR':>6} | {'Baseline':>16}"
    for r in RATIOS:
        header += f" | {'SL='+str(r)+'Ã—ATR':>14}"
    header += f" | {'Meilleur':>14}"
    print(header)

    for r in results:
        if r.get("error"):
            continue
        row = f"{r['asset']:<10} | {r['atr_mean_pips']:>6.1f} | "
        b = r["baseline"]
        row += f"WR{b['wr']:>4.0%}/Sh{b['sharpe']:+.1f}".rjust(16)
        sharpes = [g["sharpe"] for g in r["grid"]]
        for g in r["grid"]:
            row += f" | WR{g['wr']:>4.0%}/Sh{g['sharpe']:+.1f}".rjust(17)
        best_idx = int(np.argmax(sharpes))
        best = r["grid"][best_idx]
        row += f" | r={best['sl_atr_ratio']}/Sh{best['sharpe']:+.1f}".rjust(17)
        print(row)

    out_json = Path("predictions/diagnose_donchian_atr_grid.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardÃ© : {out_json}")

    # InterprÃ©tation automatique
    print("\n" + "=" * 70)
    print("INTERPRÃ‰TATION")
    print("=" * 70)
    any_edge = False
    for r in results:
        if r.get("error"):
            continue
        best = max(r["grid"], key=lambda g: g["sharpe"])
        if best["sharpe"] > 0.5 and best["wr"] >= 0.30 and best["n_trades"] >= 30:
            print(f"  âœ… {r['asset']} : meilleur ratio={best['sl_atr_ratio']}Ã—ATR "
                  f"(SL={best['sl_pips']} TP={best['tp_pips']}) â†’ "
                  f"Sharpe {best['sharpe']:+.2f}, WR {best['wr']:.1%} sur train")
            any_edge = True
        else:
            print(f"  âŒ {r['asset']} : aucun ratio ne produit d'edge sur train. "
                  f"Meilleur = Sharpe {best['sharpe']:+.2f}, WR {best['wr']:.1%}")

    if not any_edge:
        print("\nðŸ”´ AUCUN COUPLE ne montre d'edge Donchian D1 train, quel que soit le ratio.")
        print("   â†’ Donchian D1 fondamentalement dÃ©faillant. Passer au plan d'amÃ©lioration.")
    else:
        print("\nðŸ’¡ Au moins un couple a un edge potentiel aprÃ¨s re-calibration.")
        print("   â†’ Possible nouvelle hypothÃ¨se OOS unique avec ces paramÃ¨tres.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
