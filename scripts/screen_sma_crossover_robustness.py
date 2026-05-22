"""Robustesse paramÃ©trique de SmaCrossover sur H4 â€” train â‰¤ 2022 (0 n_trial).

Question : EURUSD H4 SmaCrossover_10_50 Sharpe +0.75 OOS est-il :
  A) un vrai signal robuste (apparaÃ®t sur d'autres params ET/OU autres actifs)
  B) un coup de chance (unique Ã  EURUSD/10/50)

Test : 6 jeux de params SMA Ã— 6 actifs Ã— 4 ratios TP/SL = 144 backtests.

Verdict de robustesse :
  - ParamÃ©trique : si EURUSD H4 montre Sharpe â‰¥ 0.3 sur â‰¥ 3/6 jeux de params
    â†’ robuste au paramÃ©trage (pas un cherry-pick).
  - Cross-asset : si SmaCrossover montre Sharpe â‰¥ 0.3 sur â‰¥ 3/6 actifs
    â†’ vraie famille de stratÃ©gies.
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
from app.strategies.sma_crossover import SmaCrossover  # noqa: E402

from scripts.run_validation_finale import TRAIN_CUTOFF  # noqa: E402

# 6 jeux de params SmaCrossover (du plus rapide au plus lent)
SMA_PARAMS: list[tuple[int, int]] = [
    (5, 20), (10, 30), (10, 50), (20, 60), (20, 100), (50, 200),
]
# Tous les actifs disponibles dans data/raw avec H4
ASSETS: list[str] = [
    "EURUSD", "GBPUSD", "USDCHF",  # Forex (les 3 dispos)
    "BUND",                          # Obligations
    "USOIL",                         # Ã‰nergies
    "XAGUSD",                        # MÃ©taux secondaires
]
TF = "H4"
TP_SL_RATIOS = [0.5, 0.7, 1.0, 1.5]


def _analyze(trades: list[dict], capital_pips: float = 10_000.0) -> dict[str, float]:
    if not trades:
        return {"sharpe": 0.0, "wr": 0.0, "n_trades": 0, "mean_pnl": 0.0}
    pnls = np.array([t["pips_net"] for t in trades])
    return {
        "sharpe": float(sharpe_daily_from_trades(trades, initial_capital_pips=capital_pips)),
        "wr": float((pnls > 0).mean()),
        "n_trades": int(len(trades)),
        "mean_pnl": float(pnls.mean()),
    }


def test_one(fast: int, slow: int, asset: str) -> list[dict[str, Any]]:
    try:
        df = load_asset(asset, TF)
    except Exception as exc:
        return [{"fast": fast, "slow": slow, "asset": asset, "error": str(exc)[:60]}]
    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    if df_train.empty:
        return [{"fast": fast, "slow": slow, "asset": asset, "error": "train vide"}]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    atr14 = atr(df_train["High"], df_train["Low"], df_train["Close"], 14)
    atr_mean = float((atr14 / cfg.pip_size).dropna().mean())
    if atr_mean <= 0:
        return [{"fast": fast, "slow": slow, "asset": asset, "error": "ATR=0"}]

    strat = SmaCrossover(fast=fast, slow=slow)
    signals = strat.generate_signals(df_train)
    n_signals = int((signals != 0).sum())
    if n_signals < 30:
        return [{"fast": fast, "slow": slow, "asset": asset, "error": f"{n_signals} signaux"}]

    results = []
    for ratio in TP_SL_RATIOS:
        sl_pips = max(round(ratio * atr_mean), 1)
        tp_pips = max(round(sl_pips * 2.0), 1)
        bt = run_deterministic_backtest(
            df=df_train, signals=signals,
            tp_pips=tp_pips, sl_pips=sl_pips,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
        )
        m = _analyze(bt.get("trades", []))
        m.update({
            "fast": fast, "slow": slow, "asset": asset, "tf": TF,
            "sl_atr_ratio": ratio, "sl_pips": sl_pips, "tp_pips": tp_pips,
        })
        results.append(m)
    return results


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print(f"ROBUSTESSE SmaCrossover H4 â€” {len(SMA_PARAMS)} params Ã— {len(ASSETS)} actifs Ã— "
          f"{len(TP_SL_RATIOS)} ratios = {len(SMA_PARAMS)*len(ASSETS)*len(TP_SL_RATIOS)} backtests")
    print(f"Train â‰¤ {TRAIN_CUTOFF.date()} â€” 0 n_trial consommÃ©")
    print("=" * 70)

    all_results: list[dict[str, Any]] = []
    for (fast, slow) in SMA_PARAMS:
        print(f"\nâ”€â”€ SMA {fast}/{slow} â”€â”€")
        for asset in ASSETS:
            rows = test_one(fast, slow, asset)
            ok = [r for r in rows if "error" not in r]
            err = [r for r in rows if "error" in r]
            for r in ok:
                all_results.append(r)
            if err:
                print(f"  {asset}: âŒ {err[0]['error']}")
            elif ok:
                best = max(ok, key=lambda x: x["sharpe"])
                print(f"  {asset}: meilleur Sharpe={best['sharpe']:+.2f} "
                      f"(WR={best['wr']:.0%}, n={best['n_trades']}, "
                      f"SL={best['sl_atr_ratio']}Ã—ATR)")

    # â”€â”€ Matrice : pour chaque (params, asset), garder le meilleur Sharpe â”€â”€
    print("\n" + "=" * 90)
    print("MATRICE â€” Meilleur Sharpe par (params, asset)")
    print("=" * 90)
    asset_set = sorted({r["asset"] for r in all_results})
    print(f"{'Params':<10}", end="")
    for a in asset_set:
        print(f"{a:>10}", end="")
    print()
    matrix: dict[tuple[int, int], dict[str, float]] = {}
    for (fast, slow) in SMA_PARAMS:
        params_str = f"{fast}/{slow}"
        print(f"{params_str:<10}", end="")
        matrix[(fast, slow)] = {}
        for a in asset_set:
            subset = [r for r in all_results
                      if r["fast"] == fast and r["slow"] == slow and r["asset"] == a]
            if subset:
                best = max(subset, key=lambda x: x["sharpe"])
                matrix[(fast, slow)][a] = best["sharpe"]
                marker = "âœ…" if best["sharpe"] >= 0.5 else (
                    "ðŸŸ¡" if best["sharpe"] >= 0.3 else "  "
                )
                print(f"{marker}{best['sharpe']:>+7.2f}", end="")
            else:
                matrix[(fast, slow)][a] = float("nan")
                print(f"{'N/A':>10}", end="")
        print()

    # â”€â”€ Verdict robustesse â”€â”€
    print("\n" + "=" * 70)
    print("VERDICT ROBUSTESSE")
    print("=" * 70)

    # Robustesse paramÃ©trique par actif (combien de params Sharpe â‰¥ 0.3 ?)
    print("\nRobustesse paramÃ©trique (combien de jeux de params Sharpe â‰¥ 0.3) :")
    for a in asset_set:
        sharpes = [matrix[(f, s)].get(a) for (f, s) in SMA_PARAMS if not np.isnan(matrix[(f, s)].get(a, float("nan")))]
        n_good = sum(1 for s in sharpes if s >= 0.3)
        n_excellent = sum(1 for s in sharpes if s >= 0.5)
        verdict = ("ðŸŽ¯ ROBUSTE" if n_good >= 3 else
                   ("âš ï¸  marginal" if n_good >= 2 else "âŒ pas robuste"))
        print(f"  {a:<10} : {n_good}/{len(sharpes)} params Shâ‰¥0.3, "
              f"{n_excellent}/{len(sharpes)} Shâ‰¥0.5 â†’ {verdict}")

    # Robustesse cross-asset par params (combien d'actifs Sharpe â‰¥ 0.3 ?)
    print("\nRobustesse cross-asset (combien d'actifs Sharpe â‰¥ 0.3 pour un jeu de params) :")
    for (fast, slow) in SMA_PARAMS:
        sharpes = [matrix[(fast, slow)].get(a) for a in asset_set
                   if not np.isnan(matrix[(fast, slow)].get(a, float("nan")))]
        n_good = sum(1 for s in sharpes if s >= 0.3)
        n_excellent = sum(1 for s in sharpes if s >= 0.5)
        verdict = ("ðŸŽ¯ ROBUSTE" if n_good >= 3 else
                   ("âš ï¸  marginal" if n_good >= 2 else "âŒ pas robuste"))
        print(f"  SMA {fast:>2}/{slow:>3} : {n_good}/{len(sharpes)} actifs Shâ‰¥0.3, "
              f"{n_excellent}/{len(sharpes)} Shâ‰¥0.5 â†’ {verdict}")

    out_json = Path("predictions/screen_sma_crossover_robustness.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"all_results": all_results, "matrix": {f"{k[0]}_{k[1]}": v for k, v in matrix.items()}},
                   indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardÃ© : {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
