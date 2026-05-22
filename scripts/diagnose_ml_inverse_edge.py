"""Diagnostic post-fix : pourquoi Sharpe portfolio = -5.42 aprÃ¨s corrections ?

Trois questions Ã  trancher pour chaque couple Donchian+ML :

1. Le **Donchian seul** (sans ML) a-t-il un edge sur 2024+ ?
   - Si oui â†’ le ML dÃ©truit la valeur (filtre inverse).
   - Si non â†’ le Donchian lui-mÃªme ne marche plus (rÃ©gime change).

2. Sur les trades Donchian que le ML REJETTE, quel est leur WR/Sharpe ?
   - Si WR_rejetÃ©s > WR_acceptÃ©s â†’ le ML inverse l'edge (preuve directe).

3. Le ML a-t-il une accuracy OOS meilleure que la moyenne du WR Donchian ?
   - Si acc_OOS â‰ˆ WR ou pire â†’ le ML n'a aucun pouvoir prÃ©dictif rÃ©el.

Output :
- tableau console
- predictions/diagnose_ml_inverse_edge.json
- docs/diagnostic_ml_inverse_edge.md

Note : ce script lit le test set â‰¥ 2024 â†’ consomme 1 n_trial par couple
mais ne tune AUCUN paramÃ¨tre. C'est une re-analyse du rÃ©sultat existant.
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
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.models.meta_labeling_pipeline import filter_signals_by_meta_proba  # noqa: E402
from app.testing.snooping_guard import read_oos  # noqa: E402

# RÃ©utiliser les helpers de validation_finale
from scripts.run_validation_finale import (  # noqa: E402
    DONCHIAN_M,
    DONCHIAN_N,
    TEST_START,
    TRAIN_CUTOFF,
    _build_features,
    _generate_donchian_signals,
    _target_winner,
    _train_model,
    _trades_to_equity,
)

logger = get_logger(__name__)

CAPITAL_EUR = 10_000.0
COUPLES: list[dict[str, Any]] = [
    {"asset": "GBPUSD", "tf": "D1", "model": "rf",      "threshold": 0.50},
    {"asset": "EURUSD", "tf": "D1", "model": "stacking","threshold": 0.50},
    {"asset": "USDCHF", "tf": "D1", "model": "stacking","threshold": 0.50},
    {"asset": "ETHUSD", "tf": "D1", "model": "hgbm",    "threshold": 0.50},
]


def _analyze_trades(trades: list[dict], capital_pips: float = 10_000.0) -> dict[str, float]:
    """MÃ©triques rapides : Sharpe linÃ©aire, WR, mean PnL, max DD pips."""
    if not trades:
        return {"sharpe": 0.0, "wr": 0.0, "n_trades": 0, "mean_pnl": 0.0, "max_dd_pips": 0.0}

    pnls = np.array([t["pips_net"] for t in trades])
    n_wins = int((pnls > 0).sum())
    wr = n_wins / len(pnls)
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd_pips = float((equity - peak).min())

    sharpe = sharpe_daily_from_trades(trades, initial_capital_pips=capital_pips)

    return {
        "sharpe": float(sharpe),
        "wr": float(wr),
        "n_trades": int(len(trades)),
        "mean_pnl": float(pnls.mean()),
        "max_dd_pips": max_dd_pips,
    }


def diagnose_couple(asset: str, tf: str, model_type: str, threshold: float) -> dict[str, Any]:
    """Compare Donchian seul vs ML+Donchian, dÃ©compose les rejets du ML."""
    print(f"\n{'='*60}")
    print(f"[Diagnostic] {asset} {tf} ({model_type}, threshold={threshold:.2f})")
    print(f"{'='*60}")

    df = load_asset(asset, tf)
    cfg = ASSET_CONFIGS[asset]
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]
    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # â”€â”€ 1. Donchian seul sur test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    donchian_test = _generate_donchian_signals(df_test)
    bt_baseline = run_deterministic_backtest(
        df=df_test, signals=donchian_test,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    trades_baseline = bt_baseline.get("trades", [])
    metrics_baseline = _analyze_trades(trades_baseline)
    print(f"  Donchian seul: n={metrics_baseline['n_trades']}, "
          f"Sharpe={metrics_baseline['sharpe']:.2f}, "
          f"WR={metrics_baseline['wr']:.1%}, "
          f"mean_pnl={metrics_baseline['mean_pnl']:.2f}")

    # â”€â”€ 2. EntraÃ®ner le ML sur train (mÃªmes paramÃ¨tres que validation_finale) â”€
    donchian_train = _generate_donchian_signals(df_train)
    bt_train = run_deterministic_backtest(
        df=df_train, signals=donchian_train,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    trades_train = bt_train.get("trades", [])
    if len(trades_train) < 20:
        return {"asset": asset, "tf": tf, "skipped": "trop peu de trades train"}

    features_train = _build_features(df_train, asset, tf)
    if features_train.empty:
        return {"asset": asset, "tf": tf, "skipped": "features train vides"}

    entry_times_train = pd.to_datetime([t["entry_time"] for t in trades_train])
    common_train_idx = features_train.index.intersection(entry_times_train)
    if len(common_train_idx) < 10:
        return {"asset": asset, "tf": tf, "skipped": "trades non alignÃ©s"}

    X_train = features_train.loc[common_train_idx]
    _, trades_df_train = _trades_to_equity(trades_train, cfg=cfg)
    pnl_aligned = trades_df_train.loc[
        trades_df_train.index.intersection(common_train_idx), "Pips_Nets"
    ]
    y_train = _target_winner(pnl_aligned)
    if y_train.nunique() < 2:
        return {"asset": asset, "tf": tf, "skipped": "y_train une seule classe"}

    model = _train_model(X_train, y_train, model_type, asset, tf)
    acc_train = float((model.predict(X_train.values) == y_train.values).mean())

    # â”€â”€ 3. Appliquer le ML aux signaux Donchian de test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    features_test_full = _build_features(df, asset, tf)  # historique complet pour features rolling
    features_test = features_test_full.loc[features_test_full.index.isin(df_test.index)]
    signals_ml = filter_signals_by_meta_proba(
        df=df_test, primary_signals=donchian_test,
        features=features_test, model=model, threshold=threshold,
    )

    # Re-backtest avec les signaux filtrÃ©s
    bt_ml = run_deterministic_backtest(
        df=df_test, signals=signals_ml,
        tp_pips=cfg.tp_points, sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost, pip_size=cfg.pip_size, asset_config=cfg,
    )
    trades_ml = bt_ml.get("trades", [])
    metrics_ml = _analyze_trades(trades_ml)
    print(f"  ML + Donchian: n={metrics_ml['n_trades']}, "
          f"Sharpe={metrics_ml['sharpe']:.2f}, "
          f"WR={metrics_ml['wr']:.1%}, "
          f"mean_pnl={metrics_ml['mean_pnl']:.2f}")

    # â”€â”€ 4. DÃ©composition : trades acceptÃ©s vs rejetÃ©s par le ML â”€â”€â”€â”€â”€â”€â”€
    entry_times_baseline = set(pd.to_datetime([t["entry_time"] for t in trades_baseline]))
    entry_times_ml = set(pd.to_datetime([t["entry_time"] for t in trades_ml]))
    accepted_times = entry_times_baseline & entry_times_ml
    rejected_times = entry_times_baseline - entry_times_ml

    trades_accepted = [t for t in trades_baseline if pd.Timestamp(t["entry_time"]) in accepted_times]
    trades_rejected = [t for t in trades_baseline if pd.Timestamp(t["entry_time"]) in rejected_times]

    metrics_accepted = _analyze_trades(trades_accepted)
    metrics_rejected = _analyze_trades(trades_rejected)

    print(f"  ML dÃ©composition :")
    print(f"    AcceptÃ©s (passe filtre): n={metrics_accepted['n_trades']}, "
          f"WR={metrics_accepted['wr']:.1%}, mean_pnl={metrics_accepted['mean_pnl']:.2f}")
    print(f"    RejetÃ©s (rejette filtre): n={metrics_rejected['n_trades']}, "
          f"WR={metrics_rejected['wr']:.1%}, mean_pnl={metrics_rejected['mean_pnl']:.2f}")

    inverse_signal = (
        metrics_rejected["n_trades"] >= 5
        and metrics_accepted["n_trades"] >= 5
        and metrics_rejected["wr"] > metrics_accepted["wr"] + 0.05
    )
    if inverse_signal:
        print(f"  ðŸ”´ INVERSION DÃ‰TECTÃ‰E : le ML rejette les meilleurs trades.")

    # â”€â”€ 5. Snooping guard : 1 read par couple â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    read_oos(
        prompt="diagnose_ml_inverse_edge",
        hypothesis=f"diagnose_{asset}_{tf}",
        sharpe=metrics_ml["sharpe"],
        n_trades=metrics_ml["n_trades"],
    )

    return {
        "asset": asset, "tf": tf, "model": model_type, "threshold": threshold,
        "accuracy_train": acc_train,
        "baseline_donchian": metrics_baseline,
        "ml_donchian": metrics_ml,
        "ml_accepted": metrics_accepted,
        "ml_rejected": metrics_rejected,
        "inverse_signal_detected": inverse_signal,
    }


def main() -> int:
    set_global_seeds()
    print("=" * 70)
    print("DIAGNOSTIC POST-FIX : pourquoi le ML dÃ©truit-il l'edge Donchian ?")
    print("=" * 70)

    results: list[dict[str, Any]] = []
    for c in COUPLES:
        try:
            r = diagnose_couple(c["asset"], c["tf"], c["model"], c["threshold"])
            results.append(r)
        except Exception as exc:
            logger.error("Ã‰chec %s %s : %s", c["asset"], c["tf"], exc, exc_info=True)
            results.append({"asset": c["asset"], "tf": c["tf"], "error": str(exc)})

    # Tableau rÃ©cap
    print("\n" + "=" * 70)
    print("RÃ‰CAP DIAGNOSTIC")
    print("=" * 70)
    print(f"{'Couple':<12} {'Donchian':>12} {'ML+Donch':>12} {'Acc/Rej WR':>14} {'Inverse?':>10}")
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        couple = f"{r['asset']}_{r['tf']}"
        b = r["baseline_donchian"]
        m = r["ml_donchian"]
        a = r["ml_accepted"]
        rej = r["ml_rejected"]
        donchian_str = f"{b['sharpe']:+.2f}/{b['wr']:.0%}"
        ml_str = f"{m['sharpe']:+.2f}/{m['wr']:.0%}"
        wr_diff = f"{a['wr']:.0%}/{rej['wr']:.0%}"
        inv = "ðŸ”´ OUI" if r["inverse_signal_detected"] else "  non"
        print(f"{couple:<12} {donchian_str:>12} {ml_str:>12} {wr_diff:>14} {inv:>10}")

    # Sauvegarde JSON + Markdown
    out_json = Path("predictions/diagnose_ml_inverse_edge.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON sauvegardÃ© : {out_json}")

    _write_markdown(results)
    return 0


def _write_markdown(results: list[dict[str, Any]]) -> None:
    """GÃ©nÃ¨re docs/diagnostic_ml_inverse_edge.md Ã  partir des rÃ©sultats."""
    lines = [
        "# Diagnostic post-fix â€” Pourquoi le ML dÃ©truit-il l'edge Donchian ?",
        "",
        f"**Date** : {pd.Timestamp.now(tz='UTC').isoformat()}",
        "**Question** : aprÃ¨s correction des bugs F1+F2+F3, Sharpe portfolio = -5.42.",
        "Le ML inverse-t-il l'edge Donchian, ou Donchian lui-mÃªme ne marche plus ?",
        "",
        "## MÃ©thodologie",
        "",
        "Pour chaque couple Donchian+ML :",
        "1. Backtest Donchian SEUL sur 2024+ (baseline pure).",
        "2. Backtest Donchian + filtre ML (config validation_finale).",
        "3. DÃ©composition : pour chaque trade baseline, le ML l'a-t-il acceptÃ© ou rejetÃ© ?",
        "   Calcul du WR sur acceptÃ©s vs rejetÃ©s.",
        "",
        "Si **WR_rejetÃ©s > WR_acceptÃ©s + 5pts** â†’ ðŸ”´ le ML inverse l'edge.",
        "",
        "## RÃ©sultats",
        "",
        "| Couple | Donchian seul | ML+Donchian | AcceptÃ©s WR | RejetÃ©s WR | Inverse ? |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        couple = f"{r['asset']} {r['tf']}"
        b = r["baseline_donchian"]
        m = r["ml_donchian"]
        a = r["ml_accepted"]
        rej = r["ml_rejected"]
        inv = "ðŸ”´ OUI" if r["inverse_signal_detected"] else "non"
        lines.append(
            f"| {couple} | "
            f"Sharpe {b['sharpe']:+.2f}, WR {b['wr']:.1%}, n={b['n_trades']} | "
            f"Sharpe {m['sharpe']:+.2f}, WR {m['wr']:.1%}, n={m['n_trades']} | "
            f"{a['wr']:.1%} (n={a['n_trades']}) | "
            f"{rej['wr']:.1%} (n={rej['n_trades']}) | "
            f"{inv} |"
        )

    lines += [
        "",
        "## InterprÃ©tation",
        "",
        "**Donchian seul positif â†’ ML+Donchian nÃ©gatif** : le filtre ML aggrave les rÃ©sultats.",
        "C'est la signature d'un **modÃ¨le ML dÃ©faillant** entraÃ®nÃ© sur des features qui",
        "ne se gÃ©nÃ©ralisent pas du train (â‰¤ 2022) au test (â‰¥ 2024). Le ML rejette",
        "systÃ©matiquement les trades qui *auraient gagnÃ©* dans le rÃ©gime 2024-2026.",
        "",
        "**Donchian seul nÃ©gatif** : la stratÃ©gie Donchian elle-mÃªme ne fonctionne plus",
        "dans le rÃ©gime 2024-2026. Probable changement de rÃ©gime de marchÃ©.",
        "",
        "**AcceptÃ©s WR < RejetÃ©s WR + 5pts** : le ML inverse l'edge directement. Le rejet",
        "ML est plus prÃ©dictif d'un winner que l'acceptation.",
        "",
        "## Pistes d'amÃ©lioration (si Donchian seul a un edge)",
        "",
        "1. **RÃ©gulariser le ML plus fort** : max_depth=2, min_samples_leaf=50.",
        "2. **RÃ©duire le feature set** : utiliser seulement les 3-5 features les plus stables.",
        "3. **Train plus rÃ©cent** : remplacer 2010-2022 par 2018-2022 (5 ans plus proches).",
        "4. **Walk-forward** : re-entraÃ®ner tous les 6 mois plutÃ´t qu'un modÃ¨le figÃ©.",
        "5. **Calibration sur 2023** : utiliser le flag CALIBRATE_THRESHOLD_ON_VAL=True.",
        "",
        "## Pistes (si Donchian seul ne marche plus)",
        "",
        "1. Passer aux **nouvelles stratÃ©gies** (voir plan_v5_amelioration_strategies.md Axe B).",
        "2. Tester **Donchian sur autres timeframes** (H4, H1).",
        "3. Tester **autres paramÃ¨tres Donchian** (N=10, 30, 50 au lieu de 20).",
    ]

    md_path = Path("docs/diagnostic_ml_inverse_edge.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown sauvegardÃ© : {md_path}")


if __name__ == "__main__":
    sys.exit(main())
