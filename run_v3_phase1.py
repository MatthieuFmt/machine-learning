"""Orchestrateur v3 Phase 1 — H06, H07, H08.

H06 : Grid search Donchian sur tous les actifs D1 disponibles (sauf US30 déjà GO).
H07 : Grid search 4 stratégies trend-following additionnelles sur US30 + actifs GO.
H08 : Portefeuille equal-risk weight combinant toutes les stratégies GO.

Règles:
- Split temporel strict : train <= 2022, val = 2023, test >= 2024.
- Sélection meilleur paramétrage sur train uniquement.
- Sharpe calculé sur equity curve quotidienne (jamais PnL/trade).
- Coûts réalistes modélisés dans le backtest stateful.
- DSR cumulatif : n_trials = 5 (v2 H01–H05) + 3 (H06, H07, H08) = 8.
- Pas d'arrêt au premier GO (on veut un portefeuille multi-actif).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.strategies import ALL_STRATEGIES_V3
from app.strategies.donchian import DonchianBreakout
from app.backtest.grid_search import grid_search_asset
from app.backtest.deterministic import run_deterministic_backtest

# -- Actifs D1 disponibles (tout ce qui est dans data/) -----------------
# US30 exclu de H06 car déjà GO en v2. Il reste la baseline.
# USDCHF réservé comme hold-out gate final (jamais testé avant H18).
# Ordre : priorité aux actifs décorrélés.

H06_ASSETS: list[dict[str, Any]] = [
    {
        "name": "XAUUSD",
        "tf": "D1",
        "csv": "data/XAUUSD_D1.csv",
        "tp_pips": 60,      # ~2x ATR D1 (~30 pips-or)
        "sl_pips": 30,      # ~1x ATR
        "window_hours": 120,
        "commission_pips": 0.5,
        "slippage_pips": 0.3,  # spread D1 or ~0.3-0.5 USD
        "pip_size": 1.0,
    },
    {
        "name": "GBPUSD",
        "tf": "D1",
        "csv": "data/GBPUSD_D1.csv",
        "tp_pips": 160,     # ~2x ATR D1 (~80 pips)
        "sl_pips": 80,
        "window_hours": 120,
        "commission_pips": 3,
        "slippage_pips": 5,
        "pip_size": 0.0001,
    },
    {
        "name": "ETHUSD",
        "tf": "D1",
        "csv": "data/ETHUSD_D1.csv",
        "tp_pips": 100,     # ~2x ATR D1 (~50 USD)
        "sl_pips": 50,
        "window_hours": 120,
        "commission_pips": 15,
        "slippage_pips": 10,
        "pip_size": 1.0,
    },
    {
        "name": "EURUSD",
        "tf": "D1",
        "csv": "data/EURUSD_D1.csv",
        "tp_pips": 140,     # ~2x ATR D1 (~70 pips)
        "sl_pips": 70,
        "window_hours": 120,
        "commission_pips": 1.5,
        "slippage_pips": 1,
        "pip_size": 0.0001,
    },
    {
        "name": "BTCUSD",
        "tf": "D1",
        "csv": "data/BTCUSD_D1.csv",
        "tp_pips": 4000,    # ~2x ATR D1 (~2000 USD)
        "sl_pips": 2000,
        "window_hours": 120,
        "commission_pips": 15,
        "slippage_pips": 10,
        "pip_size": 1.0,
    },
]

# Baseline US30 D1 (déjà GO v2)
US30_ASSET: dict[str, Any] = {
    "name": "USA30IDXUSD",
    "tf": "D1",
    "csv": "data/USA30IDXUSD_D1.csv",
    "tp_pips": 600,         # ~2x ATR D1 (~300 points)
    "sl_pips": 300,
    "window_hours": 120,
    "commission_pips": 3,
    "slippage_pips": 5,
    "pip_size": 1.0,
}

TRAIN_END = "2023-01-01"
VAL_END = "2024-01-01"

# -- Data loading -----------------------------------------------------

def load_asset_csv(csv_path: str) -> pd.DataFrame | None:
    """Charge un CSV D1 (tab-separated) et retourne un DataFrame OHLC indexe.

    Utilise le parsing natif pandas avec les 5 premieres colonnes (Time + OHLC).
    """
    path = Path(csv_path)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(
            str(path),
            sep="\t",
            usecols=[0, 1, 2, 3, 4],
            names=["Time", "Open", "High", "Low", "Close"],
            header=0,
            parse_dates=["Time"],
            dayfirst=False,
        )
    except Exception:
        return None

    if df.empty or len(df.columns) < 5:
        return None

    df = df.set_index("Time")
    df = df.sort_index()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if len(df) < 100:
        return None

    return df

def split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporel strict."""
    train = df[df.index < TRAIN_END].copy()
    val = df[(df.index >= TRAIN_END) & (df.index < VAL_END)].copy()
    test = df[df.index >= VAL_END].copy()
    return train, val, test


# -- Sharpe portfolio (returns quotidiens) ----------------------------

def daily_returns_from_trades(
    trades_list: list[dict[str, Any]],
) -> pd.Series:
    """Calcule les returns quotidiens depuis une liste de trades stateful.

    Args:
        trades_list: Liste de dicts avec 'exit_time' (str) et 'pips_net' (float).

    Returns:
        pd.Series de returns quotidiens (pct_change de l'equity curve).
    """
    if not trades_list or len(trades_list) < 2:
        return pd.Series(dtype=float)

    pnls = np.array([t["pips_net"] for t in trades_list], dtype=np.float64)
    equity = np.cumsum(pnls)
    exit_times = pd.to_datetime([t["exit_time"] for t in trades_list])
    equity_series = pd.Series(equity, index=exit_times).sort_index()
    equity_daily = equity_series.resample("D").last().ffill()
    if len(equity_daily) < 2:
        return pd.Series(dtype=float)
    return equity_daily.pct_change().dropna()


def sharpe_from_returns(daily_returns: pd.Series, annual_factor: float = 252.0) -> float:
    """Sharpe annualisé depuis les returns quotidiens."""
    if len(daily_returns) < 2:
        return 0.0
    std = daily_returns.std()
    if std == 0:
        return 0.0
    return float(daily_returns.mean() / std * np.sqrt(annual_factor))


# -- H08 — Portefeuille equal-risk ------------------------------------

def build_equal_risk_portfolio(
    go_strategies: list[dict[str, Any]],
    target_vol: float = 0.10,
    annual_factor: float = 252.0,
) -> dict[str, Any]:
    """Combine N stratégies GO en portefeuille equal-risk weight.

    Chaque stratégie reçoit un poids prop. 1 / volatility_réalisée.
    Les returns quotidiens sont moyennés (pondérés) pour produire
    la courbe d'equity du portefeuille.

    Args:
        go_strategies: Liste de dicts avec clés :
            - 'asset_name': str
            - 'strategy_name': str
            - 'trades': list[dict] (stateful backtest)
            - 'sharpe_test': float
        target_vol: Volatilité cible annuelle (défaut 10%).
        annual_factor: Facteur d'annualisation (252).

    Returns:
        dict avec sharpe_portfolio, n_strategies, weights, daily_returns, etc.
    """
    if not go_strategies:
        return {"sharpe_portfolio": 0.0, "n_strategies": 0, "weights": [], "error": "Aucune stratégie GO"}

    # Calcul des returns quotidiens et volatilités par stratégie
    strat_returns: dict[str, pd.Series] = {}
    strat_vols: dict[str, float] = {}

    for i, strat in enumerate(go_strategies):
        key = f"{strat['asset_name']}_{strat['strategy_name']}"
        returns = daily_returns_from_trades(strat["trades"])
        if returns.empty:
            continue
        strat_returns[key] = returns
        strat_vols[key] = returns.std() * np.sqrt(annual_factor)

    if not strat_returns:
        return {"sharpe_portfolio": 0.0, "n_strategies": 0, "weights": [], "error": "Pas de returns quotidiens exploitables"}

    # Poids equal-risk : w_i prop. 1/sigma_i
    inv_vols = {k: 1.0 / max(v, 1e-10) for k, v in strat_vols.items()}
    total_inv = sum(inv_vols.values())
    weights = {k: v / total_inv for k, v in inv_vols.items()}

    # Alignement des dates
    all_dates = sorted(set().union(*(r.index for r in strat_returns.values())))
    portfolio_daily = pd.Series(0.0, index=all_dates, dtype=float)

    for key, returns in strat_returns.items():
        aligned = returns.reindex(all_dates).fillna(0.0)
        portfolio_daily += aligned * weights[key]

    # Vol targeting : scale pour atteindre target_vol
    pf_vol = portfolio_daily.std() * np.sqrt(annual_factor)
    if pf_vol > 1e-10:
        scale = target_vol / pf_vol
        portfolio_daily = portfolio_daily * scale

    pf_sharpe = sharpe_from_returns(portfolio_daily)

    return {
        "sharpe_portfolio": pf_sharpe,
        "n_strategies": len(strat_returns),
        "weights": {k: round(float(w), 4) for k, w in weights.items()},
        "daily_returns_mean": float(portfolio_daily.mean()),
        "daily_returns_std": float(portfolio_daily.std()),
        "vol_target": target_vol,
        "scale_factor": round(float(scale) if pf_vol > 1e-10 else 1.0, 4),
    }


# -- Main orchestrateur -----------------------------------------------

def run_h06(df_dict: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """H06 : Grid search Donchian sur tous les actifs D1 disponibles.

    Retourne la liste des résultats GO (Sharpe val > 0 ET Sharpe test > 0).
    """
    print("=" * 80)
    print("V3 H06 — Extension univers CFD : Grid Search Donchian multi-actif")
    print(f"N actifs testés : {len(H06_ASSETS)}")
    print("=" * 80)

    param_grid = {"N": [20, 50, 100], "M": [10, 20, 50]}
    go_results: list[dict[str, Any]] = []

    for asset in H06_ASSETS:
        name = asset["name"]
        csv_path = asset["csv"]

        df = df_dict.get(name)
        if df is None:
            print(f"\n[SKIP] {name} : données non chargées ({csv_path})")
            continue

        train, val, test = split_df(df)
        if len(train) < 500:
            print(f"\n[SKIP] {name} : train insuffisant ({len(train)} barres)")
            continue

        print(f"\n{'-' * 60}")
        print(f"[{name} D1]  Train={len(train)} ({train.index.min().date()}->{train.index.max().date()})")
        print(f"           Val={len(val)} ({val.index.min().date()}->{val.index.max().date()})")
        print(f"           Test={len(test)} ({test.index.min().date()}->{test.index.max().date()})")
        print(f"           TP={asset['tp_pips']} SL={asset['sl_pips']} "
              f"Comm={asset['commission_pips']} Slip={asset['slippage_pips']}")

        try:
            result = grid_search_asset(
                df_train=train,
                df_val=val,
                df_test=test,
                strategy_class=DonchianBreakout,
                param_grid=param_grid,
                tp_pips=asset["tp_pips"],
                sl_pips=asset["sl_pips"],
                window_hours=asset["window_hours"],
                commission_pips=asset["commission_pips"],
                slippage_pips=asset["slippage_pips"],
                pip_size=asset["pip_size"],
            )
        except Exception as exc:
            print(f"   [ERREUR] {exc}")
            continue

        s_train = result["sharpe_train"]
        s_val = result["sharpe_val"]
        s_test = result["sharpe_test"]
        wr_test = result["wr_test"]

        print(f"   Donchian best: {result['best_params']}")
        print(f"   Sharpe: train={s_train:+.3f}  val={s_val:+.3f}  test={s_test:+.3f}")
        print(f"   WR:     test={wr_test:.1%}  trades_test={result['total_trades_test']}")

        if s_val > 0 and s_test > 0:
            print(f"   >>> [GO] Donchian sur {name} D1 !")
            # Re-run backtest complet pour obtenir la liste des trades
            best_strat = DonchianBreakout(**result["best_params"])
            signals_test = best_strat.generate_signals(test)
            bt_result = run_deterministic_backtest(
                test, signals_test,
                tp_pips=asset["tp_pips"], sl_pips=asset["sl_pips"],
                window_hours=asset["window_hours"],
                commission_pips=asset["commission_pips"],
                slippage_pips=asset["slippage_pips"],
                pip_size=asset["pip_size"],
            )
            go_results.append({
                "asset_name": name,
                "tf": "D1",
                "strategy_name": "DonchianBreakout",
                "best_params": result["best_params"],
                "sharpe_train": s_train,
                "sharpe_val": s_val,
                "sharpe_test": s_test,
                "wr_test": wr_test,
                "trades": bt_result.get("trades", []),
                "asset_config": {k: v for k, v in asset.items() if k != "csv"},
            })
        else:
            print(f"   [NO-GO]")

    print(f"\n{'-' * 60}")
    print(f"H06 terminé : {len(go_results)} actif(s) GO (hors US30 baseline)")
    return go_results


def run_h07(
    df_dict: dict[str, pd.DataFrame],
    h06_go_assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """H07 : Grid search 4 stratégies additionnelles sur US30 + actifs GO de H06."""
    print(f"\n{'=' * 80}")
    print("V3 H07 — Stratégies trend-following additionnelles")
    print(f"N stratégies : {len(ALL_STRATEGIES_V3)}")
    print("=" * 80)

    # Actifs à tester : US30 (baseline) + actifs GO de H06
    h07_assets: list[dict[str, Any]] = [US30_ASSET]
    for go in h06_go_assets:
        # Récupère la config complète depuis H06_ASSETS
        name = go["asset_name"]
        matching = [a for a in H06_ASSETS if a["name"] == name]
        if matching:
            h07_assets.append(matching[0])

    go_results: list[dict[str, Any]] = []

    for asset in h07_assets:
        name = asset["name"]
        df = df_dict.get(name)
        if df is None:
            continue

        train, val, test = split_df(df)
        if len(train) < 500:
            continue

        print(f"\n{'-' * 60}")
        print(f"[{name} D1]  TP={asset['tp_pips']} SL={asset['sl_pips']}")

        for strategy_class, param_grid in ALL_STRATEGIES_V3:
            strat_name = strategy_class.__name__
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)

            print(f"   {strat_name} ({n_combos} combinaisons)...", end=" ", flush=True)

            try:
                result = grid_search_asset(
                    df_train=train, df_val=val, df_test=test,
                    strategy_class=strategy_class,
                    param_grid=param_grid,
                    tp_pips=asset["tp_pips"], sl_pips=asset["sl_pips"],
                    window_hours=asset["window_hours"],
                    commission_pips=asset["commission_pips"],
                    slippage_pips=asset["slippage_pips"],
                    pip_size=asset["pip_size"],
                )
            except Exception as exc:
                print(f"ERREUR: {exc}")
                continue

            s_train = result["sharpe_train"]
            s_val = result["sharpe_val"]
            s_test = result["sharpe_test"]
            wr_test = result["wr_test"]

            print(f"Sharpe: val={s_val:+.3f} test={s_test:+.3f} | best={result['best_params']}")

            if s_val > 0 and s_test > 0:
                print(f"   >>> [GO] {strat_name} sur {name} D1 !")
                best_strat = strategy_class(**result["best_params"])
                signals_test = best_strat.generate_signals(test)
                bt_result = run_deterministic_backtest(
                    test, signals_test,
                    tp_pips=asset["tp_pips"], sl_pips=asset["sl_pips"],
                    window_hours=asset["window_hours"],
                    commission_pips=asset["commission_pips"],
                    slippage_pips=asset["slippage_pips"],
                    pip_size=asset["pip_size"],
                )
                go_results.append({
                    "asset_name": name,
                    "tf": "D1",
                    "strategy_name": strat_name,
                    "best_params": result["best_params"],
                    "sharpe_train": s_train,
                    "sharpe_val": s_val,
                    "sharpe_test": s_test,
                    "wr_test": wr_test,
                    "trades": bt_result.get("trades", []),
                    "asset_config": {k: v for k, v in asset.items() if k != "csv"},
                })

    print(f"\nH07 terminé : {len(go_results)} stratégie(s) GO supplémentaires")
    return go_results


def run_h08(
    all_go_strategies: list[dict[str, Any]],
) -> dict[str, Any]:
    """H08 : Portefeuille equal-risk combinant toutes les stratégies GO."""
    print(f"\n{'=' * 80}")
    print("V3 H08 — Portefeuille equal-risk multi-actif")
    print(f"N stratégies GO : {len(all_go_strategies)}")
    print("=" * 80)

    for i, s in enumerate(all_go_strategies):
        print(f"   {i+1}. {s['asset_name']} {s['strategy_name']} "
              f"(Sharpe test={s['sharpe_test']:+.3f}, WR={s['wr_test']:.1%})")

    pf_result = build_equal_risk_portfolio(all_go_strategies, target_vol=0.10)

    print(f"\n[Portefeuille]")
    if "error" in pf_result:
        print(f"   Erreur : {pf_result['error']}")
    else:
        print(f"   Sharpe portfolio : {pf_result['sharpe_portfolio']:+.3f}")
        print(f"   N stratégies      : {pf_result['n_strategies']}")
        print(f"   Poids :")
        for k, w in pf_result["weights"].items():
            print(f"      {k}: {w:.1%}")

    return pf_result


def main() -> None:
    print("=" * 80)
    print("V3 Phase 1 — H06 / H07 / H08")
    print(f"Démarrage : {datetime.now().isoformat()}")
    print(f"Split : train < {TRAIN_END}, val = [{TRAIN_END}, {VAL_END}[, test >= {VAL_END}")
    print(f"DSR n_trials cumulatif = 8 (5 v2 + 3 v3 Phase 1)")
    print("=" * 80)

    # -- Chargement de toutes les données D1 ------------------------
    all_asset_names = {a["name"] for a in H06_ASSETS} | {US30_ASSET["name"]}
    df_dict: dict[str, pd.DataFrame] = {}

    for name in sorted(all_asset_names):
        if name == US30_ASSET["name"]:
            csv_path = US30_ASSET["csv"]
        else:
            matching = [a for a in H06_ASSETS if a["name"] == name]
            if not matching:
                continue
            csv_path = matching[0]["csv"]

        df = load_asset_csv(csv_path)
        if df is not None:
            df_dict[name] = df
            print(f"[LOAD] {name}: {len(df)} barres ({df.index.min().date()} -> {df.index.max().date()})")
        else:
            print(f"[LOAD] {name}: INTROUVABLE ({csv_path})")

    # -- H06 : Donchian grid search multi-actif ---------------------
    h06_go = run_h06(df_dict)

    # -- H07 : Stratégies additionnelles ----------------------------
    h07_go = run_h07(df_dict, h06_go)

    # -- H08 : Portefeuille equal-risk ------------------------------
    all_go = h06_go + h07_go
    pf_result = run_h08(all_go)

    # -- Sauvegarde -------------------------------------------------
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "phase": "v3_phase1",
        "hypotheses": ["H06", "H07", "H08"],
        "dsr_n_trials_cumulative": 8,
        "split": {"train_end": TRAIN_END, "val_end": VAL_END},
        "h06_go_assets": [{k: v for k, v in s.items() if k != "trades"} for s in h06_go],
        "h07_go_strategies": [{k: v for k, v in s.items() if k != "trades"} for s in h07_go],
        "h08_portfolio": {k: v for k, v in pf_result.items() if k != "daily_returns"},
        "all_go_strategies": [
            {
                "asset": s["asset_name"],
                "strategy": s["strategy_name"],
                "sharpe_test": s["sharpe_test"],
                "wr_test": s["wr_test"],
                "best_params": s["best_params"],
            }
            for s in all_go
        ],
    }

    output_path = output_dir / "v3_phase1_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 80}")
    go_count = len(all_go)
    if go_count > 0:
        print(f"[FIN] {go_count} stratégies GO. "
              f"Sharpe PF: {pf_result.get('sharpe_portfolio', 0):+.3f}")
        if pf_result.get("sharpe_portfolio", 0) >= 1.0:
            print(">>> [GO] GO V3 Phase 1 : Sharpe portfolio >= 1.0 !")
        elif pf_result.get("sharpe_portfolio", 0) >= 0.5:
            print(">>> [OK] Sharpe portfolio >= 0.5, poursuite Phase 2 recommandée.")
        else:
            print(">>> [NO] Sharpe portfolio < 0.5, Phase 2 conditionnelle.")
    else:
        print("[FIN] Aucune stratégie GO. NO-GO.")
    print(f"Rapport : {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
