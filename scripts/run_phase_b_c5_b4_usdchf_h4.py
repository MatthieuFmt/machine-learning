"""Pivot v4 Phase B C5 â€” B4 : USDCHF H4 walk-forward simple (sans mÃ©ta-labeling).

Flow simplifiÃ© :
1. Charge USDCHF H4, cutoff train â‰¤ 2022-12-31, test â‰¥ 2024-01-01.
2. Construit le superset de features, sÃ©lectionne le top 15 C5 pour USDCHF H4.
3. GÃ©nÃ¨re la target Donchian (N=20, M=20) sur train â†’ backtest â†’ trades binaires
   (winner = pips_net > 0).
4. EntraÃ®ne rf (hyperparams C5) sur les features aux barres d'entrÃ©e des trades train.
5. PrÃ©dit sur test, gÃ©nÃ¨re des signaux directionnels (prob > threshold,
   direction via momentum features).
6. Backtest dÃ©terministe sur test.
7. Calcule mÃ©triques (Sharpe, trades, WR, max DD).
8. Sauvegarde dans predictions/phase_b_c5_b4_usdchf_h4.json.

ðŸš« Pas de mÃ©ta-labeling. Pas de walk-forward complexe. Train une fois, test une fois.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.backtest.deterministic import run_deterministic_backtest  # noqa: E402
from app.backtest.metrics import compute_metrics  # noqa: E402
from app.config.features_selected import FEATURES_SELECTED  # noqa: E402
from app.config.hyperparams_tuned import HYPERPARAMS_TUNED  # noqa: E402
from app.config.instruments import ASSET_CONFIGS  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.seeds import set_global_seeds  # noqa: E402
from app.data.loader import load_asset  # noqa: E402
from app.features.superset import build_superset  # noqa: E402
from app.strategies.donchian import DonchianBreakout  # noqa: E402

logger = get_logger(__name__)

# â”€â”€ Constantes du couple â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ASSET = "USDCHF"
TF = "H4"
COUPLE_KEY = (ASSET, TF)
TRAIN_CUTOFF = pd.Timestamp("2022-12-31", tz="UTC")
TEST_START = pd.Timestamp("2024-01-01", tz="UTC")
CAPITAL_EUR = 10_000.0
RISK_PCT = 0.02
DONCHIAN_N = 20
DONCHIAN_M = 20


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_features_for_split(df_split: pd.DataFrame) -> pd.DataFrame:
    """Construit le superset et sÃ©lectionne le top 15 C5 pour USDCHF H4."""
    superset = build_superset(df_split, asset=ASSET)
    selected = list(FEATURES_SELECTED[COUPLE_KEY])
    available = [c for c in selected if c in superset.columns]
    missing = set(selected) - set(available)
    if missing:
        logger.warning("Features C5 manquantes dans le superset : %s", sorted(missing))
    return superset[available].dropna()


def _generate_donchian_signals(df: pd.DataFrame) -> pd.Series:
    """GÃ©nÃ¨re les signaux Donchian (N=20, M=20) pour H4."""
    strategy = DonchianBreakout(params={"N": DONCHIAN_N, "M": DONCHIAN_M})
    return strategy.generate_signals(df)


def _build_target_winner(pnl_net: pd.Series) -> pd.Series:
    """Cible binaire : 1 si trade gagnant (pips_net > 0), 0 sinon."""
    return (pnl_net > 0).astype(int)


def _train_rf_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """EntraÃ®ne le modÃ¨le RandomForest avec les hyperparams C5 pour USDCHF H4."""
    hp = HYPERPARAMS_TUNED[COUPLE_KEY]
    params = hp["params"]
    model = RandomForestClassifier(
        max_depth=params.get("max_depth", 3),
        min_samples_leaf=params.get("min_samples_leaf", 10),
        n_estimators=params.get("n_estimators", 100),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train.values, y_train.values)
    return model


def _generate_model_signals(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    primary_signals: pd.Series,
) -> pd.Series:
    """MÃ©ta-labeling fidÃ¨le (fix F1) : filtre les signaux Donchian primaires."""
    from app.models.meta_labeling_pipeline import filter_signals_by_meta_proba

    hp = HYPERPARAMS_TUNED[COUPLE_KEY]
    threshold = hp["threshold"]

    features = _build_features_for_split(df)
    if features.empty:
        return pd.Series(0, index=df.index, dtype=int)

    return filter_signals_by_meta_proba(
        df=df,
        primary_signals=primary_signals,
        features=features,
        model=model,
        threshold=threshold,
    )


def _trades_to_dataframe(
    trades: list[dict],
    cfg: Any,
    capital_eur: float = CAPITAL_EUR,
    risk_pct: float = RISK_PCT,
) -> pd.DataFrame:
    """Convertit la liste de trades du backtest en DataFrame avec sizing."""
    if not trades:
        return pd.DataFrame(columns=["Pips_Nets", "Pips_Bruts", "result", "position_size_lots", "pnl"])

    from app.backtest.sizing import compute_position_size, expected_pnl_eur

    df_t = pd.DataFrame(trades)
    df_t["entry_time"] = pd.to_datetime(df_t["entry_time"])
    df_t = df_t.set_index("entry_time")
    df_t["Pips_Nets"] = df_t["pips_net"].astype(float)
    df_t["Pips_Bruts"] = df_t["pips_net"].astype(float)
    df_t["result"] = df_t["result"].astype(str)

    entry_prices = df_t["entry_price"].astype(float).values
    signals_signed = df_t["signal"].astype(int).values
    sl_prices = np.where(
        signals_signed == 1,
        entry_prices - cfg.sl_points * cfg.pip_size,
        entry_prices + cfg.sl_points * cfg.pip_size,
    )
    lots = np.array([
        compute_position_size(ep, sl, capital_eur, risk_pct, cfg)
        for ep, sl in zip(entry_prices, sl_prices, strict=True)
    ], dtype=float)
    df_t["position_size_lots"] = lots
    df_t["pnl"] = expected_pnl_eur(df_t["Pips_Nets"].values, lots, cfg)
    return df_t


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> int:
    set_global_seeds()

    # â”€â”€ 1. Chargement USDCHF H4 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"Chargement {ASSET} {TF}...")
    df = load_asset(ASSET, TF)
    cfg = ASSET_CONFIGS[ASSET]
    print(f"  {len(df)} barres, {df.index.min().date()} â†’ {df.index.max().date()}")

    # â”€â”€ 2. Split train / test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df_train = df.loc[:TRAIN_CUTOFF]
    df_test = df.loc[TEST_START:]

    if df_train.empty:
        logger.error("Train vide aprÃ¨s cutoff %s", TRAIN_CUTOFF.date())
        return 1
    if df_test.empty:
        logger.error("Test vide aprÃ¨s %s", TEST_START.date())
        return 1

    print(f"\nTrain: {df_train.index.min().date()} â†’ {df_train.index.max().date()} ({len(df_train)} barres)")
    print(f"Test:  {df_test.index.min().date()} â†’ {df_test.index.max().date()} ({len(df_test)} barres)")

    half_cost = (cfg.spread_pips + cfg.slippage_pips) / 2.0

    # â”€â”€ 3. GÃ©nÃ©ration target Donchian sur train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\nGÃ©nÃ©ration target Donchian (N={DONCHIAN_N}, M={DONCHIAN_M}) sur train...")
    donchian_signals_train = _generate_donchian_signals(df_train)
    n_signals = int((donchian_signals_train != 0).sum())
    print(f"  {n_signals} signaux Donchian gÃ©nÃ©rÃ©s sur train")

    bt_donchian_train = run_deterministic_backtest(
        df=df_train,
        signals=donchian_signals_train,
        tp_pips=cfg.tp_points,
        sl_pips=cfg.sl_points,
        window_hours=cfg.window_hours,
        commission_pips=cfg.commission_pips,
        slippage_pips=half_cost,
        pip_size=cfg.pip_size,
        asset_config=cfg,
    )
    trades_donchian_train: list[dict] = bt_donchian_train.get("trades", [])
    print(f"  {len(trades_donchian_train)} trades Donchian sur train")

    if len(trades_donchian_train) < 20:
        logger.warning("Seulement %d trades Donchian train, insuffisant.", len(trades_donchian_train))
        return 1

    # â”€â”€ 4. PrÃ©paration features + labels pour entraÃ®nement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    features_train = _build_features_for_split(df_train)
    if features_train.empty:
        logger.error("Aucune feature pour train.")
        return 1

    entry_times_train = pd.to_datetime([t["entry_time"] for t in trades_donchian_train])
    common_train_idx = features_train.index.intersection(entry_times_train)
    if len(common_train_idx) < 10:
        logger.warning("Seulement %d features alignÃ©es avec trades train.", len(common_train_idx))
        return 1

    X_train = features_train.loc[common_train_idx]
    trades_df_train = _trades_to_dataframe(trades_donchian_train, cfg=cfg)
    pnl_aligned = trades_df_train.loc[
        trades_df_train.index.intersection(common_train_idx), "Pips_Nets"
    ]
    y_train = _build_target_winner(pnl_aligned)

    if y_train.nunique() < 2:
        logger.warning("Une seule classe dans y_train, impossible d'entraÃ®ner.")
        return 1

    # â”€â”€ 5. EntraÃ®nement rf â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\nEntraÃ®nement rf sur {len(X_train)} trades train...")
    model = _train_rf_model(X_train, y_train)

    # VÃ©rification rapide sur train
    pred_train = model.predict(X_train.values)
    acc_train = float((pred_train == y_train.values).mean())
    print(f"  Accuracy train: {acc_train:.3f}")
    print(f"  Distribution y_train: win={y_train.sum()}/{len(y_train)} ({y_train.mean():.1%})")

    # â”€â”€ 6. PrÃ©diction sur test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\nGÃ©nÃ©ration signaux sur test (â‰¥ {TEST_START.date()})...")

    # Construire features sur historique complet pour rolling indicators
    df_test_with_history = df.loc[:df_test.index[-1]]
    features_test_full = _build_features_for_split(df_test_with_history)
    features_test = features_test_full.loc[features_test_full.index.isin(df_test.index)]

    # Fix F1 : signaux primaires Donchian sur test
    donchian_signals_test = _generate_donchian_signals(df_test)
    n_primary_test = int((donchian_signals_test != 0).sum())
    print(f"  {n_primary_test} signaux Donchian primaires sur test")

    signals_test = _generate_model_signals(df_test, model, donchian_signals_test)
    n_test_signals = int((signals_test != 0).sum())
    print(f"  {n_test_signals} signaux conservÃ©s aprÃ¨s mÃ©ta-filter")

    if n_test_signals == 0:
        logger.warning("Aucun signal sur test.")
        bt_test = {"sharpe": 0.0, "wr": 0.0, "total_trades": 0, "total_pnl_pips": 0.0,
                     "max_drawdown_pips": 0.0, "profit_factor": 0.0, "mean_pnl_per_trade": 0.0, "trades": []}
    else:
        bt_test = run_deterministic_backtest(
            df=df_test,
            signals=signals_test,
            tp_pips=cfg.tp_points,
            sl_pips=cfg.sl_points,
            window_hours=cfg.window_hours,
            commission_pips=cfg.commission_pips,
            slippage_pips=half_cost,
            pip_size=cfg.pip_size,
            asset_config=cfg,
        )

    trades_test: list[dict] = bt_test.get("trades", [])
    print(f"  {len(trades_test)} trades sur test")

    # â”€â”€ 7. MÃ©triques â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    trades_test_df = _trades_to_dataframe(trades_test, cfg=cfg)

    metrics: dict[str, Any] = {}
    if not trades_test_df.empty:
        metrics = compute_metrics(trades_test_df, asset_cfg=cfg, capital_eur=CAPITAL_EUR, df=df_test)
    else:
        metrics = {"sharpe": 0.0, "trades": 0, "win_rate": 0.0, "max_dd_pct": 0.0}

    sharpe_bt = float(bt_test.get("sharpe", 0.0))
    wr_bt = float(bt_test.get("wr", 0.0))
    total_trades_bt = int(bt_test.get("total_trades", 0))
    max_dd_pips = float(bt_test.get("max_drawdown_pips", 0.0))

    print(f"\n{'='*60}")
    print(f"Phase B C5 â€” B4 USDCHF H4 terminÃ©.")
    print(f"  Sharpe (trades)       : {sharpe_bt:.3f}")
    print(f"  Sharpe (compute_metr) : {metrics.get('sharpe', 0):.3f}")
    print(f"  Trades                : {total_trades_bt}")
    print(f"  Win rate              : {wr_bt:.1%}")
    print(f"  Max DD (pips)         : {max_dd_pips:.1f}")
    print(f"  Profit factor         : {bt_test.get('profit_factor', 0):.2f}")

    # â”€â”€ 8. Sauvegarde â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    out: dict[str, Any] = {
        "hypothesis": "B4_C5_USDCHF_H4",
        "phase": "Phase B C5 â€” B4",
        "asset": ASSET,
        "tf": TF,
        "model": "rf",
        "hyperparams": HYPERPARAMS_TUNED[COUPLE_KEY]["params"],
        "threshold": HYPERPARAMS_TUNED[COUPLE_KEY]["threshold"],
        "features": list(FEATURES_SELECTED[COUPLE_KEY]),
        "donchian_params": {"N": DONCHIAN_N, "M": DONCHIAN_M},
        "train_cutoff": str(TRAIN_CUTOFF.date()),
        "test_start": str(TEST_START.date()),
        "capital_eur": CAPITAL_EUR,
        "risk_per_trade": RISK_PCT,
        "config": {
            "spread_pips": cfg.spread_pips,
            "slippage_pips": cfg.slippage_pips,
            "tp_points": cfg.tp_points,
            "sl_points": cfg.sl_points,
            "window_hours": cfg.window_hours,
            "pip_size": cfg.pip_size,
            "pip_value_eur": cfg.pip_value_eur,
        },
        "metrics": {
            "sharpe_backtest": sharpe_bt,
            "sharpe_compute_metrics": float(metrics.get("sharpe", 0.0)),
            "win_rate_backtest": wr_bt,
            "win_rate_compute_metrics": float(metrics.get("win_rate", 0.0)),
            "total_trades": total_trades_bt,
            "max_dd_pips": max_dd_pips,
            "profit_factor": float(bt_test.get("profit_factor", 0.0)),
            "max_dd_pct": float(metrics.get("max_dd_pct", 0.0)),
            "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
        },
        "n_train_trades_donchian": len(trades_donchian_train),
        "n_train_samples": len(X_train),
        "accuracy_train": float(acc_train),
    }

    out_path = Path("predictions/phase_b_c5_b4_usdchf_h4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nRÃ©sultats sauvegardÃ©s : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
