"""Moteur de backtest déterministe stateful bar-by-bar pour H03.

Règles :
- Une seule position à la fois (stateful).
- Entrée au Close de la barre où signal ≠ 0 ET pas de position ouverte.
- Sortie au premier de : TP touché, SL touché, window_hours écoulé.
- Si TP et SL touchés dans la même barre : SL prime (conservateur — pire cas
  réaliste, aligné avec app/backtest/simulator.py). Fix v5 du finding F3.
- Commission + slippage déduits à l'entrée ET à la sortie.
- Timeout : sortie au Close de la barre d'expiration (prix réel).
- Zéro ML, zéro filtre de régime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from app.config.instruments import AssetConfig


def run_deterministic_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    tp_pips: float,
    sl_pips: float,
    window_hours: int,
    commission_pips: float,
    slippage_pips: float,
    pip_size: float = 1.0,
    swap_long_pips_per_night: float = 0.0,
    swap_short_pips_per_night: float = 0.0,
    asset_config: AssetConfig | None = None,
    entry_on_next_open: bool = True,
) -> dict[str, Any]:
    """Backtest bar-by-bar avec TP/SL fixes, moteur stateful.

    Args:
        df: DataFrame avec colonnes Open, High, Low, Close, index=Time (datetime).
        signals: pd.Series 1=LONG, -1=SHORT, 0=FLAT, même index que df.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        window_hours: Durée max d'un trade en heures.
        commission_pips: Commission en pips (aller-retour par trade).
        slippage_pips: Slippage estimé en pips.
        pip_size: Taille d'un pip (ex: 0.0001 pour EURUSD, 1.0 pour XAUUSD).
        swap_long_pips_per_night: Audit v6 F1 — charge swap par nuit, position long.
            Convention signée : > 0 = crédit, < 0 = débit. Défaut 0 (legacy).
        swap_short_pips_per_night: Idem, position short.
        asset_config: F6 — si fourni, écrase silencieusement les valeurs par
            défaut de swap_long/swap_short à partir de la config. Permet
            d'éviter le bug du swap=0 silencieux quand l'appelant oublie
            les paramètres swap_* explicites. Si un appelant a passé des
            swap_*_pips_per_night non-zéro, ces valeurs explicites priment
            sur la config (= override manuel).
        entry_on_next_open: Fix C2-bt (2026-05-29) — fill honnête sans look-ahead
            d'exécution. Le signal n'étant connu qu'à la CLÔTURE de la barre i,
            l'entrée réaliste se fait à l'OUVERTURE de la barre i+1 (et la barre
            d'entrée est elle-même scannée pour le risque de gap intra-barre).
            Défaut False = comportement legacy (entrée au Close de la barre de
            signal, légèrement optimiste). **La recherche d'edge (Phase 1) DOIT
            passer True.** Le défaut restera False tant que l'ensemble du suite
            de tests/scripts n'aura pas été re-validé sur données réelles.

    Returns:
        dict avec clés:
            sharpe: float — Sharpe ratio annualisé.
            wr: float — Win rate (0.0 à 1.0).
            total_trades: int.
            total_pnl_pips: float.
            trades: list[dict] — détail de chaque trade.
    """
    # F6 — auto-extraction du swap depuis la config si fournie ET si l'appelant
    # n'a pas explicitement passé un swap non-zéro. Évite le bug du swap=0
    # silencieux quand un script de backtest oublie de passer les paramètres.
    if asset_config is not None:
        if swap_long_pips_per_night == 0.0:
            swap_long_pips_per_night = asset_config.swap_long_pips_per_night
        if swap_short_pips_per_night == 0.0:
            swap_short_pips_per_night = asset_config.swap_short_pips_per_night

    if "Time" in df.columns:
        df = df.set_index("Time")
    elif "time" in df.columns:
        df = df.set_index("time")

    # Alignement
    common_idx = df.index.intersection(signals.index)
    df = df.loc[common_idx]
    signals = signals.loc[common_idx]

    n = len(df)
    if n == 0:
        return _empty_result()

    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    times = df.index

    tp_dist = tp_pips * pip_size
    sl_dist = sl_pips * pip_size
    cost_per_side = commission_pips + slippage_pips  # entrée + sortie
    cost_total = cost_per_side * 2  # entrée ET sortie

    # Calcul du window en nombre de barres (fix F18)
    # Utilise le MODE (espacement le plus fréquent) au lieu de la moyenne.
    # La moyenne inclut les gaps weekends (48h) qui pour H1 donnent
    # typical_hours ≈ 1.5 → window_bars 30 % trop court.
    if n >= 2:
        diffs = pd.Series(times).diff().dt.total_seconds() / 3600.0
        diffs = diffs.dropna()
        if len(diffs) > 0:
            mode_vals = diffs.mode()
            typical_hours = float(mode_vals.iloc[0]) if len(mode_vals) > 0 else float(diffs.median())
        else:
            typical_hours = 0.0
        window_bars = max(1, int(round(window_hours / typical_hours))) if typical_hours > 0 else window_hours
    else:
        window_bars = window_hours

    trades: list[dict[str, Any]] = []
    i = 0

    while i < n:
        sig_val = int(signals.iloc[i])

        if sig_val == 0:
            i += 1
            continue

        signal = sig_val  # 1 ou -1

        # ── Détermination de la barre et du prix d'entrée ──────────────
        if entry_on_next_open:
            # Fill honnête (C2-bt) : entrée à l'ouverture de la barre suivante.
            # La barre d'entrée e est elle-même scannée (risque de gap).
            e = i + 1
            if e >= n:
                break  # pas de barre suivante : impossible d'entrer
            entry_time = times[e]
            entry_price = opens[e]
            scan_start, scan_end = e, e + window_bars
        else:
            # Legacy (optimiste) : entrée au Close de la barre de signal.
            entry_time = times[i]
            entry_price = closes[i]
            scan_start, scan_end = i + 1, i + window_bars

        if signal == 1:
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist

        pips_net = -cost_total
        result_type = "loss_timeout"
        exit_time = entry_time
        exit_price = entry_price
        hit = False

        # Boucle forward sur la fenêtre de détention.
        for idx in range(scan_start, scan_end + 1):
            if idx >= n:
                # Fin des données : sortie au dernier Close disponible.
                exit_idx = n - 1
                exit_time = times[exit_idx]
                exit_price = closes[exit_idx]
                if signal == 1:
                    pips_net = (exit_price - entry_price) / pip_size - cost_total
                else:
                    pips_net = (entry_price - exit_price) / pip_size - cost_total
                result_type = "loss_timeout"
                hit = True
                i = n
                break

            curr_high = highs[idx]
            curr_low = lows[idx]

            if signal == 1:
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low <= sl_price
            else:  # signal == -1
                tp_hit = curr_low <= tp_price
                sl_hit = curr_high >= sl_price

            if sl_hit:
                # SL-prime même barre (fix F3) : pire cas réaliste,
                # aligné avec _simulate_stateful_core.
                exit_time = times[idx]
                exit_price = sl_price
                pips_net = -sl_pips - cost_total
                result_type = "loss_sl"
                hit = True
                i = idx
                break
            if tp_hit:
                exit_time = times[idx]
                exit_price = tp_price
                pips_net = tp_pips - cost_total
                result_type = "win"
                hit = True
                i = idx
                break

        if not hit:
            # Timeout : sortie au Close de la dernière barre de la fenêtre.
            exit_idx = min(scan_end, n - 1)
            exit_time = times[exit_idx]
            exit_price = closes[exit_idx]
            if signal == 1:
                pips_net = (exit_price - entry_price) / pip_size - cost_total
            else:
                pips_net = (entry_price - exit_price) / pip_size - cost_total
            result_type = "loss_timeout"
            i = exit_idx

        # ── Audit v6 F1 — Charge swap overnight ────────────────────────
        # nights_held = nombre de jours civils entre entry et exit (V1 simple).
        _et = entry_time if isinstance(entry_time, pd.Timestamp) else pd.Timestamp(entry_time)
        _xt = exit_time if isinstance(exit_time, pd.Timestamp) else pd.Timestamp(exit_time)
        nights_held = max(0, (_xt.normalize() - _et.normalize()).days)
        swap_per_night = (
            swap_long_pips_per_night if signal == 1 else swap_short_pips_per_night
        )
        if nights_held > 0 and swap_per_night != 0.0:
            pips_net += nights_held * swap_per_night

        trades.append({
            "entry_time": str(entry_time),
            "exit_time": str(exit_time),
            "signal": signal,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pips_net": float(pips_net),
            "result": result_type,
            "nights_held": int(nights_held),
        })

    return _compute_metrics(trades)


def _compute_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcule les métriques à partir de la liste de trades."""
    if not trades:
        return _empty_result()

    pnls = np.array([t["pips_net"] for t in trades])
    total_trades = len(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    wr = float(len(wins) / total_trades) if total_trades > 0 else 0.0
    total_pnl = float(pnls.sum())

    # Sharpe annualisé — calculé sur les returns quotidiens de la courbe d'equity
    from app.backtest.metrics import sharpe_daily_from_trades
    sharpe = sharpe_daily_from_trades(trades)

    # Maximum drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = float(drawdown.min())

    # Profit factor
    if len(wins) > 0 and len(losses) > 0:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    elif len(wins) > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "sharpe": sharpe,
        "wr": wr,
        "total_trades": total_trades,
        "total_pnl_pips": total_pnl,
        "max_drawdown_pips": max_dd,
        "profit_factor": profit_factor,
        "mean_pnl_per_trade": float(pnls.mean()),
        "trades": trades,
    }


def _empty_result() -> dict[str, Any]:
    """Résultat vide quand aucun trade n'est généré."""
    return {
        "sharpe": 0.0,
        "wr": 0.0,
        "total_trades": 0,
        "total_pnl_pips": 0.0,
        "max_drawdown_pips": 0.0,
        "profit_factor": 0.0,
        "mean_pnl_per_trade": 0.0,
        "trades": [],
    }


def backtest_with_config(
    df: pd.DataFrame,
    signals: pd.Series,
    tp_pips: float,
    sl_pips: float,
    cfg: AssetConfig,
    *,
    half_cost: float | None = None,
    entry_on_next_open: bool = True,
    use_measured_pct: bool = True,
) -> dict[str, Any]:
    """Wrapper de `run_deterministic_backtest` qui extrait tous les paramètres
    de coût depuis une AssetConfig — incluant les swaps overnight.

    À utiliser dans tous les nouveaux scripts pour éviter le bug du swap=0
    silencieux. Le coût total (spread+slippage+commission) est convenu
    comme `cfg.total_cost_pips / 2` réparti à l'entrée et à la sortie,
    sauf override explicite via `half_cost`.

    Args:
        df: OHLCV avec index DatetimeIndex.
        signals: pd.Series 1=LONG, -1=SHORT, 0=FLAT, même index que df.
        tp_pips: Take-profit en pips.
        sl_pips: Stop-loss en pips.
        cfg: AssetConfig de l'instrument (fournit pip_size, window_hours,
            commission_pips, slippage_pips, spread_pips, swap_long/short).
        half_cost: Optionnel — override pour `slippage_pips` (défaut :
            coût total / 2). Permet de garder la convention existante
            "half_cost appliqué à l'entrée ET à la sortie".
        entry_on_next_open: Fill honnête à l'open de la barre suivante
            (défaut True). False = fill au close de la barre du signal, soit
            un prix connu seulement à l'instant du calcul du signal.
        use_measured_pct: Utilise les coûts MESURÉS en % du notionnel quand
            ils existent (US500, GER30, BTCUSD), évalués au prix médian de la
            fenêtre. Défaut True — un coût relevé prime toujours sur une
            constante estimée.

    Returns:
        Même format que `run_deterministic_backtest`.
    """
    if half_cost is None:
        if use_measured_pct and cfg.spread_pct is not None:
            # Coût MESURÉ en % du notionnel, évalué au prix médian de la fenêtre
            # testée. Une constante en pips n'est juste qu'à UN niveau de prix :
            # sur US500 2012-2026 (1290 -> 7493) elle se trompe d'un facteur ~5.7.
            ref_price = float(df["Close"].median())
            total_cost = cfg.total_cost_pips_at(ref_price)
        else:
            total_cost = cfg.total_cost_pips
        half_cost = total_cost / 2.0

    if use_measured_pct and cfg.swap_long_pct_per_night is not None:
        ref_price = float(df["Close"].median())
        swap_long = cfg.swap_pips_per_night_at(ref_price, direction="long")
        swap_short = cfg.swap_pips_per_night_at(ref_price, direction="short")
    else:
        swap_long = cfg.swap_long_pips_per_night
        swap_short = cfg.swap_short_pips_per_night

    return run_deterministic_backtest(
        df=df,
        signals=signals,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        # ⚠️ commission_pips=0.0 VOULU : `half_cost` dérive de total_cost_pips,
        #    qui INCLUT déjà la commission. Le moteur calcule
        #    cost_per_side = commission + slippage, donc repasser
        #    cfg.commission_pips ici la compterait DEUX FOIS (bug latent tant
        #    que toutes les commissions valent 0 — il ne l'aurait pas été
        #    longtemps sur un compte Pro).
        commission_pips=0.0,
        slippage_pips=half_cost,
        pip_size=cfg.pip_size,
        swap_long_pips_per_night=swap_long,
        swap_short_pips_per_night=swap_short,
        entry_on_next_open=entry_on_next_open,
    )
