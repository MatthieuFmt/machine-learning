"""Harnais de recherche d'edge honnête — Phase 1.

Protocole appliqué (non négociable, cf. CLAUDE.md §5) :
1. Backtest HONNÊTE : `entry_on_next_open=True` (entrée à l'ouverture de la barre
   suivant le signal), coûts round-trip = `cfg.total_cost_pips`, swap overnight
   depuis la config.
2. SÉLECTION sur l'In-Sample uniquement (trades dont l'entrée < `oos_start`).
3. UN SEUL regard sur l'Out-Of-Sample : le meilleur candidat IS est évalué une
   fois sur l'OOS via `validate_edge`, et l'événement est journalisé dans le
   registre anti-snooping (`read_oos`) → incrémente automatiquement `n_trials`.
4. Sharpe annualisé routé par fréquence (anti-inflation E4).

Aucune décision GO ne doit être prise hors de ce harnais.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from app.analysis.edge_validation import validate_edge
from app.backtest.deterministic import run_deterministic_backtest
from app.backtest.metrics import sharpe_daily_from_trades

if TYPE_CHECKING:
    from app.config.instruments import AssetConfig

DEFAULT_CAPITAL_EUR = 10_000.0


@dataclass(frozen=True)
class EdgeResult:
    """Verdict honnête d'une stratégie sur un couple (actif, timeframe)."""

    asset: str
    timeframe: str
    label: str  # identifiant du candidat retenu (stratégie + params + TP/SL)
    tp_pips: float
    sl_pips: float
    is_sharpe: float
    is_trades: int
    oos_sharpe: float
    oos_trades: int
    oos_win_rate: float
    oos_dsr: float
    oos_p_value: float
    oos_max_dd_pct: float
    n_trials: int
    go: bool
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "✅ GO" if self.go else "❌ NO-GO"
        return (
            f"{verdict}  {self.asset}/{self.timeframe}  [{self.label}]  "
            f"IS Sharpe={self.is_sharpe:.2f} ({self.is_trades} tr) | "
            f"OOS Sharpe={self.oos_sharpe:.2f} DSR={self.oos_dsr:.2f} "
            f"(p={self.oos_p_value:.3f}) WR={self.oos_win_rate:.0%} "
            f"DD={self.oos_max_dd_pct:.0%} ({self.oos_trades} tr) "
            f"[n_trials={self.n_trials}]"
        )


def run_honest_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    cfg: AssetConfig,
    tp_pips: float,
    sl_pips: float,
) -> dict:
    """Backtest avec fill honnête + coûts XTB réels + swap.

    Le coût round-trip total vaut `cfg.total_cost_pips` (spread+slippage+commission) :
    on le répartit en passant `commission_pips=0` et `slippage_pips=total/2`, ce qui
    donne `cost_total = total_cost_pips` dans `run_deterministic_backtest`.
    """
    return run_deterministic_backtest(
        df=df,
        signals=signals,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        window_hours=cfg.window_hours,
        commission_pips=0.0,
        slippage_pips=cfg.total_cost_pips / 2.0,
        pip_size=cfg.pip_size,
        swap_long_pips_per_night=cfg.swap_long_pips_per_night,
        swap_short_pips_per_night=cfg.swap_short_pips_per_night,
        entry_on_next_open=True,
    )


def _split_trades(trades: list[dict], oos_start: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    """Sépare les trades en (in-sample, out-of-sample) selon l'heure d'ENTRÉE."""
    oos_start = pd.Timestamp(oos_start)
    if oos_start.tz is None:
        oos_start = oos_start.tz_localize("UTC")
    is_tr, oos_tr = [], []
    for t in trades:
        entry = pd.Timestamp(t["entry_time"])
        if entry.tz is None:
            entry = entry.tz_localize("UTC")
        (oos_tr if entry >= oos_start else is_tr).append(t)
    return is_tr, oos_tr


def _equity_and_df(
    trades: list[dict], cfg: AssetConfig, capital: float
) -> tuple[pd.Series, pd.DataFrame]:
    """Construit (equity €, trades_df['pnl']) pour `validate_edge`."""
    exit_times = pd.to_datetime([t["exit_time"] for t in trades], utc=True)
    pnl_eur = np.array([t["pips_net"] for t in trades], dtype=float) * cfg.pip_value_eur
    order = np.argsort(exit_times.values)
    exit_times = exit_times[order]
    pnl_eur = pnl_eur[order]
    equity = pd.Series(capital + np.cumsum(pnl_eur), index=exit_times)
    trades_df = pd.DataFrame({"pnl": pnl_eur}, index=exit_times)
    return equity, trades_df


def record_and_resolve_n_trials(
    prompt: str,
    hypothesis: str,
    sharpe: float,
    n_trades: int,
) -> int:
    """Journalise une lecture OOS et retourne le n_trials cumulé pour le DSR.

    À appeler par tout screen autonome JUSTE AVANT `validate_edge` : la lecture
    courante est enregistrée dans le registre anti-snooping (TEST_SET_LOCK.json)
    puis comptée dans la pénalité. n_trials = nombre d'hypothèses UNIQUES
    (clé prompt+hypothesis) → re-lancer le même screen n'inflate pas la pénalité,
    tester une nouvelle configuration si. Remplace les `n_trials=len(assets)`
    locaux qui sous-comptaient l'historique cumulé du projet (fix C4).
    """
    from app.testing.snooping_guard import n_unique_hypotheses, read_oos

    read_oos(prompt=prompt, hypothesis=hypothesis, sharpe=sharpe, n_trades=n_trades)
    return max(1, n_unique_hypotheses())


def evaluate_oos(
    df: pd.DataFrame,
    signals: pd.Series,
    cfg: AssetConfig,
    *,
    asset: str,
    timeframe: str,
    label: str,
    tp_pips: float,
    sl_pips: float,
    oos_start: pd.Timestamp,
    n_trials: int | None = None,
    capital: float = DEFAULT_CAPITAL_EUR,
    record_read: bool = True,
) -> EdgeResult:
    """Évalue UN candidat : backtest honnête, split IS/OOS, verdict OOS.

    Si `record_read=True`, journalise la lecture OOS dans le registre anti-snooping
    (compte comme un essai pour le DSR). N'appeler qu'une fois par hypothèse.
    """
    result = run_honest_backtest(df, signals, cfg, tp_pips, sl_pips)
    is_tr, oos_tr = _split_trades(result["trades"], oos_start)

    is_sharpe = sharpe_daily_from_trades(is_tr) if is_tr else 0.0

    if len(oos_tr) < 2:
        return EdgeResult(
            asset=asset, timeframe=timeframe, label=label,
            tp_pips=tp_pips, sl_pips=sl_pips,
            is_sharpe=is_sharpe, is_trades=len(is_tr),
            oos_sharpe=0.0, oos_trades=len(oos_tr), oos_win_rate=0.0,
            oos_dsr=float("nan"), oos_p_value=float("nan"), oos_max_dd_pct=0.0,
            n_trials=n_trials or 0, go=False,
            reasons=["OOS: moins de 2 trades — non évaluable"],
        )

    oos_sharpe = sharpe_daily_from_trades(oos_tr)
    equity, trades_df = _equity_and_df(oos_tr, cfg, capital)

    if record_read:
        from app.testing.snooping_guard import read_oos

        read_oos(
            prompt="edge_harness",
            hypothesis=f"{asset}/{timeframe}:{label}",
            sharpe=oos_sharpe,
            n_trades=len(oos_tr),
        )

    # Résolution de n_trials APRÈS la lecture OOS, pour que l'essai courant soit
    # compté dans la pénalité de data-snooping du DSR. On compte les hypothèses
    # UNIQUES (clé prompt+hypothesis) : re-lancer le même screen n'inflate pas la
    # pénalité, mais tester de nouvelles configurations si.
    if n_trials is None:
        from app.testing.snooping_guard import n_unique_hypotheses

        resolved_n_trials = max(1, n_unique_hypotheses())
    else:
        resolved_n_trials = n_trials

    # On passe le Sharpe annualisé HONNÊTE (routé par fréquence) pour le critère
    # « ≥ 1.0 » ; le DSR, lui, recalcule en interne un Sharpe par-période.
    report = validate_edge(
        equity, trades_df, n_trials=resolved_n_trials, annualized_sharpe=oos_sharpe
    )

    return EdgeResult(
        asset=asset, timeframe=timeframe, label=label,
        tp_pips=tp_pips, sl_pips=sl_pips,
        is_sharpe=is_sharpe, is_trades=len(is_tr),
        oos_sharpe=oos_sharpe, oos_trades=len(oos_tr),
        oos_win_rate=report.metrics.get("wr", 0.0),
        oos_dsr=report.metrics.get("dsr", float("nan")),
        oos_p_value=report.metrics.get("p_value", float("nan")),
        oos_max_dd_pct=report.metrics.get("max_dd", 0.0),
        n_trials=resolved_n_trials,
        go=report.go,
        reasons=list(report.reasons),
    )


def screen_candidates(
    df: pd.DataFrame,
    candidates: dict[str, pd.Series],
    cfg: AssetConfig,
    *,
    asset: str,
    timeframe: str,
    tp_sl_grid: list[tuple[float, float]],
    oos_start: pd.Timestamp,
    capital: float = DEFAULT_CAPITAL_EUR,
) -> EdgeResult:
    """Sélectionne le meilleur (candidat × TP/SL) sur l'IS, puis lit l'OOS UNE fois.

    `candidates` : {label: signaux}. La sélection se fait STRICTEMENT sur le
    Sharpe in-sample (entrée < oos_start). Seul le gagnant est confronté à l'OOS,
    ce qui ajoute exactement 1 essai au registre anti-snooping.

    Retourne l'EdgeResult du gagnant. Si aucun candidat n'a de trades IS, retourne
    un verdict NO-GO explicite (sans lecture OOS).
    """
    best_label: str | None = None
    best_signals: pd.Series | None = None
    best_tp = best_sl = 0.0
    best_is_sharpe = -np.inf

    for label, signals in candidates.items():
        for tp_pips, sl_pips in tp_sl_grid:
            result = run_honest_backtest(df, signals, cfg, tp_pips, sl_pips)
            is_tr, _ = _split_trades(result["trades"], oos_start)
            if len(is_tr) < 2:
                continue
            sr = sharpe_daily_from_trades(is_tr)
            if sr > best_is_sharpe:
                best_is_sharpe = sr
                best_label = f"{label} TP{tp_pips:g}/SL{sl_pips:g}"
                best_signals = signals
                best_tp, best_sl = tp_pips, sl_pips

    if best_signals is None:
        return EdgeResult(
            asset=asset, timeframe=timeframe, label="(aucun candidat IS)",
            tp_pips=0.0, sl_pips=0.0, is_sharpe=0.0, is_trades=0,
            oos_sharpe=0.0, oos_trades=0, oos_win_rate=0.0,
            oos_dsr=float("nan"), oos_p_value=float("nan"), oos_max_dd_pct=0.0,
            n_trials=0, go=False,
            reasons=["Aucun candidat ne produit ≥ 2 trades in-sample"],
        )

    return evaluate_oos(
        df, best_signals, cfg,
        asset=asset, timeframe=timeframe, label=best_label,
        tp_pips=best_tp, sl_pips=best_sl,
        oos_start=oos_start, n_trials=None, capital=capital,
        record_read=True,
    )
