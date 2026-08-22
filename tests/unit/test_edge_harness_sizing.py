"""Sizing au risque dans `build_equity_curve` (correctif 2026-08-22).

L'ancien `_equity_and_df` faisait `pips_net x pip_value_eur`, soit 1.00 lot en
dur. Le gate `MaxDD < 15 %` mesurait donc un levier arbitraire, pas la
stratégie — et DSR/PSR/t-stat, calculés sur `equity.pct_change()`, en
héritaient. Ces tests verrouillent le comportement corrigé.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.analysis.edge_validation import max_drawdown
from app.config.instruments import ASSET_CONFIGS
from app.research.edge_harness import build_equity_curve

CFG = ASSET_CONFIGS["EURUSD"]


def _trades(pnls: list[float], entry_price: float = 1.1000, signal: int = 1) -> list[dict]:
    return [
        {
            "entry_time": f"2024-01-{i + 1:02d}T00:00:00Z",
            "exit_time": f"2024-01-{i + 1:02d}T12:00:00Z",
            "entry_price": entry_price,
            "signal": signal,
            "pips_net": p,
        }
        for i, p in enumerate(pnls)
    ]


def test_empty_trades_returns_empty() -> None:
    eq, td = build_equity_curve([], CFG, capital=5_000.0, sl_pips=10.0)
    assert len(eq) == 0 and len(td) == 0


def test_wider_stop_gives_smaller_position() -> None:
    """Le cœur du sizing au risque : stop 2x plus large -> position 2x plus petite."""
    tr = _trades([10.0])
    _, tight = build_equity_curve(tr, CFG, 5_000.0, sl_pips=10.0)
    _, wide = build_equity_curve(tr, CFG, 5_000.0, sl_pips=20.0)
    assert wide["pnl"].iloc[0] == pytest.approx(tight["pnl"].iloc[0] / 2.0, rel=1e-6)


def test_risk_per_trade_matches_risk_pct() -> None:
    """Un trade perdant exactement du SL doit coûter ~risk_pct du capital."""
    capital, sl = 5_000.0, 20.0
    eq, td = build_equity_curve(_trades([-sl]), CFG, capital, sl_pips=sl, risk_pct=0.02)
    assert td["pnl"].iloc[0] == pytest.approx(-capital * 0.02, rel=0.02)


def test_fixed_lot_mode_reproduces_legacy() -> None:
    """Le mode hérité reste disponible et inchangé (9 screens en dépendent)."""
    tr = _trades([10.0, -5.0])
    _, td = build_equity_curve(tr, CFG, 5_000.0, sizing="fixed_lot")
    assert td["pnl"].tolist() == pytest.approx([10.0 * CFG.pip_value_eur, -5.0 * CFG.pip_value_eur])


def test_equity_never_negative_and_stops_after_ruin() -> None:
    """L'equity ne doit jamais passer sous zéro.

    En dessous, `equity.pct_change()` change de signe et DSR/PSR/t-stat
    deviennent du bruit — silencieusement, dans l'ancienne version.
    """
    catastrophic = _trades([-100_000.0] * 5)
    eq, _ = build_equity_curve(catastrophic, CFG, 1_000.0, sl_pips=10.0)
    assert (eq >= 0).all(), f"equity négative : {eq.tolist()}"
    assert eq.iloc[-1] == 0.0


def test_sizing_changes_maxdd_materially() -> None:
    """MaxDD doit dépendre du sizing — c'est tout l'objet du correctif."""
    tr = _trades([50.0, -20.0, 40.0, -20.0, 30.0, -20.0] * 5)
    eq_fixed, _ = build_equity_curve(tr, CFG, 10_000.0, sizing="fixed_lot")
    eq_risk, _ = build_equity_curve(tr, CFG, 10_000.0, sl_pips=20.0)
    assert max_drawdown(eq_fixed) != pytest.approx(max_drawdown(eq_risk), rel=1e-3)


def test_short_trades_size_on_correct_stop_side() -> None:
    """Un short a son stop AU-DESSUS de l'entrée : la distance doit être la même."""
    long_tr = _trades([10.0], signal=1)
    short_tr = _trades([10.0], signal=-1)
    _, tdl = build_equity_curve(long_tr, CFG, 5_000.0, sl_pips=15.0)
    _, tds = build_equity_curve(short_tr, CFG, 5_000.0, sl_pips=15.0)
    assert tdl["pnl"].iloc[0] == pytest.approx(tds["pnl"].iloc[0], rel=1e-9)


def test_sizing_respects_lot_bounds() -> None:
    """La taille reste bornée par [min_lot, max_lot], même à capital extrême."""
    tr = _trades([-20.0])
    _, tiny = build_equity_curve(tr, CFG, 50.0, sl_pips=20.0)
    _, huge = build_equity_curve(tr, CFG, 10_000_000.0, sl_pips=20.0)
    per_lot = 20.0 * CFG.pip_value_eur
    assert abs(tiny["pnl"].iloc[0]) == pytest.approx(per_lot * CFG.min_lot, rel=1e-6)
    assert abs(huge["pnl"].iloc[0]) == pytest.approx(per_lot * CFG.max_lot, rel=1e-6)


def test_micro_lots_make_fx_viable_at_small_capital() -> None:
    """EURUSD au min_lot risque ~1.4 % d'un compte de 140 € — donc tradable.

    Contraste voulu avec les indices : c'est la raison pour laquelle le forex
    est la seule famille de ce panier qui passe la contrainte de taille de
    position à petit capital.
    """
    _, td = build_equity_curve(_trades([-20.0]), CFG, 140.0, sl_pips=20.0, risk_pct=0.02)
    assert abs(td["pnl"].iloc[0]) / 140.0 < 0.02


@pytest.mark.xfail(
    strict=True,
    reason="min_lot/pip_value_eur ne codent PAS la taille minimale réelle chez XTB. "
           "La config implique ~68 EUR de notionnel minimum sur US500 (prix 7400) "
           "alors que le relevé mainteneur donne 1 636 EUR, soit un facteur 24 — "
           "encore dans le sens favorable, qui fait paraître viables des comptes "
           "qui ne le sont pas. RELEVER contract_size + taille de lot minimale en "
           "euros de notionnel, puis retirer ce xfail.",
)
def test_min_lot_matches_real_notional_us500() -> None:
    """Verrou sur la granularité de position — non mesurée à ce jour."""
    cfg = ASSET_CONFIGS["US500"]
    eur_per_index_point = cfg.pip_value_eur / cfg.pip_size
    implied_notional = 7_400.0 * eur_per_index_point * cfg.min_lot
    assert implied_notional == pytest.approx(1_636.0, rel=0.10), (
        f"notionnel minimum implicite {implied_notional:.0f} EUR != 1 636 EUR relevé"
    )
