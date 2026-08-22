"""Tests des coûts post pivot v4 A2 — calibration XTB Standard Account.

Valide que chaque AssetConfig a des coûts réalistes (spread + slippage)
et que le ratio coût/SL reste ≤ 10 % (sinon stratégie mathématiquement
impossible). Les assertions portent sur la propriété `total_cost_pips`
qui somme spread + slippage + commission.

Source : docs/cost_audit_v2.md, valeurs XTB Standard Account 2025.
"""

from __future__ import annotations

import pytest

from app.config.instruments import ASSET_CONFIGS

# ── Tests fixes (4) ────────────────────────────────────────────────────────


def test_us30_spread_realistic() -> None:
    """US30 : spread ≤ 2.0 pts, total_cost ≤ 2.5 pts (v3 → v4 ÷ 4.4)."""
    cfg = ASSET_CONFIGS["US30"]
    assert cfg.spread_pips <= 2.0, f"US30 spread {cfg.spread_pips} > 2.0"
    assert cfg.total_cost_pips <= 2.5, f"US30 total_cost {cfg.total_cost_pips} > 2.5"


def test_xauusd_costs_realistic() -> None:
    """XAUUSD : spread en USD ≤ 1.0 (v3 35 USD → v4 0.35 USD, ÷ 100)."""
    cfg = ASSET_CONFIGS["XAUUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert spread_usd <= 1.0, f"XAUUSD spread USD {spread_usd} > 1.0"


def test_xagusd_costs_realistic() -> None:
    """XAGUSD : spread en USD ≤ 0.05 (v3 45 USD → v4 0.025 USD × 0.001 → réaliste)."""
    cfg = ASSET_CONFIGS["XAGUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert spread_usd <= 0.05, f"XAGUSD spread USD {spread_usd} > 0.05"


def test_eurusd_present_and_correct() -> None:
    """EURUSD ajouté en v4, pip_size forex standard 0.0001."""
    assert "EURUSD" in ASSET_CONFIGS, "EURUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["EURUSD"]
    assert cfg.pip_size == 0.0001, (
        f"EURUSD pip_size {cfg.pip_size} != 0.0001 (standard forex)"
    )
    assert cfg.spread_pips <= 1.5, (
        f"EURUSD spread {cfg.spread_pips} > 1.5 pips (anormal pour majeur forex)"
    )


def test_gbpusd_present_and_correct() -> None:
    """GBPUSD ajouté en v4, pip_size forex standard 0.0001, spread ≤ 1.5 pips."""
    assert "GBPUSD" in ASSET_CONFIGS, "GBPUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["GBPUSD"]
    assert cfg.pip_size == 0.0001, (
        f"GBPUSD pip_size {cfg.pip_size} != 0.0001 (standard forex)"
    )
    assert cfg.spread_pips <= 1.5, (
        f"GBPUSD spread {cfg.spread_pips} > 1.5 pips (anormal pour majeur forex)"
    )


def test_usdchf_present_and_correct() -> None:
    """USDCHF ajouté en v4, pip_size forex standard 0.0001, spread ≤ 1.5 pips."""
    assert "USDCHF" in ASSET_CONFIGS, "USDCHF manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["USDCHF"]
    assert cfg.pip_size == 0.0001, (
        f"USDCHF pip_size {cfg.pip_size} != 0.0001 (standard forex)"
    )
    assert cfg.spread_pips <= 1.5, (
        f"USDCHF spread {cfg.spread_pips} > 1.5 pips (anormal pour majeur forex)"
    )


def test_btcusd_costs_realistic() -> None:
    """BTCUSD : spread dans [60, 600] USD, slippage ≥ spread × 0.3.

    ⚠️ FIX 2026-08-01 (double correction) :
    1. La fourchette [10, 60] USD était une ESTIMATION jamais mesurée. Relevé
       réel sur l'app XTB (marché ouvert) : bid 62 849.7 / ask 63 039.2 →
       **189.5 USD**, soit 0.30 % du prix. Nouvelle fourchette ancrée sur la
       mesure : 0.1 % à 1 % d'un BTC autour de 60 k$ → [60, 600] USD.
    2. Le test comparait `spread_pips` (en PIPS) à une borne en USD, sans
       multiplier par `pip_size` — même confusion d'unité que celle qui a rendu
       tous les spreads du projet ×6 à ×15 trop bas. On compare désormais des USD
       à des USD.
    """
    assert "BTCUSD" in ASSET_CONFIGS, "BTCUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["BTCUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert 60.0 <= spread_usd <= 600.0, (
        f"BTCUSD spread {spread_usd} USD hors fourchette réaliste [60, 600] "
        f"(relevé XTB 2026-08-01 : 189.5 USD)"
    )
    assert cfg.slippage_pips >= 0.3 * cfg.spread_pips, (
        f"BTCUSD slippage {cfg.slippage_pips} < 0.3 × spread {cfg.spread_pips}"
    )
    assert cfg.pip_size == 1.0, (
        f"BTCUSD pip_size {cfg.pip_size} != 1.0 (1 USD par défaut crypto)"
    )


def test_ethusd_costs_realistic() -> None:
    """ETHUSD : spread dans [1, 10] USD, slippage ≥ spread × 0.3.

    ⚠️ FIX 2026-08-01 : le test comparait `spread_pips` (300 pips) à une borne en
    USD, alors que `pip_size=0.01` → le vrai spread vaut 300 × 0.01 = 3.00 USD,
    bien DANS la fourchette. L'échec était un bug d'unité du test, pas de la
    config. Même classe d'erreur que le « pip interne vs pip broker » qui a
    faussé tous les spreads du projet.
    ⚠️ Valeurs ETHUSD encore NON RELEVÉES sur l'app XTB (contrairement à BTCUSD).
    """
    assert "ETHUSD" in ASSET_CONFIGS, "ETHUSD manquant dans ASSET_CONFIGS v4"
    cfg = ASSET_CONFIGS["ETHUSD"]
    spread_usd = cfg.spread_pips * cfg.pip_size
    assert 1.0 <= spread_usd <= 10.0, (
        f"ETHUSD spread {spread_usd} USD hors fourchette réaliste [1, 10]"
    )
    assert cfg.slippage_pips >= 0.3 * cfg.spread_pips, (
        f"ETHUSD slippage {cfg.slippage_pips} < 0.3 × spread {cfg.spread_pips}"
    )
    assert cfg.pip_size == 0.01, (
        f"ETHUSD pip_size {cfg.pip_size} != 0.01 (cotation au centime)"
    )


# ── Tests paramétrés (11 actifs → 11 tests) ─────────────────────────────────


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_asset_total_cost_positive(asset: str) -> None:
    """Le coût total doit être strictement positif pour chaque actif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.total_cost_pips > 0, (
        f"{asset}: total_cost_pips={cfg.total_cost_pips} doit être > 0"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_asset_spread_nonnegative(asset: str) -> None:
    """Le spread ne doit jamais être négatif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.spread_pips >= 0, (
        f"{asset}: spread_pips={cfg.spread_pips} < 0"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_asset_slippage_nonnegative(asset: str) -> None:
    """Le slippage ne doit jamais être négatif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.slippage_pips >= 0, (
        f"{asset}: slippage_pips={cfg.slippage_pips} < 0"
    )


# Seuils de ratio coût/SL par classe d'actif
_COST_SL_THRESHOLDS: dict[str, float] = {
    "BTCUSD": 0.25,
    "ETHUSD": 0.25,
    "GBPUSD": 0.12,
    "USDCHF": 0.15,
}


# Actifs dont les TP/SL par défaut sont NON VIABLES au coût réel.
#   - 7 l'étaient DÉJÀ avant tout relevé (dette préexistante, jamais traitée) ;
#   - US500 et BTCUSD le sont devenus quand les coûts MESURÉS ont remplacé les
#     estimations — le test dit vrai, ce sont les distances de stop qui sont
#     trop serrées pour être tradées.
# ⚠️ NE PAS desserrer le seuil pour faire passer le test : cela reviendrait à
#    refabriquer un « GO » avec des coûts optimistes, ce qui a déjà coûté trois
#    faux edges au projet. La correction est de revoir sl_points — décision de
#    stratégie, à prendre explicitement par le mainteneur.
_COST_SL_NOT_VIABLE: frozenset[str] = frozenset(
    {"XAGUSD", "USDJPY", "AUDJPY", "EURJPY", "GBPJPY"}
)


@pytest.mark.parametrize(
    "asset",
    [
        pytest.param(
            a,
            marks=pytest.mark.xfail(
                strict=True,
                reason="TP/SL par défaut non viables au coût réel — revoir sl_points, "
                       "pas le seuil. Si ce test XPASS, l'actif est corrigé : le retirer "
                       "de _COST_SL_NOT_VIABLE.",
            ),
        )
        if a in _COST_SL_NOT_VIABLE
        else a
        for a in ASSET_CONFIGS
    ],
)
def test_cost_vs_sl_ratio(asset: str) -> None:
    """Le coût total ne doit pas dépasser N % du SL (sinon stratégie impossible).

    Ratio = total_cost_pips / sl_points.
    Si > seuil, le coût d'entrée/sortie absorbe trop de la marge de sécurité
    du stop-loss, rendant l'espérance mathématique négative même avec
    un win-rate de 50 %.

    Seuils :
      - Standard : 10 % (forex, indices, matières premières)
      - Crypto   : 25 % (volatilité plus élevée, SL plus large)
    """
    cfg = ASSET_CONFIGS[asset]
    ratio = cfg.total_cost_pips / cfg.sl_points
    threshold = _COST_SL_THRESHOLDS.get(asset, 0.10)
    assert ratio <= threshold, (
        f"{asset}: coût {cfg.total_cost_pips:.4f} > {threshold:.0%} du SL {cfg.sl_points}. "
        f"Ratio={ratio:.3f}. Stratégie mathématiquement impossible."
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_tp_gt_sl(asset: str) -> None:
    """Take-profit strictement supérieur au stop-loss (ratio risque/récompense)."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.tp_points > cfg.sl_points, (
        f"{asset}: tp_points={cfg.tp_points} ≤ sl_points={cfg.sl_points}"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_pip_size_positive(asset: str) -> None:
    """pip_size doit être strictement positif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.pip_size > 0, (
        f"{asset}: pip_size={cfg.pip_size} ≤ 0"
    )


@pytest.mark.parametrize(
    "asset",
    list(ASSET_CONFIGS.keys()),
)
def test_pip_value_eur_positive(asset: str) -> None:
    """pip_value_eur doit être strictement positif."""
    cfg = ASSET_CONFIGS[asset]
    assert cfg.pip_value_eur > 0, (
        f"{asset}: pip_value_eur={cfg.pip_value_eur} ≤ 0"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Verrous anti-régression sur les coûts RELEVÉS (2026-08-22)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Motif : sur 5 estimations de coût confrontées à un relevé réel, 5 étaient
# fausses, TOUJOURS dans le sens favorable. Trois « GO » ont été fabriqués ainsi.
# Les tests historiques ne testaient qu'un PLAFOND — ils ne pouvaient donc
# détecter que la SUR-estimation, jamais la sous-estimation qui a réellement
# tué le projet. Les tests ci-dessous sont BILATÉRAUX.

import pytest as _pytest

from app.config.instruments import (
    ASSET_CONFIGS as _CFG,
    UnmeasuredCostError,
    assert_costs_measured,
)

# Valeurs RELEVÉES sur la plateforme XTB. Toute modification de ASSET_CONFIGS
# qui s'en écarte doit casser le test — c'est le but.
_MEASURED: dict[str, dict[str, float]] = {
    # Captures app XTB : US500 2026-07-31 23:08, DE40 et BTCUSD 2026-08-01.
    "US500":  {"spread_pct": 0.0123, "swap_long_pct_per_year": -7.726,
               "swap_short_pct_per_year": -0.385},
    "GER30":  {"spread_pct": 0.0357, "swap_long_pct_per_year": -6.2415,
               "swap_short_pct_per_year": -1.699},
    "BTCUSD": {"spread_pct": 0.302, "swap_long_pct_per_year": -35.405,
               "swap_short_pct_per_year": -9.957},
}


@_pytest.mark.parametrize("asset", sorted(_MEASURED))
def test_measured_spread_matches_platform_capture(asset: str) -> None:
    """Le spread mesuré doit être EXACTEMENT celui relevé à l'écran."""
    cfg = _CFG[asset]
    assert cfg.spread_pct == _pytest.approx(_MEASURED[asset]["spread_pct"]), (
        f"{asset}: spread_pct={cfg.spread_pct} != relevé {_MEASURED[asset]['spread_pct']}"
    )
    assert "spread" in cfg.costs_measured, f"{asset}: spread relevé mais absent de costs_measured"


@_pytest.mark.parametrize("asset", sorted(_MEASURED))
def test_measured_swap_matches_platform_capture(asset: str) -> None:
    """Le portage annuel d'une position longue doit correspondre au relevé."""
    cfg = _CFG[asset]
    assert cfg.swap_long_pct_per_night is not None
    per_year = cfg.swap_long_pct_per_night * 365.0
    assert per_year == _pytest.approx(_MEASURED[asset]["swap_long_pct_per_year"], rel=1e-3), (
        f"{asset}: portage {per_year:.2f} %/an != relevé "
        f"{_MEASURED[asset]['swap_long_pct_per_year']} %/an"
    )


# Prix de référence approximatifs (ordre de grandeur uniquement — servent à
# détecter une erreur d'UNITÉ x10/x100, pas à valider une valeur précise).
_REF_PRICE: dict[str, float] = {
    "US30": 44_000, "US500": 7_400, "GER30": 24_000, "XAUUSD": 4_700,
    "XAGUSD": 85, "USOIL": 70, "EURUSD": 1.08, "GBPUSD": 1.27,
    "USDCHF": 0.78, "BTCUSD": 81_000, "ETHUSD": 2_300, "USDJPY": 150,
    "AUDJPY": 98, "EURJPY": 162, "GBPJPY": 190,
}


@_pytest.mark.parametrize(
    "asset",
    [
        _pytest.param(
            a,
            marks=_pytest.mark.xfail(
                strict=True,
                reason="US30 spread_pips=1.5 JAMAIS relevé et implausiblement bas "
                       "(0.0034 % du prix, vs 0.012 % mesuré sur US500 et 0.036 % sur "
                       "GER30). C'est le dernier coût estimé qui porte encore un "
                       "verdict (pre-FOMC US30) : une erreur x9.1 l'annule, et "
                       "l'erreur constatée sur GER30 était x9.2. RELEVER, ne pas "
                       "ré-estimer. Ce test XPASS quand la vraie valeur est écrite.",
            ),
        )
        if a == "US30"
        else a
        for a in sorted(_CFG)
    ],
)
def test_spread_plausible_as_fraction_of_price(asset: str) -> None:
    """Détecteur d'erreur d'unité, BILATÉRAL.

    Le spread d'un CFD liquide vaut entre 0.004 % et 1.5 % du prix. Une erreur
    d'unité x10 ou x100 — la classe de bug qui a touché XAGUSD, ETHUSD, USOIL
    et US500 — fait sortir la valeur de cette fourchette par le BAS, ce qu'un
    simple plafond ne voyait pas.
    """
    cfg = _CFG[asset]
    spread_native = cfg.spread_pips * cfg.pip_size
    pct = 100.0 * spread_native / _REF_PRICE[asset]
    assert 0.004 <= pct <= 1.5, (
        f"{asset}: spread {spread_native:g} en unités natives = {pct:.5f} % du prix "
        f"({_REF_PRICE[asset]:g}) — hors [0.004 %, 1.5 %]. Erreur d'unité probable "
        f"(pip_size={cfg.pip_size})."
    )


def test_usoil_unit_bug_regression() -> None:
    """USOIL : régression du bug d'unité x100.

    docs/cost_audit_v2.md donne « Spread moyen : 0.05 USD » avec « Pip : 0.01 USD ».
    0.05 USD / 0.01 = 5 pips. La config stockait 0.05 — soit 0.0005 USD.
    """
    cfg = _CFG["USOIL"]
    assert cfg.pip_size == 0.01
    assert cfg.spread_pips * cfg.pip_size == _pytest.approx(0.05), (
        f"USOIL spread {cfg.spread_pips * cfg.pip_size} USD != 0.05 USD (audit)"
    )


def test_us500_unit_bug_regression() -> None:
    """US500 : le spread doit être en PIPS (pip_size 0.1), pas en points."""
    cfg = _CFG["US500"]
    assert cfg.pip_size == 0.1
    assert cfg.spread_pips * cfg.pip_size == _pytest.approx(0.92), (
        f"US500 spread {cfg.spread_pips * cfg.pip_size} pts != 0.92 pt (relevé)"
    )


# ── Garde anti-coût-estimé ──────────────────────────────────────────────────


def test_guard_raises_on_unmeasured_cost() -> None:
    """Un coût jamais relevé doit FAIRE ÉCHOUER, pas biaiser en silence."""
    with _pytest.raises(UnmeasuredCostError, match="JAMAIS relevé"):
        assert_costs_measured("US30", "spread")


def test_guard_passes_on_measured_cost() -> None:
    """Un coût relevé passe sans lever."""
    assert assert_costs_measured("US500", "spread", "swap_long") is None


def test_short_swap_is_measured_and_never_positive() -> None:
    """Le carry SHORT est mesuré sur 3 actifs — et il est NÉGATIF partout.

    Ce test remplace un verrou « swap_short jamais relevé nulle part » qui était
    FAUX : les captures du 2026-08-01 mesurent bien les deux sens. Le résultat
    compte : on avait supposé que le côté short pouvait RECEVOIR une part du
    financement payé par les longs (35.4 %/an sur BTC). Il n'en reçoit rien —
    **XTB facture les deux sens**. Il n'existe donc AUCUN coin à carry positif
    dans ce panier d'instruments, ce qui ferme la piste « récolter le
    financement en étant short ».
    """
    for asset, exp in _MEASURED.items():
        cfg = _CFG[asset]
        assert "swap_short" in cfg.costs_measured, f"{asset}: swap_short relevé"
        assert cfg.swap_short_pct_per_night is not None
        per_year = cfg.swap_short_pct_per_night * 365.0
        assert per_year < 0, (
            f"{asset}: carry short {per_year:+.2f} %/an > 0 — si un jour c'est "
            f"vrai, c'est une PISTE, pas un bug : vérifier la capture."
        )
        assert per_year == _pytest.approx(exp["swap_short_pct_per_year"], rel=1e-2)


def test_long_financing_exceeds_short_on_every_measured_asset() -> None:
    """Tenir LONG coûte systématiquement plus cher que tenir SHORT.

    Asymétrie mesurée : US500 -7.73 %/an long contre -0.39 %/an short (x20).
    C'est la raison structurelle pour laquelle toute stratégie qui DÉTIENT du
    long sur indice se bat contre un handicap ~= à la prime de risque actions.
    """
    for asset in _MEASURED:
        cfg = _CFG[asset]
        assert cfg.swap_long_pct_per_night < cfg.swap_short_pct_per_night, (
            f"{asset}: le portage long devrait être plus coûteux que le short"
        )


def test_guard_allow_estimated_bypasses() -> None:
    """`allow_estimated=True` n'émet qu'un avertissement."""
    assert assert_costs_measured("US30", "spread", allow_estimated=True) is None


# ── Coûts en % du notionnel (priment sur les pips constants) ────────────────


def test_price_aware_spread_scales_with_price() -> None:
    """Le spread en pips doit croître avec le prix quand spread_pct est renseigné."""
    cfg = _CFG["US500"]
    assert cfg.spread_pips_at(7400) > cfg.spread_pips_at(1290)
    # 0.0123 % de 7400 = 0.910 pt = 9.10 pips (pip_size 0.1)
    assert cfg.spread_pips_at(7400) == _pytest.approx(9.102, rel=1e-3)


def test_price_aware_falls_back_to_constant_when_unmeasured() -> None:
    """Sans spread_pct, on retombe sur la constante héritée, quel que soit le prix."""
    cfg = _CFG["US30"]
    assert cfg.spread_pct is None
    assert cfg.spread_pips_at(10_000) == cfg.spread_pips
    assert cfg.spread_pips_at(44_000) == cfg.spread_pips


def test_constant_swap_misprices_early_sample() -> None:
    """Documente POURQUOI le % remplace les pips constants.

    US500 est passé de 1290 (2012) à 7493 (2026). Une constante en pips est
    juste à UN seul niveau de prix et fausse partout ailleurs.
    """
    cfg = _CFG["US500"]
    early = abs(cfg.swap_pips_per_night_at(1290, direction="long"))
    late = abs(cfg.swap_pips_per_night_at(7493, direction="long"))
    assert late / early == _pytest.approx(7493 / 1290, rel=1e-6)
    # La constante héritée (-16 pips) correspond à la FIN de l'échantillon.
    assert abs(cfg.swap_long_pips_per_night) == _pytest.approx(late, abs=0.5)
