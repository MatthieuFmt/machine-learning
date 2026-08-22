#!/usr/bin/env python3
"""Re-verdict analytique des 3 signaux Phase 1, SANS les bougies brutes.

Pourquoi ce script existe
-------------------------
Les screens (`screen_pre_fomc`, `screen_orb_fine`, `screen_carry`) ont besoin des
CSV Dukascopy, qui vivent sur la machine du mainteneur. Depuis une session cloud,
les sources de données sont bloquées par la politique réseau → impossible de
re-lancer les screens.

MAIS le bug corrigé le 2026-06-09 (« DSR ×√252 ») est un bug d'ALGÈBRE, pas de
données : il portait uniquement sur la conversion Sharpe annualisé → Sharpe
par-période. On peut donc recalculer le verdict corrigé à partir des seules
statistiques résumées déjà publiées dans `docs/signaux_reels_phase1.md`
(Sharpe annualisé, fréquence, durée d'échantillon), car :

    SR_par_période = SR_annualisé / √(périodes par an)
    t_stat         = SR_par_période × √n_obs
    DSR_z          = SR_pp·√(n_obs−1)/√(1 − γ₃·SR_pp + (γ₄−1)/4·SR_pp²) − z_mix

Ce que ce script NE remplace PAS
--------------------------------
- Le bootstrap stationnaire (exige les retours trade par trade).
- Les valeurs exactes de skewness/kurtosis (ici : balayage de sensibilité).
- Une éventuelle erreur dans les screens eux-mêmes (backtest, coûts, fill).
→ Les screens en local restent la mesure de référence. Ce script donne le
  verdict ATTENDU, avec ses barres d'erreur, pour savoir quoi prioriser.

USAGE :
    python scripts/recheck_signals_from_stats.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analysis.edge_validation import deflated_sharpe  # noqa: E402


@dataclass(frozen=True)
class RecordedSignal:
    """Statistiques résumées d'un signal, telles que publiées (source citée)."""

    name: str
    asset: str
    annualized_sharpe: float
    periods_per_year: float  # trades/an (event-driven) ou 252 (position continue)
    years: float
    max_dd: float
    pre_registered: bool  # hypothèse publiée AVANT nos tests (littérature) ?
    source: str

    @property
    def n_obs(self) -> int:
        return int(round(self.periods_per_year * self.years))

    @property
    def sharpe_per_period(self) -> float:
        """Le coeur du fix : dé-annualisation du Sharpe."""
        return self.annualized_sharpe / np.sqrt(self.periods_per_year)


# Statistiques figées, recopiées de docs/signaux_reels_phase1.md (état 2026-06-09).
# Aucune n'est ré-optimisée ici : on ne fait que RE-JUGER des chiffres déjà publiés.
SIGNALS: list[RecordedSignal] = [
    RecordedSignal(
        name="Pre-FOMC drift",
        asset="US500",
        annualized_sharpe=0.70,
        periods_per_year=8.0,  # ~8 réunions FOMC par an
        years=16.0,  # calendrier FOMC chargé 2010 → 2026
        max_dd=0.05,  # « faible » dans le mémo
        pre_registered=True,  # Lucca & Moench (2015), Journal of Finance
        source="signaux_reels_phase1.md : Sharpe ~0,7 ; ~8 trades/an ; DD faible",
    ),
    RecordedSignal(
        name="Carry JPY (panier)",
        asset="AUDJPY+GBPJPY+EURJPY",
        annualized_sharpe=0.40,
        periods_per_year=252.0,  # position continue → retours quotidiens
        years=16.0,
        max_dd=0.235,  # milieu de la fourchette 17-30 %
        pre_registered=True,  # anomalie carry FX, littérature abondante
        source="signaux_reels_phase1.md : Sharpe ~0,4 ; continu ; DD 17-30 %",
    ),
    RecordedSignal(
        name="ORB US500 5 min",
        asset="US500",
        annualized_sharpe=0.17,
        periods_per_year=257.0,  # ~257 trades/an (1 par séance)
        years=11.0,  # données M5 2015 → 2026
        max_dd=0.067,
        pre_registered=True,  # Zarattini & Aziz (2023)
        source="signaux_reels_phase1.md : Sharpe 0,17 ; ~257 trades/an ; DD 6,7 %",
    ),
]

# Grille de sensibilité : on ne connaît pas skew/kurtosis exacts sans les trades.
# (0, 3) = gaussien ; (−0.5, 6) = queues épaisses réalistes pour du CFD.
MOMENT_SCENARIOS: list[tuple[str, float, float]] = [
    ("gaussien", 0.0, 3.0),
    ("queues épaisses", -0.5, 6.0),
]

# n_trials : 1 = hypothèse pré-enregistrée (publiée avant nos tests, on n'a rien
# cherché) ; 15 / 60 = ordres de grandeur du data-snooping cumulé du projet.
N_TRIALS_GRID: list[int] = [1, 15, 60]


def one_sided_p(t_stat: float, dof: int) -> float:
    """p-value unilatérale (H₁ : moyenne > 0) d'un t de Student."""
    return float(scipy_stats.t.sf(t_stat, df=max(dof, 1)))


def main() -> int:
    print("=" * 78)
    print("RE-VERDICT ANALYTIQUE DES 3 SIGNAUX — pile statistique corrigée")
    print("(recalcul depuis les stats publiées ; les screens locaux restent la")
    print(" mesure de référence, ceci en donne le résultat ATTENDU)")
    print("=" * 78)

    for sig in SIGNALS:
        sr_pp = sig.sharpe_per_period
        n = sig.n_obs
        t_stat = sr_pp * np.sqrt(n)
        p_t = one_sided_p(t_stat, dof=n - 1)

        print(f"\n{'─' * 78}")
        print(f"■ {sig.name}  [{sig.asset}]")
        print(f"  source : {sig.source}")
        print(f"  Sharpe annualisé publié : {sig.annualized_sharpe:.2f}")
        print(
            f"  → Sharpe PAR PÉRIODE (le fix) : {sr_pp:.4f}"
            f"   (= {sig.annualized_sharpe:.2f} / √{sig.periods_per_year:.0f})"
        )
        print(f"  n_obs : {n}  ({sig.periods_per_year:.0f}/an × {sig.years:.0f} ans)")
        print("\n  PREUVE PRIMAIRE (t-test unilatéral, indépendant du data-snooping) :")
        verdict_t = "significatif" if p_t < 0.05 else "NON significatif"
        print(f"     t = {t_stat:.2f}   p = {p_t:.4f}   → {verdict_t}")

        print("\n  DSR corrigé (pénalité de data-snooping) :")
        header = "     scénario        " + "".join(f"  n_trials={k:<3d}" for k in N_TRIALS_GRID)
        print(header)
        for label, skew, kurt in MOMENT_SCENARIOS:
            cells = []
            for n_trials in N_TRIALS_GRID:
                z, p = deflated_sharpe(
                    sr=sr_pp, n_trials=n_trials, n_obs=n, skew=skew, kurtosis=kurt
                )
                mark = "✅" if (p < 0.05) else "❌"
                cells.append(f"  {mark} z={z:+.2f} p={p:.3f}")
            print(f"     {label:<15s}" + "".join(cells))

        # Rappel des autres critères GO de la constitution (§5 CLAUDE.md).
        print("\n  Autres critères GO :")
        ok_sharpe = sig.annualized_sharpe >= 1.0
        ok_dd = sig.max_dd < 0.15
        ok_freq = sig.periods_per_year >= 30.0
        print(
            f"     Sharpe ≥ 1,0 : {'✅' if ok_sharpe else '❌'} ({sig.annualized_sharpe:.2f})   "
            f"MaxDD < 15 % : {'✅' if ok_dd else '❌'} ({sig.max_dd:.1%})   "
            f"≥ 30 trades/an : {'✅' if ok_freq else '❌'} ({sig.periods_per_year:.0f})"
        )
        if sig.pre_registered:
            print(
                "     ℹ️  Hypothèse PRÉ-ENREGISTRÉE (publiée avant nos tests) → la"
                " colonne n_trials=1 est la lecture défendable."
            )

    print(f"\n{'=' * 78}")
    print("LECTURE : le t-test est la preuve la plus robuste (il ne dépend pas du")
    print("nombre d'hypothèses testées). Un signal dont le t-test échoue est mort,")
    print("quelle que soit la colonne DSR choisie.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
