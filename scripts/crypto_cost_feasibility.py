#!/usr/bin/env python3
"""Quelles stratégies crypto restent possibles avec les VRAIS coûts XTB ?

Contexte
--------
Une seule famille crypto a été screenée à ce jour : `crypto_trend` (time-series
momentum D1, position continue long/short). Le relevé BTCUSD du 2026-08-01 la tue
sur le portage. Mais « la crypto » ≠ « le momentum multi-jours ». Ce script
répond à la vraie question : parmi les FAMILLES de stratégies crypto, lesquelles
sont éliminées par les coûts seuls, et lesquelles restent ouvertes ?

Méthode : « quel Sharpe faut-il JUSTE pour couvrir les frais ? »
---------------------------------------------------------------
Comparer un coût à un rendement absolu ne veut rien dire, et le comparer à
l'amplitude σ(h) non plus (ça supposerait que la stratégie capture TOUT le
mouvement — aucune ne le fait). La bonne mesure est le **Sharpe de seuil** :

    coût(h)   = spread_A/R + h × swap_par_nuit                 (% du notionnel)
    σ(h)      = σ_journalier × √h        (marche aléatoire)     (%)
    SR_trade  = coût(h) / σ(h)      ← Sharpe PAR TRADE requis pour être à zéro
    SR_annuel = SR_trade × √(trades par an)   ← Sharpe ANNUEL de seuil

Interprétation : si ta stratégie a un Sharpe brut de 0.6 et que le seuil est 0.43,
il te reste ~0.17 net. Si le seuil dépasse ton Sharpe brut, tu perds de l'argent
avec un signal pourtant correct.

Résultat structurel : le coût croît en **h** (le swap s'accumule chaque nuit),
l'amplitude en **√h**. Tenir longtemps dégrade donc mécaniquement le ratio —
exactement à l'envers de ce dont le suivi de tendance a besoin.

⚠️ Ce script élimine des familles sur le COÛT seul. Il ne prouve l'existence
d'aucun edge : passer ce filtre est nécessaire, pas suffisant.

USAGE :
    python scripts/crypto_cost_feasibility.py
    python scripts/crypto_cost_feasibility.py --daily-vol 3.0
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.instruments import ASSET_CONFIGS  # noqa: E402

# ── Relevé app XTB, BTCUSD, 2026-08-01 12:14, marché OUVERT ──────────────────
# bid 62 849.7 / ask 63 039.2 ; contrat 329.93 EUR pour 0.006 lot ;
# swap Vente −0.09 EUR / Achat −0.32 EUR.
SPREAD_RT_PCT = 189.5 / 62_849.7 * 100.0     # 0.302 % — un aller-retour = UN spread
SWAP_LONG_PCT = 0.32 / 329.93 * 100.0        # 0.0970 %/nuit
SWAP_SHORT_PCT = 0.09 / 329.93 * 100.0       # 0.0273 %/nuit

# Sharpe BRUT de référence pour une stratégie crypto correcte (littérature
# momentum crypto : 0.5-0.8 avant coûts). On prend le milieu, volontairement
# optimiste : si une famille ne passe pas même à 0.6, elle ne passera jamais.
GROSS_SHARPE_REF = 0.6
# Il faut qu'il reste quelque chose d'exploitable après frais.
NET_SHARPE_MIN = 0.20


@dataclass(frozen=True)
class Family:
    """Une famille de stratégies, caractérisée par sa durée de détention."""

    name: str
    hold_days: float          # nuits payées par trade (0 = clôturé le jour même)
    trades_per_year: float
    side: str                 # "long", "short" ou "both"
    tested: bool
    note: str

    def swap_pct_per_night(self) -> float:
        if self.side == "long":
            return SWAP_LONG_PCT
        if self.side == "short":
            return SWAP_SHORT_PCT
        return (SWAP_LONG_PCT + SWAP_SHORT_PCT) / 2.0

    def cost_per_trade_pct(self) -> float:
        return SPREAD_RT_PCT + self.hold_days * self.swap_pct_per_night()

    def cost_per_year_pct(self) -> float:
        return self.cost_per_trade_pct() * self.trades_per_year


FAMILIES: list[Family] = [
    Family("Momentum D1 (TSMOM)", 60.0, 6.0, "both", True,
           "LA seule famille screenée (screen_crypto_trend) — position continue"),
    Family("Tendance long-only", 45.0, 8.0, "long", False,
           "buy-the-trend classique"),
    Family("Swing 3 jours", 3.0, 40.0, "both", False,
           "cassure/pullback tenu quelques jours"),
    Family("Effet week-end", 3.0, 52.0, "long", False,
           "acheter vendredi soir, vendre lundi"),
    Family("Intraday quotidien", 0.0, 250.0, "both", False,
           "1 trade/jour, clôturé le soir → ZÉRO swap"),
    Family("Intraday hebdomadaire", 0.0, 52.0, "both", False,
           "1 trade/semaine, clôturé le jour même"),
    Family("Intraday rare (événementiel)", 0.0, 12.0, "both", False,
           "~1/mois, clôturé le jour même — le seul 'petit budget frais'"),
    Family("Tendance short-only", 45.0, 8.0, "short", False,
           "le swap short est 3.5× moins cher que le long"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Faisabilité des familles crypto vs coûts XTB.")
    p.add_argument("--daily-vol", default=3.0, type=float,
                   help="Volatilité journalière du BTC en %% (défaut 3.0 ≈ 57 %%/an).")
    args = p.parse_args()
    dv = args.daily_vol

    cfg = ASSET_CONFIGS["BTCUSD"]
    print("=" * 92)
    print("FAMILLES DE STRATÉGIES CRYPTO — éliminées par les COÛTS seuls ?")
    print(f"Relevé XTB BTCUSD 2026-08-01 : spread A/R {SPREAD_RT_PCT:.3f} %  ·  "
          f"swap long {SWAP_LONG_PCT:.4f} %/nuit  ·  short {SWAP_SHORT_PCT:.4f} %/nuit")
    print(f"Config du repo (alignée) : spread {cfg.spread_pips:.1f} pips  ·  "
          f"swap long {cfg.swap_long_pips_per_night:.1f} pips/nuit")
    print(f"Volatilité journalière supposée : {dv:.1f} %  →  σ(h) = {dv:.1f} × √h")
    print(f"Référence : Sharpe BRUT {GROSS_SHARPE_REF} (optimiste) ; il faut qu'il "
          f"reste ≥ {NET_SHARPE_MIN} net")
    print("=" * 92)
    print(f"\n{'Famille':<30}{'testée':>7}{'coût/trade':>12}{'coût/an':>9}"
          f"{'SR seuil':>10}{'SR net':>8}  verdict")
    print("-" * 92)

    survivors: list[tuple[Family, float]] = []
    for f in FAMILIES:
        cpt = f.cost_per_trade_pct()
        cpy = f.cost_per_year_pct()
        # σ sur l'horizon de détention ; un intraday « voit » ~1 journée d'amplitude.
        sigma = dv * max(f.hold_days, 1.0) ** 0.5
        sr_threshold = (cpt / sigma) * f.trades_per_year**0.5   # Sharpe ANNUEL de seuil
        sr_net = GROSS_SHARPE_REF - sr_threshold
        ok = sr_net >= NET_SHARPE_MIN
        if ok:
            survivors.append((f, sr_net))
        print(f"{f.name:<30}{'oui' if f.tested else '—':>7}{cpt:>11.3f}%{cpy:>8.1f}%"
              f"{sr_threshold:>10.2f}{sr_net:>8.2f}  "
              f"{'🟢 possible' if ok else '☠️ éliminée'}")

    print("-" * 92)
    print("\n« SR seuil » = Sharpe annuel que la stratégie doit atteindre JUSTE pour")
    print("payer les frais.  « SR net » = ce qu'il reste si le brut vaut "
          f"{GROSS_SHARPE_REF}.  Négatif = tu")
    print("perds de l'argent même avec un signal correct.")

    print("\nPOURQUOI TENIR LONGTEMPS EST STRUCTURELLEMENT PERDANT ICI :")
    print("  le coût croît en h (le swap s'accumule chaque nuit), l'amplitude en √h.")
    for h in (1, 3, 7, 30, 90):
        c = SPREAD_RT_PCT + h * SWAP_LONG_PCT
        s = dv * h**0.5
        print(f"    tenir {h:>3} jour(s) : frais {c:5.2f} %  vs  amplitude {s:5.1f} %"
              f"  →  {c / s:.0%} de l'amplitude")
    print("  C'est l'inverse exact de ce dont le suivi de tendance a besoin.")

    print(f"\n{'=' * 92}")
    if survivors:
        print("FAMILLES ENCORE OUVERTES (coût seul — l'edge reste ENTIÈREMENT à prouver) :")
        for f, net in sorted(survivors, key=lambda kv: -kv[1]):
            print(f"  🟢 SR net ≈ {net:.2f}  {f.name} — {f.note}")
    else:
        print("☠️  AUCUNE famille crypto ne passe le filtre de coût chez XTB.")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
