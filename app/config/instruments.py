"""Configuration par instrument — dataclasses immuables.

Un InstrumentConfig définit toutes les propriétés spécifiques à un actif :
taille du pip, timeframes, instruments macro corrélés, etc.
Ajouter un nouvel actif = créer une nouvelle sous-classe.

AssetConfig (v3 / Prompt 07) définit les coûts et paramètres
spécifiques au backtest déterministe multi-actif (spread, slippage, TP/SL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.logging import get_logger

logger = get_logger(__name__)

TargetMode = Literal[
    "triple_barrier",
    "forward_return",
    "directional_clean",
    "cost_aware_v2",
]


@dataclass(frozen=True)
class InstrumentConfig:
    """Configuration immuable pour un instrument de trading.

    Tous les champs sont gelés après construction. Pour dériver une config,
    utiliser `dataclasses.replace(config, name='BTCUSD')`.
    """

    name: str
    pip_size: float
    pip_value_eur: float
    timeframes: frozenset[str]
    primary_tf: str
    macro_instruments: frozenset[str]
    features_dropped: tuple[str, ...] = ()
    tp_sl_scale_factor: float = 1.0
    cost_aware_labeling: bool = False
    friction_pips: float = 1.5
    min_profit_pips_cost_aware: float = 3.0

    # ── Step 01 — Redéfinition de la cible ─────────────────────────────
    target_mode: TargetMode = "triple_barrier"
    target_horizon_hours: int = 24
    target_noise_threshold_atr: float = 0.5
    target_atr_period: int = 14
    target_k_atr: float = 1.0

    # ── Step 04 — Features de session ────────────────────────────
    session_encoding: Literal["ordinal", "one_hot"] = "one_hot"

    def __post_init__(self) -> None:
        if self.pip_size <= 0:
            raise ValueError(f"pip_size doit être > 0, reçu {self.pip_size}")
        if self.pip_value_eur <= 0:
            raise ValueError(f"pip_value_eur doit être > 0, reçu {self.pip_value_eur}")
        if self.primary_tf not in self.timeframes:
            raise ValueError(
                f"primary_tf '{self.primary_tf}' doit être dans timeframes {self.timeframes}"
            )
        if not self.timeframes:
            raise ValueError("timeframes ne peut pas être vide")
        if self.target_horizon_hours < 1:
            raise ValueError(
                f"target_horizon_hours doit être >= 1, reçu {self.target_horizon_hours}"
            )
        if self.target_noise_threshold_atr <= 0:
            raise ValueError(
                f"target_noise_threshold_atr doit être > 0, "
                f"reçu {self.target_noise_threshold_atr}"
            )
        if self.target_atr_period < 1:
            raise ValueError(
                f"target_atr_period doit être >= 1, reçu {self.target_atr_period}"
            )
        if self.target_k_atr <= 0:
            raise ValueError(
                f"target_k_atr doit être > 0, reçu {self.target_k_atr}"
            )
        if self.tp_sl_scale_factor <= 0:
            raise ValueError(
                f"tp_sl_scale_factor doit être > 0, reçu {self.tp_sl_scale_factor}"
            )

    def path_suffix(self, timeframe: str) -> str:
        """Retourne le suffixe de chemin pour un timeframe donné.

        >>> EurUsdConfig().path_suffix('H1')
        'EURUSD_H1_cleaned.csv'
        """
        return f"{self.name}_{timeframe}_cleaned.csv"


@dataclass(frozen=True)
class EurUsdConfig(InstrumentConfig):
    """Configuration EUR/USD — l'actif principal actuel."""

    name: str = "EURUSD"
    pip_size: float = 0.0001
    pip_value_eur: float = 1.0
    timeframes: frozenset[str] = frozenset({"H1", "H4", "D1"})
    primary_tf: str = "H1"
    macro_instruments: frozenset[str] = frozenset({"XAUUSD", "USDCHF"})
    features_dropped: tuple[str, ...] = (
        "Dist_EMA_9",
        "Dist_EMA_21",
        "Dist_EMA_20",
        "Log_Return",
        "CHF_Return",
        "Dist_EMA_50_D1",
        "BB_Width",
        "Hour_Cos",
        "Hour_Sin",
        "RSI_14_H4",
        "Dist_EMA_20_H4",
        "Dist_EMA_50_H4",
        "ATR_Norm",
        "Volatilite_Realisee_24h",
        "Range_ATR_ratio",
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "EMA_20_50_cross",
        "Volatility_Ratio",
    )


@dataclass(frozen=True)
class BtcUsdConfig(InstrumentConfig):
    """Configuration BTC/USD — futur actif cible.

    BTC n'a pas de pip au sens forex : 1$ = 1 unité.
    pip_size=1.0, pip_value_eur=0.92 (taux EUR/USD ≈ 1.08).
    Pas d'instruments macro corrélés connus.
    """

    name: str = "BTCUSD"
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    timeframes: frozenset[str] = frozenset({"H1", "H4", "D1"})
    primary_tf: str = "H1"
    macro_instruments: frozenset[str] = frozenset()
    features_dropped: tuple[str, ...] = (
        "Dist_EMA_9",
        "Dist_EMA_21",
        "Dist_EMA_20",
        "Log_Return",
        "CHF_Return",
        "Dist_EMA_50_D1",
        "BB_Width",
        "Hour_Cos",
        "Hour_Sin",
        "RSI_14_H4",
        "Dist_EMA_20_H4",
        "Dist_EMA_50_H4",
        "ATR_Norm",
        "Volatilite_Realisee_24h",
        "Range_ATR_ratio",
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "EMA_20_50_cross",
        "Volatility_Ratio",
    )
    tp_sl_scale_factor: float = 5.0


@dataclass(frozen=True)
class Us30Config(InstrumentConfig):
    """Configuration US30 (Dow Jones CFD) — indice, D1 primaire.

    US30 n'a pas de pip au sens forex : 1 point = 1 unité.
    pip_value_eur=0.92 (taux EUR/USD ≈ 1.08).
    Pas de macro_instruments pour les indices.
    features_dropped vide — on part de zéro en v2.
    """

    name: str = "USA30IDXUSD"
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    timeframes: frozenset[str] = frozenset({"D1", "H4"})
    primary_tf: str = "D1"
    macro_instruments: frozenset[str] = frozenset()
    features_dropped: tuple[str, ...] = ()
    tp_sl_scale_factor: float = 1.0

@dataclass(frozen=True)
class XauUsdConfig(InstrumentConfig):
    """Configuration XAUUSD (Or spot) — H4 primaire, mono-TF.

    Pip or : 1 pip-or = 1 cent = $0.01, donc pip_size=1.0.
    pip_value_eur=0.92 (taux EUR/USD ≈ 1.08).
    Pas de macro_instruments, pas de features_dropped.
    D1 chargé mais non utilisé dans build_features (gardé pour v2-02b).
    """

    name: str = "XAUUSD"
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    timeframes: frozenset[str] = frozenset({"H4", "D1"})
    primary_tf: str = "H4"
    macro_instruments: frozenset[str] = frozenset()
    features_dropped: tuple[str, ...] = ()
    tp_sl_scale_factor: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# AssetConfig — coûts et paramètres backtest déterministe multi-actif (Prompt 07)
# ═══════════════════════════════════════════════════════════════════════════════


_MEASURABLE_COSTS: frozenset[str] = frozenset(
    {"spread", "swap_long", "swap_short", "pip_value", "min_lot", "commission"}
)


class UnmeasuredCostError(RuntimeError):
    """Levée quand un backtest s'appuie sur un coût jamais relevé sur la plateforme."""


@dataclass(frozen=True)
class AssetConfig:
    """Coûts et paramètres spécifiques au backtest Donchian multi-actif.

    Champs obligatoires pour le moteur `run_deterministic_backtest` :
        spread_pips, slippage_pips, commission_pips, pip_size, pip_value_eur.

    Champs pour le sizing :
        min_lot, max_lot — limites de position.

    Champs pour TP/SL adaptatifs (ATR-based, prompts futurs) :
        tp_atr_multiplier, sl_atr_multiplier.
    """

    spread_pips: float
    slippage_pips: float
    commission_pips: float = 0.0
    pip_size: float = 1.0
    pip_value_eur: float = 0.92
    min_lot: float = 0.01
    max_lot: float = 10.0
    tp_atr_multiplier: float = 2.0
    sl_atr_multiplier: float = 1.0

    # TP/SL fixes en points (utilisés pour le backtest déterministe)
    tp_points: float = 200
    sl_points: float = 100
    window_hours: int = 120

    # ── Audit v6 F1 — Swap overnight (audit_v6_data_gaps.md §4) ─────────
    # Charge appliquée par nuit de détention. Convention signée :
    #   > 0 → crédit (carry favorable, ex: long AUDJPY ≈ +1.5 pip/nuit)
    #   < 0 → débit  (carry défavorable, ex: long EURUSD ≈ -0.5 pip/nuit)
    # Le PnL final est : pips_brut + nights_held × swap_*_pips_per_night.
    swap_long_pips_per_night: float = 0.0
    swap_short_pips_per_night: float = 0.0

    # ── Coûts MESURÉS, exprimés en % du notionnel ────────────────────────
    # Un spread/swap est physiquement un POURCENTAGE du notionnel, pas un
    # nombre de pips constant. Stocker des pips fixes sur une série où l'actif
    # est passé de 1290 à 7493 (US500, 2012→2026) sous-facture le début de
    # l'échantillon d'un facteur ~5.7. Ces champs, quand ils sont renseignés,
    # PRIMENT sur les champs en pips ci-dessus.
    #   spread_pct           : spread aller-simple, en % du prix (0.012 = 0.012 %)
    #   swap_*_pct_per_night : charge par nuit, en % du notionnel (signé :
    #                          < 0 = débit, > 0 = crédit)
    spread_pct: float | None = None
    swap_long_pct_per_night: float | None = None
    swap_short_pct_per_night: float | None = None

    # ── Provenance des coûts (leçon la plus coûteuse du projet) ──────────
    # Sur 5 estimations de coût confrontées à un relevé réel, 5 étaient
    # fausses, TOUJOURS dans le sens qui arrangeait l'hypothèse testée
    # (spreads ×15, ×9.2, ×6.3 ; swap ×3.8 ; carry JPY jusqu'au signe
    # inverse). Un coût non relevé à l'écran doit donc être traité comme FAUX.
    # Ce champ liste les grandeurs effectivement RELEVÉES sur la plateforme.
    # Noms reconnus : "spread", "swap_long", "swap_short", "pip_value",
    # "min_lot", "commission".  Voir `assert_costs_measured`.
    costs_measured: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.spread_pips < 0:
            raise ValueError(f"spread_pips doit être >= 0, reçu {self.spread_pips}")
        if self.slippage_pips < 0:
            raise ValueError(f"slippage_pips doit être >= 0, reçu {self.slippage_pips}")
        if self.tp_points <= 0:
            raise ValueError(f"tp_points doit être > 0, reçu {self.tp_points}")
        if self.sl_points <= 0:
            raise ValueError(f"sl_points doit être > 0, reçu {self.sl_points}")
        if self.window_hours <= 0:
            raise ValueError(f"window_hours doit être > 0, reçu {self.window_hours}")
        if self.spread_pct is not None and self.spread_pct < 0:
            raise ValueError(f"spread_pct doit être >= 0, reçu {self.spread_pct}")
        unknown = self.costs_measured - _MEASURABLE_COSTS
        if unknown:
            raise ValueError(
                f"costs_measured contient des noms inconnus : {sorted(unknown)}. "
                f"Attendu parmi {sorted(_MEASURABLE_COSTS)}."
            )

    @property
    def total_cost_pips(self) -> float:
        """Coût total aller-retour (spread + slippage + commission).

        ⚠️ Valeur CONSTANTE, héritée. Préférer `total_cost_pips_at(price)` dès
        qu'un `spread_pct` mesuré existe : le spread est un % du notionnel, et
        une constante en pips est fausse partout sauf au prix de référence.
        """
        return self.spread_pips + self.slippage_pips + self.commission_pips

    def spread_pips_at(self, price: float) -> float:
        """Spread aller-simple en pips au niveau de prix `price`.

        Utilise `spread_pct` (mesuré) s'il existe, sinon retombe sur la
        constante `spread_pips`.
        """
        if self.spread_pct is None:
            return self.spread_pips
        return (self.spread_pct / 100.0) * price / self.pip_size

    def total_cost_pips_at(self, price: float) -> float:
        """Coût aller-retour en pips au niveau de prix `price`.

        Le slippage reste proportionnel au spread (règle docs/cost_audit_v2.md
        §2) : on conserve le ratio slippage/spread de la config héritée.
        """
        spread = self.spread_pips_at(price)
        ratio = (self.slippage_pips / self.spread_pips) if self.spread_pips > 0 else 0.0
        return spread + spread * ratio + self.commission_pips

    def swap_pips_per_night_at(self, price: float, *, direction: str) -> float:
        """Swap overnight en pips (signé) au niveau de prix `price`.

        Args:
            direction: "long" ou "short".
        """
        if direction not in ("long", "short"):
            raise ValueError(f"direction doit être 'long' ou 'short', reçu {direction!r}")
        pct = self.swap_long_pct_per_night if direction == "long" else self.swap_short_pct_per_night
        if pct is None:
            return (
                self.swap_long_pips_per_night
                if direction == "long"
                else self.swap_short_pips_per_night
            )
        return (pct / 100.0) * price / self.pip_size

    def unmeasured(self, *fields: str) -> tuple[str, ...]:
        """Parmi `fields`, ceux qui n'ont JAMAIS été relevés sur la plateforme."""
        unknown = set(fields) - _MEASURABLE_COSTS
        if unknown:
            raise ValueError(f"Grandeurs inconnues : {sorted(unknown)}")
        return tuple(f for f in fields if f not in self.costs_measured)


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET_CONFIGS v4 (pivot A2) — coûts XTB Standard Account, capture 2026-05-15
# Source : docs/cost_audit_v2.md
# ═══════════════════════════════════════════════════════════════════════════════
#
# Légende :
#   spread_pips  : spread XTB Standard en unités natives de l'actif (points, pips, USD)
#   slippage_pips : estimé selon règle (0.2× spread majeures, 0.5× mineures)
#   commission    : 0 sur Standard Account (spreads variables, pas de commission)
#   total_cost_pips = spread_pips + slippage_pips + commission_pips
#
# Règle slippage (cf docs/cost_audit_v2.md §2) :
#   - Majeures liquides (US30, US500, GER30, EURUSD, XAUUSD) : slippage ≈ 0.2 × spread
#   - Mineures (XAGUSD, USOIL)                           : slippage ≈ 0.5 × spread
#   - Crypto (BTCUSD, ETHUSD)                            : slippage ≈ 1.0 × spread
#
# ⚠️ BUND désactivé (données indisponibles).
# ⚠️ BTCUSD/ETHUSD ajoutables sur demande utilisateur.

ASSET_CONFIGS: dict[str, AssetConfig] = {
    # ── US30 (Dow Jones CFD) ─────────────────────────────────────────────
    # v3: spread=3.0 + slippage=5.0 = 8.0  ← surestimation × 4.4
    # v4: vrai XTB Standard ~1.5 pts, slippage majeure 0.2×
    "US30": AssetConfig(
        spread_pips=1.5,
        slippage_pips=0.3,
        commission_pips=0.0,
        pip_size=1.0,          # 1 pt US30 = 1 USD
        pip_value_eur=0.92,
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # F6 — swaps estimés indices CFD (financement SOFR + ~5% = ~10%/an)
        # US30 ≈ 40000 pts × 10%/an / 365 ≈ $11/jour ≈ 11 pips (pip=$1)
        swap_long_pips_per_night=-11.0,
        swap_short_pips_per_night=1.0,
    ),
    # ── US500 (S&P 500 CFD) ──────────────────────────────────────────────
    # v3: spread=1.5 + slippage=2.0 = 3.5  ← surestimation × 5.8
    # v4: vrai XTB ~0.5 pts, slippage majeure 0.2×
    # ⚠️ pip_size = 0.1 (le S&P cote au dixième de point)
    "US500": AssetConfig(
        # ✅ RELEVÉ XTB : spread 0.92 pt = 0.012 % (pip_size 0.1 -> 9.2 pips).
        #    ⚠️ relevé en PRÉ-OUVERTURE = pire cas, donc conservateur. À raffiner
        #    par un relevé EN SÉANCE.
        #    L'ancienne valeur 0.5 cumulait DEUX erreurs : bug d'unité (l'audit
        #    disait "0.5 pts" avec "Pip: 0.1 pt", soit 5 pips, pas 0.5) ET
        #    sous-estimation x15 face au réel.
        spread_pips=9.2,
        slippage_pips=1.84,    # 0.2 x spread (majeure liquide)
        commission_pips=0.0,
        pip_size=0.1,          # 1 pt S&P = 0.1 (cotation au dixième)
        pip_value_eur=0.092,   # 0.1 pt x 0.92 ≈ 0.092 EUR
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # ⚠️ swap SHORT jamais relevé — cf. assert_costs_measured.
        swap_long_pips_per_night=-16.0,
        swap_short_pips_per_night=2.0,
        # ✅ MESURÉS (% du notionnel) — priment sur les pips ci-dessus.
        #    L'indice est passé de 1290 (2012) à 7493 (2026) : une constante en
        #    pips sous-facture le début de l'échantillon d'un facteur ~5.7.
        spread_pct=0.012,
        swap_long_pct_per_night=-0.021,     # = -7.7 %/an sur 365 nuits
        costs_measured=frozenset({"spread", "swap_long"}),
    ),
    # ── GER30 (DAX 40 CFD) ───────────────────────────────────────────────
    # v3: spread=2.0 + slippage=3.0 = 5.0  ← surestimation × 4.2
    # v4: vrai XTB ~1.0 pt, slippage majeure 0.2×
    "GER30": AssetConfig(
        # ✅ RELEVÉ XTB : spread 9.2 pts = 0.036 % (l'estimation 1.0 était x9.2 trop basse).
        #    ⚠️ relevé en PRÉ-OUVERTURE = pire cas, conservateur. À raffiner en séance.
        spread_pips=9.2,
        slippage_pips=1.84,
        commission_pips=0.0,
        pip_size=1.0,          # 1 pt DAX = 1 EUR
        pip_value_eur=1.0,
        tp_points=400,
        sl_points=200,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # ⚠️ swap SHORT jamais relevé — cf. assert_costs_measured.
        swap_long_pips_per_night=-5.0,
        swap_short_pips_per_night=0.5,
        # ✅ MESURÉS (% du notionnel).
        spread_pct=0.036,
        swap_long_pct_per_night=-6.2 / 365.0,   # -6.2 %/an relevé
        costs_measured=frozenset({"spread", "swap_long"}),
    ),
    # ── XAUUSD (Or spot) ─────────────────────────────────────────────────
    # v3: spread=25.0 + slippage=10.0 = 35.0  ← surestimation × 100
    # v4: spread XTB ≈ 0.30 USD, slippage majeure 0.2× ≈ 0.05
    # Convention : pip_size = 1.0 USD (1 "big figure")
    "XAUUSD": AssetConfig(
        spread_pips=0.30,
        slippage_pips=0.05,
        commission_pips=0.0,
        pip_size=1.0,          # 1 pip XTB GOLD = 1 USD (big figure)
        pip_value_eur=0.92,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
        # F6 — swaps estimés
        # XAU $3000 × ~3%/an / 365 ≈ $0.25/jour ≈ 0.25 pips/nuit (pip=$1)
        # Marge broker → arrondi à -1 long, -0.5 short
        swap_long_pips_per_night=-1.0,
        swap_short_pips_per_night=-0.5,
    ),
    # ── XAGUSD (Argent spot) ─────────────────────────────────────────────
    # v3: spread=30.0 + slippage=15.0 = 45.0  ← surestimation × 1285
    # v4: spread XTB ≈ 0.025 USD, slippage mineure 0.5× ≈ 0.0125 USD
    # ⚠️ pip_size = 0.001 (1 "pip" SILVER = 1 millième de USD)
    # FIX (2026-05-30) — spread/slippage doivent être EN PIPS (×pip_size pour
    # obtenir des USD), comme le swap. Les anciennes valeurs (0.025 / 0.01)
    # étaient en USD → coût en prix = 0.025×0.001 ≈ 0.000025 USD, soit ~1000×
    # trop faible (silver « gratuit »). Pour 0.025 USD de spread avec
    # pip_size=0.001 il faut spread_pips=25 ; slippage 0.5× → 12.5.
    # ⚠️ Toujours des estimations XTB — à confirmer en démo.
    "XAGUSD": AssetConfig(
        spread_pips=25.0,      # = 25 × 0.001 = 0.025 USD aller-retour
        slippage_pips=12.5,    # = 0.0125 USD (mineure : 0.5 × spread)
        commission_pips=0.0,
        pip_size=0.001,        # 1 pip XTB SILVER = 0.001 USD (pipette)
        pip_value_eur=0.92,
        tp_points=300,         # = 0.30 USD soit ~1.5 % du prix spot typique
        sl_points=150,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
        # F6 — swaps estimés
        # XAG $30 × ~3%/an / 365 ≈ $0.0025/jour ≈ 2.5 pips/nuit (pip=$0.001)
        # Avec markup broker
        swap_long_pips_per_night=-8.0,
        swap_short_pips_per_night=-4.0,
    ),
    # ── USOIL (WTI Crude CFD) ────────────────────────────────────────────
    # v3: spread=4.0 + slippage=3.0 = 7.0  ← surestimation × 100
    # v4: spread XTB ≈ 0.05 USD, slippage mineure 0.5× ≈ 0.02
    "USOIL": AssetConfig(
        # 🔧 BUG D'UNITÉ CORRIGÉ (x100) : docs/cost_audit_v2.md donne
        #    "Spread moyen : 0.05 USD" avec "Pip : 0.01 USD" -> 5 pips, pas 0.05.
        #    Même classe de bug que XAGUSD et ETHUSD, corrigés en leur temps.
        #    ⚠️ TOUJOURS PAS RELEVÉ sur la plateforme : valeur d'audit, pas de mesure.
        spread_pips=5.0,
        slippage_pips=2.0,     # 0.02 USD / 0.01 = 2 pips (mineure, 0.5 x spread)
        commission_pips=0.0,
        pip_size=0.01,         # 1 pip WTI = 0.01 USD
        pip_value_eur=0.92,
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
        # F6 — swaps estimés (contango futures, défavorable des deux côtés)
        swap_long_pips_per_night=-0.5,
        swap_short_pips_per_night=-0.5,
    ),
    # ── EURUSD (Forex) — NOUVEAU en v4, absent de v3 ────────────────────
    "EURUSD": AssetConfig(
        spread_pips=0.7,
        slippage_pips=0.2,
        commission_pips=0.0,
        pip_size=0.0001,       # 1 pip forex standard = 4ème décimale
        pip_value_eur=10.0,    # 1 pip × 1 lot standard (100k) ≈ 10 USD ≃ 9.2 EUR
        tp_points=20,          # 20 pips
        sl_points=10,          # 10 pips
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # F6 — swaps estimés (différentiel Fed-BCE 2026)
        swap_long_pips_per_night=-0.8,
        swap_short_pips_per_night=0.1,
    ),
    # ⚠️ BUND désactivé : données indisponibles
    # "BUND": AssetConfig(...),

    # ── GBPUSD (Forex) — NOUVEAU C1, PROVISOIRE ──────────────────────────
    # ⚠️ PROVISOIRE — à valider en démo MT5 (Symbol Specifications)
    "GBPUSD": AssetConfig(
        spread_pips=0.9,       # ≈ 0.9 pip XTB Standard
        slippage_pips=0.2,     # majeure : 0.2× spread
        commission_pips=0.0,
        pip_size=0.0001,       # 1 pip forex = 4ème décimale
        pip_value_eur=9.2,     # 1 pip × 1 lot standard ≈ 10 USD ≈ 9.2 EUR
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # F6 — swaps estimés (différentiel BoE-Fed)
        swap_long_pips_per_night=-1.2,
        swap_short_pips_per_night=0.4,
    ),
    # ── USDCHF (Forex) — NOUVEAU C1, PROVISOIRE ──────────────────────────
    # ⚠️ PROVISOIRE — à valider en démo
    "USDCHF": AssetConfig(
        spread_pips=1.0,
        slippage_pips=0.2,
        commission_pips=0.0,
        pip_size=0.0001,
        pip_value_eur=10.5,    # CHF base, valeur EUR variable selon taux
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # F6 — swaps estimés (CHF taux SNB historiquement faibles/négatifs)
        swap_long_pips_per_night=0.3,
        swap_short_pips_per_night=-1.1,
    ),

    # ── BTCUSD (Bitcoin spot) — NOUVEAU C1, PROVISOIRE ───────────────────
    # Source : XTB.com → Crypto → BITCOIN — spread variable selon marché
    # ⚠️ PROVISOIRE — à valider en démo MT5 (Symbol Specifications)
    "BTCUSD": AssetConfig(
        # ✅ RELEVÉ XTB : spread 0.302 % du notionnel (189.5 USD au prix du relevé).
        #    L'estimation 30 USD était x6.3 trop basse.
        spread_pips=189.5,
        slippage_pips=189.5,   # crypto : 1.0x spread (forte volatilité)
        commission_pips=0.0,
        pip_size=1.0,          # 1 pip BTC = 1 USD (big figure)
        pip_value_eur=0.92,    # 1 USD ≈ 0.92 EUR
        tp_points=2000,        # 2000 USD soit ~3-5% du prix BTC typique
        sl_points=1000,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
        # ⚠️ swap SHORT jamais relevé. C'est le chiffre le PLUS important encore
        #    manquant du projet : les longs paient 35.4 %/an, donc le côté short
        #    reçoit peut-être une part substantielle de ce financement. C'est le
        #    seul endroit de tout le panier d'instruments où le carry pourrait
        #    être POSITIF. Ne PAS estimer — relever.
        swap_long_pips_per_night=-16.0,
        swap_short_pips_per_night=-3.0,
        # ✅ MESURÉS (% du notionnel). BTC est passé de 2 255 (2017) à 81 166
        #    (2026) : une constante en pips est fausse d'un facteur ~36.
        spread_pct=0.302,
        swap_long_pct_per_night=-35.4 / 365.0,   # -35.4 %/an relevé
        costs_measured=frozenset({"spread", "swap_long"}),
    ),
    # ── ETHUSD (Ethereum spot) — NOUVEAU C1, PROVISOIRE ──────────────────
    # ⚠️ PROVISOIRE — à valider en démo
    # FIX (2026-06-01) — spread/slippage doivent être EN PIPS (×pip_size pour
    # obtenir des USD), comme le swap (cf. même bug corrigé sur XAGUSD). Les
    # anciennes valeurs (3.0 / 3.0) étaient en USD → coût en prix = 3×0.01 =
    # 0.03 USD, ~100× trop faible. Pour 3 USD de spread avec pip_size=0.01 il
    # faut spread_pips=300 ; slippage crypto 1.0× → 300. ⚠️ Estimations XTB.
    "ETHUSD": AssetConfig(
        spread_pips=300.0,     # = 300 × 0.01 = 3 USD spread typique
        slippage_pips=300.0,   # = 3 USD (crypto : 1.0× spread)
        commission_pips=0.0,
        pip_size=0.01,         # 1 pip ETH = 0.01 USD (cotation au centime)
        pip_value_eur=0.92,
        tp_points=10000,       # 100 USD soit ~3-5% du prix ETH typique
        sl_points=5000,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
        # F6 — swaps estimés (financement crypto CFD ~10%/an)
        # ETH $3000 × 10%/an / 365 ≈ $0.82/jour ≈ 82 pips/nuit (pip=$0.01)
        # Short rebate faible/négatif chez CFD broker
        swap_long_pips_per_night=-80.0,
        swap_short_pips_per_night=-10.0,
    ),
    # ── USDJPY (Forex) — NOUVEAU H2 (Asian Range), PROVISOIRE ────────────
    # ⚠️ PROVISOIRE — à valider en démo XTB MT5 (Symbol Specifications)
    # Pair JPY : pip_size = 0.01 (2ème décimale), spread XTB Standard ~1.0 pip
    "USDJPY": AssetConfig(
        spread_pips=1.0,
        slippage_pips=0.2,     # majeure : 0.2× spread
        commission_pips=0.0,
        pip_size=0.01,         # pair JPY : 1 pip = 0.01 JPY (2ème décimale)
        pip_value_eur=6.1,     # 1 pip × 1 lot std (100k USD) = 1000 JPY ≈ $6.67 ≈ 6.1 EUR @USDJPY=150
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # F6 — swaps estimés (différentiel BoJ-Fed : Fed ~5%, BoJ ~0-0.5%)
        # Long USDJPY = long USD/short JPY = carry positif (Fed > BoJ)
        # Markup broker → avantage long modeste, short fortement défavorable
        swap_long_pips_per_night=0.5,
        swap_short_pips_per_night=-1.5,
    ),
    # ── Paires JPY (crosses) — NOUVEAU, PROVISOIRE (carry research) ───────
    # ⚠️ PROVISOIRE — spreads/swaps à valider en démo XTB MT5.
    # Toutes : pip_size=0.01, pip_value_eur≈6.1 (1 pip × 1 lot std = 1000 JPY).
    # Le swap encode le carry : taux locaux 2026 ≈ AUD 4.35 %, GBP 4.5 %,
    # EUR 2.5 %, JPY 0.5 % → long cross = carry positif, short = fortement négatif.
    "AUDJPY": AssetConfig(
        spread_pips=1.8,
        slippage_pips=0.4,     # mineure : ~0.2-0.5× spread
        commission_pips=0.0,
        pip_size=0.01,
        pip_value_eur=6.1,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # Carry AUD(4.35%)-JPY(0.5%) ≈ +3.85 %/an, markup broker déduit
        swap_long_pips_per_night=0.9,
        swap_short_pips_per_night=-2.4,
    ),
    "EURJPY": AssetConfig(
        spread_pips=1.6,
        slippage_pips=0.3,
        commission_pips=0.0,
        pip_size=0.01,
        pip_value_eur=6.1,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # Carry EUR(2.5%)-JPY(0.5%) ≈ +2 %/an, modeste
        swap_long_pips_per_night=0.3,
        swap_short_pips_per_night=-1.6,
    ),
    "GBPJPY": AssetConfig(
        spread_pips=2.5,       # cross volatile, spread plus large
        slippage_pips=0.6,
        commission_pips=0.0,
        pip_size=0.01,
        pip_value_eur=6.1,
        tp_points=20,
        sl_points=10,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # Carry GBP(4.5%)-JPY(0.5%) ≈ +4 %/an
        swap_long_pips_per_night=1.0,
        swap_short_pips_per_night=-2.6,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Garde anti-coût-estimé
# ═══════════════════════════════════════════════════════════════════════════════
#
# Historique qui justifie ce garde-fou (memory/cost-estimates-always-wrong.md) :
# sur 5 estimations de coût confrontées à un relevé réel, 5 étaient fausses,
# TOUJOURS dans le sens qui arrangeait l'hypothèse testée —
#   spread US500 ×15 · spread GER30 ×9.2 · spread BTCUSD ×6.3 ·
#   swap crypto ×3.8 · carry JPY jusqu'au SIGNE INVERSE.
# Trois verdicts « GO » ont été fabriqués par des coûts optimistes. Un coût non
# relevé à l'écran doit donc faire ÉCHOUER le screen, pas le biaiser en silence.


def assert_costs_measured(asset: str, *fields: str, allow_estimated: bool = False) -> None:
    """Refuse de continuer si un coût requis n'a jamais été relevé sur XTB.

    À appeler en tête de tout screen, AVANT le backtest.

    Args:
        asset: clé de `ASSET_CONFIGS`.
        *fields: grandeurs requises ("spread", "swap_long", "swap_short",
            "pip_value", "min_lot", "commission").
        allow_estimated: si True, n'émet qu'un avertissement. À n'utiliser que
            pour une exploration explicitement étiquetée non-concluante.

    Raises:
        UnmeasuredCostError: si `allow_estimated` est False et qu'au moins une
            grandeur requise est estimée.
    """
    cfg = ASSET_CONFIGS.get(asset)
    if cfg is None:
        raise KeyError(f"{asset} absent de ASSET_CONFIGS")

    missing = cfg.unmeasured(*fields)
    if not missing:
        return

    msg = (
        f"{asset} : coût(s) JAMAIS relevé(s) sur la plateforme -> {list(missing)}.\n"
        f"  Les 5 dernières estimations confrontées au réel étaient fausses (x3.8 à x15,\n"
        f"  toujours dans le sens favorable). Un verdict bâti là-dessus ne vaut rien.\n"
        f"  -> Relever ces valeurs dans l'app XTB (voir docs/checklist_couts_xtb.md),\n"
        f"    les écrire dans ASSET_CONFIGS['{asset}'] et ajouter leur nom à\n"
        f"    costs_measured. Pour une exploration non-concluante : allow_estimated=True."
    )
    if allow_estimated:
        logger.warning("COÛTS ESTIMÉS ACCEPTÉS — %s", msg)
        return
    raise UnmeasuredCostError(msg)
