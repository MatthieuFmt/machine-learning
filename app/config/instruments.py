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

    @property
    def total_cost_pips(self) -> float:
        """Coût total aller-retour (spread + slippage + commission)."""
        return self.spread_pips + self.slippage_pips + self.commission_pips


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
        spread_pips=0.5,
        slippage_pips=0.1,
        commission_pips=0.0,
        pip_size=0.1,          # 1 pt S&P = 0.1 (cotation au dixième)
        pip_value_eur=0.092,   # 0.1 pt × 0.92 ≈ 0.092 EUR
        tp_points=200,
        sl_points=100,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # F6 — swaps estimés indices CFD (financement SOFR + ~5% = ~10%/an)
        # US500 ≈ 6000 × 10%/an / 365 ≈ $1.64/jour = 16 pips (pip=$0.1)
        swap_long_pips_per_night=-16.0,
        swap_short_pips_per_night=2.0,
    ),
    # ── GER30 (DAX 40 CFD) ───────────────────────────────────────────────
    # v3: spread=2.0 + slippage=3.0 = 5.0  ← surestimation × 4.2
    # v4: vrai XTB ~1.0 pt, slippage majeure 0.2×
    "GER30": AssetConfig(
        spread_pips=1.0,
        slippage_pips=0.2,
        commission_pips=0.0,
        pip_size=1.0,          # 1 pt DAX = 1 EUR
        pip_value_eur=1.0,
        tp_points=400,
        sl_points=200,
        window_hours=120,
        min_lot=0.01,
        max_lot=10.0,
        # F6 — swaps estimés indices CFD (financement ESTR + ~5% = ~8%/an)
        # DAX ≈ 23000 × 8%/an / 365 ≈ €5/jour = 5 pips (pip=€1)
        swap_long_pips_per_night=-5.0,
        swap_short_pips_per_night=0.5,
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
    # v4: spread XTB ≈ 0.025 USD, slippage mineure 0.5× ≈ 0.01
    # ⚠️ pip_size = 0.001 (1 "pip" SILVER = 1 millième de USD)
    "XAGUSD": AssetConfig(
        spread_pips=0.025,
        slippage_pips=0.01,
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
        spread_pips=0.05,
        slippage_pips=0.02,
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
        spread_pips=30.0,      # ≈ 30 USD spread typique heures actives
        slippage_pips=30.0,    # crypto : 1.0× spread (forte volatilité)
        commission_pips=0.0,
        pip_size=1.0,          # 1 pip BTC = 1 USD (big figure)
        pip_value_eur=0.92,    # 1 USD ≈ 0.92 EUR
        tp_points=2000,        # 2000 USD soit ~3-5% du prix BTC typique
        sl_points=1000,
        window_hours=120,
        min_lot=0.01,
        max_lot=5.0,
        # F6 — swaps estimés (financement crypto CFD ~10%/an)
        # BTC $60k × 10%/an / 365 ≈ $16/jour ≈ 16 pips/nuit (pip=$1)
        # Short rebate généralement faible/négatif chez CFD broker
        swap_long_pips_per_night=-16.0,
        swap_short_pips_per_night=-3.0,
    ),
    # ── ETHUSD (Ethereum spot) — NOUVEAU C1, PROVISOIRE ──────────────────
    # ⚠️ PROVISOIRE — à valider en démo
    "ETHUSD": AssetConfig(
        spread_pips=3.0,       # ≈ 3 USD spread typique
        slippage_pips=3.0,     # crypto : 1.0× spread
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
