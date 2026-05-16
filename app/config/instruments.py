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
    ),
    # ⚠️ BUND désactivé : données indisponibles
    # "BUND": AssetConfig(...),
    # ⚠️ BTCUSD et ETHUSD ajoutables si demande utilisateur
}
