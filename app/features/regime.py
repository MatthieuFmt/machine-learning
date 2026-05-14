"""Features de regime — volatilite, range/ATR, momentum macro, tendance long-terme.

Ces features sont calculees principalement sur D1 puis reparties sur H1 via merge_asof.
Elles ne doivent PAS etre utilisees comme features d'entrainement du modele
(risque de surapprentissage), mais servent aux filtres de regime (TrendFilter, VolFilter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def calc_volatilite_realisee(
    log_returns: pd.Series, window: int = 24
) -> pd.Series:
    """Volatilite realisee : ecart-type des log-returns sur `window` periodes.

    Args:
        log_returns: Serie de log-returns H1.
        window: Fenetre de calcul (24 = 24h).

    Returns:
        Serie de volatilite realisee.
    """
    return log_returns.rolling(window=window).std()


def calc_range_atr_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Ratio Range / ATR : detecte expansion (>1) vs contraction (<1).

    Args:
        high, low, close: Series OHLC.
        length: Periode ATR.

    Returns:
        Serie du ratio. +1e-10 evite la division par zero.
    """
    import pandas_ta as ta

    atr = ta.atr(high, low, close, length=length)
    return (high - low) / (atr + 1e-10)


def calc_rsi_d1_delta(
    rsi_d1: pd.Series, diff_periods: int = 3
) -> pd.Series:
    """Variation du RSI D1 sur `diff_periods` jours — momentum macro.

    Args:
        rsi_d1: Serie RSI calculee sur D1.
        diff_periods: Nombre de jours pour la difference.

    Returns:
        Serie de la variation du RSI.
    """
    return rsi_d1.diff(diff_periods)


def calc_dist_sma200_d1(
    close_d1: pd.Series, length: int = 200
) -> pd.Series:
    """Distance a la SMA200 D1 — vraie tendance long-terme (~9 mois).

    Utilisee par TrendFilter pour n'autoriser que les trades dans le sens
    de la tendance macro.

    Args:
        close_d1: Serie Close D1.
        length: Periode SMA (200 par defaut).

    Returns:
        Serie de la distance normalisee (Close - SMA) / Close.
    """
    import pandas_ta as ta

    sma200 = ta.sma(close_d1, length=length)
    return (close_d1 - sma200) / close_d1


# ══════════════════════════════════════════════════════════════════════
# Session-Aware Features (Step 04) — Microstructure FX
# ══════════════════════════════════════════════════════════════════════

def compute_session_id(index: pd.DatetimeIndex) -> pd.Series:
    """Détermine la session de trading pour chaque barre H1.

    Mapping (heures UTC) :
        0 = Tokyo     01:00-07:00 (+ gap 21:00-22:00)
        1 = London    07:00-12:00
        2 = NY        16:00-21:00
        3 = Overlap   12:00-16:00 (prioritaire sur London et NY)
        4 = Low liq   22:00-01:00 (prioritaire sur tout)

    Priorités d'assignation (ordre) :
        1. Low liq (22h-01h) — priorité maximale
        2. Overlap London-NY (12h-16h)
        3. London pure (7h-12h)
        4. NY pure (16h-21h)
        5. Tokyo (catch-all : 1h-7h, 21h-22h)

    Args:
        index: DatetimeIndex des barres H1.

    Returns:
        pd.Series[int8] de session_id, même index.
    """
    hours = index.hour
    sid = pd.Series(0, index=index, dtype=np.int8)

    # Priorité 1 : low liquidity (22h-01h)
    sid[(hours >= 22) | (hours == 0)] = 4
    # Priorité 2 : overlap London-NY (12h-16h)
    sid[(hours >= 12) & (hours < 16)] = 3
    # Priorité 3 : London pure (7h-12h)
    sid[(hours >= 7) & (hours < 12)] = 1
    # Priorité 4 : NY pure (16h-21h)
    sid[(hours >= 16) & (hours < 21)] = 2
    # Reste : Tokyo (1h-7h, 21h-22h) — déjà 0 par défaut

    return sid


def compute_session_open_range(
    high: pd.Series,
    low: pd.Series,
    session_id: pd.Series,
) -> pd.Series:
    """Range cumulé (High - Low) depuis le début de la session courante.

    Reset à chaque changement de session_id. À t=0 de session, retourne
    le range de la première barre.

    Args:
        high: Série High H1.
        low: Série Low H1.
        session_id: Série session_id (sortie de compute_session_id).

    Returns:
        pd.Series du range cumulé en prix natifs (pas normalisé).
    """
    # Détecter les changements de session
    session_change = session_id.diff().fillna(1).ne(0)
    segment_id = session_change.cumsum()

    cummax = high.groupby(segment_id, sort=False).cummax()
    cummin = low.groupby(segment_id, sort=False).cummin()

    return cummax - cummin


_SESSION_START: dict[int, int] = {0: 23, 1: 7, 2: 16, 3: 12, 4: 22}
_SESSION_DURATION: dict[int, int] = {0: 9, 1: 5, 2: 5, 3: 4, 4: 3}


def compute_relative_position_in_session(
    index: pd.DatetimeIndex,
    session_id: pd.Series,
) -> pd.Series:
    """Position relative dans la session ∈ [0, 1].

    0 = début de session, 1 = dernière barre de la session.
    Basé sur l'heure de la barre et les bornes fixes UTC.

    Args:
        index: DatetimeIndex des barres H1.
        session_id: Série session_id.

    Returns:
        pd.Series float ∈ [0, 1].
    """
    hours = index.hour
    result = pd.Series(0.0, index=index)

    for sid_val in [0, 1, 2, 3, 4]:
        mask = session_id == sid_val
        if not mask.any():
            continue
        start = _SESSION_START[sid_val]
        duration = _SESSION_DURATION[sid_val]
        # Conversion en Series pour permettre l'indexation modifiable
        elapsed = pd.Series(hours[mask].astype(float), index=index[mask])
        elapsed = elapsed - start
        # Correction midnight wrap (heures 0-7 pour Tokyo)
        elapsed[elapsed < 0] += 24.0
        result[mask] = elapsed / duration

    return result.clip(0.0, 1.0)


class SessionVolatilityScaler:
    """Scaler de volatilité conditionnel à la session — fit train-only.

    Analogue à StandardScaler mais calcule μ et σ par session.
    Le fit() ne doit voir QUE les données d'entraînement.

    Usage:
        scaler = SessionVolatilityScaler()
        scaler.fit(atr_norm_train, session_id_train)
        zscored = scaler.transform(atr_norm_all, session_id_all)
    """

    def __init__(self) -> None:
        self._stats: dict[int, tuple[float, float]] = {}  # session_id → (μ, σ)
        self._default_mu: float = 0.0
        self._default_sigma: float = 1.0

    def fit(
        self, atr_norm: pd.Series, session_id: pd.Series
    ) -> "SessionVolatilityScaler":
        """Calcule μ et σ par session sur les données d'entraînement uniquement.

        Args:
            atr_norm: Série ATR_Norm (déjà normalisée par Close).
            session_id: Série session_id correspondante.

        Returns:
            self pour chaînage.
        """
        valid = atr_norm.notna() & session_id.notna()
        atr_valid = atr_norm[valid]
        sid_valid = session_id[valid]

        for sid_val in [0, 1, 2, 3, 4]:
            mask = sid_valid == sid_val
            if mask.sum() < 10:
                continue
            self._stats[sid_val] = (
                float(atr_valid[mask].mean()),
                float(atr_valid[mask].std()) + 1e-10,  # éviter /0
            )

        # Fallback global si une session manque
        all_mu = atr_valid.mean()
        all_sigma = atr_valid.std()
        self._default_mu = float(all_mu) if not pd.isna(all_mu) else 0.0
        self._default_sigma = float(all_sigma) + 1e-10 if not pd.isna(all_sigma) else 1.0

        return self

    def transform(self, atr_norm: pd.Series, session_id: pd.Series) -> pd.Series:
        """Applique la standardisation par session.

        Args:
            atr_norm: Série ATR_Norm à transformer.
            session_id: Série session_id correspondante.

        Returns:
            pd.Serie de z-scores (ATR_session_zscore).
        """
        result = pd.Series(0.0, index=atr_norm.index)

        for sid_val in [0, 1, 2, 3, 4]:
            mask = session_id == sid_val
            if not mask.any():
                continue
            mu, sigma = self._stats.get(
                sid_val, (self._default_mu, self._default_sigma)
            )
            result[mask] = (atr_norm[mask] - mu) / sigma

        return result

    def fit_transform(
        self, atr_norm: pd.Series, session_id: pd.Series
    ) -> pd.Series:
        """Fit + transform en un appel."""
        self.fit(atr_norm, session_id)
        return self.transform(atr_norm, session_id)
