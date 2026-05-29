"""Framework de validation statistique unifié — Prompt 06.

Fonctions de base :
    - sharpe_ratio(returns, freq) -> float
    - sortino_ratio(returns, freq) -> float
    - max_drawdown(equity) -> float (en %)

Tests statistiques avancés :
    - bootstrap_sharpe(returns, n_iter, seed) -> tuple[float, float]
    - deflated_sharpe(sr, n_trials, n_obs, skew, kurtosis) -> tuple[float, float]
    - probabilistic_sharpe(sr, n_obs, skew, kurtosis, sr_benchmark) -> float

Générateurs de splits temporels :
    - purged_kfold_cv(df, k, embargo_pct) -> Iterator[tuple[np.ndarray, np.ndarray]]
    - walk_forward_split(df, train_months, val_months, step_months) -> Iterator

Validation combinée :
    - EdgeReport (dataclass): go, reasons, metrics
    - validate_edge(equity, trades, n_trials) -> EdgeReport

Tous les Sharpe sont calculés sur retours quotidiens (equity.pct_change()),
jamais sur PnL/trade (Constitution Règle 10).

Note v5 (fix F16) : `deflated_sharpe_ratio_from_distribution` et
`validate_edge_distribution` ont été supprimés (code mort, bug latent
d'annualisation, violation Règle 10).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Fonctions de base (stateless)
# ═══════════════════════════════════════════════════════════════════════════════


def sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Sharpe ratio annualisé sur retours.

    Préconditions :
        - returns est une pd.Series de retours (pas de PnL par trade)
        - freq est le facteur d'annualisation (252 = trading days)

    Edge cases :
        - std(returns) == 0 → 0.0
        - len(returns) < 2 → 0.0
        - NaN internes → dropna() avant calcul

    Formule : mean(returns) / std(returns, ddof=1) * sqrt(freq)
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 2:
        return 0.0
    std = float(clean.std(ddof=1))
    if np.isnan(std) or np.isclose(std, 0.0):
        return 0.0
    return float(clean.mean() / std * np.sqrt(freq))


def sortino_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Sortino ratio — pénalise uniquement la volatilité baissière.

    Args:
        returns: pd.Series de retours.
        freq: Facteur d'annualisation.

    Returns:
        Sortino ratio annualisé. 0.0 si aucun return négatif ou n < 2.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 2:
        return 0.0
    mean_ret = float(clean.mean())
    downside = clean[clean < 0.0]
    n_down = len(downside)
    if n_down < 2:
        return 0.0
    downside_std = float(downside.std(ddof=1))
    if downside_std == 0.0 or np.isnan(downside_std):
        return 0.0
    return float(mean_ret / downside_std * np.sqrt(freq))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown en pourcentage (valeur dans [0, 1]).

    Formule : max((cummax - equity) / cummax).

    Args:
        equity: Courbe d'equity, index chronologique.

    Returns:
        Float dans [0, 1] (ex: 0.15 = 15 %). 0.0 si equity vide ou constant.
    """
    if len(equity) == 0:
        return 0.0
    cummax = equity.cummax()
    if len(cummax.dropna()) == 0:
        return 0.0
    drawdown = (cummax - equity) / cummax.replace(0.0, np.nan)
    dd_max = drawdown.max()
    return float(dd_max) if not np.isnan(dd_max) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tests statistiques avancés
# ═══════════════════════════════════════════════════════════════════════════════


def _stationary_bootstrap_indices(
    n: int,
    avg_block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Génère une séquence d'indices via stationary bootstrap (Politis-Romano 1994).

    À chaque pas :
    - probabilité 1/avg_block_size : redémarrer à une position aléatoire.
    - sinon : avancer d'un cran (modulo n).

    Args:
        n: Longueur de la série originale.
        avg_block_size: Taille moyenne de bloc (1/p). Doit être ≥ 1.
        rng: Générateur aléatoire numpy.

    Returns:
        Array de longueur n d'indices ∈ [0, n).
    """
    if avg_block_size < 1:
        raise ValueError(f"avg_block_size doit être >= 1, reçu {avg_block_size}")
    p_restart = 1.0 / avg_block_size
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(0, n)
    restarts = rng.random(n - 1) < p_restart
    randoms = rng.integers(0, n, size=n - 1)
    for i in range(1, n):
        if restarts[i - 1]:
            indices[i] = randoms[i - 1]
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def bootstrap_sharpe(
    returns: pd.Series,
    n_iter: int = 10_000,
    seed: int = 42,
    block_size: int = 10,
) -> tuple[float, float]:
    """Bootstrap stationnaire du Sharpe ratio annualisé (fix F15).

    Préserve l'autocorrélation des returns financiers en ré-échantillonnant
    par blocs (Politis-Romano 1994). Le bootstrap iid classique sous-estime
    la variance sur des séries autocorrélées.

    Args:
        returns: pd.Series de retours.
        n_iter: Nombre d'itérations bootstrap.
        seed: Graine aléatoire.
        block_size: Taille moyenne du bloc géométrique. Défaut 10
            (compromis raisonnable pour returns daily ; pour minute/H1
            data, utiliser 20-50).

    Returns:
        (sharpe_moyen_bootstrap, p_value_le_0).
        p_value = proportion de Sharpe bootstrap ≤ 0.
    """
    rng = np.random.default_rng(seed)
    clean = returns.dropna().values.astype(np.float64)
    n = len(clean)
    if n < 2:
        return (float("nan"), float("nan"))
    if block_size < 1:
        raise ValueError(f"block_size doit être >= 1, reçu {block_size}")

    boot_sharpes = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = _stationary_bootstrap_indices(n, block_size, rng)
        sample = clean[idx]
        std = float(np.std(sample, ddof=1))
        if std == 0.0 or np.isnan(std):
            boot_sharpes[i] = 0.0
        else:
            boot_sharpes[i] = float(np.mean(sample) / std * np.sqrt(252))

    mean_boot = float(np.mean(boot_sharpes))
    p_val = float(np.mean(boot_sharpes <= 0.0))
    return (mean_boot, p_val)


def deflated_sharpe(
    sr: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Corrige le Sharpe observé pour le nombre de stratégies testées (data-snooping).

    Args:
        sr: Sharpe ratio observé ANNUALISÉ et HONNÊTE (annualisation routée par
            fréquence). ⚠️ Ne PAS passer un Sharpe annualisé ×√252 calculé sur une
            série par-trade basse fréquence : il serait gonflé et le DSR deviendrait
            faussement positif (bug historique USDJPY/H4).
        n_trials: Nombre total de stratégies/configurations testées (n_trials_cumul).
        n_obs: Nombre d'observations (retours), ≥ 30 requis.
        skew: Skewness des retours (scipy.stats.skew, bias=False).
        kurtosis: Kurtosis RAW (scipy.stats.kurtosis, bias=False, fisher=True) + 3.

    Returns:
        (dsr_z, p_value) où p_value = 1 − Φ(dsr_z).
        (NaN, NaN) si n_trials < 1, n_obs < 30, ou dénominateur ≤ 0.

    Formule (López de Prado 2014) :
        γ = constante d'Euler-Mascheroni ≈ 0.5772156649
        SR₀ = √((1−γ)Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))
        numérateur = (ŜR − SR₀) · √(n_obs − 1)
        dénominateur = √(1 − γ₃·ŜR + (γ₄−1)/4 · ŜR²)
        DSR = numérateur / dénominateur
    """
    if n_trials < 1:
        logger.warning("DSR: n_trials=%d < 1, retourne NaN.", n_trials)
        return (float("nan"), float("nan"))

    if n_obs < 30:
        logger.warning("DSR: n_obs=%d < 30, retourne NaN.", n_obs)
        return (float("nan"), float("nan"))

    # Constante d'Euler-Mascheroni
    euler = 0.5772156649015328606

    # Quantiles gaussiens
    z_alpha = scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_alpha_e = scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))

    # SR₀ — Sharpe maximum attendu sous H₀ (dû au data-snooping)
    sr_zero_sq = (1.0 - euler) * z_alpha + euler * z_alpha_e
    # Protection : si sr_zero_sq est négatif (numériquement improbable), on clamp
    sr_zero = float(np.sqrt(max(sr_zero_sq, 0.0)))

    numerator = (sr - sr_zero) * np.sqrt(max(n_obs - 1, 0))

    denominator_sq = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if denominator_sq <= 0.0:
        logger.warning(
            "DSR: dénominateur² = %.6f ≤ 0 (skew=%.3f, kurt=%.3f, sr=%.4f).",
            denominator_sq, skew, kurtosis, sr,
        )
        return (float("nan"), float("nan"))

    denominator = np.sqrt(denominator_sq)
    dsr_z = numerator / denominator
    p_value = 1.0 - scipy_stats.norm.cdf(dsr_z)

    return (float(dsr_z), float(p_value))


def probabilistic_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    Probabilité que le vrai Sharpe ratio dépasse sr_benchmark, étant donné
    les moments estimés (ŜR, skewness, kurtosis).

    Args:
        sr: Sharpe ratio observé (annualisé).
        n_obs: Nombre d'observations.
        skew: Skewness des retours.
        kurtosis: Kurtosis RAW (excess + 3).
        sr_benchmark: Sharpe de référence (0 = H₀: SR ≤ 0).

    Returns:
        P(SR_vrai > sr_benchmark). NaN si n < 2 ou dénominateur ≤ 0.
    """
    if n_obs < 2:
        return float("nan")

    denominator_sq = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if denominator_sq <= 0.0:
        return float("nan")

    numerator = (sr - sr_benchmark) * np.sqrt(max(n_obs - 1, 0))
    denominator = np.sqrt(denominator_sq)
    psr_val = float(scipy_stats.norm.cdf(numerator / denominator))

    logger.debug(
        "PSR: ŜR=%.4f, skew=%.3f, kurt=%.3f, n=%d → PSR=%.6f",
        sr, skew, kurtosis, n_obs, psr_val,
    )
    return psr_val


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Générateurs de splits temporels
# ═══════════════════════════════════════════════════════════════════════════════


def purged_kfold_cv(
    df: pd.DataFrame,
    k: int = 5,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold Cross-Validation (López de Prado ch.7).

    Découpe l'index chronologique en k folds. Pour chaque fold i :
        - test = fold i
        - train = tous les folds < i (purgés de embargo_pct % avant test)
        - Les folds > i sont exclus (pas de look-ahead).

    Args:
        df: DataFrame indexé par DatetimeIndex trié chronologiquement.
        k: Nombre de folds (≥ 2).
        embargo_pct: Fraction de l'index à purger avant chaque test (doit être > 0).

    Yields:
        Tuple (train_indices: np.ndarray, test_indices: np.ndarray).

    Raises:
        ValueError: si embargo_pct == 0.
        ValueError: si k < 2.
        ValueError: si l'index n'est pas trié ou trop court.

    Invariants garantis :
        - max(train) + embargo < min(test) pour chaque split
        - Aucun chevauchement train/test
    """
    if embargo_pct <= 0.0:
        raise ValueError(f"embargo_pct doit être > 0, reçu {embargo_pct}")
    if k < 2:
        raise ValueError(f"k doit être >= 2, reçu {k}")

    n = len(df)
    if n < k:
        raise ValueError(f"DataFrame trop court ({n} barres) pour {k} folds.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'index du DataFrame doit être un DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("L'index doit être trié chronologiquement (monotonic increasing).")

    # Découpage en k folds de taille ~égale
    fold_size = n // k
    embargo_bars = max(1, int(n * embargo_pct))

    for i in range(1, k):
        # test = fold i (barres [test_start, test_end))
        test_start = i * fold_size
        test_end = test_start + fold_size if i < k - 1 else n  # dernier fold prend le reste
        test_idx = np.arange(test_start, test_end, dtype=np.int64)

        # train = folds 0 à i-1, mais on purge les embargo_bars dernières barres
        train_end = test_start - embargo_bars
        if train_end <= 0:
            logger.warning(
                "purged_kfold: split %d/%d — embargo consomme tout le train, skip.",
                i, k - 1,
            )
            continue
        train_idx = np.arange(0, train_end, dtype=np.int64)

        yield train_idx, test_idx

    logger.debug("purged_kfold: %d splits émis (k=%d, embargo=%.1f%%).", k - 1, k, embargo_pct * 100)


def walk_forward_split(
    df: pd.DataFrame,
    train_months: int = 36,
    val_months: int = 6,
    step_months: int = 6,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Générateur de splits walk-forward ancrés (expanding window).

    Le train initial couvre les `train_months` premiers mois.
    La validation avance par pas de `step_months`, chaque fenêtre val
    dure `val_months` mois. Le train est expanding (grossit à chaque step).

    Args:
        df: DataFrame indexé par DatetimeIndex trié.
        train_months: Durée du train initial en mois.
        val_months: Durée de chaque fenêtre de validation en mois.
        step_months: Pas d'avancement entre deux validations.

    Yields:
        Tuple (train_indices: np.ndarray, val_indices: np.ndarray).

    Invariants :
        - train_end < val_start (pas de chevauchement)
        - Les fenêtres val ne se chevauchent pas entre elles
        - Le train grossit à chaque step (expanding window)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'index du DataFrame doit être un DatetimeIndex.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("L'index doit être trié chronologiquement (monotonic increasing).")

    if len(df) == 0:
        return

    index = df.index
    t0 = index[0]
    t_end = index[-1]

    # Le premier train va de t0 à t0 + train_months
    train_end = t0 + pd.DateOffset(months=train_months)

    # Première val : (train_end, train_end + val_months]
    val_start = train_end
    val_end = val_start + pd.DateOffset(months=val_months)

    step_offset = pd.DateOffset(months=step_months)
    val_offset = pd.DateOffset(months=val_months)

    while val_start < t_end:
        # Trouver les indices correspondants
        train_mask = index <= train_end
        val_mask = (index > val_start) & (index <= val_end)

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(train_idx) > 0 and len(val_idx) > 0:
            yield train_idx, val_idx

        # Avancer : train_end recule à val_end (expanding)
        val_start = val_start + step_offset
        val_end = val_start + val_offset
        train_end = val_start  # expanding : tout avant val_start est train

    logger.debug("walk_forward_split: génération terminée.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Validation combinée — EdgeReport + validate_edge
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EdgeReport:
    """Rapport de validation d'edge selon les 5 critères de la constitution.

    Attributes:
        go: True si TOUS les critères sont satisfaits.
        reasons: Liste des raisons d'échec (vide si go=True).
        metrics: Dict des métriques chiffrées.
    """
    go: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def validate_edge(
    equity: pd.Series,
    trades: pd.DataFrame,
    n_trials: int | None = None,
    annualized_sharpe: float | None = None,
) -> EdgeReport:
    """Validation complète selon les 5 critères de la constitution.

    Critères (Constitution §2) :
        1. Sharpe walk-forward ≥ 1.0
        2. DSR > 0 ET p < 0.05
        3. Max drawdown < 15 %
        4. Win rate > 30 %
        5. Trades par an ≥ 30

    ⚠️ DEUX Sharpe distincts, à ne pas confondre :
      - Critère 1 (« ≥ 1.0 ») = Sharpe ANNUALISÉ. À passer via ``annualized_sharpe``
        (calculé par l'appelant avec l'annualisation routée par fréquence, cf.
        ``sharpe_daily_from_trades`` — Règle 5). À défaut, repli sur
        ``equity.pct_change()`` × √252 (legacy).
      - DSR (critère 2) = la formule de Bailey & López de Prado attend le Sharpe
        PAR-PÉRIODE (NON annualisé) : le terme √(n_obs−1) fait déjà l'échelle.
        Lui passer un Sharpe ×√252 le double-annualise et peut INVERSER son signe
        (bug historique). On calcule donc ici ``sr_per_periode = mean/std`` brut.

    Args:
        equity: Courbe d'equity, pd.Series indexée par datetime.
        trades: DataFrame avec colonne obligatoire 'pnl' (PnL par trade).
        n_trials: Nombre total de configurations testées (pénalité de
            data-snooping du DSR). Fix C4 : si ``None`` (défaut), il est lu
            AUTOMATIQUEMENT depuis le registre anti-snooping
            (``snooping_guard.n_trials_from_history()``) au lieu d'une constante
            en dur — chaque ``read_oos()` enregistré compte comme un essai.
            Passer un entier explicite reste possible (override).
        annualized_sharpe: Sharpe annualisé honnête pour le critère 1. Si None,
            repli legacy (``equity.pct_change()`` × √252, possiblement biaisé en
            basse fréquence).

    Returns:
        EdgeReport avec go=True si et seulement si TOUS les critères passent.
    """
    if n_trials is None:
        # Fix C4 — n_trials ne doit JAMAIS être une constante en dur : on le
        # dérive du nombre de lectures OOS effectivement enregistrées.
        from app.testing.snooping_guard import n_trials_from_history

        n_trials = n_trials_from_history()

    reasons: list[str] = []
    metrics: dict[str, float] = {}

    # ── 1. Sharpe ANNUALISÉ (critère ≥ 1.0) ────────────────────────────────
    period_returns = equity.pct_change().dropna()
    # Sharpe annualisé honnête fourni par l'appelant (fréquence-aware) ; sinon
    # repli legacy ×√252 sur les retours par-trade.
    sr_annual = (
        annualized_sharpe
        if annualized_sharpe is not None
        else sharpe_ratio(period_returns)
    )
    metrics["sharpe"] = sr_annual

    if sr_annual < 1.0:
        reasons.append(f"Sharpe {sr_annual:.2f} < 1.0")

    # ── 2. DSR ──────────────────────────────────────────────────────────────
    # `deflated_sharpe` est calibrée pour un Sharpe ANNUALISÉ (cf. SR0 avec V≈1 ;
    # cf. tests). Le bug historique n'était PAS l'annualisation en soi, mais le
    # fait d'annualiser ×√252 une série PAR-TRADE (~17 trades/an) → Sharpe gonflé
    # à 2.4 au lieu de 0.52 → DSR faussement positif. On lui passe donc le MÊME
    # Sharpe annualisé HONNÊTE (routé par fréquence) que le critère 1.
    n_obs = len(period_returns)
    skew = float(period_returns.skew())
    # scipy kurtosis donne l'excess kurtosis (fisher=True), on ajoute 3
    kurt_raw = float(period_returns.kurtosis()) + 3.0

    dsr, p_dsr = deflated_sharpe(
        sr=sr_annual,
        n_trials=n_trials,
        n_obs=n_obs,
        skew=skew,
        kurtosis=kurt_raw,
    )
    metrics["dsr"] = dsr if not np.isnan(dsr) else float("nan")
    metrics["p_value"] = p_dsr if not np.isnan(p_dsr) else float("nan")

    if not (not np.isnan(dsr) and dsr > 0 and not np.isnan(p_dsr) and p_dsr < 0.05):
        dsr_str = f"{dsr:.2f}" if not np.isnan(dsr) else "NaN"
        p_str = f"{p_dsr:.3f}" if not np.isnan(p_dsr) else "NaN"
        reasons.append(f"DSR={dsr_str} (p={p_str}) non significatif")

    # ── 3. Max drawdown ───────────────────────────────────────────────────
    dd = max_drawdown(equity)
    metrics["max_dd"] = dd

    if dd >= 0.15:
        reasons.append(f"Max DD {dd:.1%} >= 15%")

    # ── 4. Win rate ───────────────────────────────────────────────────────
    if "pnl" not in trades.columns:
        raise KeyError("trades doit contenir la colonne 'pnl' (PnL par trade)")

    n_trades = len(trades)
    n_wins = int((trades["pnl"] > 0).sum())
    wr = n_wins / n_trades if n_trades > 0 else 0.0
    metrics["wr"] = wr
    metrics["n_trades"] = float(n_trades)

    if wr <= 0.30:
        reasons.append(f"WR {wr:.1%} <= 30%")

    # ── 5. Trades par an ──────────────────────────────────────────────────
    if n_trades == 0 or equity.index is None or len(equity.index) < 2:
        trades_per_year = 0.0
    else:
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        trades_per_year = n_trades / n_years if n_years > 0 else 0.0
    metrics["trades_per_year"] = trades_per_year

    if trades_per_year < 30:
        reasons.append(f"Trades/an {trades_per_year:.1f} < 30")

    return EdgeReport(
        go=len(reasons) == 0,
        reasons=reasons,
        metrics=metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Fonctions supprimées en v5 (fix F16)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Les fonctions suivantes ont été supprimées car :
# - `_compute_sharpe_from_returns(returns)` : ne faisait PAS l'annualisation
#   (oubli de √252). Bug latent qui aurait sous-estimé tous les Sharpe ×16.
# - `validate_edge_distribution(trades, ...)` : calculait le Sharpe sur PnL
#   par trade (violation Règle 10 de la Constitution).
# - `deflated_sharpe_ratio_from_distribution(observed_sr, sharpe_distribution)` :
#   uniquement utilisé par validate_edge_distribution. Plus aucun appel actif.
#
# Aucun script du projet ne les importait au moment de la suppression
# (vérification grep 2026-05-18). Si vous avez besoin du DSR distributionnel
# CPCV, utiliser `deflated_sharpe(sr, n_trials, n_obs, skew, kurtosis)` au-dessus
# avec n_trials = nombre de splits CPCV.
