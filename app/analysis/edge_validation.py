"""Validation statistique de l'edge — Breakeven WR, Bootstrap Sharpe, DSR, t-test.

Implémente les tests complets :
1. Breakeven WR vs WR observé
2. Bootstrap Sharpe ratio (10 000 itérations)
3. Deflated Sharpe Ratio (López de Prado, ch.14) + PSR
4. t-statistique sur l'expectancy par trade
5. PSR de Bailey & López de Prado (2012) — formule avec skewness/kurtosis
6. DSR via distribution CPCV (López de Prado 2014)
7. validate_edge_distribution — wrapper combiné CPCV + tests classiques
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from learning_machine_learning.config.backtest import BacktestConfig
from learning_machine_learning.core.logging import get_logger

logger = get_logger(__name__)


def _compute_sharpe_from_returns(returns: np.ndarray) -> float:
    """Calcule le Sharpe ratio annualisé à partir d'un vecteur de returns.

    Le Sharpe est calculé comme mean(returns) / std(returns, ddof=1).
    Retourne 0.0 si écart-type nul (distribution dégénérée).
    """
    std = np.std(returns, ddof=1)
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(np.mean(returns) / std)


def _expected_max_sr(n_trials: int, n_observations: int) -> float:
    """Estime E[max(SR)] sous H0 (returns i.i.d. normaux, moyenne nulle).

    Utilise la théorie des valeurs extrêmes (López de Prado, ch.14).
    Formule : E[max(Z)] / sqrt(T) où E[max(Z)] approxime le maximum
    de N tirages gaussiens standards.

    Args:
        n_trials: Nombre de stratégies/configurations testées (data-snooping).
        n_observations: Nombre d'observations (trades) dans l'échantillon.

    Returns:
        Espérance du Sharpe ratio maximum sous H0.
    """
    if n_trials <= 1 or n_observations <= 1:
        return 0.0

    # Constante d'Euler-Mascheroni
    gamma_euler = 0.5772156649015328606

    # E[max(Z)] pour N tirages gaussiens standards (Extreme Value Theory)
    z_alpha = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z_alpha_e = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))

    expected_max_z = (1.0 - gamma_euler) * z_alpha + gamma_euler * z_alpha_e

    return float(expected_max_z / np.sqrt(n_observations))


def _var_max_sr(n_trials: int, n_observations: int) -> float:
    """Estime la variance de SR_max sous H0.

    Var[Z_max] ≈ π²/6 · 1/(2·log(N+1)), puis Var[SR_max] = Var[Z_max] / T.
    """
    if n_trials <= 1 or n_observations <= 1:
        return 1.0 / max(n_observations, 1)

    var_max_z = (np.pi**2 / 6.0) / (2.0 * np.log(n_trials + 1))
    return float(var_max_z / n_observations)


def validate_edge(
    trades_df: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    n_trials_searched: int = 10,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Valide statistiquement l'edge d'une stratégie de trading.

    Quatre tests :
    1. Breakeven WR : win-rate minimum pour être rentable compte tenu
       du ratio TP/SL et des coûts de friction.
    2. Bootstrap Sharpe : distribution empirique du Sharpe par
       ré-échantillonnage avec remise (10 000 itérations).
    3. Deflated Sharpe Ratio (DSR) : corrige le Sharpe observé pour
       le nombre de trials testés (data-snooping). Inclut le PSR
       (Probabilistic Sharpe Ratio).
    4. t-test sur l'expectancy : H0: E[pnl par trade] = 0.

    Args:
        trades_df: DataFrame des trades avec colonne 'Pips_Nets'
                   (pnl net par trade en pips, déjà pondéré).
        backtest_cfg: Configuration de backtest (TP, SL, coûts).
        n_trials_searched: Nombre de stratégies/configurations testées
                           avant celle-ci. Corrige le data-snooping.
        n_bootstrap: Nombre d'itérations bootstrap (défaut: 10 000).
        random_state: Graine aléatoire pour reproductibilité.

    Returns:
        Dictionnaire structuré :
        {
            "n_trades": int,
            "breakeven": {"wr_pct", "observed_wr_pct", "margin_pct"},
            "bootstrap_sharpe": {"observed", "mean_bootstrap", "p_value_gt_0",
                                 "ci_95_lower", "ci_95_upper"},
            "deflated_sharpe": {"dsr", "psr", "e_max_sr", "n_trials"},
            "t_statistic": {"mean_pnl", "std_pnl", "t_stat", "p_value"},
        }
    """
    rng = np.random.default_rng(random_state)

    # ── Extraction des PnL nets ────────────────────────────────────────
    pnl: np.ndarray = trades_df["Pips_Nets"].values.astype(np.float64)
    n_trades = len(pnl)

    if n_trades == 0:
        return {
            "n_trades": 0,
            "breakeven": {"wr_pct": np.nan, "observed_wr_pct": np.nan, "margin_pct": np.nan},
            "bootstrap_sharpe": {
                "observed": np.nan, "mean_bootstrap": np.nan,
                "p_value_gt_0": np.nan, "ci_95_lower": np.nan, "ci_95_upper": np.nan,
            },
            "deflated_sharpe": {"dsr": np.nan, "psr": np.nan, "e_max_sr": np.nan, "n_trials": n_trials_searched},
            "t_statistic": {"mean_pnl": np.nan, "std_pnl": np.nan, "t_stat": np.nan, "p_value": np.nan},
        }

    # ── 1. Breakeven WR ────────────────────────────────────────────────
    friction = backtest_cfg.commission_pips + backtest_cfg.slippage_pips
    breakeven_wr = (backtest_cfg.sl_pips + friction) / (backtest_cfg.sl_pips + backtest_cfg.tp_pips)

    # WR observé : trades gagnants / total
    n_wins = int((pnl > 0).sum())
    observed_wr = n_wins / n_trades

    # ── 2. Bootstrap Sharpe ────────────────────────────────────────────
    observed_sharpe = _compute_sharpe_from_returns(pnl)

    bootstrap_sharpes = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_trades, size=n_trades)
        bootstrap_sharpes[b] = _compute_sharpe_from_returns(pnl[idx])

    mean_bootstrap_sharpe = float(np.mean(bootstrap_sharpes))
    p_value_sharpe_gt_0 = float(np.mean(bootstrap_sharpes <= 0.0))
    ci_lower = float(np.percentile(bootstrap_sharpes, 2.5))
    ci_upper = float(np.percentile(bootstrap_sharpes, 97.5))

    # ── 3. Deflated Sharpe Ratio ───────────────────────────────────────
    e_max_sr = _expected_max_sr(n_trials_searched, n_trades)
    se_max_sr = float(np.sqrt(_var_max_sr(n_trials_searched, n_trades)))

    # DSR : écart entre SR observé et E[SR_max], normalisé par SE(SR_max)
    if se_max_sr > 0.0:
        dsr = (observed_sharpe - e_max_sr) / se_max_sr
    else:
        dsr = float("nan")

    psr = float(stats.norm.cdf(dsr)) if not np.isnan(dsr) else float("nan")

    # ── 4. t-test sur l'expectancy ─────────────────────────────────────
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1))
    if std_pnl > 0.0 and n_trades > 1:
        t_stat = mean_pnl / (std_pnl / np.sqrt(n_trades))
        # Test bilatéral : H0: E[pnl] = 0
        p_value_t = float(2.0 * stats.t.sf(np.abs(t_stat), df=n_trades - 1))
    else:
        t_stat = 0.0 if std_pnl == 0.0 else mean_pnl
        p_value_t = 1.0 if std_pnl == 0.0 else float("nan")

    return {
        "n_trades": n_trades,
        "breakeven": {
            "wr_pct": round(breakeven_wr * 100.0, 2),
            "observed_wr_pct": round(observed_wr * 100.0, 2),
            "margin_pct": round((observed_wr - breakeven_wr) * 100.0, 2),
        },
        "bootstrap_sharpe": {
            "observed": round(observed_sharpe, 6),
            "mean_bootstrap": round(mean_bootstrap_sharpe, 6),
            "p_value_gt_0": round(p_value_sharpe_gt_0, 6),
            "ci_95_lower": round(ci_lower, 6),
            "ci_95_upper": round(ci_upper, 6),
        },
        "deflated_sharpe": {
            "dsr": round(dsr, 6) if not np.isnan(dsr) else float("nan"),
            "psr": round(psr, 6) if not np.isnan(psr) else float("nan"),
            "e_max_sr": round(e_max_sr, 6),
            "n_trials": n_trials_searched,
        },
        "t_statistic": {
            "mean_pnl": round(mean_pnl, 4),
            "std_pnl": round(std_pnl, 4),
            "t_stat": round(t_stat, 6) if not np.isnan(t_stat) else float("nan"),
            "p_value": round(p_value_t, 6) if not np.isnan(p_value_t) else float("nan"),
        },
    }


# ── Step 02 — PSR pur (Bailey & López de Prado 2012) ───────────────────

def psr_from_returns(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    PSR(SR*) = Φ( (ŜR − SR*) · √(n−1) / √(1 − γ̂₃·ŜR + (γ̂₄−1)/4 · ŜR²) )

    où ŜR est le Sharpe estimé, γ̂₃ skewness, γ̂₄ kurtosis,
    SR* le benchmark (0 pour H₀: SR ≤ 0).

    Args:
        returns: Vecteur de returns (trades), float64.
        sr_benchmark: Sharpe benchmark à tester (défaut: 0).

    Returns:
        Probabilité P(SR_vrai > sr_benchmark). NaN si n < 2 ou dénominateur nul.
    """
    n = len(returns)
    if n < 2:
        return float("nan")

    sr_hat = _compute_sharpe_from_returns(returns)
    if sr_hat == 0.0 and np.std(returns, ddof=1) == 0.0:
        # Distribution dégénérée (tous les returns identiques)
        return 1.0 if np.mean(returns) > sr_benchmark else 0.0

    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, bias=False, fisher=True))  # excess kurtosis
    # kurtosis "classique" = excess + 3
    kurt_full = kurt + 3.0

    denominator_sq = 1.0 - skew * sr_hat + (kurt_full - 1.0) / 4.0 * sr_hat**2
    if denominator_sq <= 0.0:
        return float("nan")

    numerator = (sr_hat - sr_benchmark) * np.sqrt(n - 1)
    denominator = np.sqrt(denominator_sq)

    psr_val = float(stats.norm.cdf(numerator / denominator))
    logger.info(
        "PSR: ŜR=%.4f, skew=%.3f, kurt=%.3f, n=%d → PSR=%.6f",
        sr_hat, skew, kurt_full, n, psr_val,
    )
    return psr_val


# ── Step 02 — DSR via distribution CPCV ──────────────────────────────────

def deflated_sharpe_ratio_from_distribution(
    observed_sr: float,
    sharpe_distribution: np.ndarray,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """DSR à partir d'une distribution empirique de Sharpe (issue CPCV).

    Implémente López de Prado (2014) §II, utilisant la variance empirique
    des splits CPCV plutôt qu'une estimation EVT théorique.

    SR₀* = √Var({SRᵢ}) · ((1−γ)Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))

    Args:
        observed_sr: Sharpe observé sur le split principal (ex: train≤2023, test=2025).
        sharpe_distribution: Array float64 des Sharpe de chaque split CPCV.
        ci_level: Niveau de confiance (défaut: 0.95).

    Returns:
        {
            "dsr": float,              # Deflated Sharpe Ratio (= PSR(SR₀*))
            "psr_zero": float,         # PSR(SR*=0) classique
            "sr0_star": float,         # Seuil de Sharpe déflaté
            "e_max_sr": float,         # E[max(SR)] sous H₀ (EVT)
            "var_sr": float,           # Variance des Sharpe CPCV
            "n_splits": int,           # Nombre de splits valides
            "pct_profitable": float,   # % splits avec Sharpe > 0
            "mean_sr": float,          # E[SR] sur distribution
            "std_sr": float,           # σ[SR] sur distribution
            "min_sr": float,
            "max_sr": float,
            "median_sr": float,
            "ci_95_lower": float,
            "ci_95_upper": float,
        }
    """
    # Filtrer les NaN
    valid = sharpe_distribution[~np.isnan(sharpe_distribution)]
    n_splits = len(valid)

    if n_splits < 2:
        logger.warning("DSR distributionnel: moins de 2 splits valides (%d)", n_splits)
        return {
            "dsr": float("nan"), "psr_zero": float("nan"),
            "sr0_star": float("nan"), "e_max_sr": float("nan"),
            "var_sr": float("nan"), "n_splits": n_splits,
            "pct_profitable": float("nan"),
            "mean_sr": float("nan"), "std_sr": float("nan"),
            "min_sr": float("nan"), "max_sr": float("nan"),
            "median_sr": float("nan"),
            "ci_95_lower": float("nan"), "ci_95_upper": float("nan"),
        }

    mean_sr = float(np.mean(valid))
    std_sr = float(np.std(valid, ddof=1))
    var_sr = float(np.var(valid, ddof=1))
    pct_profitable = float(np.mean(valid > 0.0)) * 100.0
    min_sr = float(np.min(valid))
    max_sr = float(np.max(valid))
    median_sr = float(np.median(valid))
    ci_lower = float(np.percentile(valid, 2.5))
    ci_upper = float(np.percentile(valid, 97.5))

    # ── SR₀* (Deflated Sharpe threshold) ──────────────────────────────
    # γ (gamma) = poids entre normal et EVT
    # Plus γ est proche de 1 → plus on utilise l'approximation EVT
    gamma_euler = 0.5772156649015328606

    z_alpha = stats.norm.ppf(1.0 - 1.0 / n_splits)
    z_alpha_e = stats.norm.ppf(1.0 - 1.0 / (n_splits * np.e))

    # E[max Z] approximation EVT
    expected_max_z = (1.0 - gamma_euler) * z_alpha + gamma_euler * z_alpha_e

    # SR₀* = std(SR) · E[max Z]
    sr0_star = float(np.sqrt(var_sr) * expected_max_z)
    e_max_sr = sr0_star

    # ── DSR = PSR(SR₀*) ──────────────────────────────────────────────
    # On utilise la formule PSR standard avec skewness/kurtosis
    # sur le Sharpe observé vs sr0_star
    # Mais ici on ne peut pas calculer skew/kurt des "returns" individuels
    # → on utilise une version simplifiée : DSR = (ŜR − SR₀*) / σ(SR)
    if std_sr > 0.0:
        dsr = (observed_sr - sr0_star) / std_sr
    else:
        dsr = float("nan")

    # PSR(SR*=0) : probabilité que le vrai Sharpe > 0
    # Même approche simplifiée
    if std_sr > 0.0:
        psr_zero = float(stats.norm.cdf(observed_sr / std_sr))
    else:
        psr_zero = float("nan")

    logger.info(
        "DSR distrib: ŜR=%.4f, SR₀*=%.4f, DSR=%.4f, E[SR]=%.4f±%.4f, "
        "profitable=%.1f%%, n=%d",
        observed_sr, sr0_star, dsr, mean_sr, std_sr, pct_profitable, n_splits,
    )

    return {
        "dsr": round(dsr, 6) if not np.isnan(dsr) else float("nan"),
        "psr_zero": round(psr_zero, 6) if not np.isnan(psr_zero) else float("nan"),
        "sr0_star": round(sr0_star, 6),
        "e_max_sr": round(e_max_sr, 6),
        "var_sr": round(var_sr, 6),
        "n_splits": n_splits,
        "pct_profitable": round(pct_profitable, 2),
        "mean_sr": round(mean_sr, 6),
        "std_sr": round(std_sr, 6),
        "min_sr": round(min_sr, 6),
        "max_sr": round(max_sr, 6),
        "median_sr": round(median_sr, 6),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
    }


# ── Step 02 — Wrapper combiné CPCV + validate_edge ──────────────────────

def validate_edge_distribution(
    trades_df: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    sharpe_distribution: np.ndarray | None = None,
    n_trials_searched: int = 10,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Valide l'edge en combinant tests classiques + DSR distributionnel CPCV.

    Si `sharpe_distribution` est fournie, le DSR est calculé à partir
    de la distribution CPCV. Sinon, seul le DSR classique (via
    n_trials_searched) est utilisé.

    Args:
        trades_df: DataFrame des trades du split principal.
        backtest_cfg: Configuration backtest.
        sharpe_distribution: Distribution Sharpe CPCV (optionnel).
        n_trials_searched: Nombre de stratégies testées (DSR classique).
        n_bootstrap: Itérations bootstrap.
        random_state: Graine aléatoire.

    Returns:
        Dict combinant validate_edge() + champ "cpcv_dsr" si applicable.
    """
    # Tests classiques
    result = validate_edge(
        trades_df=trades_df,
        backtest_cfg=backtest_cfg,
        n_trials_searched=n_trials_searched,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # PSR pur (Bailey) sur les PnL
    if result["n_trades"] > 1:
        pnl: np.ndarray = trades_df["Pips_Nets"].values.astype(np.float64)
        result["psr_bailey"] = round(psr_from_returns(pnl, sr_benchmark=0.0), 6)
    else:
        result["psr_bailey"] = float("nan")

    # DSR distributionnel CPCV
    if sharpe_distribution is not None and len(sharpe_distribution) > 0:
        observed_sr = result["bootstrap_sharpe"]["observed"]
        cpcv_dsr = deflated_sharpe_ratio_from_distribution(
            observed_sr=observed_sr,
            sharpe_distribution=sharpe_distribution,
        )
        result["cpcv_dsr"] = cpcv_dsr
    else:
        result["cpcv_dsr"] = None

    return result
