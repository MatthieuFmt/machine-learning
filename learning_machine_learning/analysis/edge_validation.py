"""Validation statistique de l'edge — Breakeven WR, Bootstrap Sharpe, DSR, t-test.

Implémente les trois tests du prompt 1 :
1. Breakeven WR vs WR observé
2. Bootstrap Sharpe ratio (10 000 itérations)
3. Deflated Sharpe Ratio (López de Prado, ch.14) + PSR
4. t-statistique sur l'expectancy par trade
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from learning_machine_learning.config.backtest import BacktestConfig


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
