"""Portfolio equal-risk weight + filtre correlation (pivot v4 B4).

Reusable portfolio construction for multi-sleeve deployment.
When < 2 sleeves GO, single-sleeve fallback applies automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_risk_weights(
    sleeve_returns: dict[str, pd.Series],
    target_vol_annual: float = 0.10,
    leverage_cap: float = 2.0,
    vol_lookback: int = 60,
) -> pd.DataFrame:
    """Equal-risk weights: each sleeve contributes target_vol/sqrt(n) to portfolio vol.

    Args:
        sleeve_returns: Dict sleeve_name -> daily return Series (fraction of capital).
        target_vol_annual: Target annualized portfolio volatility (default 10%).
        leverage_cap: Maximum leverage per sleeve (default 2.0).
        vol_lookback: Rolling window (days) for realized volatility estimation.

    Returns:
        DataFrame (index=sleeve_returns index, columns=sleeve names) of weights.
    """
    df = pd.DataFrame(sleeve_returns).fillna(0.0)
    n = len(sleeve_returns)
    if n == 0:
        return pd.DataFrame()
    target_vol_per_sleeve = target_vol_annual / np.sqrt(n)
    realized_vol = df.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)
    # Avoid division by zero: wherever realized_vol is 0, weight = 0
    weights = (
        target_vol_per_sleeve / realized_vol.replace(0.0, np.nan)
    ).clip(upper=leverage_cap).fillna(0.0)
    return weights


def correlation_filter(
    sleeve_returns: dict[str, pd.Series],
    cap: float = 0.7,
    corr_window: int = 60,
    sharpe_window: int = 126,
) -> pd.DataFrame:
    """Enable/disable sleeves based on rolling correlation and 6M Sharpe.

    If ρ_ij > cap over corr_window → disable the sleeve with the worse 6M Sharpe.
    Rebalanced weekly (Friday).

    Args:
        sleeve_returns: Dict sleeve_name -> daily return Series.
        cap: Correlation threshold above which one sleeve is disabled.
        corr_window: Rolling window (days) for correlation estimation.
        sharpe_window: Rolling window (days) for Sharpe comparison.

    Returns:
        Boolean DataFrame (True = active), reindexed to original index, forward-filled.
    """
    df = pd.DataFrame(sleeve_returns).fillna(0.0)
    if df.shape[1] <= 1:
        # Single sleeve: always active
        active = pd.DataFrame(True, index=df.index, columns=df.columns)
        return active

    weekly_idx = df.resample("W-FRI").last().index
    active = pd.DataFrame(True, index=weekly_idx, columns=df.columns)

    rolling_sharpe = df.rolling(sharpe_window, min_periods=1).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0.0,
    )

    for date in weekly_idx[corr_window:]:
        window = df.loc[:date].tail(corr_window)
        if window.shape[0] < 2:
            continue
        corr = window.corr()
        active_this = set(df.columns)
        for i in df.columns:
            for j in df.columns:
                if i >= j or i not in active_this or j not in active_this:
                    continue
                if corr.loc[i, j] > cap:
                    sr_i = rolling_sharpe.loc[date, i] if date in rolling_sharpe.index else 0.0
                    sr_j = rolling_sharpe.loc[date, j] if date in rolling_sharpe.index else 0.0
                    drop = i if sr_i < sr_j else j
                    active_this.discard(drop)
        for c in df.columns:
            active.loc[date, c] = c in active_this

    # Forward-fill to original index
    return active.reindex(df.index, method="ffill").fillna(True).astype(bool)


def build_portfolio_equity(
    sleeve_returns: dict[str, pd.Series],
    weights: pd.DataFrame,
    active_mask: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> pd.Series:
    """Build portfolio equity curve from sleeve returns, weights and active mask.

    Args:
        sleeve_returns: Dict sleeve_name -> daily return Series.
        weights: Equal-risk weights DataFrame.
        active_mask: Boolean active mask from correlation_filter.
        initial_capital: Starting capital in EUR.

    Returns:
        Portfolio equity curve (pd.Series, index=common dates).
    """
    df = pd.DataFrame(sleeve_returns).fillna(0.0)
    effective_weights = weights * active_mask.astype(float)
    portfolio_returns = (df * effective_weights).sum(axis=1)
    return initial_capital * (1.0 + portfolio_returns).cumprod()
