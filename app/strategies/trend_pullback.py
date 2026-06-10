"""Trend-Pullback D1/H4 — port mécanique de la stratégie MANUELLE TradingView.

Réplique exacte des deux indicateurs Pine de ``strategie-forex/`` pour pouvoir
MESURER la stratégie au lieu de la croire :

  Indicateur 1 (« Boussole de Tendance », lu en D1) :
    - haussier  : Close > EMA200  ET  EMA20 > EMA50  ET  ADX14 > 25
    - baissier  : Close < EMA200  ET  EMA20 < EMA50  ET  ADX14 > 25
    - sinon     : neutre (ne pas trader)

  Indicateur 2 (« Signal d'Entrée », lu en H4) :
    - tendance H4   : EMA20 vs EMA50
    - repli         : le prix a touché la zone [min(EMA20,EMA50), max(EMA20,EMA50)]
                      dans les `pullback_lookback` dernières barres
    - déclencheur   : RSI14 recroise 50 (haussier : vers le haut) + bougie de
                      même couleur (close > open pour un achat)
    - risque        : SL = 1.5 × ATR14, TP = 2 × le risque (R:R 1:2)

Anti-fuite (différence volontaire avec un usage naïf de l'indicateur) :
  - le régime D1 utilisé pendant le jour J est celui calculé sur la barre D1 de
    J−1 (``shift(1)`` AVANT le reindex H4) — la barre D1 de J n'est connue qu'à
    la clôture de J ;
  - le signal est connu à la CLÔTURE de la barre H4 → entrée à l'OPEN de la
    barre suivante ;
  - conflit TP/SL dans la même barre → le SL gagne (règle 7, CLAUDE.md §5).

Coûts : spread + 2×(slippage+commission) du AssetConfig, swap signé par nuit.
``cost_multiplier`` (>1) applique la marge de sécurité « coûts XTB non relevés
en démo » : coûts ×m et swaps NÉGATIFS ×m (les swaps positifs sont divisés).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config.instruments import AssetConfig
from app.core.logging import get_logger
from app.features.indicators import adx, atr, ema, rsi

logger = get_logger(__name__)

_OHLC = ("Open", "High", "Low", "Close")


def _check_frame(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name}.index doit être DatetimeIndex")
    if df.index.tz is None:
        raise ValueError(f"{name}.index doit être tz-aware (UTC)")
    for col in _OHLC:
        if col not in df.columns:
            raise KeyError(f"{name} doit contenir la colonne '{col}'")


def d1_regime(
    df_d1: pd.DataFrame,
    *,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """Régime D1 (« Boussole ») : +1 haussier, −1 baissier, 0 neutre/range.

    ⚠️ La valeur à la date J est calculée AVEC la barre de J : elle n'est connue
    qu'à la clôture de J. Pour un usage intraday sans fuite, décaler d'une barre
    (``.shift(1)``) avant de projeter sur l'index intraday — c'est ce que fait
    ``compute_trend_pullback_signals``.
    """
    _check_frame(df_d1, "df_d1")
    close = df_d1["Close"]
    e_fast = ema(close, ema_fast)
    e_mid = ema(close, ema_mid)
    e_slow = ema(close, ema_slow)
    adx_line = adx(df_d1["High"], df_d1["Low"], close, adx_period)["adx_line"]

    trending = adx_line > adx_threshold
    bull = (close > e_slow) & (e_fast > e_mid) & trending
    bear = (close < e_slow) & (e_fast < e_mid) & trending
    regime = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(regime, index=df_d1.index, dtype="int64", name="d1_regime")


def compute_trend_pullback_signals(
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    *,
    ema_fast: int = 20,
    ema_mid: int = 50,
    pullback_lookback: int = 5,
    rsi_period: int = 14,
    rsi_mid: float = 50.0,
    ema_slow_d1: int = 200,
    adx_period_d1: int = 14,
    adx_threshold_d1: float = 25.0,
) -> pd.Series:
    """Signaux à la CLÔTURE de chaque barre H4 : +1 achat, −1 vente, 0 rien.

    Filtre D1 intégré (ce que la stratégie manuelle fait « à l'œil ») : un achat
    n'est possible que si le régime D1 CONNU (barre de la veille) est haussier.
    """
    _check_frame(df_h4, "df_h4")
    close, high, low, open_ = (df_h4[c] for c in ("Close", "High", "Low", "Open"))

    e_fast = ema(close, ema_fast)
    e_mid = ema(close, ema_mid)
    rsi_s = rsi(close, rsi_period)

    bull_h4 = e_fast > e_mid
    bear_h4 = e_fast < e_mid

    # Repli : le prix a touché la zone EMA20-EMA50 dans les N dernières barres.
    hi_zone = pd.concat([e_fast, e_mid], axis=1).max(axis=1)
    lo_zone = pd.concat([e_fast, e_mid], axis=1).min(axis=1)
    touched = (low <= hi_zone) & (high >= lo_zone)
    idx = np.arange(len(df_h4), dtype=np.float64)
    last_touch = pd.Series(np.where(touched, idx, np.nan), index=df_h4.index).ffill()
    bars_since = pd.Series(idx, index=df_h4.index) - last_touch  # NaN avant 1ʳᵉ touche
    pulled_back = (bars_since <= pullback_lookback).fillna(False)

    # Déclencheur : RSI recroise 50 (sémantique ta.crossover) + bougie alignée.
    rsi_up = (rsi_s > rsi_mid) & (rsi_s.shift(1) <= rsi_mid)
    rsi_down = (rsi_s < rsi_mid) & (rsi_s.shift(1) >= rsi_mid)
    candle_up = close > open_
    candle_down = close < open_

    # Régime D1 de la VEILLE (anti-fuite), projeté sur l'index H4.
    regime_known = (
        d1_regime(
            df_d1,
            ema_fast=ema_fast,
            ema_mid=ema_mid,
            ema_slow=ema_slow_d1,
            adx_period=adx_period_d1,
            adx_threshold=adx_threshold_d1,
        )
        .shift(1)
        .reindex(df_h4.index, method="ffill")
        .fillna(0)
        .astype("int64")
    )

    long_sig = (regime_known == 1) & bull_h4 & pulled_back & rsi_up & candle_up
    short_sig = (regime_known == -1) & bear_h4 & pulled_back & rsi_down & candle_down
    signals = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return pd.Series(signals, index=df_h4.index, dtype="int64", name="signal")


def _margined_costs(
    cfg: AssetConfig, cost_multiplier: float
) -> tuple[float, float, float]:
    """(coût round-trip, swap_long, swap_short) avec marge de sécurité.

    La marge dégrade toujours : coûts ×m ; swap négatif ×m ; swap positif ÷m.
    """
    if cost_multiplier < 1.0:
        raise ValueError(f"cost_multiplier doit être ≥ 1, reçu {cost_multiplier}")
    cost_total = (
        cfg.spread_pips + 2.0 * (cfg.slippage_pips + cfg.commission_pips)
    ) * cost_multiplier

    def _margin_swap(swap: float) -> float:
        return swap * cost_multiplier if swap < 0 else swap / cost_multiplier

    return (
        cost_total,
        _margin_swap(cfg.swap_long_pips_per_night),
        _margin_swap(cfg.swap_short_pips_per_night),
    )


def simulate_trend_pullback_trades(
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    asset_config: AssetConfig,
    *,
    atr_period: int = 14,
    atr_mult_sl: float = 1.5,
    rr: float = 2.0,
    cost_multiplier: float = 1.0,
    signals: pd.Series | None = None,
) -> list[dict]:
    """Simule la stratégie manuelle : 1 position à la fois, SL/TP ATR, swing.

    Règles d'exécution (honnêtes) :
        - signal à la clôture de la barre i → entrée à l'Open de i+1 ;
        - SL/TP calculés depuis la CLÔTURE du signal (comme l'indicateur Pine
          les affiche au trader) : SL = close ∓ atr_mult_sl×ATR, TP = ±rr× le risque ;
        - scan depuis la barre d'entrée incluse ; même barre touche TP ET SL →
          SL gagne ; gap d'ouverture au-delà du SL → sortie à l'Open (pire) ;
          gap au-delà du TP → sortie au TP (conservateur) ;
        - pas de time-stop (fidèle au manuel : on tient jusqu'au TP/SL) ;
          fin des données → sortie à la dernière clôture (« eot ») ;
        - coûts round-trip + swap signé × nuits (marge via cost_multiplier).

    Args:
        df_h4 / df_d1: OHLC tz-aware UTC (H4 = exécution, D1 = filtre).
        asset_config: Coûts XTB de l'instrument.
        atr_period / atr_mult_sl / rr: Paramètres risque de l'indicateur 2.
        cost_multiplier: Marge de sécurité sur les coûts (≥1 ; 1.5 recommandé
            tant que les coûts réels n'ont pas été relevés en démo).
        signals: Série de signaux pré-calculée (tests/variantes). Défaut :
            ``compute_trend_pullback_signals`` avec les paramètres standard.

    Returns:
        Liste de dicts compatibles ``sharpe_daily_from_trades`` : signal,
        entry_time, exit_time, entry_price, exit_price, sl_price, tp_price,
        pips_brut, pips_net, nights_held, exit_reason ∈ {"tp","sl","eot"}.
    """
    _check_frame(df_h4, "df_h4")
    if signals is None:
        signals = compute_trend_pullback_signals(df_h4, df_d1)
    signals = signals.reindex(df_h4.index).fillna(0).astype("int64")

    pip_size = asset_config.pip_size
    cost_total, swap_long, swap_short = _margined_costs(asset_config, cost_multiplier)

    atr_s = atr(df_h4["High"], df_h4["Low"], df_h4["Close"], atr_period)
    opens = df_h4["Open"].to_numpy(dtype=np.float64)
    highs = df_h4["High"].to_numpy(dtype=np.float64)
    lows = df_h4["Low"].to_numpy(dtype=np.float64)
    closes = df_h4["Close"].to_numpy(dtype=np.float64)
    atr_v = atr_s.to_numpy(dtype=np.float64)
    sig_v = signals.to_numpy()
    index = df_h4.index

    trades: list[dict] = []
    n = len(df_h4)
    i = 0
    while i < n - 1:
        direction = int(sig_v[i])
        a = atr_v[i]
        if direction == 0 or not np.isfinite(a) or a <= 0.0:
            i += 1
            continue

        # SL/TP depuis la clôture du signal (ce que voit le trader sur le chart).
        sig_close = closes[i]
        sl_dist = atr_mult_sl * a
        tp_dist = sl_dist * rr
        if direction == 1:
            sl_price, tp_price = sig_close - sl_dist, sig_close + tp_dist
        else:
            sl_price, tp_price = sig_close + sl_dist, sig_close - tp_dist

        entry_i = i + 1
        entry_price = opens[entry_i]
        entry_ts = index[entry_i]

        exit_price: float | None = None
        exit_ts = index[-1]
        exit_reason = "eot"
        j = entry_i
        for j in range(entry_i, n):
            o, h, lo_ = opens[j], highs[j], lows[j]
            if direction == 1:
                sl_hit, tp_hit = lo_ <= sl_price, h >= tp_price
                gap_through_sl = o <= sl_price
            else:
                sl_hit, tp_hit = h >= sl_price, lo_ <= tp_price
                gap_through_sl = o >= sl_price
            if sl_hit:  # SL prioritaire (règle 7) ; gap → fill à l'open (pire)
                exit_price = o if gap_through_sl else sl_price
                exit_ts, exit_reason = index[j], "sl"
                break
            if tp_hit:  # gap au-delà du TP → on prend le TP (conservateur)
                exit_price = tp_price
                exit_ts, exit_reason = index[j], "tp"
                break
        if exit_price is None:
            exit_price, exit_ts, exit_reason = closes[-1], index[-1], "eot"
            j = n - 1

        pips_brut = direction * (exit_price - entry_price) / pip_size
        pips_net = pips_brut - cost_total
        nights_held = max(0, (exit_ts.normalize() - entry_ts.normalize()).days)
        if nights_held > 0:
            swap = swap_long if direction == 1 else swap_short
            pips_net += nights_held * swap

        trades.append({
            "signal": direction,
            "entry_time": entry_ts.isoformat(),
            "exit_time": exit_ts.isoformat(),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "pips_brut": float(pips_brut),
            "pips_net": float(pips_net),
            "nights_held": int(nights_held),
            "exit_reason": exit_reason,
        })
        i = j + 1  # une seule position à la fois

    logger.info(
        "trend_pullback_simulated",
        extra={"context": {
            "n_trades": len(trades),
            "n_bars": n,
            "atr_mult_sl": atr_mult_sl,
            "rr": rr,
            "cost_multiplier": cost_multiplier,
        }},
    )
    return trades
