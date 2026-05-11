"""Utilitaires partagés entre 4_backtest_triple_barrier.py et optimize_sizing.py.

Le backtest est *stateful* : un seul trade ouvert à la fois. Tant qu'un trade
n'est pas fermé (TP, SL ou timeout dans la fenêtre WINDOW_HOURS), les nouveaux
signaux générés pendant cette fenêtre sont ignorés. Cela évite le double-comptage
sur des barres successives qui produiraient des signaux corrélés.
"""
import os

import numpy as np
import pandas as pd

from config import (
    COMMISSION_PIPS,
    DIR_PREDICTIONS,
    DIR_RESULTS,
    FILE_EURUSD_H1_CLEAN,
    FILE_ML_READY,
    INITIAL_CAPITAL,
    PIP_SIZE,
    PIP_VALUE_EUR,
    SEUIL_CONFIANCE,
    SL_PIPS,
    SLIPPAGE_PIPS,
    SMA200_LOOKBACK,
    SESSION_EXCLUDE_END,
    SESSION_EXCLUDE_START,
    TP_PIPS,
    USE_SESSION_FILTER,
    USE_TREND_FILTER,
    USE_VOL_FILTER,
    VOL_FILTER_MULTIPLIER,
    VOL_FILTER_WINDOW,
    WINDOW_HOURS,
)


def _pips_to_return(pips):
    """Conversion pips → fraction de capital (voir audit I2 et config)."""
    return pips * PIP_VALUE_EUR / INITIAL_CAPITAL


def log_row_loss(label, before, after, threshold_pct=5.0):
    """Log la perte de lignes après un merge / dropna (audit I8).

    Affiche before → after avec le pourcentage perdu. Préfixe ⚠️ si la perte
    dépasse `threshold_pct` (par défaut 5%). Ne lève pas d'exception : à
    l'utilisateur de juger si le warning est tolérable ou pas.
    """
    delta = after - before  # négatif = lignes perdues, positif = ajoutées
    lost = max(0, -delta)
    pct = (lost / before * 100) if before else 0.0
    flag = " ⚠️" if pct > threshold_pct else ""
    print(f"  {label}: {before} → {after} ({delta:+d} rows, {pct:.2f}% loss){flag}")


def load_backtest_inputs(annee):
    """Charge prédictions + prix + features pour une année donnée.

    Retourne None si le fichier de prédictions n'existe pas (à gérer par l'appelant).
    Sinon retourne un DataFrame indexé par Time.
    """
    preds_path = f'{DIR_RESULTS}/Predictions_{annee}_TripleBarrier.csv'
    if not os.path.exists(preds_path):
        return None

    preds = pd.read_csv(preds_path, index_col='Time', parse_dates=True)
    prices = pd.read_csv(FILE_EURUSD_H1_CLEAN, index_col='Time', parse_dates=True)
    ml = pd.read_csv(FILE_ML_READY, index_col='Time', parse_dates=True)

    X_cols = [c for c in ml.columns if c not in ['Target', 'Spread']]
    n_preds = len(preds)
    df = preds.join(prices[['High', 'Low', 'Close']], how='inner')
    log_row_loss(f"[{annee}] join prix H1", n_preds, len(df))
    n_after_prices = len(df)
    df = df.join(ml[X_cols], how='inner')
    log_row_loss(f"[{annee}] join features ML", n_after_prices, len(df))
    return df


def _normalize_seuil(seuil):
    return seuil if seuil < 1 else seuil / 100


def simulate_trades(
    df,
    weight_func,
    tp_pips=TP_PIPS,
    sl_pips=SL_PIPS,
    window=WINDOW_HOURS,
    pip_size=PIP_SIZE,
    seuil_confiance=SEUIL_CONFIANCE,
    commission_pips=COMMISSION_PIPS,
    slippage_pips=SLIPPAGE_PIPS,
    use_trend_filter=USE_TREND_FILTER,
    use_vol_filter=USE_VOL_FILTER,
    use_session_filter=USE_SESSION_FILTER,
    sma200_lookback=SMA200_LOOKBACK,
    vol_filter_window=VOL_FILTER_WINDOW,
    vol_filter_multiplier=VOL_FILTER_MULTIPLIER,
    session_exclude_start=SESSION_EXCLUDE_START,
    session_exclude_end=SESSION_EXCLUDE_END,
):
    """Simule la stratégie en mode stateful (un trade à la fois).

    `weight_func(proba_max_array)` -> array de poids par trade (multiplie pips_brut).

    Filtres de régime (Priorité 4) :
    - Trend : Close > SMA200 → LONG only ; Close < SMA200 → SHORT only
    - Volatilité : ignorer signaux si ATR_Norm > multiplier × médiane glissante
    - Session : ignorer signaux entre session_exclude_start et session_exclude_end (GMT)

    Retourne un tuple (trades_df, n_signaux, n_filtres_appliques) :
    - trades_df indexé par Time d'entrée, colonnes Pips_Nets, Pips_Bruts, Weight, result, filter_rejected.
    - n_signaux = nombre total de barres avec un signal franchissant le seuil
      (avant filtrage stateful). Le ratio n_signaux / len(trades_df) mesure
      l'effet de la logique "un trade à la fois" (audit I6).
    - n_filtres_appliques = dict {filtre: nombre de signaux rejetés}
    """
    df = df.copy()
    df['proba_max'] = df[
        ['Confiance_Hausse_%', 'Confiance_Neutre_%', 'Confiance_Baisse_%']
    ].max(axis=1) / 100
    df['Signal'] = 0
    df['Weight'] = 0.0
    df['Filter_Rejected'] = ''  # trace le filtre qui a rejeté le signal

    n_filtres_appliques = {'trend': 0, 'vol': 0, 'session': 0}

    seuil = _normalize_seuil(seuil_confiance)
    mask_long = (df['Prediction_Modele'] == 1) & (df['Confiance_Hausse_%'] / 100 >= seuil)
    mask_short = (df['Prediction_Modele'] == -1) & (df['Confiance_Baisse_%'] / 100 >= seuil)

    # --- Filtre de tendance : SMA200 ---
    if use_trend_filter and 'Dist_SMA200' in df.columns:
        trend_mask_long = df['Dist_SMA200'] > 0   # Close > SMA200 → tendance haussière
        trend_mask_short = df['Dist_SMA200'] < 0  # Close < SMA200 → tendance baissière
        rejected_trend_long = mask_long & ~trend_mask_long
        rejected_trend_short = mask_short & ~trend_mask_short
        df.loc[rejected_trend_long, 'Filter_Rejected'] = 'trend'
        df.loc[rejected_trend_short, 'Filter_Rejected'] = 'trend'
        n_filtres_appliques['trend'] = int((rejected_trend_long | rejected_trend_short).sum())
        mask_long = mask_long & trend_mask_long
        mask_short = mask_short & trend_mask_short

    # --- Filtre de volatilité : ATR_Norm ---
    if use_vol_filter and 'ATR_Norm' in df.columns:
        atr_median = df['ATR_Norm'].rolling(window=vol_filter_window, min_periods=1).median()
        vol_threshold = atr_median * vol_filter_multiplier
        high_vol = df['ATR_Norm'] > vol_threshold
        rejected_vol_long = mask_long & high_vol
        rejected_vol_short = mask_short & high_vol
        df.loc[rejected_vol_long, 'Filter_Rejected'] = df.loc[rejected_vol_long, 'Filter_Rejected'] + ';vol'
        df.loc[rejected_vol_short, 'Filter_Rejected'] = df.loc[rejected_vol_short, 'Filter_Rejected'] + ';vol'
        n_filtres_appliques['vol'] = int((rejected_vol_long | rejected_vol_short).sum())
        mask_long = mask_long & ~high_vol
        mask_short = mask_short & ~high_vol

    # --- Filtre de session : liquidité faible ---
    if use_session_filter:
        hours_gmt = df.index.hour
        if session_exclude_start > session_exclude_end:
            # Plage qui traverse minuit (ex: 22h → 1h)
            session_mask = (hours_gmt >= session_exclude_start) | (hours_gmt < session_exclude_end)
        else:
            session_mask = (hours_gmt >= session_exclude_start) & (hours_gmt < session_exclude_end)
        rejected_session_long = mask_long & session_mask
        rejected_session_short = mask_short & session_mask
        df.loc[rejected_session_long, 'Filter_Rejected'] = df.loc[rejected_session_long, 'Filter_Rejected'] + ';session'
        df.loc[rejected_session_short, 'Filter_Rejected'] = df.loc[rejected_session_short, 'Filter_Rejected'] + ';session'
        n_filtres_appliques['session'] = int((rejected_session_long | rejected_session_short).sum())
        mask_long = mask_long & ~session_mask
        mask_short = mask_short & ~session_mask

    # Nettoyer les ';' initiaux dans Filter_Rejected
    df['Filter_Rejected'] = df['Filter_Rejected'].str.strip(';')

    df.loc[mask_long, 'Signal'] = 1
    df.loc[mask_short, 'Signal'] = -1

    signal_mask = df['Signal'] != 0
    df.loc[signal_mask, 'Weight'] = weight_func(df.loc[signal_mask, 'proba_max'])
    n_signaux = int(signal_mask.sum())

    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    signals = df['Signal'].values
    weights = df['Weight'].values
    spreads = df['Spread'].values
    filter_rejected = df['Filter_Rejected'].values

    trades = []
    i = 0
    while i < len(df):
        if signals[i] != 0:
            entry_time = dates[i]
            entry_price = closes[i]
            signal = signals[i]
            spread_cost = spreads[i] / 10.0 + commission_pips + slippage_pips
            weight = weights[i]
            filter_info = filter_rejected[i]

            if signal == 1:
                tp = entry_price + tp_pips * pip_size
                sl = entry_price - sl_pips * pip_size
            else:
                tp = entry_price - tp_pips * pip_size
                sl = entry_price + sl_pips * pip_size

            pips_brut = -sl_pips - spread_cost
            result_type = 'loss_sl'

            for j in range(1, window + 1):
                if i + j >= len(df):
                    i = len(df)
                    break
                curr_high, curr_low = highs[i + j], lows[i + j]
                if signal == 1:
                    if curr_low <= sl:
                        i += j
                        break
                    elif curr_high >= tp:
                        pips_brut = tp_pips - spread_cost
                        result_type = 'win'
                        i += j
                        break
                else:
                    if curr_high >= sl:
                        i += j
                        break
                    elif curr_low <= tp:
                        pips_brut = tp_pips - spread_cost
                        result_type = 'win'
                        i += j
                        break
            else:
                i += window
                result_type = 'loss_timeout'

            trades.append({
                'Time': entry_time,
                'Pips_Nets': pips_brut * weight,
                'Pips_Bruts': pips_brut,
                'Weight': weight,
                'result': result_type,
                'filter_rejected': filter_info,
            })
            continue
        i += 1

    if not trades:
        empty = pd.DataFrame(columns=['Time', 'Pips_Nets', 'Pips_Bruts', 'Weight', 'result', 'filter_rejected'])
        return empty.set_index('Time'), n_signaux, n_filtres_appliques

    return pd.DataFrame(trades).set_index('Time'), n_signaux, n_filtres_appliques


def _buy_and_hold_pips(df):
    """Profit en pips d'un buy & hold long (achat à la 1re close, vente à la dernière)."""
    if df is None or df.empty:
        return 0.0
    closes = df['Close'].dropna()
    if len(closes) < 2:
        return 0.0
    return (closes.iloc[-1] - closes.iloc[0]) / PIP_SIZE


def compute_metrics(trades_df, annee=None, df=None):
    """Métriques agrégées sur un DataFrame de trades.

    Si `df` (le DataFrame d'entrée du backtest, avec 'Close') est fourni, ajoute
    les métriques de benchmark buy & hold (audit I1) et un alpha = stratégie - B&H.

    Toutes les métriques `*_pct` sont exprimées en fraction de INITIAL_CAPITAL
    (audit I2). `sharpe` et `pips_sharpe` sont mathématiquement identiques
    (scaling linéaire), mais `sharpe` est désormais sur une mesure standard de
    returns ce qui le rend comparable inter-stratégies / vs benchmark.
    """
    base = {
        'annee': annee,
        'profit_net': 0.0,
        'dd': 0.0,
        'trades': 0,
        'win_rate': 0.0,
        'sharpe': 0.0,
        'pips_sharpe': 0.0,
        'sharpe_per_trade': 0.0,
        'total_return_pct': 0.0,
        'max_dd_pct': 0.0,
        'bh_pips': 0.0,
        'bh_return_pct': 0.0,
        'alpha_pips': 0.0,
        'alpha_return_pct': 0.0,
    }
    if trades_df.empty:
        return base

    win_rate = (trades_df['Pips_Bruts'] > 0).mean() * 100
    profit_net = trades_df['Pips_Nets'].sum()
    cum = trades_df['Pips_Nets'].cumsum()
    dd = (cum - cum.cummax()).min()

    n_trades = len(trades_df)

    daily_pips = trades_df['Pips_Nets'].resample('D').sum().dropna()
    daily_returns = _pips_to_return(daily_pips)

    pips_sharpe = (
        (daily_pips.mean() / daily_pips.std()) * np.sqrt(252)
        if (len(daily_pips) > 1 and daily_pips.std() != 0) else 0.0
    )
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if (len(daily_returns) > 1 and daily_returns.std() != 0) else 0.0
    )

    # Sharpe per-trade : neutralise l'effet de lissage par diversification intraday.
    # Avec beaucoup de trades par jour (ex: 4+), le resample('D').sum() écrase la
    # volatilité réelle car les trades se compensent dans la même journée.
    # Le Sharpe per-trade mesure le résultat de chaque décision indépendante.
    trade_returns = _pips_to_return(trades_df['Pips_Nets'])
    sharpe_per_trade = (
        (trade_returns.mean() / trade_returns.std()) * np.sqrt(n_trades)
        if (n_trades > 1 and trade_returns.std() != 0) else 0.0
    )

    total_return_pct = _pips_to_return(profit_net) * 100
    max_dd_pct = _pips_to_return(dd) * 100

    bh_pips = _buy_and_hold_pips(df)
    bh_return_pct = _pips_to_return(bh_pips) * 100
    alpha_pips = profit_net - bh_pips
    alpha_return_pct = total_return_pct - bh_return_pct

    return {
        'annee': annee,
        'profit_net': profit_net,
        'dd': dd,
        'trades': len(trades_df),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'pips_sharpe': pips_sharpe,
        'sharpe_per_trade': sharpe_per_trade,
        'total_return_pct': total_return_pct,
        'max_dd_pct': max_dd_pct,
        'bh_pips': bh_pips,
        'bh_return_pct': bh_return_pct,
        'alpha_pips': alpha_pips,
        'alpha_return_pct': alpha_return_pct,
    }


def save_trades_detailed(trades_df, annee, df=None, output_dir=DIR_RESULTS):
    """Persiste la liste des trades pour analyse ex-post (utilisée par 5_analyze_losses.py).

    Si `df` (DataFrame d'entrée du backtest, avec features et probas) est fourni,
    on enrichit chaque trade avec les features et probas observées au moment de l'entrée.
    """
    if trades_df.empty:
        return None
    out = trades_df.copy()
    if df is not None:
        proba_cols = ['Confiance_Hausse_%', 'Confiance_Neutre_%', 'Confiance_Baisse_%']
        feature_cols = [
            c for c in df.columns
            if c not in (
                ['High', 'Low', 'Close', 'Spread', 'Signal', 'Weight', 'proba_max',
                 'Prediction_Modele', 'Close_Reel_Direction', 'Filter_Rejected'] + proba_cols
            )
        ]
        enrich_cols = feature_cols + proba_cols + ['Filter_Rejected']
        enrich_cols = [c for c in enrich_cols if c in df.columns]
        enrich = df.loc[df.index.intersection(out.index), enrich_cols].copy()
        enrich = enrich.rename(columns={
            'Confiance_Hausse_%': 'proba_hausse',
            'Confiance_Neutre_%': 'proba_neutre',
            'Confiance_Baisse_%': 'proba_baisse',
        })
        out = out.join(enrich, how='left')
    os.makedirs(output_dir, exist_ok=True)
    path = f'{output_dir}/Trades_Detailed_{annee}.csv'
    out.to_csv(path)
    return path


def save_report_md(metrics, annee, output_dir=DIR_PREDICTIONS, version=None,
                   notes=None, n_signaux=None):
    """Génère un rapport Markdown à partir du dict de métriques.

    Évite le copier-coller manuel des V1/V2 (audit C2). Si `version` est fourni,
    le rapport est sauvé sous `{output_dir}/{version}/Rapport_Performance_{annee}.md`,
    sinon directement sous `{output_dir}/Rapport_Performance_{annee}.md`.
    """
    target_dir = f'{output_dir}/{version}' if version else output_dir
    os.makedirs(target_dir, exist_ok=True)
    path = f'{target_dir}/Rapport_Performance_{annee}.md'

    n_trades = metrics.get('trades', 0)
    profit_net = metrics.get('profit_net', 0.0)
    win_rate = metrics.get('win_rate', 0.0)
    dd = metrics.get('dd', 0.0)
    sharpe = metrics.get('sharpe', 0.0)
    pips_sharpe = metrics.get('pips_sharpe', 0.0)
    sharpe_per_trade = metrics.get('sharpe_per_trade', 0.0)
    total_return_pct = metrics.get('total_return_pct', 0.0)
    max_dd_pct = metrics.get('max_dd_pct', 0.0)
    bh_pips = metrics.get('bh_pips', 0.0)
    bh_return_pct = metrics.get('bh_return_pct', 0.0)
    alpha_pips = metrics.get('alpha_pips', 0.0)
    alpha_return_pct = metrics.get('alpha_return_pct', 0.0)
    esperance = profit_net / n_trades if n_trades else 0.0

    config_block = (
        f"TP={TP_PIPS:g}p / SL={SL_PIPS:g}p / Window={WINDOW_HOURS}h / "
        f"Seuil confiance={SEUIL_CONFIANCE:g} / Commission={COMMISSION_PIPS:g}p / "
        f"Capital ref={INITIAL_CAPITAL:g}€"
        f"{' / FiltreTrend=ON' if USE_TREND_FILTER else ''}"
        f"{' / FiltreVol=ON' if USE_VOL_FILTER else ''}"
        f"{' / FiltreSession=ON' if USE_SESSION_FILTER else ''}"
    )

    lines = [
        f"# 📈 Rapport de Performance — EURUSD H1",
        f"**Année testée :** {annee}",
    ]
    if version:
        lines.append(f"**Version :** {version}")
    lines += [
        f"**Configuration :** {config_block}",
        "",
        "## 📊 Stratégie",
        "| Métrique | Valeur |",
        "| :--- | :--- |",
        f"| Nombre de Trades | {n_trades} |",
        f"| Win Rate | {win_rate:.2f}% |",
        f"| Résultat Net | **{profit_net:.1f} pips** ({total_return_pct:+.2f}%) |",
        f"| Max Drawdown | {dd:.1f} pips ({max_dd_pct:.2f}%) |",
        f"| Espérance par trade | {esperance:.2f} pips/trade |",
        f"| Sharpe (returns annualisés) | {sharpe:.2f} |",
        f"| Pips Sharpe (cf. audit I2) | {pips_sharpe:.2f} |",
        f"| Sharpe per-trade | {sharpe_per_trade:.2f} |",
    ]
    if n_signaux is not None:
        ratio = n_signaux / n_trades if n_trades else 0
        lines.append(
            f"| Signaux générés / trades exécutés | {n_signaux} / {n_trades} (×{ratio:.2f}) |"
        )
    lines += [
        "",
        "## 📊 Benchmark Buy & Hold",
        "| Métrique | Valeur |",
        "| :--- | :--- |",
        f"| Buy & Hold Net | {bh_pips:+.1f} pips ({bh_return_pct:+.2f}%) |",
        f"| Alpha (stratégie − B&H) | **{alpha_pips:+.1f} pips ({alpha_return_pct:+.2f}%)** |",
    ]
    if notes:
        lines += ["", "## 📝 Notes", notes]
    lines += ["", "*Généré automatiquement par backtest_utils.save_report_md*"]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return path
