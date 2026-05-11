"""Diagnostic directionnel des trades OOS (VAL_YEAR + TEST_YEAR).

Croise Trades_Detailed_{annee}.csv avec Predictions_{annee}_TripleBarrier.csv
pour reconstituer la direction (Prediction_Modele 1=LONG / -1=SHORT) et
ventiler le PnL. Compare avec Dist_SMA200_D1 (même feature que le filtre
tendance) pour vérifier que les trades respectent bien le filtre.

Standalone — ne modifie aucun pipeline. Lecture pure des CSV results/ + ML_Ready.
"""
import pandas as pd

from config import (
    DIR_RESULTS,
    EVAL_YEARS,
    FILE_EURUSD_H1_CLEAN,
    FILE_ML_READY,
    PIP_SIZE,
)


def diagnostic_annee(annee, prices_all, ml_all):
    trades_path = f'{DIR_RESULTS}/Trades_Detailed_{annee}.csv'
    preds_path = f'{DIR_RESULTS}/Predictions_{annee}_TripleBarrier.csv'

    trades = pd.read_csv(trades_path, index_col='Time', parse_dates=True)
    preds = pd.read_csv(preds_path, index_col='Time', parse_dates=True)

    # Direction depuis Prediction_Modele à l'instant d'entrée du trade
    trades = trades.join(preds[['Prediction_Modele']], how='left')
    trades['direction'] = trades['Prediction_Modele'].map({1.0: 'LONG', -1.0: 'SHORT'})

    # Contexte marché : B&H + % du temps Close > SMA200_D1 (même feature que le filtre)
    p_year = prices_all.loc[f'{annee}-01-01':f'{annee}-12-31']
    bh_pips = (p_year['Close'].iloc[-1] - p_year['Close'].iloc[0]) / PIP_SIZE
    ml_year = ml_all.loc[f'{annee}-01-01':f'{annee}-12-31']
    pct_above = (ml_year['Dist_SMA200_D1'] > 0).mean() * 100

    print(f"\n{'=' * 60}")
    print(f"=== Année {annee} ===")
    print(f"{'=' * 60}")
    print(f"Total trades        : {len(trades)}")
    print(f"B&H (EURUSD)        : {bh_pips:+.0f} pips")
    print(f"Close > SMA200_D1   : {pct_above:.1f}% du temps")
    print(f"PnL stratégie       : {trades['Pips_Nets'].sum():+.1f} pips")

    print(f"\n{'Direction':10s} {'N':>4s} {'WR':>6s} {'PnL':>9s} {'Esp':>8s} "
          f"{'Wins':>5s} {'SL':>5s} {'TO':>4s}")
    print('-' * 60)
    for direction in ['LONG', 'SHORT']:
        sub = trades[trades['direction'] == direction]
        if sub.empty:
            print(f"{direction:10s} {0:>4d} {'  -  ':>6s} {'    -    ':>9s} "
                  f"{'   -   ':>8s} {0:>5d} {0:>5d} {0:>4d}")
            continue
        wins = (sub['result'] == 'win').sum()
        sl = (sub['result'] == 'loss_sl').sum()
        timeout = (sub['result'] == 'loss_timeout').sum()
        wr = wins / len(sub) * 100
        pnl = sub['Pips_Nets'].sum()
        esp = sub['Pips_Nets'].mean()
        print(f"{direction:10s} {len(sub):>4d} {wr:>5.1f}% {pnl:>+8.1f}p "
              f"{esp:>+7.2f}p {wins:>5d} {sl:>5d} {timeout:>4d}")

    # Sanity check : filtre tendance respecté ?
    # Si USE_TREND_FILTER=ON → LONG ne devrait apparaître que quand
    # Dist_SMA200_D1 > 0 à l'entrée, et SHORT seulement quand < 0.
    # Dist_SMA200_D1 est déjà dans trades (sauvegardé par 4_backtest_triple_barrier),
    # mais on tolère aussi le cas où la colonne serait absente.
    if 'Dist_SMA200_D1' not in trades.columns:
        trades = trades.join(ml_all[['Dist_SMA200_D1']], how='left')
    enriched = trades
    enriched['above_sma'] = enriched['Dist_SMA200_D1'] > 0

    long_trades = enriched[enriched['direction'] == 'LONG']
    short_trades = enriched[enriched['direction'] == 'SHORT']

    if len(long_trades) > 0:
        pct_long_in_trend = long_trades['above_sma'].mean() * 100
        print(f"\nLONG pris quand Close>SMA200_D1  : {pct_long_in_trend:.0f}% "
              f"(attendu 100% si filtre actif)")
    if len(short_trades) > 0:
        pct_short_in_trend = (~short_trades['above_sma']).mean() * 100
        print(f"SHORT pris quand Close<SMA200_D1 : {pct_short_in_trend:.0f}% "
              f"(attendu 100% si filtre actif)")


def main():
    prices_all = pd.read_csv(
        FILE_EURUSD_H1_CLEAN, index_col='Time', parse_dates=True
    )
    ml_all = pd.read_csv(
        FILE_ML_READY, index_col='Time', parse_dates=True
    )
    if 'Dist_SMA200_D1' not in ml_all.columns:
        print("⚠️  'Dist_SMA200_D1' absente du CSV ML. Relancer "
              "2_master_feature_engineering.py avant ce diagnostic.")
        return
    for annee in EVAL_YEARS:
        try:
            diagnostic_annee(annee, prices_all, ml_all)
        except FileNotFoundError as e:
            print(f"\n⚠️  Fichier manquant pour {annee} : {e}")


if __name__ == '__main__':
    main()
