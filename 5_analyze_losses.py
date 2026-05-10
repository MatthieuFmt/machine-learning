import pandas as pd
import matplotlib.pyplot as plt

from config import DIR_RESULTS, TEST_YEAR

df = pd.read_csv(f'{DIR_RESULTS}/Trades_Detailed_{TEST_YEAR}.csv', index_col='Time', parse_dates=True)

losses = df[df['Pips_Nets'] < 0]
wins   = df[df['Pips_Nets'] > 0]

# Résumé
print(f"Total trades : {len(df)} | Pertes : {len(losses)} | Win rate : {len(wins)/len(df)*100:.1f}%")
print(f"Pertes SL : {(losses['result']=='loss_sl').sum()} | Pertes Timeout : {(losses['result']=='loss_timeout').sum()}")

# Top 5 features où les moyennes diffèrent le plus
features = [c for c in df.columns if c not in (
    'Signal', 'Pips_Nets', 'Pips_Bruts', 'Weight', 'result',
    'proba_hausse', 'proba_neutre', 'proba_baisse'
)]
diff = {f: abs(wins[f].mean() - losses[f].mean()) for f in features}
top5 = sorted(diff, key=diff.get, reverse=True)[:5]

print("\nTop 5 features avec le plus grand écart moyen (Gagnants vs Perdants) :")
for f in top5:
    print(f"{f:20s} : Gagnants moy={wins[f].mean():.4f}, Perdants moy={losses[f].mean():.4f}")

# Distribution des probas max
df['proba_max'] = df[['proba_hausse','proba_neutre','proba_baisse']].max(axis=1)
ax = df.boxplot(column='proba_max', by='result')
ax.set_title('Probabilité max par résultat')
ax.set_ylabel('proba_max')
plt.suptitle('')
fig_path = f'{DIR_RESULTS}/Loss_Analysis_{TEST_YEAR}.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\n💾 Boxplot sauvegardé : {fig_path}")