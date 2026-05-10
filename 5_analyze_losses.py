import pandas as pd
import matplotlib.pyplot as plt

ANNEE_TEST = 2025
df = pd.read_csv(f'./results/Trades_Detailed_{ANNEE_TEST}.csv', index_col='Time', parse_dates=True)

losses = df[df['Pips_Nets'] < 0]
wins   = df[df['Pips_Nets'] > 0]

# Résumé
print(f"Total trades : {len(df)} | Pertes : {len(losses)} | Win rate : {len(wins)/len(df)*100:.1f}%")
print(f"Pertes SL : {(losses['result']=='loss_sl').sum()} | Pertes Timeout : {(losses['result']=='loss_timeout').sum()}")

# Top 5 features où les moyennes diffèrent le plus
features = [c for c in df.columns if c not in ('Signal','Pips_Nets','result','proba_hausse','proba_neutre','proba_baisse')]
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
plt.show()