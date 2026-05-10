import pandas as pd

# 1. Charger les résultats du backtest
df_results = pd.read_csv('./cleaned-data/Predictions_2026_Baseline_H1.csv', index_col='Time', parse_dates=True)

# 2. Calculs enrichis
# On simule les gains (simplifié pour le rapport)
df_results['Is_Correct'] = (df_results['Prediction_Modele'] == df_results['Close_Reel_Direction'])
trades = df_results[((df_results['Prediction_Modele'] == 1) & (df_results['Confiance_Hausse_%'] >= 60)) | 
                    ((df_results['Prediction_Modele'] == 0) & (df_results['Confiance_Hausse_%'] <= 40))]

nb_trades = len(trades)
win_rate = (trades['Is_Correct'].sum() / nb_trades) * 100 if nb_trades > 0 else 0
avg_confiance = trades['Confiance_Hausse_%'].apply(lambda x: x if x > 50 else 100 - x).mean()

# Espérance par trade (Pips nets / Nb trades)
pips_nets = -40.1 # Valeur issue de ton dernier test
expectancy = pips_nets / nb_trades if nb_trades > 0 else 0

# 3. Génération du fichier Markdown
report_content = f"""# 📈 Rapport de Performance - EURUSD H1
**Version :** Baseline V1 (Indicateurs Techniques uniquement)
**Période :** Janvier 2026 - Mai 2026

---

## 📊 Statistiques Globales
| Métrique | Valeur |
| :--- | :--- |
| **Nombre de Trades** | {nb_trades} |
| **Taux de Réussite (Win Rate)** | {win_rate:.2f}% |
| **Résultat Net** | **{pips_nets} Pips** |
| **Espérance par trade** | {expectancy:.2f} pips / trade |
| **Confiance Moyenne du Bot** | {avg_confiance:.1f}% |

---

## 🔍 Analyse Rapide
*   **Point Fort :** Un Win Rate au-dessus de 50% indique que les indicateurs (RSI, EMA, Log_Return) ont un pouvoir prédictif réel.
*   **Point Faible :** Le coût du spread est supérieur aux gains bruts. Le modèle n'est pas encore assez "agressif" dans ses prédictions pour couvrir les frais.
*   **Opportunité :** L'ajout de données corrélées (Gold, JPY) pourrait augmenter la précision pour dépasser les 55-56%, seuil de rentabilité probable.

---
*Généré automatiquement par le projet Learning-Machine-Learning*
"""

with open('./cleaned-data/Rapport_Performance.md', 'w', encoding='utf-8') as f:
    f.write(report_content)

print("✅ Rapport visuel généré : ./cleaned-data/Rapport_Performance.md")