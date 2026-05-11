# 📈 Rapport de Performance — EURUSD H1
**Année testée :** 2025
**Configuration :** TP=20p / SL=10p / Window=24h / Seuil confiance=0.45 / Commission=0.5p / Capital ref=10000€

## 📊 Stratégie
| Métrique | Valeur |
| :--- | :--- |
| Nombre de Trades | 33 |
| Win Rate | 39.39% |
| Résultat Net | **-4.9 pips** (-0.05%) |
| Max Drawdown | -83.2 pips (-0.83%) |
| Espérance par trade | -0.15 pips/trade |
| Sharpe (returns annualisés) | -0.05 |
| Pips Sharpe (cf. audit I2) | -0.05 |
| Sharpe per-trade | -0.06 |
| Signaux générés / trades exécutés | 72 / 33 (×2.18) |

## 📊 Benchmark Buy & Hold
| Métrique | Valeur |
| :--- | :--- |
| Buy & Hold Net | +1390.5 pips (+13.90%) |
| Alpha (stratégie − B&H) | **-1395.4 pips (-13.95%)** |

---

## 🔍 Analyse post-backtest (2026-05-10)

### 1. Structure des pertes
| Résultat | Nombre | Pips nets moyens | Proba_Hausse moy | Proba_Baisse moy |
| :--- | :--- | :--- | :--- | :--- |
| win | 13 | +16.0 | 24.5% | 45.6% |
| loss_sl | 20 | -10.7 | 27.0% | 42.8% |
| loss_timeout | 0 | — | — | — |

### 2. Déséquilibre directionnel
| Direction | Trades | Profit | Win Rate |
| :--- | :--- | :--- | :--- |
| SHORT | 28 | +25.3 pips | 42.9% |
| LONG | 5 | -30.2 pips | 20.0% |

### 3. Proba_max (confiance modèle)
| Année | Mean | Médiane | ≥0.45 | ≥0.50 | ≥0.55 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2022 (in-sample) | 0.422 | 0.399 | 1612 | — | — |
| 2023 (in-sample) | 0.419 | 0.397 | 1511 | — | — |
| 2024 (OOS) | 0.373 | 0.364 | 257 | 94 | — |
| 2025 (OOS) | 0.364 | 0.359 | 101 | 16 | 4 |

### 4. Feature Importance (permutation sur 2024)
| Rang | Feature | Permutation_mean | Significatif ? |
| :--- | :--- | :--- | :--- |
| 1 | RSI_14_D1 | 0.02086 | ✅ |
| 2 | Dist_EMA_20_D1 | 0.01835 | ✅ |
| 3 | ADX_14 | 0.01781 | ✅ |
| 4 | Dist_EMA_50 | 0.01085 | ⚠️ borderline |
| 5 | RSI_14 | 0.01002 | ⚠️ borderline |
| 6 | Dist_EMA_50_H4 | 0.00685 | ⚠️ borderline |
| 7 | ATR_Norm | 0.00675 | ⚠️ borderline |
| 8 | Dist_EMA_20_H4 | 0.00341 | ❌ bruit |
| 9 | RSI_14_H4 | 0.00125 | ❌ bruit |

### 5. OOB vs Test Accuracy
| Métrique | Valeur |
| :--- | :--- |
| OOB score (train ≤ 2023) | 0.4417 |
| Test accuracy 2024 | 0.3477 (Δ -0.094) |
| Test accuracy 2025 | 0.3317 (Δ -0.110) |

*Généré automatiquement par backtest_utils.save_report_md*
