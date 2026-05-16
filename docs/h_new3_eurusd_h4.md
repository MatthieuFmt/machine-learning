# H_new3 — EURUSD H4 mean-reversion + meta-labeling

## Question
Un signal RSI(14, 30/70) + Bollinger(20, 2) filtré par méta-labeling RF sur EURUSD H4 produit-il un edge statistiquement significatif en walk-forward ?

## Hypothèses sous-jacentes
- Les extrêmes RSI (< 30 ou > 70) coïncidant avec une cassure des bandes de Bollinger signalent un retour à la moyenne exploitable.
- Le méta-labeling RF (winner/loser) filtre suffisamment de faux signaux pour maintenir WR > 50 %.
- La fréquence de trades en H4 reste ≥ 30/an malgré des signaux plus rares que H1.

## Résultats walk-forward OOS

| Métrique | Valeur | Seuil | Statut |
|----------|--------|-------|--------|
| Sharpe (per-trade) | +1.73 | ≥ 1.0 | ✅ |
| Sharpe (annualisé, validate_edge) | +5.39 | ≥ 1.0 | ✅ |
| DSR | +23.41 | > 0 | ✅ |
| p-value | 0.0 | < 0.05 | ✅ |
| Drawdown max | 8.1% | < 15% | ✅ |
| Win Rate | 53.7% | > 30% | ✅ |
| **Trades/an** | **25.2** | **≥ 30** | **❌** |
| Profit net | 281.4 pips (€5,628) | — | — |
| Capital final | €15,628 (+56.3%) | — | — |

**Verdict : ❌ NO-GO** — seul critère échoué : 25.2 trades/an < 30.

## Segments walk-forward

| Période | n_train | Trades OOS | Sharpe OOS | WR OOS | Méta actif | Seuil |
|---------|---------|------------|------------|--------|------------|-------|
| 2024-01 → 2024-06 | 779 | 3 | +2.35 | 100% | ✅ | 0.50 |
| 2024-07 → 2024-12 | 794 | 6 | +0.60 | 16.7% | ✅ | 0.50 |
| 2025-01 → 2025-06 | 822 | 30 | +2.31 | 56.7% | ✅ | 0.50 |
| 2025-07 → 2025-12 | 861 | 4 | −2.50 | 50.0% | ✅ | 0.50 |
| 2026-01 → 2026-05 | 885 | 11 | +2.48 | 54.5% | ✅ | 0.50 |

## Décision

**NO-GO.** La stratégie a un edge réel (Sharpe +5.39 annualisé, DSR +23.41 p=0.0) mais la fréquence de trades est insuffisante pour valider la robustesse statistique. 54 trades sur 26 mois = 25.2/an, sous le seuil de 30.

Le méta-labeling RF n'a jamais été désactivé (threshold=0.50 sur tous les segments), confirmant qu'il trouve un signal utilisable — mais les signaux bruts sont tout simplement trop rares.

## Causes possibles d'échec

1. **Rareté des conditions RSI+Bollinger simultanées en H4** : le croisement RSI extrême + cassure BB est un événement peu fréquent sur cette timeframe.
2. **Seuil de trades/an trop exigeant pour du mean-reversion H4** : 25 trades/an avec Sharpe > 5 est statistiquement plus informatif que 30 trades/an avec Sharpe < 1. Le critère pourrait être modulé par le Sharpe.
3. **Pipeline ML figé (A9)** : pas de tuning possible du RF, seuil fixe à 0.55. Un seuil plus bas pourrait laisser passer plus de trades.

---

## 2026-05-16 — Pivot v4 B2 : H_new3 EURUSD H4 mean-reversion + meta

- **n_trials** : 26
- **Sharpe walk-forward** : +1.73 per-trade
- **Trades** : 54
- **Verdict** : NO-GO (trades/an)
- **Fichiers** : `app/strategies/mean_reversion.py`, `scripts/run_h_new3_eurusd_h4.py`, `predictions/h_new3_eurusd_h4.json`

## Critères go/no-go

| Critère | Seuil | Réel | GO ? |
|---------|-------|------|------|
| Sharpe annualisé | ≥ +1.0 | +5.39 | ✅ |
| DSR (p < 0.05) | > 0 | +23.41 | ✅ |
| Max DD | < 15% | 8.1% | ✅ |
| Win Rate | > 30% | 53.7% | ✅ |
| Trades/an | ≥ 30 | 25.2 | ❌ |

---

### A1 — Pourquoi EURUSD H4 et pas H1 ?
H4 réduit le bruit intraday et favorise les configurations techniques plus propres. Le mean-reversion fonctionne mieux sur des barres plus larges où les extrêmes sont plus significatifs.

### A2 — Pourquoi mean-reversion et pas trend-following ?
Complémentarité avec H_new1 (Donchian trend-following US30). EURUSD range 70% du temps → mean-reversion plus adapté que trend-following sur cet actif.

### A3 — RSI(14) seuils 30/70
Seuils classiques de Wilder. 30 = survente, 70 = surachat. Testés et validés depuis 1978.

### A4 — Bollinger(20, 2) seuils
20 périodes = ~1 mois de trading H4. 2 écarts-types = 95% des prix dans les bandes. Cassure = événement rare (2.5% de probabilité).

### A5 — Filtre session Londres/NY
Feature `Is_London_NY_Overlap` (13h–17h UTC) comme input du méta-labeling RF, pas comme filtre dur. Le RF apprend si cette information est pertinente.

### A6 — Timeout 32 barres H4 = 8 jours
Géré par le simulateur déterministe stateful (`app/backtest/deterministic.py`).

## Fin du rapport B2.
