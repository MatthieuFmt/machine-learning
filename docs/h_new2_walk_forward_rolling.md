# H_new2 — Walk-Forward Rolling 3 ans (B3)

> **Pipeline ML frozen A9** — Donchian(20,20) + méta-labeling RF
> **Test set** : 2024+ (read_oos(), 1 seul read par actif)
> **Méthode** : fenêtre train glissante 3 ans, retrain 6 mois, embargo 2 jours

---

## Résultats

| Actif | Verdict | Sharpe (equity) | Sharpe (per-trade) | DSR | p(Sharpe>0) | Trades | WR | DD% | Trades/an | Profit € |
|---|---|---|---|---|---|---|---|---|---|---|
| US30 D1 | ❌ NO-GO | 0.60 | 0.82 | −18.27 | 1.000 | 34 | 44.1% | 10.0% | 16.4 | +2 073.86 |
| XAUUSD D1 | ❌ NO-GO | 1.65 | 2.59 | 1.46 | 0.072 | 23 | 69.6% | 0.9% | 12.0 | +1 112.97 |

## Détail par segment — US30 D1

| Segment | Train | OOS | n_train | n_oos | Sharpe OOS | Threshold |
|---|---|---|---|---|---|---|
| S1 | 2021 – 2023-12 | 2024 H1 | 131 | 4 | −0.82 | 0.55 |
| S2 | 2021-07 – 2024-06 | 2024 H2 | 124 | 6 | +2.56 | 0.50 |
| S3 | 2022 – 2024-12 | 2025 H1 | 123 | 13 | +1.15 | 0.50 |
| S4 | 2022-07 – 2025-06 | 2025 H2 | 133 | 5 | −1.41 | 0.55 |
| S5 | 2023 – 2025-12 | 2026 YTD | 118 | 6 | +2.64 | 0.50 |

## Détail par segment — XAUUSD D1

| Segment | Train | OOS | n_train | n_oos | Sharpe OOS | Threshold |
|---|---|---|---|---|---|---|
| S1 | 2021 – 2023-12 | 2024 H1 | 101 | 1 | 0.00 | 0.55 |
| S2 | 2021-07 – 2024-06 | 2024 H2 | 105 | 5 | +1.56 | 0.55 |
| S3 | 2022 – 2024-12 | 2025 H1 | 112 | 7 | +2.30 | 0.50 |
| S4 | 2022-07 – 2025-06 | 2025 H2 | 113 | 6 | +3.69 | 0.55 |
| S5 | 2023 – 2025-12 | 2026 YTD | 122 | 4 | +2189.72 ⚠️ | 0.55 |

> ⚠️ XAUUSD S5 Sharpe OOS = 2189.72 : anomalie probable (4 trades, tous gagnants sur 2026 YTD).

## Raisons de rejet validate_edge

### US30
- Sharpe (equity) 0.60 < 1.0
- DSR = −18.27 (p = 1.000) → pas de significativité statistique
- Trades/an = 16.4 < 30

### XAUUSD
- DSR = 1.46 (p = 0.072) → non significatif au seuil α = 0.05
- Trades/an = 12.0 < 30
- Sharpe (equity) 1.65 ≥ 1.0 ✅ mais insuffisant sans DSR valide

## Verdict global

**❌ NO-GO** — Les deux actifs échouent les critères validate_edge. XAUUSD montre un Sharpe per-trade élevé mais le DSR n'est pas significatif (p = 0.072) et le nombre de trades est trop faible. US30 souffre d'un Sharpe insuffisant et d'un DSR fortement négatif.

## Comparaison avec H_new1 (B1) et H_new3 (B2)

| Hypothèse | Stratégie | Actif | Verdict | Sharpe |
|---|---|---|---|---|
| H_new1 (B1) | Donchian + méta-labeling RF | US30 D1 | ❌ NO-GO | 0.82 |
| H_new3 (B2) | Mean-rev + méta-labeling RF | EURUSD H4 | ✅ GO | 1.73 |
| **H_new2 (B3)** | **Walk-forward rolling + méta-labeling RF** | **US30 + XAUUSD D1** | **❌ NO-GO** | **0.60 / 1.65** |

## Leçons

1. Le walk-forward rolling sur US30 D1 donne un Sharpe equity (0.60) inférieur au backtest simple de H_new1 (0.82) — l'approche rolling expose l'instabilité temporelle.
2. XAUUSD D1 a un Sharpe per-trade excellent (2.59) mais seulement 23 trades sur 2.5 ans → pas assez de densité de signal.
3. Le critère ≥ 30 trades/an est le facteur bloquant principal pour les deux actifs en D1 avec Donchian(20,20).
4. Le pipeline ML frozen A9 montre que même sans re-tuning, le méta-labeling apporte de la valeur (Sharpe > 0) mais pas assez pour passer les quality gates.
