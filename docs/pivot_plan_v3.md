# Pivot Plan v3 — Post Phase 2

## Diagnostic racine
- Le méta-labeling RF v2 (H05, Sharpe +8.84 WF) n'a jamais été retesté avec coûts v3
- Toute la roadmap H06-H18 testait des stratégies pures sans méta-labeling → 0 GO
- XAUUSD D1 a Sharpe brut +1.46, DSR +2.88 (p=0.002) — edge significatif, seul WR bloque
- 0 hypothèse H4/H1 testée en v3. BTCUSD/ETHUSD jamais testés.

## Règles absolues (10 commandements)
1. Pas de stratégie déterministe pure sans méta-labeling
2. Sweep seuil méta sur TRAIN UNIQUEMENT (pas val)
3. Pas de features de contexte de marché dans méta-modèle
4. RF uniquement pour méta-labeling (pas GBM)
5. Pas de portfolio tant qu'aucune stratégie n'a Sharpe > 0
6. Abandonner roadmap H09-H18
7. Split figé : train ≤ 2022, val = 2023, test ≥ 2024
8. Test set lu 1 seule fois par hypothèse
9. Toujours validate_edge() avec n_trials incrémenté
10. Sharpe sur pct_change de l'equity curve

## Paramètres figés
- Capital : 10 000 €
- Risque : 2% / trade
- Coûts v3 XAUUSD : spread 25 + slippage 10 pips
- Coûts v3 US30 : spread 3.0 + slippage 5.0
- Coûts v3 GER30 : spread 2.0 + slippage 5.0
- Méta-labeling : RF 200 arbres, max_depth=4, 8 features SANS contexte
- Seuil méta : sweep [0.50-0.60] sur train, plancher 0.50

## Hypothèses

### H1 — 🔴 P1 : Méta-labeling RF sur XAUUSD D1 (Donchian)
- Question : Le méta-labeling RF (v2) sur Donchian(N=100, M=20) XAUUSD D1 → Sharpe ≥ 1.0, WR ≥ 30%, ≥ 30 trades/an ?
- Méthode : RF binaire, 8 features sans contexte, sweep seuil train, TP/SL 600/300
- GO : Sharpe test ≥ 1.0, WR ≥ 30%, ≥ 30 trades/an, DSR > 0
- NO-GO : Sharpe < 0.5, WR < 25%, < 20 trades/an
- Statut : ❌ NO-GO (2026-05-14)
- Résumé : 0 trade en test méta — RF rejette tous les signaux. Train ≤2022 : 1 win / 68 samples, RF ne peut rien apprendre. Split figé crée une distribution train/test inversée pour XAUUSD.
- Résultats :

| Période | Sharpe | WR | Trades |
|---------|--------|-----|--------|
| Train base (≤2022) | +1.03 | 1.5% | 68 |
| Val base (2023) | 0.00 | 0.0% | 4 |
| Test base (≥2024) | +2.06 | 25.8% | 31 |
| Test méta | 0.00 | 0.0% | 0 |

- Critères : Sharpe 0.00 ✗, WR 0.0% ✗, Trades/an 0.0 ✗
- Leçon : Split figé unique tous actifs = fragilité structurelle. Chaque actif a son propre régime de profitabilité temporelle.

### H2 — 🔴 P1 : Méta-labeling RF sur US30 D1 (Donchian)
- Question : Le méta-labeling RF v2 (+8.84) survit-il aux coûts v3 ?
- Méthode : Donchian(N=20, M=20) ET (N=100, M=10), RF 200 arbres, sweep seuil train, plancher 0.50
- GO : Sharpe test ≥ 1.0, WR ≥ 35%, ≥ 30 trades/an, DSR > 0
- NO-GO : Sharpe < 0.3, WR < 30%, < 15 trades/an
- Statut : ⏳ À exécuter (dépend de H1 pour l'archi)

### H3 — 🟠 P2 : Donchian + méta-labeling US30/XAUUSD H4
- Question : H4 → ≥ 50 trades/an, Sharpe ≥ 1.0 ?
- Méthode : Grid search Donchian H4, TP/SL = 2×/1× ATR(20), RF identique
- GO : Sharpe ≥ 1.0, ≥ 50 trades/an, WR ≥ 30%, DSR > 0
- NO-GO : Sharpe < 0.5, < 25 trades/an
- Statut : ⏳ À exécuter (dépend H1, H2)

### H4 — 🟡 P3 : Donchian + méta-labeling BTCUSD/ETHUSD D1
- Question : Crypto → Sharpe ≥ 1.0 avec coûts v3 ?
- Méthode : Grid search N ∈ {20,50,100,200}, TP/SL = 3×/1.5× ATR(20)
- GO : Sharpe ≥ 1.0, ≥ 25 trades/an, WR ≥ 30%
- NO-GO : Sharpe < 0.3, < 15 trades/an
- Statut : ⏳ À exécuter (dépend H1 pour l'archi)

### H5 — 🟢 P4 : Mean-reversion RSI(2) extrême US30 H1
- Question : Mean-reversion décorrélé (ρ < 0.3) → Sharpe ≥ 1.0 ?
- Méthode : RSI(2) < 10 → LONG, RSI(2) > 90 → SHORT, filtre ATR, sortie RSI traverse 50 ou SL 1.5× ATR
- GO : Sharpe ≥ 1.0, ≥ 100 trades/an, ρ < 0.3, Max DD < 20%
- NO-GO : Sharpe < 0.3, < 50 trades/an, ρ ≥ 0.5
- Statut : ❌ NO-GO (2026-05-15)
- Résumé : 1765 trades en test, Sharpe −0.95, PnL moyen −4.0 pts/trade. 72% des sorties par RSI cross — le prix dérive contre la position. RSI(2) extrême = générateur de bruit, pas d'edge.
- Résultats :

| Période | Sharpe | WR | Trades | PnL (pts) | Max DD (pts) | T/an |
|---------|--------|-----|--------|-----------|-------------|------|
| Train (≤2022) | +0.16 | 53.8% | 6120 | −44,255 | −44,358 | 637 |
| Val (2023) | −1.17 | 57.8% | 725 | −2,476 | −3,987 | 736 |
| Test (≥2024) | −0.95 | 55.1% | 1765 | −7,074 | −10,392 | 748 |

- Critères : Sharpe −0.95 ✗, Trades/an 748 ✓, Max DD 92.8% ✗
- Leçon : Le mean-reversion RSI(2) sur US30 H1 ne capture aucun edge. Même avec méta-labeling, le signal sous-jacent est trop faible. Abandonner les approches mean-reversion pures.

## Ordre d'exécution
| Ordre | ID | Priorité | Effort | Dépend de | Statut |
|--------|-----|----------|--------|-----------|--------|
| 1 | H1 | P1 | 1j | — | ❌ NO-GO |
| 2 | H2 | P1 | 0.5j | H1 | ⏳ |
| 3 | H3 | P2 | 1.5j | H1,H2 | ⏳ |
| 4 | H4 | P3 | 1j | H1 | ⏳ |
| 5 | H5 | P4 | 1.5j | — | ❌ NO-GO |

## Compteurs
- n_trials actuel : 24
- n_trials après pivot : 26 (23 + 3)
- DSR critique corrigé à reporter

## Note — Split structurel
Le split figé unique (train ≤2022, val=2023, test ≥2024) s'est révélé cassant pour XAUUSD D1 : la période rentable (test ≥2024) est exclue de l'apprentissage, la période non-rentable (train ≤2022, 1 win / 68) domine le méta-modèle. Une réévaluation par actif est nécessaire avant de poursuivre H2, H3, H4.
