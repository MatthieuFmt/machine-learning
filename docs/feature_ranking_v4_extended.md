# Feature ranking v4 — Extension multi-actifs (Phase C2)

**Date** : 2026-05-17
**Périmètre** : 9 couples nouveaux + 3 couples d'origine A6
**Train cutoff** : 2022-12-31

## Tableau récapitulatif

| Actif | TF | Donchian (N,M) | Trades train | WR | Stab top1 | Stab moy top 15 | Statut |
|---|---|---|---|---|---|---|---|
| US30 | D1 | (20, 20) | 232 | 48.3% | 1.00 | 0.72 | A6 (original) |
| EURUSD | H4 | (20, 20) | 506 | 38.7% | 0.80 | 0.59 | A6 (original) |
| XAUUSD | D1 | (100, 20) | 85 | 11.8% | 0.80 | 0.56 | A6 (original) |
| BTCUSD | D1 | (20, 20) | 150 | 43.3% | 0.80 | 0.53 | C2 |
| ETHUSD | D1 | (20, 20) | 145 | 50.3% | 0.60 | 0.56 | C2 |
| ETHUSD | H4 | (20, 20) | 475 | 40.6% | 1.00 | 0.61 | C2 |
| ETHUSD | H1 | (50, 20) | 845 | 40.7% | 1.00 | 0.71 | C2 |
| EURUSD | D1 | (20, 20) | 501 | 64.5% | 0.80 | 0.60 | C2 |
| GBPUSD | D1 | (20, 20) | 482 | 69.3% | 1.00 | 0.60 | C2 |
| GBPUSD | H4 | (20, 20) | 2190 | 46.6% | 0.80 | 0.63 | C2 |
| USDCHF | D1 | (20, 20) | 495 | 59.2% | 0.80 | 0.55 | C2 |
| USDCHF | H4 | (20, 20) | 2040 | 39.9% | 1.00 | 0.61 | C2 |

## Patterns dominants par classe d'actif

### Crypto (BTCUSD, ETHUSD) — stabilité 0.53–0.71

- **Volatilité + price action** : `atr_14`, `atr_pct_14`, `kc_width_20`, `bb_width_20`, `upper_shadow_ratio` dominent systématiquement. Les cryptos sont pilotées par les régimes de volatilité.
- **ETHUSD H1** (stab 0.71) : meilleure stabilité de la phase C2. `kurt_returns_20` et `atr_pct_14` sont les seules features à stability = 1.0 sur 845 trades — l'excès de kurtosis est fortement prédictif du succès Donchian en H1.
- **ETHUSD H4** : `body_to_range_ratio` (stability 1.0) — la taille relative du corps de bougie est le meilleur prédicteur en H4.
- **BTCUSD D1** (stab 0.53) : stabilité la plus faible du groupe. `atr_14` (stability 0.8) domine, suivi par `sma_50`, `cci_20`, `dist_ema_12`.
- **Cross-asset** : `btcusd_return_5` apparaît dans ETHUSD H4 (stab 0.6), `usdchf_return_5` dans ETHUSD H1 (stab 0.6).

### Forex majeures (EURUSD, GBPUSD, USDCHF) — stabilité 0.55–0.63

- **Price action dominante** : `body_to_range_ratio`, `range_atr_ratio`, `upper_shadow_ratio`, `lower_shadow_ratio` dans le top 5 de tous les forex D1.
- **GBPUSD D1** (stab 0.60) : `usdchf_return_5` en #1 (stability 1.0), `range_atr_ratio` #2 (stability 1.0). La paire GBPUSD est fortement corrélée aux mouvements du USDCHF.
- **GBPUSD H4** (stab 0.63) : 2190 trades — le plus gros échantillon. `autocorr_returns_lag1_20` et `kurt_returns_20` en tête (stability 0.8). `session_tokyo` apparaît pour la première fois dans un top 15 (stability 0.6) — les sessions deviennent informatives en H4 sur GBPUSD.
- **EURUSD D1** (stab 0.60) : WR 64.5% impressionnante. `body_to_range_ratio`, `range_atr_ratio`, `upper_shadow_ratio` en top 3 (stability 0.8). `vol_percentile_60` fait son apparition (stability 0.6) — absent des tops A6.
- **USDCHF H4** (stab 0.61) : `atr_zscore_60` #1 (stability 1.0). `xauusd_return_5` #4 (stability 0.8) — lien or/CHF documenté. `close_zscore_60` et `sma_200` apparaissent (features long-terme).
- **USDCHF D1** (stab 0.55) : stabilité la plus faible du groupe forex. `williams_r_14` #1 (stability 0.8), `upper_shadow_ratio` #2 (stability 0.8). Mix price action + momentum.

### Indices (US30 uniquement dans C2) — stabilité 0.72 (A6)

- **US30 D1** (A6) reste le benchmark avec stab 0.72. Distances aux MAs dominent — profil trend-following pur.
- Aucun nouvel indice (GER30, US500) dans C2 car non inclus dans le périmètre `new_phase_c`.

### Métaux (XAUUSD uniquement dans C2) — stabilité 0.56 (A6)

- **XAUUSD D1** (A6) : 85 trades seulement, WR 11.8%. MAs brutes + price action + cross-asset. Stabilité 0.56 — échantillon trop faible pour un ranking robuste.
- XAGUSD non inclus dans le périmètre C2.

### Features cross-asset : confirmation

Les features cross-asset (`usdchf_return_5`, `xauusd_return_5`, `btcusd_return_5`) apparaissent dans **8 des 9** tops C2 (absentes uniquement de BTCUSD D1). C'est la confirmation de l'observation A6 : les retours 5-barres d'actifs corrélés sont systématiquement informatifs pour le méta-labeling Donchian.

### Features cycliques : première apparition

- `day_sin` et `day_cos` apparaissent dans GBPUSD D1 (stability 0.8, rang #4–#5) — première fois que des features cycliques passent le cutoff.
- `month_cos` présent dans XAUUSD D1 A6 (rang #15, stability 0.4).
- `session_tokyo` dans GBPUSD H4 (stability 0.6, rang #13).

### Features absentes (tous couples C2)

- **Economic** (9 features) : toujours stability 0.0 — confirmé sur 9 nouveaux couples.
- **Vol Regime** (3 features) : stability 0.0 — sauf `vol_percentile_60` qui apparaît dans EURUSD D1.
- **Patterns chandeliers rares** (inside_bar, outside_bar, doji) : stability 0.0 — confirmé.
- **Sessions** (hors session_tokyo) : stability 0.0 sur D1/H4/H1.

## Shortlist C3 (stability moyenne ≥ 0.5)

Les 9 couples C2 passent tous le seuil stability ≥ 0.5. Classés par stabilité décroissante :

| Rang | Actif | TF | Stab moy | Trades train | WR | Statut C3 |
|------|-------|-----|----------|-------------|-----|-----------|
| 1 | ETHUSD | H1 | 0.71 | 845 | 40.7% | ✅ GO |
| 2 | GBPUSD | H4 | 0.63 | 2190 | 46.6% | ✅ GO |
| 3 | ETHUSD | H4 | 0.61 | 475 | 40.6% | ✅ GO |
| 4 | USDCHF | H4 | 0.61 | 2040 | 39.9% | ✅ GO |
| 5 | EURUSD | D1 | 0.60 | 501 | 64.5% | ✅ GO |
| 6 | GBPUSD | D1 | 0.60 | 482 | 69.3% | ✅ GO |
| 7 | ETHUSD | D1 | 0.56 | 145 | 50.3% | ✅ GO |
| 8 | USDCHF | D1 | 0.55 | 495 | 59.2% | ✅ GO |
| 9 | BTCUSD | D1 | 0.53 | 150 | 43.3% | ✅ GO |

**Total shortlist C3** : 9 couples (auxquels s'ajoutent les 3 couples A6 US30/D1, EURUSD/H4, XAUUSD/D1 déjà en Phase B).

## Couples exclus (raison)

- **Aucun** — 9/9 couples `new_phase_c` ont passé le cutoff stability ≥ 0.5.
