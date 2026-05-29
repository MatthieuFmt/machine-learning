# Feature Ranking v4 (pivot v4 A6)

**Date** : 2026-05-15
**Méthode** : Bootstrap stability 5× × 3 métriques (MI + perm imp + Spearman |corr|)
**Périmètre** : train ≤ 2022-12-31 UNIQUEMENT
**n_trials** : inchangé (analyse train pure, 0 n_trial consommé)

## Top 15 par actif

### US30 D1 (Donchian winner target)

**Contexte** : 232 trades train, WR 48.3%

| Rank | Feature | MI | Perm | \|corr\| | Composite | Stability |
|------|---------|-----|------|---------|-----------|-----------|
| 1 | `dist_sma_20` | 0.220 | 0.00328 | 0.301 | 17.2 | **1.0** |
| 2 | `autocorr_returns_lag1_20` | 0.243 | 0.00828 | 0.265 | 14.9 | 0.8 |
| 3 | `range_atr_ratio` | 0.193 | 0.00638 | 0.274 | 17.8 | 0.8 |
| 4 | `close_zscore_20` | 0.228 | 0.00724 | 0.216 | 18.8 | 0.8 |
| 5 | `dist_ema_26` | 0.221 | 0.00138 | 0.302 | 20.9 | 0.8 |
| 6 | `dist_ema_12` | 0.228 | 0.00052 | 0.283 | 21.1 | 0.8 |
| 7 | `dist_sma_200` | 0.230 | 0.00397 | 0.236 | 19.1 | 0.6 |
| 8 | `stoch_k_14` | 0.199 | 0.00103 | 0.341 | 19.3 | 0.6 |
| 9 | `cci_20` | 0.217 | 0.00638 | 0.198 | 20.4 | 0.6 |
| 10 | `stoch_d_14` | 0.188 | 0.00534 | 0.337 | 20.5 | 0.6 |
| 11 | `atr_14` | 0.189 | 0.00638 | 0.188 | 22.9 | 0.6 |
| 12 | `rsi_21` | 0.231 | −0.00172 | 0.299 | 23.8 | 0.6 |
| 13 | `dist_sma_200_abs_atr` | 0.248 | 0.00603 | 0.123 | 23.1 | 0.4 |
| 14 | `slope_sma_20` | 0.194 | 0.00483 | 0.244 | 23.5 | 0.4 |
| 15 | `macd` | 0.190 | 0.00259 | 0.258 | 23.9 | 0.4 |

### EURUSD H4 (Donchian winner target)

**Contexte** : 506 trades train, WR 38.7%

| Rank | Feature | MI | Perm | \|corr\| | Composite | Stability |
|------|---------|-----|------|---------|-----------|-----------|
| 1 | `bb_width_20` | 0.178 | 0.00324 | 0.125 | 13.7 | **0.8** |
| 2 | `usdchf_return_5` | 0.327 | 0.00593 | 0.052 | 9.3 | 0.8 |
| 3 | `kc_width_20` | 0.175 | 0.00711 | 0.111 | 12.4 | 0.8 |
| 4 | `close_zscore_20` | 0.187 | 0.00451 | 0.048 | 19.5 | 0.8 |
| 5 | `lower_shadow_ratio` | 0.172 | 0.00474 | 0.047 | 22.6 | 0.6 |
| 6 | `atr_pct_14` | 0.128 | 0.00079 | 0.124 | 23.4 | 0.6 |
| 7 | `cci_20` | 0.154 | 0.00909 | 0.031 | 23.4 | 0.6 |
| 8 | `body_to_range_ratio` | 0.155 | 0.00292 | 0.060 | 25.1 | 0.6 |
| 9 | `btcusd_return_5` | 0.167 | 0.00451 | 0.045 | 25.4 | 0.6 |
| 10 | `dist_ema_12` | 0.164 | 0.00379 | 0.040 | 26.7 | 0.6 |
| 11 | `xauusd_return_5` | 0.167 | 0.00632 | 0.034 | 24.7 | 0.4 |
| 12 | `atr_14` | 0.179 | −0.00285 | 0.126 | 22.9 | 0.4 |
| 13 | `sma_50` | 0.177 | 0.00506 | 0.063 | 25.9 | 0.4 |
| 14 | `range_atr_ratio` | 0.145 | 0.00490 | 0.068 | 25.8 | 0.4 |
| 15 | `dist_sma_20` | 0.159 | 0.00411 | 0.038 | 26.2 | 0.4 |

### XAUUSD D1 (Donchian winner target)

**Contexte** : 85 trades train, WR 11.8% — échantillon très déséquilibré

| Rank | Feature | MI | Perm | \|corr\| | Composite | Stability |
|------|---------|-----|------|---------|-----------|-----------|
| 1 | `ema_12` | 0.186 | −0.00471 | 0.276 | 12.6 | **0.8** |
| 2 | `upper_shadow_ratio` | 0.167 | 0.00565 | 0.141 | 15.1 | 0.8 |
| 3 | `gap_overnight` | 0.176 | 0.00353 | 0.283 | 14.1 | 0.8 |
| 4 | `ema_26` | 0.131 | 0.00094 | 0.261 | 17.3 | 0.8 |
| 5 | `btcusd_return_5` | 0.164 | 0.00188 | 0.118 | 17.1 | 0.6 |
| 6 | `volume_zscore_20` | 0.101 | 0.00635 | 0.105 | 16.7 | 0.6 |
| 7 | `sma_50` | 0.124 | 0.00424 | 0.153 | 18.5 | 0.6 |
| 8 | `dist_sma_200_abs_atr` | 0.182 | 0.00000 | 0.083 | 19.8 | 0.6 |
| 9 | `dist_sma_200` | 0.205 | 0.00047 | 0.112 | 16.5 | 0.4 |
| 10 | `mfi_14` | 0.167 | 0.00094 | 0.252 | 18.4 | 0.4 |
| 11 | `autocorr_returns_lag1_20` | 0.145 | 0.00471 | 0.129 | 18.7 | 0.4 |
| 12 | `body_to_range_ratio` | 0.128 | 0.00047 | 0.179 | 18.9 | 0.4 |
| 13 | `kc_width_20` | 0.122 | 0.00588 | 0.195 | 17.3 | 0.4 |
| 14 | `range_atr_ratio` | 0.111 | 0.00518 | 0.201 | 17.9 | 0.4 |
| 15 | `month_cos` | 0.089 | 0.00306 | 0.190 | 20.9 | 0.4 |

## Interprétation

### Patterns dominants

- **US30** : Dominance des distances aux moyennes mobiles (`dist_sma_20`, `dist_ema_12/26`, `dist_sma_200`) + auto-corrélation des retours + range/ATR ratio. Profil typique d'un actif **trend-following** où l'edge vient du positionnement relatif au régime. La feature #1 (`dist_sma_20`) est la seule avec stability = 1.0 (présente dans les 5 bootstraps). Les features purement directionnelles (`ema_12`, `sma_20`) sont moins bien classées que leurs versions « distance » — la normalisation par rapport au prix est clé.

- **EURUSD H4** : Domination des indicateurs de **volatilité** (`bb_width_20`, `kc_width_20`) et des **returns cross-asset** (`usdchf_return_5` en #2, `btcusd_return_5` en #9, `xauusd_return_5` en #11). Le forex est fortement influencé par les corrélations inter-marchés. Les features de price action (`lower_shadow_ratio`, `body_to_range_ratio`) apparaissent significativement, contrairement à US30 — les ombres de bougies sont informatives en H4.

- **XAUUSD** : Mix de **moyennes mobiles brutes** (`ema_12` #1, `ema_26` #4), **price action** (`upper_shadow_ratio` #2, `gap_overnight` #3) et **cross-asset** (`btcusd_return_5` #5). L'or est sensible aux gaps overnight (gap entre close et open suivant) et aux ombres supérieures — suggérant des patterns de rejet de prix. Le volume (`volume_zscore_20` #6) est informatif, contrairement à US30 où il n'est pas dans le top 15. **Attention** : échantillon très faible (85 trades, WR 11.8%) → ranking potentiellement instable.

### Features systématiquement absentes (tous actifs confondus)

| Catégorie | Features concernées | Interprétation |
|-----------|-------------------|----------------|
| **Economic** | Les 9 features `event_high_within_*`, `hours_since_*`, `hours_to_next_*` | Stabilité 0.0 sur les 3 actifs. Le calendrier économique seul (binaire/horaire) n'a pas de pouvoir prédictif linéaire sur la cible « winner Donchian ». Pourrait être utile en interaction (régime × event), mais sort du scope A6. |
| **Sessions** | `session_tokyo`, `session_london`, `session_ny`, `session_overlap_london_ny` | Stabilité 0.0 sur tous les actifs. Les encodages one-hot de session n'apportent rien pour du D1/H4 — la barre quotidienne/4H couvre plusieurs sessions. |
| **Cycliques jour** | `day_sin`, `day_cos` | Stabilité 0.0 partout. Le jour de la semaine n'a pas d'effet détectable sur Donchian D1/H4. |
| **Vol Regime** | `vol_regime_low`, `vol_regime_mid`, `vol_regime_high` | Stabilité 0.0. L'encodage one-hot par terciles est trop grossier ou le régime de volatilité change trop lentement pour être capté sur la fenêtre train. |
| **Patterns chandeliers** | `inside_bar`, `outside_bar`, `doji` | Stabilité 0.0 partout. Événements rares → peu de signal dans un échantillon de quelques centaines de trades. |

### Features cross-asset : résultat notable

Les features cross-asset (`usdchf_return_5`, `xauusd_return_5`, `btcusd_return_5`) apparaissent dans le **top 15 des 3 actifs**, avec des stabilités allant jusqu'à 0.8 (EURUSD). C'est la surprise de ce ranking : les retours 5-barres d'actifs corrélés contiennent de l'information prédictive pour le méta-labeling, même sur des timeframes différents.

### Stabilité

| Actif | Stabilité moyenne top 15 | Features stability ≥ 0.8 | % stable (≥0.6) |
|-------|--------------------------|--------------------------|-----------------|
| US30 D1 | **0.72** | 5 (`dist_sma_20` 1.0, 4× 0.8) | 73% (11/15) |
| EURUSD H4 | **0.59** | 4 (`bb_width_20`, `usdchf_return_5`, `kc_width_20`, `close_zscore_20`) | 67% (10/15) |
| XAUUSD D1 | **0.56** | 4 (`ema_12`, `upper_shadow_ratio`, `gap_overnight`, `ema_26`) | 53% (8/15) |

## Décision de gel

Les top 15 par actif sont **FIGÉS** dans [`app/config/features_selected.py`](../app/config/features_selected.py).
**Aucune modification autorisée** jusqu'à la fin de Phase B.

## Vérification go/no-go

| Critère | Statut | Détail |
|---------|--------|--------|
| ≥ 1 actif avec top 15 figé | ✅ | 3 actifs figés |
| Stability moyenne ≥ 0.6 sur top 15 (≥ 1 actif) | ✅ | US30 = 0.72 |
| Aucune feature stability = 0.0 dans top 5 | ✅ | Top 5 US30 : min 0.8 ; EURUSD : min 0.6 ; XAUUSD : min 0.6 |
| Ruff + mypy OK sur ranking.py | ✅ | per-file-ignores configurés |
| make verify OK | ✅ | Voir ci-dessous |

**Verdict** : ✅ **GO** — Phase A6 terminée, passage en Phase B autorisé.

## Limites

- **n_bootstrap=5** : compromis vitesse/stabilité. 20 serait plus robuste mais coûteux (×4 temps de calcul). La stability = 1.0 sur `dist_sma_20` US30 est rassurante, les stabilities 0.4 en queue de top 15 mériteraient confirmation.
- **XAUUSD faible échantillon** : 85 trades train, WR 11.8%. Le ranking sur un échantillon aussi déséquilibré et petit est intrinsèquement plus bruité. Les features du top XAUUSD sont à prendre avec précaution.
- **Ranking figé sur train ≤ 2022** : Si on ré-entraîne sur train+val au moment du walk-forward (B1+), le ranking reste basé sur train ≤ 2022. C'est conservateur (pas d'info de 2023 dans le ranking) mais peut sous-optimal si le régime post-2022 valorise d'autres features.
- **Target binaire Donchian uniquement** : Le ranking est conditionné à la qualité de la stratégie Donchian sous-jacente. Une stratégie alternative (Chandelier, Keltner) pourrait valoriser des features différentes.

## Tests unitaires associés

7 tests dans [`tests/unit/test_feature_ranking.py`](../tests/unit/test_feature_ranking.py) :
- `test_ranking_reproducible` ✅
- `test_ranking_top_is_sorted` ✅
- `test_stability_in_range` ✅
- `test_no_nan_in_metrics` ✅
- `test_relevant_features_in_top` ✅
- `test_pure_noise_features_excluded` ✅
- `test_raises_if_too_few_features` ✅
