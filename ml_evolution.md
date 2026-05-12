# Historique des Améliorations — Pipeline EURUSD H1

## Date/Version: 1 — 2026-05-11
- **Modification** : Application effective du filtrage `features_dropped` dans `build_ml_ready()` (bug: défini dans `EurUsdConfig` mais jamais appliqué). Ajout de `Range_ATR_ratio` à la liste d'exclusion. Réduction de 19 → 7 features d'entraînement (gardées: RSI_14, ADX_14, Dist_EMA_50, RSI_14_D1, Dist_EMA_20_D1, RSI_D1_delta, Dist_SMA200_D1).
- **Hypothèse** : La réduction drastique de dimensionalité (15 features bruit exclues) diminue l'overfitting en forçant le modèle à se concentrer sur les 5 features significatives identifiées par permutation importance. Les features de régime (ATR_Norm, Volatilite_Realisee_24h, Range_ATR_ratio) restent disponibles pour les filtres backtest mais ne polluent plus l'entraînement.
- **Résultats (pré-fix — baseline)** :
  - 2024 OOS : 216 trades, WR 34.3%, Net −258.7 pips, Sharpe −1.19
  - 2025 OOS : 349 trades, WR 32.7%, Net −659.0 pips, Sharpe −2.62
  - Accuracy OOS 2025 : 0.332 (vs aléatoire 0.333)
  - Δ OOB−OOS : −0.110
  - Biais directionnel : 85% SHORT en marché haussier 2025
- **Résultats (post-fix — 2026-05-11)** :
  - Features actives : 7 (`Dist_EMA_50`, `RSI_14`, `ADX_14`, `Dist_EMA_50_H4`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`)
  - 2024 OOS : 521 trades (+141% vs baseline 216), WR 35.5%, Net −515.5 pips, Sharpe −1.74, Alpha +175.7 pips
  - 2025 OOS : 698 trades (+100% vs baseline 349), WR 32.2%, Net −1365.1 pips, Sharpe −3.41, Alpha −2755.6 pips
  - **Analyse** : ✅ Trades/an ≥ 100 atteint et dépassé (521/698). ❌ Sharpe, profit net, win rate dégradés. L'exclusion des features de régime (ATR_Norm, Volatilite_Realisee_24h, Range_ATR_ratio, Dist_SMA200_D1) a supprimé des signaux de régularisation implicite. `Dist_EMA_50_H4` (borderline dans l'audit) est resté actif et pollue probablement encore. Le biais directionnel SHORT persiste (85%→?).
- **Target Metrics pour v2** : Sharpe OOS ≥ −1.0, Win Rate ≥ 38%, Profit net OOS ≥ 0. Restaurer `Dist_SMA200_D1` comme feature d'entraînement ? Ajouter méta-labeling ?

## Date/Version: 2 — 2026-05-11
- **Modification** : Swap dans `EurUsdConfig.features_dropped` : retrait de `Dist_SMA200_D1`, ajout de `Dist_EMA_50_H4`. Dimensionalité inchangée (7 features actives). Les features d'entraînement deviennent : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`, `Dist_SMA200_D1`.
- **Hypothèse** : Le RandomForest n'avait **aucune information de tendance long-terme** dans ses features (`Dist_SMA200_D1` exclue, `TrendFilter` binary-only trop frustre). Le modèle shortait par défaut (biais historique EURUSD 2010-2023), désastreux en marché haussier 2025. `Dist_EMA_50_H4` était du bruit pur (permutation_mean < 0.007). `Dist_SMA200_D1` encode la tendance ~9 mois en continu, complémentaire au `TrendFilter`. Swap 1-pour-1 → pas de risque d'overfitting supplémentaire.
- **Résultats (pré-fix — v1 post-fix)** :
  - Features actives v1 : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `Dist_EMA_50_H4`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`
  - 2024 OOS : 521 trades, WR 35.5%, Net −515.5 pips, Sharpe −1.74, Alpha +175.7 pips
  - 2025 OOS : 698 trades, WR 32.2%, Net −1365.1 pips, Sharpe −3.41, Alpha −2755.6 pips
- **Résultats (post-fix — à confirmer après run pipeline)** :
  - Features actives v2 : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`, `Dist_SMA200_D1`
  - 2024 OOS : [À MESURER]
  - 2025 OOS : [À MESURER]
- **Target Metrics pour v3** : Sharpe OOS ≥ −1.0, Win Rate ≥ 38%, Réduction biais SHORT < 60%, Profit net OOS ≥ 0.

## Date/Version: 3 — 2026-05-11
- **Modification** : Injection du `FilterPipeline` (TrendFilter + VolFilter + SessionFilter) dans `BasePipeline.run_backtest()`. Construction conditionnelle selon `BacktestConfig.use_trend_filter`, `use_vol_filter`, `use_session_filter`. Passage explicite à `simulate_trades(filter_pipeline=...)`.
- **Bug corrigé (v3.1)** : `ValueError: TrendFilter nécessite la colonne 'Dist_SMA200_D1'` — `preds_df` ne contenait que `Prediction_Modele`, `Confiance_*_%`, `Spread`. Fix : join `FILTER_COLS` depuis `ml_data`.
- **Bug corrigé (v3.2)** : `ValueError: VolFilter nécessite la colonne 'ATR_Norm'` — `ATR_Norm` dans `features_dropped` donc exclu de `ml_data`. Fix : `FILTER_KEEP` dans `pipeline.py` préserve `ATR_Norm` + `Dist_SMA200_D1` dans `ml_data` ; `_FILTER_ONLY_COLS` dans `training.py` exclut ces colonnes de l'entraînement via `extra_drop_cols`.
- **Hypothèse** : Les filtres existaient dans le code source (`filters.py`) et étaient testés unitairement mais **jamais connectés au pipeline**. `TrendFilter` bloque les LONG si Close < SMA200 et les SHORT si Close > SMA200 → corrige le biais directionnel (85% SHORT en marché haussier 2025). `VolFilter` bloque les entrées en régime de turbulences (ATR_Norm > 2× médiane glissante 168h) → réduit les SL touchés en période chaotique. `SessionFilter` exclut 22h-01h GMT (faible liquidité, spreads élargis).
- **Résultats (pré-fix — v2)** :
  - Features actives v2 : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`, `Dist_SMA200_D1`
  - 2024 OOS : 521 trades, WR 35.5%, Net −515.5 pips, Sharpe −1.74, Alpha +175.7 pips
  - 2025 OOS : 698 trades, WR 32.2%, Net −1365.1 pips, Sharpe −3.41, Alpha −2755.6 pips
- **Résultats (post-fix — 2026-05-11 mesurés)** :
  - Features actives v3 : `Dist_EMA_50`, `RSI_14`, `ADX_14`, `RSI_14_D1`, `Dist_EMA_20_D1`, `RSI_D1_delta`, `Dist_SMA200_D1` (inchangé)
  - Colonnes ml_data : 10 (ajout `ATR_Norm` préservé pour VolFilter)
  - **2024 OOS** : 512 trades (−1.7% vs 521), WR 30.7% (−4.8pp), Net **−1098.5 pips**, Sharpe **−3.38** (−1.64), Alpha −407.3 pips
  - **2025 OOS** : 848 trades (+21.5% vs 698 ⚠️), WR 31.7% (−0.5pp), Net **−1783.2 pips**, Sharpe **−4.35** (−0.94), Alpha −3173.7 pips
  - **Analyse** : ❌ Dégradation sur toutes les métriques. Le nombre de trades 2025 a **augmenté** (698→848) au lieu de baisser — le TrendFilter n'a pas bloqué les SHORT comme attendu. Cause probable : `Dist_SMA200_D1 > 0` (Close au-dessus de SMA200) est vrai pour presque toute l'année 2025 (marché haussier), donc le TrendFilter n'a **jamais** activé son blocage SHORT → aucun filtre effectif. Le TrendFilter binaire (SMA200) est trop fruste pour un marché en tendance unidirectionnelle. `VolFilter` a pu rejeter quelques trades mais pas assez. Le `SessionFilter` exclut 3h/jour soit ~12.5% des barres, insuffisant.
  - **Tests unitaires** : 192 passed ✅ (post-modifications)
  - **Target Metrics pour v4** : Sharpe OOS ≥ −1.0, Win Rate ≥ 38%, Trades/an ∈ [100, 400], Profit net OOS ≥ 0. Piste : remplacer TrendFilter binaire par un filtre de momentum normalisé, ou méta-labeling.

## Date/Version: 4 — 2026-05-11
- **Modification** : Ajout de `MomentumFilter` dans [`filters.py`](learning_machine_learning/backtest/filters.py) — filtre directionnel basé sur `RSI_D1_delta` (variation du RSI D1 sur 3 jours). Remplace `TrendFilter` dans le `FilterPipeline` construit par [`base.py`](learning_machine_learning/pipelines/base.py). Ajout de `MOMENTUM_FILTER_COL` = `"RSI_D1_delta"` dans `FILTER_COLS` de [`base.py`](learning_machine_learning/pipelines/base.py) pour que la colonne soit jointe depuis `ml_data` dans `run_backtest()`. Mise à jour de `BacktestConfig` dans [`backtest.py`](learning_machine_learning/config/backtest.py) : nouveau flag `use_momentum_filter` + paramètre `momentum_filter_threshold`.
- **Hypothèse** : Le `TrendFilter` binaire (`Dist_SMA200_D1 > 0`) est structurellement inopérant en marché tendanciel unidirectionnel — en 2025 (Close > SMA200 sur ~95% des barres), zéro SHORT bloqué. Le `MomentumFilter` est **symétrique par conception** : bloque LONG si `RSI_D1_delta < −3` (momentum baissier) et bloque SHORT si `RSI_D1_delta > +3` (momentum haussier). Fonctionne dans tous les régimes. `RSI_D1_delta` est déjà une feature d'entraînement → aucun nouveau calcul, coût zéro. Impact attendu : réduction trades de −30% à −50%, rééquilibrage LONG/SHORT, amélioration WR via alignement momentum.
- **Résultats (pré-fix — v3)** :
 - 2024 OOS : 512 trades, WR 30.7%, Net −1098.5 pips, Sharpe −3.38, Alpha −407.3 pips
 - 2025 OOS : 848 trades, WR 31.7%, Net −1783.2 pips, Sharpe −4.35, Alpha −3173.7 pips
- **Résultats (post-fix — 2026-05-12 mesurés)** :
  - 2024 OOS : 463 trades (−9.6% vs 512), WR 30.7% (=), Net **−1009.9 pips** (+88.6), Sharpe **−3.36** (+0.02), Alpha −318.7 pips (+88.6)
  - 2025 OOS : 552 trades (−34.9% vs 848 ✅), WR 31.9% (+0.2pp), Net **−1147.7 pips** (+635.5), Sharpe **−3.18** (+1.17), Alpha −2538.2 pips (+635.5)
  - **Analyse** : ✅ Le MomentumFilter fonctionne — réduction massive des trades 2025 (848→552, −35%). ✅ Toutes les métriques s'améliorent (Sharpe −4.35→−3.18, DD −18.2%→−12.1%, profit net +635 pips). ❌ Sharpe reste profondément négatif (−3.18), WR bloqué à ~31% (< seuil rentabilité 38.3%), profit net toujours à −1148 pips. Le filtre réduit le bruit mais le problème racine persiste : le RandomForest a une accuracy OOS ≈ aléatoire (0.332). **Le filtre seul ne suffira pas** — il faut s'attaquer à la qualité du signal (features, labelling, ou méta-labeling).
  - **Tests unitaires** : 204 passed ✅
- **Target Metrics pour v5** : Sharpe OOS ≥ −1.0, Win Rate ≥ 38%, Trades/an ∈ [100, 400], Profit net OOS ≥ 0. Pistes : (a) Remplacer TP/SL 20/10 par 20/20 (ratio 1:1) pour réduire l'exigence de WR à 51.5% — plus atteignable si le modèle capte un edge directionnel; (b) Méta-labeling : entraîner un second classifieur binaire (entrée/pas entrée) sur les features déjà filtrées; (c) Réduire le seuil de confiance à 0.40 pour filtrer les prédictions peu convaincues.

## Plan d'action v4 — Spécification technique

### Fichiers à modifier (4 fichiers, ~60 lignes ajoutées)

#### 1. [`learning_machine_learning/backtest/filters.py`](learning_machine_learning/backtest/filters.py:135)
Insérer la classe `MomentumFilter` avant `FilterPipeline` :

```python
class MomentumFilter:
   """Filtre directionnel basé sur le momentum macro RSI_D1_delta.

   Remplace le TrendFilter binaire (SMA200) qui est inopérant en marchés
   tendanciels unidirectionnels (ex: 2025 où Close > SMA200 sur ~95% des
   barres → zéro SHORT bloqué). Utilise la variation du RSI D1 sur 3 jours
   pour détecter les retournements de momentum, symétrique par conception.

   LONG autorisé uniquement si RSI_D1_delta >= -threshold (pas de momentum baissier).
   SHORT autorisé uniquement si RSI_D1_delta <= +threshold (pas de momentum haussier).
   """

   name = "momentum"

   def __init__(self, threshold: float = 3.0) -> None:
       self.threshold = threshold

   def apply(
       self,
       df: pd.DataFrame,
       mask_long: pd.Series,
       mask_short: pd.Series,
   ) -> tuple[pd.Series, pd.Series, int]:
       if "RSI_D1_delta" not in df.columns:
           raise ValueError(
               "MomentumFilter nécessite la colonne 'RSI_D1_delta'. "
               "Relancer le feature engineering."
           )

       momentum_long_ok = df["RSI_D1_delta"] >= -self.threshold
       momentum_short_ok = df["RSI_D1_delta"] <= self.threshold

       rejected_long = mask_long & ~momentum_long_ok
       rejected_short = mask_short & ~momentum_short_ok
       n_rejected = int((rejected_long | rejected_short).sum())

       mask_long = mask_long & momentum_long_ok
       mask_short = mask_short & momentum_short_ok

       logger.debug("MomentumFilter: %d signaux rejetés", n_rejected)
       return mask_long, mask_short, n_rejected
```

#### 2. [`learning_machine_learning/config/backtest.py`](learning_machine_learning/config/backtest.py:26)
Ajouter après `use_trend_filter` :

```python
   # Filtre de momentum (remplace progressivement trend_filter)
   use_momentum_filter: bool = True
   momentum_filter_threshold: float = 3.0
```

Et dans `__post_init__`, ajouter la validation :
```python
       if self.momentum_filter_threshold <= 0:
           raise ValueError(
               f"momentum_filter_threshold doit être > 0, reçu {self.momentum_filter_threshold}"
           )
```

#### 3. [`learning_machine_learning/pipelines/base.py`](learning_machine_learning/pipelines/base.py:81)
Remplacer l'import de `TrendFilter` par `MomentumFilter` et changer la construction du `FilterPipeline` :

```python
       from learning_machine_learning.backtest.filters import (
           FilterPipeline,
           MomentumFilter,
           VolFilter,
           SessionFilter,
       )
```

Remplacer le bloc de construction des filtres (lignes 93-94) :
```python
       if cfg.use_momentum_filter:
           filters.append(
               MomentumFilter(threshold=cfg.momentum_filter_threshold)
           )
```

Ajouter `MOMENTUM_FILTER_COL` dans `FILTER_COLS` (ligne 115) :
```python
       FILTER_COLS: tuple[str, ...] = ("Dist_SMA200_D1", "ATR_Norm", "RSI_D1_delta")
```

#### 4. [`learning_machine_learning/backtest/simulator.py`](learning_machine_learning/backtest/simulator.py:82)
Ajouter `"momentum": 0` dans le dict `n_filtres_appliques` :
```python
   n_filtres_appliques: dict[str, int] = {"trend": 0, "vol": 0, "session": 0, "momentum": 0}
```

### Tests unitaires à ajouter
- `test_momentum_filter_blocks_long_when_rsi_d1_delta_below_neg_threshold`
- `test_momentum_filter_blocks_short_when_rsi_d1_delta_above_pos_threshold`
- `test_momentum_filter_allows_all_when_rsi_d1_delta_neutral`
- `test_momentum_filter_raises_when_column_missing`
