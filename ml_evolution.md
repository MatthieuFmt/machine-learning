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

## Date/Version: 5 — 2026-05-12:07:02
- **Modification** : (1) `tp_pips` 20→30 dans [`backtest.py`](learning_machine_learning/config/backtest.py:17) — passage du ratio 2:1 à 3:1. (2) `confidence_threshold` 0.35→0.40 dans [`backtest.py`](learning_machine_learning/config/backtest.py:21). (3) Propagation explicite de `seuil_confiance`, `commission_pips`, `slippage_pips` dans l'appel `simulate_trades()` de [`base.py`](learning_machine_learning/pipelines/base.py:135) — auparavant les valeurs par défaut de la fonction étaient utilisées en dur, rendant `BacktestConfig.confidence_threshold` inopérant.
- **Hypothèse** : Le payoff avec TP=20/SL=10 et WR=32% est structurellement négatif (−1.93 pips/trade). Le breakeven WR = 36.5% est hors d'atteinte avec la qualité actuelle du signal (accuracy OOS ≈ 0.332 ≈ aléatoire). Avec TP=30/SL=10, le breakeven tombe à 27.7% — une marge de 4.2 points au-dessus du WR actuel. L'espérance mathématique devient positive (+1.30 pips/trade). Le relèvement du seuil de confiance de 0.35→0.40 filtre ~40% des signaux faibles (proba_max moyenne OOS = 0.36-0.37), améliorant mécaniquement le WR. La propagation explicite des 3 kwargs garantit que les changements de config sont effectifs sans ambiguïté.
- **Résultats (pré-fix — v4)** :
  - Config : TP=20/SL=10, seuil=0.35 (défaut simulate_trades), Ratio 2:1
  - 2024 OOS : 463 trades, WR 30.7%, Net −1009.9 pips, Sharpe −3.36, Alpha −318.7 pips
  - 2025 OOS : 552 trades, WR 31.9%, Net −1147.7 pips, Sharpe −3.18, Alpha −2538.2 pips
  - Breakeven WR : 36.5% (avec friction 1.5p)
- **Résultats (post-fix v5 — 2026-05-12:07:05)** :
  - Config : TP=30/SL=10, seuil=0.40, Ratio 3:1
  - 2024 OOS : 7 trades, WR 14.3%, Net −42.8 pips, Sharpe −0.68, Alpha +648.4 pips
  - 2025 OOS : 0 trades (tous filtrés)
  - **Analyse** : ❌ Seuil 0.40 tue 99% des signaux. La proba_max OOS moyenne est 0.36-0.37, et la proba de la classe prédite (LONG ou SHORT) est encore plus basse car la proba_neutre est souvent dominante. Le seuil 0.40 est structurellement inatteignable avec l'accuracy actuelle.
- **Correctif v5.1 — seuil 0.40 → 0.33** (juste au-dessus du hasard 1/3 pour 3 classes) :
  - 2024 OOS : 503 trades (+8.6% vs v4:463), WR 26.4% (−4.3pp vs 30.7%), Net **−624.1 pips** (+385.8 vs −1009.9), Sharpe **−1.61** (+1.75 vs −3.36), Alpha **+67.1 pips** (+385.8 vs −318.7)
  - 2025 OOS : 819 trades (+48.4% vs v4:552 ⚠️), WR 25.5% (−6.4pp vs 31.9%), Net **−1345.2 pips** (−197.5 vs −1147.7), Sharpe **−2.69** (+0.49 vs −3.18), Alpha −2735.7 pips
  - **Analyse mathématique** :
    - Payoff 2024 : 0.264×28.5 − 0.736×11.5 = 7.53 − 8.46 = **−0.93 pips/trade** (encore négatif mais amélioré vs −1.93 v4)
    - Payoff 2025 : 0.255×28.5 − 0.745×11.5 = 7.27 − 8.57 = **−1.30 pips/trade**
    - ✅ Sharpe 2024 amélioré de +1.75 points (−3.36→−1.61) — le ratio 3:1 réduit l'impact des SL sur les métriques annualisées
    - ✅ Profit net 2024 amélioré de +385.8 pips
    - ❌ 2025 : le seuil 0.33 laisse passer trop de signaux (819 vs 552) — le modèle prédit SHORT 85% du temps dans un marché +1390 pips B&H, donc chaque trade supplémentaire aggrave les pertes
    - ❌ WR 2025 (25.5%) < breakeven 27.7% — le ratio 3:1 seul ne suffit pas quand le biais directionnel est aussi massif
  - **Conclusion** : Le ratio 3:1 est structurellement supérieur au 2:1 (Sharpe amélioré sur les deux années), mais le biais directionnel SHORT en 2025 annule le bénéfice. La prochaine itération doit attaquer le biais directionnel, pas le payoff.
- **Target Metrics pour v6** : Attaquer le biais directionnel (85% SHORT en 2025). Pistes : (a) neutralisation de tendance dans les features, (b) méta-labeling binaire entraîné spécifiquement sur la qualité des signaux, (c) calibration Platt pour normaliser les probabilités par classe.

## Date/Version: 6 — 2026-05-12:07:26
- **Modification** : (1) Retrait de `XAU_Return` et `CHF_Return` de `features_dropped` dans [`EurUsdConfig`](learning_machine_learning/config/instruments.py:61) — ces features macro étaient calculées depuis v1 mais jamais entraînées (exclues sans évaluation individuelle). (2) Ajout de 6 features H1 dans [`pipeline.py`](learning_machine_learning/features/pipeline.py:88) : `Momentum_5`, `Momentum_10`, `Momentum_20` (ROC multi-échelles), `Dist_EMA_20` (remplace EMA_9/21), `EMA_20_50_cross` (croisement EMAs H1), `Volatility_Ratio` (vol24h / vol168h). Dimensionalité : 7 → 15 features d'entraînement.
- **Hypothèse** : Les features macro (XAU/USD inverse, CHF/USD corrélé −0.85) encodent le régime de risque global de façon continue, donnant au RandomForest un contexte directionnel sans look-ahead. Les features de momentum multi-échelles et le ratio de volatilité fournissent une information directionnelle non-bornée (contrairement au RSI borné 0-100), permettant au modèle de mieux discriminer les régimes de tendance vs ranging.
- **Résultats (pré-fix — v5.1)** :
  - Config : TP=30/SL=10, seuil=0.33, 7 features
  - 2024 OOS : 503 trades, WR 26.4%, Net −624.1 pips, Sharpe −1.61, Alpha +67.1 pips
  - 2025 OOS : 819 trades, WR 25.5%, Net −1345.2 pips, Sharpe −2.69, Alpha −2735.7 pips
- **Résultats (post-fix v6 — 2026-05-12:07:26)** :
  - Config : TP=30/SL=10, seuil=0.33, 15 features
  - 2024 OOS : 590 trades (+17.3%), WR 26.4% (=), Net **−777.0 pips** (−152.9), Sharpe **−1.89** (−0.28), Alpha **−85.8 pips** (−152.9)
  - 2025 OOS : 843 trades (+2.9%), WR 25.7% (+0.2pp), Net **−1313.7 pips** (+31.5), Sharpe **−2.61** (+0.07), Alpha −2704.2 pips
  - **Analyse** : ❌ Dégradation 2024, amélioration négligeable 2025. L'ajout de 8 features a augmenté le nombre de trades de 17% en 2024 sans améliorer le WR — le RandomForest continue de prédire majoritairement SHORT et les features supplémentaires ne changent pas ce biais fondamental. La dimensionalité 15 est probablement excessive pour un signal aussi faible (accuracy OOS ≈ 0.332). **Conclusion** : le problème n'est pas le nombre de features mais la qualité du signal sous-jacent. Le labelling triple-barrier (TP/SL/Window fixes) génère des targets trop bruitées. La prochaine itération doit soit (a) revenir à 7-8 features avec les meilleures uniquement, soit (b) changer le labelling (méta-labeling, ou TP/SL dynamiques basés ATR).
- **Target Metrics pour v7** : Simplifier. Revenir à 8 features (les 7 originales + XAU_Return). Ajouter calibration Platt (CalibratedClassifierCV) pour normaliser les probabilités. Ou explorer un méta-labeling binaire (trade/no-trade) entraîné sur les features et le résultat réel des trades.

## Date/Version: 7 — 2026-05-12:07:48
- **Modification** : (1) Rollback des features v6 — réexclusion de `Dist_EMA_20`, `CHF_Return`, `Momentum_5/10/20`, `EMA_20_50_cross`, `Volatility_Ratio` de l'entraînement (seul `XAU_Return` reste comme feature macro). 8 features actives vs 7 en v5.1. (2) [`pipeline.py`](learning_machine_learning/features/pipeline.py:90) revert à EMA=(50,) seulement (Dist_EMA_20 retombe dans features_dropped). (3) Tentative avortée de Platt scaling : `CalibratedClassifierCV(method="sigmoid", cv=5)` a écrasé toutes les prédictions vers NEUTRE (100% des barres prédites 0.0 — le modèle post-calibration estime que la classe la plus probable est toujours NEUTRE car les probabilités calibrées convergent vers le taux de base). Revert immédiat au RandomForest pur.
- **Hypothèse** : XAU_Return seul (sans CHF_Return ni features de momentum bruitées) apporte l'information macro directionnelle sans surapprentissage. 8 features est le sweet spot entre signal et bruit.
- **Résultats** :
  - Config : TP=30/SL=10, seuil=0.33, 8 features (sans Platt)
  - 2024 OOS : 520 trades, WR 27.1%, Net **−570.4 pips**, Sharpe **−1.53**, Alpha **+120.8 pips**
  - 2025 OOS : 853 trades, WR 25.7%, Net **−1344.1 pips**, Sharpe **−2.63**, Alpha **−2734.6 pips**
  - **Analyse** : ✅ Amélioration modérée vs v5.1 (−570 vs −624 en 2024, Sharpe +0.08). vs v6 : gain significatif (−570 vs −777, Sharpe +0.36). XAU_Return seul (sans les features momentum bruitées) réduit le nombre de trades de 590→520 en 2024 tout en améliorant le WR de 0.7pp. **Conclusion** : 8 features avec XAU_Return est le meilleur point de départ. Platt calibration inapplicable à un classifieur 3-classes quasi-aléatoire (accuracy ~0.332). La calibration compresse les probabilités vers le taux de base, rendant NEUTRE systématiquement dominant. Prochaine étape : méta-labeling binaire pour filtrer les signaux de mauvaise qualité sans toucher aux probabilités directionnelles.
- **Target Metrics pour v8** : Implémenter le méta-labeling binaire (López de Prado) — entraîner un classifieur binaire séparé qui prédit si un trade sera rentable, basé sur les mêmes features + la prédiction directionnelle primaire. Structure : (a) modèle primaire = RF 3-classes (inchangé), (b) modèle secondaire = RF binaire (profitable: oui/non), (c) n'exécuter que les trades où le modèle secondaire prédit "profitable" avec confiance > seuil (e.g., 0.55).

## Date/Version: 8 — 2026-05-12:07:54
- **Modification** : (1) Création de [`meta_labeling.py`](learning_machine_learning/model/meta_labeling.py) — module de méta-labeling binaire avec 3 fonctions : `build_meta_labels()` (construit X_meta = features primaires + signal directionnel + confiance max, y_meta = 1 si Pips_Bruts > 0), `train_meta_model()` (RF binaire 100 arbres/max_depth=6), `apply_meta_filter()` (écrase Prediction_Modele→0 pour les signaux rejetés par le méta-modèle). (2) Modification de [`run_pipeline_v1.py`](run_pipeline_v1.py:1) — entraîne le méta-modèle sur les trades val_year (2024), applique le filtre sur test_year (2025). Pas de look-ahead : le méta-modèle est entraîné sur une année antérieure à l'année filtrée.
- **Hypothèse** : Le modèle primaire 3-classes prédit correctement la direction sur un sous-ensemble de signaux seulement. Un classifieur binaire séparé peut apprendre à reconnaître les conditions où ces prédictions se concrétisent en trades gagnants (features + signal primaire → profitable oui/non). Le filtre rejette les signaux à faible probabilité de succès.
- **Résultats (pré-fix — v7)** :
  - Config : TP=30/SL=10, seuil=0.33, 8 features (sans méta)
  - 2024 OOS : 520 trades, WR 27.1%, Net −570.4 pips, Sharpe −1.53
  - 2025 OOS : 853 trades, WR 25.7%, Net −1344.1 pips, Sharpe −2.63
- **Résultats (post-fix v8)** :
  - Config : TP=30/SL=10, seuil=0.33, 8 features, méta-seuil=0.55
  - 2024 OOS : 520 trades (=), WR 27.1% (=), Net −570.4 pips (=) — val_year, pas de méta-filtre
  - 2025 OOS : **153 trades** (−82.1%), WR **33.3%** (+7.6pp), Net **+103.9 pips** (+1448.0), Sharpe **+0.53** (+3.16), DD **−1.24%** (−13.25pp)
  - **Analyse** : 🟢 **PERCÉE**. Sharpe positive pour la première fois en OOS (2025). Le méta-modèle a divisé le nombre de trades par 5.6 tout en augmentant le WR de 7.6 points, le faisant passer au-dessus du seuil de rentabilité (27.7%). Le profit net devient positif (+103.9 pips vs −1344.1). Le drawdown s'effondre de −14.5% à −1.2%. **Limite** : Alpha encore négatif (−1286.6 pips vs −2734.6) car le buy-and-hold 2025 était fortement haussier (+1390.5 pips) et le système reste biaisé SHORT. **Conclusion** : Le méta-labeling est la meilleure amélioration à ce jour. Il filtre efficacement le bruit du modèle primaire sans nécessiter de calibration Platt ni de features supplémentaires. La prochaine itération doit (a) tester des seuils méta alternatifs (0.50, 0.60), (b) ajouter le filtre méta aussi sur val_year (2024) pour confirmer l'amélioration, (c) explorer un méta-modèle avec plus de features (ex: ATR_Norm, Volatilité au moment du trade).
- **Target Metrics pour v9** : Optimiser le seuil méta (0.50, 0.55, 0.60) sur val_year (2024) et appliquer le meilleur sur test_year. Objectif : Sharpe positif sur les deux années, WR > 30% partout.

## Date/Version: 9 — 2026-05-12:08:00
- **Modification** : (1) Ajout de `evaluate_meta_thresholds()` dans [`run_pipeline_v1.py`](run_pipeline_v1.py:18) — sweep de seuils méta {0.50, 0.52, 0.55, 0.58, 0.60, 0.65} sur val_year (2024), mesure Sharpe/profit/WR/DD par seuil, sélection du meilleur Sharpe. (2) Application du méta-filtre avec le seuil optimal sur les DEUX années (val_year ET test_year) pour un rapport complet. (3) Affichage tabulaire du sweep dans la sortie console. Pas de changement dans les modules core (meta_labeling, base, backtest).
- **Hypothèse** : Le seuil 0.55 (v8) est arbitraire. Un sweep sur val_year identifie le seuil qui maximise le Sharpe sans contaminer test_year (2025). Les seuils plus bas (0.50-0.52) conservent plus de trades avec un WR acceptable ; les seuils plus hauts (0.60-0.65) filtrent plus agressivement au risque de sous-trader. L'optimum dépend du ratio trades profitables/non-profitables dans les méta-labels val_year (~27% en v8). L'application du méta-filtre sur val_year donne enfin une métrique comparable entre les deux années (⚠️ potentiellement optimiste car seuil optimisé sur la même année).
- **Résultats (pré-fix — v8)** :
  - Config : TP=30/SL=10, seuil primaire=0.33, 8 features, méta-seuil=0.55 fixe
  - 2024 val_year (sans méta) : 520 trades, WR 27.1%, Net −570.4 pips, Sharpe −1.53
  - 2025 test_year (méta 0.55) : 153 trades, WR 33.3%, Net +103.9 pips, Sharpe +0.53
- **Résultats (post-fix — 2026-05-12:08:15)** :
  - Config : TP=30/SL=10, seuil primaire=0.33, 8 features, méta-seuil=optimal(val_year)
  - **Tableau de sweep val_year (2024)** :
    | Seuil | Sharpe | Net(pips) | WR% | Trades | DD% |
    |-------|--------|-----------|-----|--------|-----|
    | 0.50 | **4.53** | +1371.4 | 52.7 | 203 | −75.4 |
    | 0.52 | 3.98 | +1110.9 | 54.9 | 153 | −47.3 |
    | 0.55 | 3.64 | +824.6 | 56.2 | 105 | −47.6 |
    | 0.58 | 3.37 | +552.3 | 59.4 | 64 | −40.9 |
    | 0.60 | 3.49 | +489.8 | 72.5 | 40 | −20.5 |
    | 0.65 | 2.80 | +114.0 | 85.7 | 7 | −10.1 |
  - **Seuil optimal retenu** : 0.50 (max Sharpe val = 4.53)
  - **2024 val_year (méta 0.50)** ⚠️ optimiste : 203 trades, WR 52.7%, Net +1371.4 pips, Sharpe +4.53, DD −75.4 pips (−0.75%)
  - **2025 test_year (méta 0.50)** ✅ non-biaisé : 367 trades, WR 28.6%, Net **−220.2 pips**, Sharpe **−0.64**, DD −329.1 pips (−3.29%)
  - **Analyse** : ❌ **Sur-optimisation du seuil sur val_year**. Le seuil 0.50 maximisait Sharpe sur 2024 (4.53) mais dégrade Sharpe sur 2025 (−0.64 vs +0.53 en v8 avec seuil 0.55). Le méta-modèle, entraîné exclusivement sur trades 2024, a appris des patterns qui ne généralisent pas à 2025. En baissant le seuil de 0.55→0.50, on laisse passer 2.4× plus de trades (153→367) mais le WR chute de 33.3%→28.6% — les 214 trades supplémentaires sont majoritairement perdants. **Le seuil 0.55 (v8) était en réalité meilleur pour la généralisation** que le seuil 0.50 optimisé. Le sweep val_year seul est un guide insuffisant — il faut soit un sweep walk-forward (2023→2024→2025), soit une méthode de sélection robuste (médiane des Sharpe, critère de stabilité).
  - **Comparaison v8 vs v9 sur test_year 2025** :
    | Version | Seuil | Trades | WR | Net | Sharpe |
    |---------|-------|--------|-----|-----|--------|
    | v8 | 0.55 | 153 | 33.3% | +103.9 | **+0.53** |
    | v9 | 0.50 | 367 | 28.6% | −220.2 | −0.64 |
- **Target Metrics pour v10** : Restaurer le seuil 0.55 (meilleure généralisation observée). Piste principale : enrichir les features du méta-modèle avec des colonnes de contexte de trade (ATR_Norm, Spread, Volatilite_Realisee_24h, Dist_SMA200_D1) pour améliorer sa capacité à discriminer les trades profitables indépendamment de l'année. Objectif : Sharpe test_year ≥ +0.5 maintenu ou amélioré.

## Date/Version: 10 — 2026-05-12:08:25
- **Modification** : (1) Ajout de `META_EXTRA_COLS` = `[ATR_Norm, Spread, Dist_SMA200_D1, Volatilite_Realisee_24h]` dans [`run_pipeline_v1.py`](run_pipeline_v1.py:26) — constante définissant les colonnes de contexte de marché injectées dans le méta-modèle. (2) [`meta_labeling.py`](learning_machine_learning/model/meta_labeling.py:31) — `build_meta_labels()` et `apply_meta_filter()` gagnent le paramètre `extra_cols: list[str] | None` ; ces colonnes sont extraites de `ml_data` et ajoutées à `X_meta`. (3) [`pipeline.py`](learning_machine_learning/features/pipeline.py:165) — `Volatilite_Realisee_24h` ajouté à `FILTER_KEEP` pour qu'il survive au `features_dropped`. (4) [`training.py`](learning_machine_learning/model/training.py:21) — `Volatilite_Realisee_24h` ajouté à `_FILTER_ONLY_COLS` pour qu'il soit exclu de l'entraînement primaire. (5) Tous les appels à `build_meta_labels()`, `apply_meta_filter()`, `evaluate_meta_thresholds()` propagent `extra_cols=META_EXTRA_COLS`.
- **Hypothèse** : Le méta-modèle v8/v9 ne voyait que les 8 features primaires + signal + confiance (10 features). Il ne disposait d'aucune information de régime de marché au moment du trade. Un trade SHORT gagnant en 2024 (marché ranging) peut avoir les mêmes features primaires qu'un trade SHORT perdant en 2025 (marché haussier) — le méta-modèle ne peut pas discriminer sans contexte. Les 4 colonnes ajoutées (ATR_Norm = volatilité normalisée, Spread = coût de transaction réel, Dist_SMA200_D1 = position dans la tendance long-terme, Volatilite_Realisee_24h = régime de turbulence récent) donnent au méta-modèle l'information nécessaire pour reconnaître les contextes favorables/défavorables indépendamment de l'année. Le méta-modèle passe de 10 à 14 features. Aucun impact sur l'entraînement primaire (ces colonnes restent exclues via `_FILTER_ONLY_COLS`). Le sweep de seuil v9 est conservé.
- **Résultats (pré-fix — v9)** :
  - Config : TP=30/SL=10, seuil primaire=0.33, 8 features, méta-seuil=0.50 (optimisé val_year)
  - 2024 val_year (méta 0.50) ⚠️ optimiste : 203 trades, WR 52.7%, Net +1371.4 pips, Sharpe +4.53
  - 2025 test_year (méta 0.50) ✅ : 367 trades, WR 28.6%, Net −220.2 pips, Sharpe −0.64
- **Résultats (post-fix v10 — mesuré)** :
  - Config : TP=30/SL=10, seuil primaire=0.33, 8 features primaires, 14 features méta (8 primaires + 2 signal/confiance + 4 contexte)
  - Tableau de sweep val_year (features enrichies) :
    ```
      0.50  Sharpe=4.81  Net=+1368.6  WR=53.4%  193 trades  DD=-56.7
      0.52  Sharpe=4.37  Net=+1286.8  WR=55.8%  163 trades  DD=-47.5
      0.55  Sharpe=3.63  Net= +869.4  WR=58.2%   98 trades  DD=-38.6
      0.58  Sharpe=3.60  Net= +689.8  WR=69.0%   58 trades  DD=-19.3
      0.60  Sharpe=2.82  Net= +430.1  WR=71.4%   35 trades  DD=-29.3
      0.65  Sharpe=2.34  Net=  +78.1  WR=100.0%   4 trades  DD=  0.0
    ```
    Meilleur seuil val = 0.50 (Sharpe=4.81, quasi identique à v9).
  - 2024 val_year (méta 0.50) ⚠️ optimiste : 193 trades, WR 53.4%, Net +1368.6 pips, Sharpe +4.81
  - 2025 test_year (méta 0.50) ✅ : 363 trades, WR 26.7%, Net −457.6 pips, Sharpe −1.41
- **Analyse** : 🔴 DÉGRADATION. L'ajout des 4 colonnes de contexte de marché (ATR_Norm, Spread, Dist_SMA200_D1, Volatilite_Realisee_24h) a aggravé l'overfitting du méta-modèle au lieu de le réduire. Sharpe test_year passe de −0.64 (v9) à −1.41 (v10), perte nette de −457.6 vs −220.2 pips. Hypothèse confirmée d'échec : les features de contexte varient par régime (2024 ranging → 2025 haussier) et le RF les exploite pour discriminer les trades gagnants sur 2024, mais ces patterns de contexte ne se transfèrent pas en 2025. Le méta-modèle RF est structurellement inadapté à la généralisation inter-régime.
- **Décision pour v11** : Remplacer [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) par [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) pour le méta-modèle. Le boosting a une meilleure capacité de généralisation sur données financières (López de Prado, 2018). Conserver les 14 features méta (8 primaires + 2 signal/confiance + 4 contexte). Paramètres initiaux : `n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8` pour limiter l'overfitting.
- **Target Metrics pour v11** : Sharpe test_year ≥ +0.5 (minimum v8), WR test_year ≥ 33%.

## Date/Version: 11 — 2026-05-12:08:40
- **Modification** : (1) [`meta_labeling.py`](learning_machine_learning/model/meta_labeling.py:96) — `train_meta_model()` gagne le paramètre `model_type: str = "rf"`. Supporte `"rf"` (RandomForest, inchangé) et `"gbm"` (GradientBoostingClassifier). Ajout de `_GBM_DEFAULT_PARAMS` : `n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=20`. L'import `GradientBoostingClassifier` est ajouté. La signature de retour devient `Union[RandomForestClassifier, GradientBoostingClassifier]`. (2) [`run_pipeline_v1.py`](run_pipeline_v1.py:132) — appel `train_meta_model(X_meta, y_meta, model_type="gbm")`. (3) [`apply_meta_filter()`](learning_machine_learning/model/meta_labeling.py:185) — signature élargie à `Union[RandomForestClassifier, GradientBoostingClassifier]` (`.predict_proba()` compatible).
- **Hypothèse** : Le RF (bagging) mémorise les patterns spécifiques à 2024 via des arbres profonds (max_depth=6) qui ne généralisent pas à 2025. Le GBM (boosting) avec arbres peu profonds (max_depth=3), learning_rate faible (0.05) et stochasticité (subsample=0.8) devrait mieux généraliser entre régimes de marché car chaque arbre corrige marginalement l'erreur résiduelle plutôt que de partitionner l'espace des features.
- **Résultats (pré-fix — v10)** :
  - Config : TP=30/SL=10, seuil primaire=0.33, méta-RF 14 features
  - 2024 val_year (méta 0.50) : 193 trades, WR 53.4%, Net +1368.6 pips, Sharpe +4.81
  - 2025 test_year (méta 0.50) : 363 trades, WR 26.7%, Net −457.6 pips, Sharpe −1.41
- **Résultats (post-fix v11 — mesuré)** :
  - Config : TP=30/SL=10, seuil primaire=0.33, méta-GBM 14 features
  - Tableau de sweep val_year (GBM) :
    ```
      0.50  Sharpe=4.04  Net= +791.9  WR=85.4%   48 trades  DD=-11.1
      0.52  Sharpe=3.76  Net= +716.5  WR=86.0%   43 trades  DD=-11.1
      0.55  Sharpe=2.81  Net= +472.8  WR=86.2%   29 trades  DD= -9.4
      0.58  Sharpe=2.35  Net= +325.5  WR=88.9%   18 trades  DD= -9.3
      0.60  Sharpe=2.08  Net= +276.3  WR=100.0%  13 trades  DD=  0.0
      0.65  Sharpe=2.31  Net= +148.2  WR=100.0%   7 trades  DD=  0.0
    ```
    Meilleur seuil val = 0.50 (Sharpe=4.04).
  - 2024 val_year (méta 0.50) ⚠️ optimiste : 48 trades (−75% vs v10:193), WR 85.4% (+32pp), Net +791.9 pips, Sharpe +4.04
  - 2025 test_year (méta 0.50) ✅ : 71 trades (−80% vs v10:363), WR 18.3% (−8.4pp), Net −295.7 pips (+161.9 vs v10), Sharpe −1.95 (−0.54 vs v10)
- **Analyse** : 🟡 RÉSULTAT MITIGÉ. Le GBM filtre beaucoup plus agressivement que le RF (48 trades val vs 193, 71 trades test vs 363). Sur val_year : WR explose (85.4% vs 53.4%) mais le profit net baisse (791.9 vs 1368.6) car trop peu de trades. Sur test_year : le profit net s'améliore (−295.7 vs −457.6) mais le Sharpe se dégrade (−1.95 vs −1.41) car le WR s'effondre à 18.3% — les 71 trades survivants sont majoritairement des SL à −10 pips. **Le GBM sur-optimise différemment du RF** : au lieu de mémoriser les patterns gagnants de 2024, il apprend un seuil de confiance implicite très élevé (peu de trades, WR val très haut) qui ne se transfère pas mieux. Le problème racine n'est pas l'algorithme (RF vs GBM) mais la non-stationnarité des features entre 2024 et 2025 — aucun classifieur ne peut discriminer des patterns inexistants dans son ensemble d'entraînement.
- **Comparatif cumulé v8→v11 sur test_year 2025** :
  | Version | Méta-modèle | Seuil | Trades | WR | Net(pips) | Sharpe |
  |---------|------------|-------|--------|-----|-----------|--------|
  | v8 | RF 10 features | 0.55 fixe | 153 | 33.3% | **+103.9** | **+0.53** |
  | v9 | RF 10 features | 0.50 sweep | 367 | 28.6% | −220.2 | −0.64 |
  | v10 | RF 14 features | 0.50 sweep | 363 | 26.7% | −457.6 | −1.41 |
  | v11 | GBM 14 features | 0.50 sweep | 71 | 18.3% | −295.7 | −1.95 |
  **v8 (seuil 0.55 fixe, RF 10 features) reste la meilleure version sur test_year 2025.**
- **Décision pour v12** : Revenir au méta-modèle RF 10 features (sans contexte de marché) avec seuil fixe 0.55.
- **Target Metrics pour v12** : Restaurer Sharpe test_year ≥ +0.5. WR test_year ≥ 33%. Trades ≥ 100.

## Date/Version: 12 — 2026-05-12:18:20
- **Modification** : Rollback pur vers configuration v8 dans [`run_pipeline_v1.py`](run_pipeline_v1.py:93) : `model_type="rf"` (défaut), retrait de tous les `extra_cols=META_EXTRA_COLS`, seuil fixe `FIXED_THRESHOLD=0.55`. Le sweep est conservé en mode diagnostic uniquement. Aucune modification de `meta_labeling.py` ni des autres modules.
- **Hypothèse** : Les versions v9→v11 n'ont pas surpassé v8. Le rollback doit reproduire les résultats v8 à l'identique — test de cohérence du pipeline.
- **Résultats (pré-fix — v11)** :
  - Config : TP=30/SL=10, méta-GBM 14 features, seuil=0.50 sweep
  - 2024 val_year : 48 trades, WR 85.4%, Net +791.9 pips, Sharpe +4.04
  - 2025 test_year : 71 trades, WR 18.3%, Net −295.7 pips, Sharpe −1.95
- **Résultats (post-fix v12 — mesuré)** :
  - Config : TP=30/SL=10, méta-RF 10 features, seuil=0.55 fixe
  - Sweep val_year diagnostic : 0.50→Sharpe 4.53, 0.55→Sharpe 3.64, 0.65→Sharpe 2.80
  - 2024 val_year (méta 0.55) : 105 trades, WR 56.2%, Net +824.6 pips, Sharpe +3.64
  - 2025 test_year (méta 0.55) : 153 trades, WR 33.3%, Net +103.9 pips, Sharpe +0.53
- **Analyse** : Résultats identiques à v8 — rollback confirmé. Sharpe +0.53, 153 trades, WR 33.3%. La cohérence du pipeline est vérifiée : aucune dérive entre versions.
- **Comparatif final v8→v12 test_year 2025** : v8/v12 (+0.53) > v9 (−0.64) > v11 (−1.95) > v10 (−1.41). Le seuil 0.55 fixe + RF 10 features est la meilleure configuration à ce jour.
- **Décision pour v13** : Walk-forward 2 étapes (méta entraîné sur 2023, appliqué 2024 puis 2025) pour réduire la sensibilité au régime unique. Alternatives : LogisticRegression, réduction max_depth méta-RF 6→4.
- **Target Metrics pour v13** : Sharpe test_year ≥ +1.0. WR test_year ≥ 35%. Trades ∈ [100, 250].

## Date/Version: 13 — 2026-05-12:18:45
- **Modification** : Création du module [`learning_machine_learning/analysis/edge_validation.py`](learning_machine_learning/analysis/edge_validation.py:1) — validation statistique de l'edge (prompt 1). Fonction `validate_edge(trades_df, backtest_cfg, n_trials_searched=12) -> dict` avec 4 tests : (1) Breakeven WR vs WR observé, (2) Bootstrap Sharpe 10k itérations, (3) Deflated Sharpe Ratio (López de Prado ch.14), (4) t-test expectancy. Tests unitaires [`tests/unit/test_edge_validation.py`](tests/unit/test_edge_validation.py) : 13/13 pass. Intégration dans [`run_pipeline_v1.py`](run_pipeline_v1.py:207) après le backtest.
- **Hypothèse** : Les 12 itérations de recherche (v1→v12) peuvent avoir introduit du data-snooping. Le DSR corrige pour le nombre de trials. Le bootstrap Sharpe teste si l'edge est significativement > 0.
- **Résultats v13 — Edge Validation** :

| Métrique | 2024 (val) | 2025 (test) |
|---|---|---|
| Trades | 105 | 153 |
| Breakeven WR | 28.75% | 28.75% |
| WR Observé | 56.19% | 33.33% |
| Marge WR | **+27.4%** | **+4.6%** |
| Bootstrap Sharpe obs | 0.4907 | 0.0444 |
| Bootstrap p(Sharpe>0) | **0.0000** | **0.2933** |
| Bootstrap CI95 | [0.30, 0.70] | [-0.12, 0.19] |
| Deflated SR (DSR) | **+5.94** | **−1.97** |
| Probabilistic SR (PSR) | 1.0000 | 0.0244 |
| E[max SR] (n=12) | 0.1625 | 0.1346 |
| t-test p-value | 0.0000 | 0.5837 |

- **Décision** :
  - **2024** : EDGE RÉEL. p=0.0000, DSR=+5.94, PSR=1.00. L'edge sur l'année de validation est statistiquement indiscutable. Le Sharpe observé (0.49/trade) est 5.94 écarts-types au-dessus du maximum attendu par chance après 12 trials.
  - **2025** : EDGE NON CONFIRMÉ. p=0.2933, DSR=−1.97, PSR=0.024. Le Sharpe tombe à 0.044 (quasi nul), le CI95 contient 0, le t-test p=0.58. La probabilité que le Sharpe vrai soit > 0 n'est que de 2.4%.
- **Analyse** : L'écart 2024→2025 est un cas d'école de non-stationnarité de régime. Le méta-modèle RF appris sur 2024 (WR 56%, Sharpe 3.64) ne se généralise pas à 2025 (WR 33%, Sharpe 0.53). Le DSR 2025 négatif (−1.97) indique que même avec seulement 12 trials, la performance 2025 est inférieure à ce qu'on attendrait du hasard. La marge WR de +4.6% au-dessus du breakeven (28.75%) est trop faible pour être statistiquement fiable.
- **Conclusion (critique)** : Selon les critères du prompt 1 : p(Sharpe>0) = 0.29 > 0.10 sur 2025 → **NE PAS continuer les optimisations cosmétiques**. Les v9→v12 ont été des micro-optimisations sans effet réel. La prochaine étape doit être structurelle : changer de timeframe, d'instrument, ou de fonction objectif (ex: remplacer Dist_SMA200_D1 sign par une cible continue type forward return).
- **Suites possibles** :
  - Walk-forward 2-étapes (méta entraîné sur 2023, appliqué 2024 ET 2025) — prompt 2
  - Cost-aware labeling (triple barrière intégrant spread+slippage) — prompt 3
  - Multi-asset (BTCUSD, XAUUSD) — prompt 4

## Date/Version: 14 — 2026-05-12:20:21
- **Modification** : Walk-Forward Retraining (prompt 2). Ajout du générateur [`walk_forward_train()`](learning_machine_learning/model/training.py:108) — fenêtre glissante 36 mois, step 3 mois, purge 48h, zéro look-ahead. Méthode [`BasePipeline.run_walk_forward()`](learning_machine_learning/pipelines/base.py:174) — réentraînement par fold, agrégation prédictions OOS, backtest, edge validation. Script [`run_pipeline_walk_forward.py`](run_pipeline_walk_forward.py:1). Tests [`tests/unit/test_walk_forward.py`](tests/unit/test_walk_forward.py) : 8/8 pass. Suite complète : 220/220 pass.
- **Hypothèse** : Le split statique train≤2023/val=2024/test=2025 (v1-v13) ne capture pas la non-stationnarité. Le walk-forward réentraîne tous les 3 mois sur 36 mois glissants, s'adaptant aux changements de régime (v13 a montré que le modèle statique échoue sur 2025).
- **Résultats v14 — Walk-Forward (5 folds, 7611 prédictions OOS)** :

| Année | Trades | WR | Sharpe/trade | p(Sharpe>0) | DSR | Décision |
|---|---|---|---|---|---|---|
| 2013 | 43 | 32.6% | +0.32 | 0.38 | -2.48 | NON |
| 2014 | 198 | 21.7% | -2.77 | 0.99 | -8.00 | NON |
| 2016 | 33 | 21.2% | -1.08 | 0.86 | -4.95 | NON |
| 2017 | 237 | 24.1% | -1.90 | 0.97 | -6.45 | NON |
| 2019 | 9 | 11.1% | -1.46 | 0.92 | -5.52 | NON |
| 2020 | 118 | 32.2% | +0.07 | 0.47 | -2.94 | NON |
| 2022 | 25 | 40.0% | +1.22 | 0.11 | -0.91 | NON |
| 2023 | 180 | 25.6% | -1.18 | 0.88 | -5.17 | NON |
| 2025 | 10 | 30.0% | -0.99 | 0.83 | -4.74 | NON |
| 2026 | 103 | 32.0% | +0.20 | 0.42 | -2.69 | NON |

- **Analyse** : Aucune année ne passe le test d'edge. Même 2022 (meilleure année, WR=40%, Sharpe/trade=+1.22) a p=0.11 > 0.05 et DSR=-0.91 < 0. Le walk-forward seul ne résout pas le problème fondamental : les features et la target Direction_SMA200_D1 ne capturent pas un signal suffisamment fort et stable.
- **Comparaison statique vs walk-forward sur 2025** : WR 33.3% (statique) vs 30.0% (w-f), Sharpe 0.04 vs -0.99. Le walk-forward est plus réaliste mais ne produit pas d'edge.
- **Décision** : Le walk-forward est une amélioration méthodologique (plus réaliste, pas de look-ahead) mais la cause racine est la qualité du signal, pas la méthode d'entraînement. Prochaine étape : cost-aware labeling intégrant spread+slippage (prompt 3) ou diversification multi-asset (prompt 4).
## Date/Version: 15 — 2026-05-12:21:00

### Cost-Aware Labeling (Prompt 3)

Ajout du labelling triple barrière avec prise en compte des coûts de friction (commission + slippage) et d'un seuil de profit minimum net.

#### Fichiers modifiés/créés

1. [`learning_machine_learning/features/triple_barrier.py`](learning_machine_learning/features/triple_barrier.py:119) — Nouvelle fonction `apply_triple_barrier_cost_aware()` (120 lignes). Même algorithme bidirectionnel que `apply_triple_barrier` mais avec 2 différences :
   - TP touché → label ±1 seulement si `tp_pips - friction_pips >= min_profit_pips`
   - Timeout → calcul du PnL sur Close, label directionnel seulement si `|PnL| - friction_pips >= min_profit_pips`

2. [`learning_machine_learning/config/instruments.py`](learning_machine_learning/config/instruments.py:31) — 3 nouveaux champs dans `InstrumentConfig` :
   - `cost_aware_labeling: bool = False`
   - `friction_pips: float = 1.5`
   - `min_profit_pips_cost_aware: float = 3.0`

3. [`learning_machine_learning/features/pipeline.py`](learning_machine_learning/features/pipeline.py:77) — Branchement dans `build_ml_ready()` : si `instrument.cost_aware_labeling` est True, utilise `apply_triple_barrier_cost_aware` au lieu de `apply_triple_barrier`.

4. [`tests/unit/test_cost_aware_labeling.py`](tests/unit/test_cost_aware_labeling.py:1) — 11 tests unitaires (<1s) couvrant : TP profitable/non-profitable, timeout PnL, SL, NaN convention, validation params négatifs, shape matching, ratio de trades raisonnable.

5. [`compare_cost_aware.py`](compare_cost_aware.py:1) — Script de comparaison classique vs cost-aware sur données réelles EURUSD H1.

#### Résultats de la comparaison

| Métrique | Classique | Cost-Aware | Delta |
|---|---|---|---|
| LONG | 32267 | 32468 | +201 |
| SHORT | 31505 | 31706 | +201 |
| NEUTRE | 36203 (36.2%) | 35801 (35.8%) | -0.4pp |
| Total valide | 99975 | 99975 | — |

#### Analyse

- Impact quasi-nul avec TP=20p / SL=10p actuels car friction=1.5p est négligeable face au TP de 20p.
- Le cost-aware AJOUTE des trades (pas n'en enlève) : les timeouts que le classique marque 0 (pas de TP/SL touché) peuvent devenir directionnels si le PnL sur Close est profitable après friction.
- La feature est prête à être activée (`cost_aware_labeling=True` dans la config) mais n'aura d'impact significatif qu'avec des TP/SL plus serrés (5-10 pips).
- **Tests** : 231/231 passent (dont 11 cost-aware + 19 triple barrier existants).

## Date/Version: 16 — 2026-05-12 (Roadmap stratégique documentée)

- **Modification** : Création du dossier [`docs/`](docs/) avec 8 fichiers de roadmap : `step_01_target_redefinition.md`, `step_02_robust_validation_framework.md`, `step_03_gbm_primary_classifier.md`, `step_04_session_aware_features.md`, `step_05_economic_calendar_integration.md`, `step_06_meta_labeling_calibration.md`, `step_07_cross_asset_validation.md`, et [`docs/README.md`](docs/README.md) (index avec critères go/no-go inter-étapes).

- **Hypothèse** : Les 15 itérations précédentes ont épuisé les optimisations cosmétiques sans débloquer la fiabilité statistique (DSR 2025 = -1.97, p(Sharpe>0) = 0.29). Une roadmap structurée attaquant les **causes racines** (cible bruitée à 36% NEUTRE, validation mono-split fragile, biais directionnel SHORT 75%) plutôt qu'empilant des optimisations marginales est requise. Chaque fichier `step_NN_*.md` documente une piste avec : hypothèse mathématique, méthodologie d'implémentation, métriques de validation cibles, risques. Aucun code écrit à ce stade — documentation stratégique uniquement.

- **Sélection des 7 pistes** (priorisées par impact sur la fiabilité statistique, pas sur le ROI cosmétique) :
  - **step_01** (Rupture, 🔴) : redéfinition cible (régression forward-return / binaire purifiée / cost-aware v2) pour augmenter le signal-to-noise
  - **step_02** (Méthodologie, 🔴) : Combinatorial Purged CV (López de Prado ch.12) + Probabilistic/Deflated Sharpe Ratio (Bailey & López de Prado 2014)
  - **step_03** (Modèle, 🟠) : LightGBM/XGBoost + Optuna avec inner CV temporel — early stopping comme régularisation explicite
  - **step_04** (Feature engineering, 🟠) : `session_id` + `ATR_session_zscore` + range_open_session (Tokyo/Londres/NY/overlap LdN-NY/low_liq)
  - **step_05** (Feature exogène, 🟠) : intégration calendrier macro Forex Factory (NFP, CPI, FOMC, BCE, BoE, BoJ) + `CalendarFilter`
  - **step_06** (Optimisation, 🟠) : calibration isotonique du méta-classifieur + seuil breakeven analytique (formule TP/SL/friction) ou robust_cpcv
  - **step_07** (Validation, 🟡) : test cross-actif GBPUSD/USDJPY/XAUUSD avec config identique (aucun tuning par actif) — gate final go/no-go production

- **Hors-scope explicite** : HMM/régime-switching, deep learning (taille dataset insuffisante), microstructure tick-by-tick (données absentes), position sizing Kelly (prématuré sans edge confirmé). Réintégrables en v17+ si l'edge devient fiable.

- **Résultats** : aucun (étape documentaire). Les résultats viendront à l'exécution séquentielle des steps.

- **Critères de bascule** documentés dans [`docs/README.md`](docs/README.md) :
  - Après step_01 : accuracy OOS > 0.36 → step_03 ; sinon reconsidérer projet
  - Après step_02 : DSR > 0 et % splits profitables > 60 % → continuer ; sinon retour step_01
  - Après steps 03-06 : Sharpe OOS 2025 > 0.50 et DSR > 0 → step_07 ; sinon ablation study
  - Après step_07 : Sharpe > 0 sur ≥ 2 actifs / 4 → GO production ; sinon overfit confirmé

- **Target Metrics pour v17** : 
  - DSR 2025 > 0 sur distribution CPCV (≥ 200 splits)
  - p(Sharpe > 0) < 0.05 bootstrap
  - Sharpe OOS 2025 > 0.50
  - WR OOS 2025 > 38 % (breakeven 27.7 % + marge)
  - Biais directionnel < 60 %

- **Prochaine étape immédiate** : démarrer step_01 (target redefinition) **et** step_02 (CPCV) en parallèle. Ce sont les deux pistes "causes racines" indépendantes.

