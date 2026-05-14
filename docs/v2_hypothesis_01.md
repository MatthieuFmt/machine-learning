# V2 Hypothesis 01 — US30 D1 : Trend-Following avec features minimales

**Date** : 2026-05-13
**Statut** : 🔴 NO-GO — Sharpe 2024-2025 = −1.27
**Priorité** : 🔴 P0 (première hypothèse v2)
**Données** : `data/USA30IDXUSD_D1.csv`, `USA30IDXUSD_H4.csv`

---

## 1. Hypothèse mathématique

### 1.1 Pourquoi US30 (Dow Jones CFD) plutôt qu'EURUSD

| Critère | EURUSD H1 (v1) | US30 D1 (v2) |
|---------|-----------------|---------------|
| Efficience de marché | Extrême (FX, ~$7T/jour) | Modérée (indice CFD, moins d'arbitragistes) |
| Ratio signal/bruit | Très faible — microstructure domine | Élevé — les flux macro dominent |
| Régime de tendance | Range-bound fréquent | Tendance directionnelle plus persistante |
| Friction relative | 1-2 pips spread / 30 pips TP = 3-7% | ~2-3 points spread / 200 points TP = 1-1.5% |
| Nombre d'échantillons (10 ans) | ~60k barres H1 | ~2 600 barres D1 |
| Trades/an attendus | 200-800 (trop, bruit) | 20-60 (OK pour validation statistique) |

### 1.2 Formalisation

Soit $P_t$ le prix de clôture US30 au jour $t$. On définit :

- **Cible** : `directional_clean` — classification binaire (HAUSSE/BAISSE sur horizon 5 jours). Seuil de bruit = 0.5 × ATR(14).
  - Pourquoi 5 jours et pas 24h : D1 → une barre par jour. L'horizon 5j capture le swing macro sans être trop long (ne pas confondre avec buy-and-hold).
  - Pourquoi `directional_clean` plutôt que `triple_barrier` : la cible `triple_barrier` avec TP=3×SL a une probabilité intrinsèque de 25% sous brownien sans drift — biais structurel. `directional_clean` est symétrique (50/50 sous H₀).

- **Features** (max 8 — parcimonie forcée) :
  | # | Feature | Timeframe | Justification |
  |---|---------|-----------|---------------|
  | 1 | `RSI_14` | D1 | Surachat/survente — testé sur indices, parfois informatif |
  | 2 | `ADX_14` | D1 | Force de tendance — utile pour filtrer le range |
  | 3 | `Dist_SMA50` | D1 | Distance à SMA 50j (% — normalisé) |
  | 4 | `Dist_SMA200` | D1 | Tendance long-terme — anomalie de momentum documentée |
  | 5 | `ATR_Norm` | D1 | Régime de volatilité normalisé |
  | 6 | `Log_Return_5d` | D1 | Momentum 5 jours |
  | 7 | `VIX_Return` | D1 | Indicateur de stress marché (si disponible, sinon exclue) |
  | 8 | `Volume_Ratio` | D1 | Volume relatif vs moyenne 20j |

  **Règle anti-overfit v1** : PAS de features H4/H1. Une seule timeframe. 8 colonnes maximum. Aucun hyperparamètre de feature (fenêtre EMA, période RSI) ne sera tuné.

- **Modèle** : RandomForest, hyperparamètres figés ex ante :
  ```python
  RF_PARAMS = {
      "n_estimators": 200,     # réduit vs v1 (500) car dataset D1 plus petit
      "max_depth": 4,          # fortement régularisé
      "min_samples_leaf": 20,  # ~1% de l'échantillon D1
      "class_weight": "balanced_subsample",
      "random_state": 42,
  }
  ```

- **Split temporel** : **NOUVEAU split** (l'ancien split ≤2023/2024/2025 est BRÛLÉ pour EURUSD H1 — mais réutilisable pour US30 D1 car instrument/timeframe différent). Par précaution, on décale quand même :
  - Train : ≤ 2022-12-31
  - Val : 2023-01-01 → 2023-12-31
  - Test : 2024-01-01 → 2025-05-13 (≈17 mois)

- **Backtest** :
  ```python
  US30_CONFIG = BacktestConfig(
      tp_pips=200.0,     # 200 points US30 (≈ 0.5% du prix actuel ~40k)
      sl_pips=100.0,     # 100 points (ratio 2:1 — plus conservateur que 3:1)
      window_hours=120,  # 5 jours
      commission_pips=3.0,   # spread US30 CFD typique
      slippage_pips=5.0,
      confidence_threshold=0.55,  # seuil conservateur
      use_momentum_filter=False,  # pas de filtre momentum (on a déjà Log_Return_5d)
      use_vol_filter=True,
      use_session_filter=False,   # pas pertinent en D1
      use_calendar_filter=False,  # pas de calendrier économique pour les indices CFD
  )
  ```

---

## 2. Protocole anti-snooping — Engagement préalable

### Règle 1 — Split fixé ex ante
Le split train ≤ 2022 / val 2023 / test 2024-2025 **ne sera pas modifié** après le premier run. Si les données ne remontent pas assez loin, on ajuste AVANT le premier run et on documente.

### Règle 2 — Un seul regard
Une fois `run_pipeline_us30.py` exécuté, les résultats OOS 2024-2025 seront lus **une seule fois**. Aucune modification de features, d'hyperparamètres, ou de cible ne sera faite en réaction aux résultats. Si Sharpe < 0 → on passe à l'hypothèse v2-02. Si Sharpe > 0 → on passe au CPCV.

### Règle 3 — DSR avec n_trials = 1
Pour cette première hypothèse v2, n_trials = 1. Si d'autres hypothèses sont testées ensuite sur le même instrument/timeframe, le DSR sera recalculé avec n_trials cumulatif.

### Règle 4 — Instrument hold-out
USDCHF D1 est réservé comme gate final. Il ne sera testé qu'à la fin de la roadmap v2, avec la config figée.

---

## 3. Métriques de succès / échec

### GO si :
- **Sharpe test (2024-2025) > 0** ET **WR > breakeven WR** (≈ 37% avec TP=200/SL=100/friction=8)
- → Passage en CPCV pour validation robuste

### NO-GO si :
- Sharpe test ≤ 0 **OU** WR ≤ breakeven WR
- → Passage à l'hypothèse v2-02 (XAUUSD H4, ou US30 H4)

---

## 4. Fichiers à créer/modifier

| # | Fichier | Action | Description |
|---|---------|--------|-------------|
| 1 | `learning_machine_learning_v2/config/instruments.py` | **Modifier** | Ajouter `Us30Config` (D1 primary, H4 fallback) |
| 2 | `learning_machine_learning_v2/features/pipeline.py` | **Créer** | Pipeline feature simplifié — une seule TF, pas de macro, pas de calendrier |
| 3 | `learning_machine_learning_v2/targets/labels.py` | **Créer** | Fonctions cibles (`directional_clean`, `triple_barrier`) |
| 4 | `learning_machine_learning_v2/models/training.py` | **Créer** | Split train/val/test sur index temporel, entraînement RF |
| 5 | `learning_machine_learning_v2/pipelines/us30.py` | **Créer** | Pipeline US30 D1 (~60 lignes) |
| 6 | `run_pipeline_us30.py` | **Créer** | Script de lancement |
| 7 | `scripts/inspect_us30_csv.py` | **Créer** | Diagnostic CSV US30 D1/H4 |

---

## 5. Prérequis données

Avant toute implémentation, vérifier :
1. `data/US30_D1.csv` existe et contient les colonnes `Time,Open,High,Low,Close,Volume` (ou `TickVol`)
2. Plage de dates couvre au minimum 2018-2025 (8 ans)
3. Pas de gaps > 5 jours consécutifs

Si les données US30 sont indisponibles :
- Fallback 1 : XAUUSD H4 (même principe, TP=500/SL=250 pips-or)
- Fallback 2 : GER30 D1 (DAX, même logique que US30)

---

## 6. Résultats

| Split | Barres | Signaux | Trades | Sharpe | WR | PnL (pips) |
|-------|--------|---------|--------|--------|-----|------------|
| Train (≤2022) | 2764 | — | — | — | — | — |
| Val (2023) | 310 | 0 | 0 | 0.000 | — | 0 |
| Test (2024) | 313 | 37 | 31 | **−1.173** | 25.8% | −1 137.6 |
| Test (2025) | 312 | 36 | 35 | **−1.480** | 22.9% | −1 526.4 |
| **Test 2024-2025** | **625** | **73** | **66** | **−1.270** | **24.2%** | **−2 664.0** |

**Verdict** : ❌ NO-GO. Sharpe agrégé = −1.27, WR = 24.2% ≪ breakeven (~37%). Signaux 2023 = 0 (le modèle n'a généré aucune prédiction au-dessus du seuil de confiance 0.55 en validation). Le RF ne trouve aucune structure prédictive dans les 6 features mono-D1.

**Analyse** : 0 signal en 2023 suggère que les features D1 (RSI, ADX, SMA distances, ATR, Log_Return_5d, Volume_Ratio) ne produisent pas de prédictions confiantes sur US30. Le problème n'est pas l'overfit (trop peu de paramètres) mais l'absence d'information prédictive — même pattern qu'en v1.

---

## 7. Différences clés vs v1

| Aspect | v1 (EURUSD H1) | v2 (US30 D1) |
|--------|----------------|---------------|
| Timeframe | H1 (~60k échantillons) | D1 (~2.6k échantillons) |
| Features | 7-20, multi-TF, macro, calendrier | 6-8, mono-TF |
| Cible | `triple_barrier` 3:1 | `directional_clean` (binaire symétrique) |
| Modèle | RF 500 arbres, depth=6 | RF 200 arbres, depth=4 |
| Split | ≤2023/2024/2025 (brûlé) | ≤2022/2023/2024-2025 (neuf pour US30) |
| Filtres | 5 filtres superposés | 1 seul filtre (vol) |
| Protocole | Aucun (15 itérations ad-hoc) | Engagement préalable, DSR, hold-out |
