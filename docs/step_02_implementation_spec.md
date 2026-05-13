# Step 02 — Spécification d'implémentation détaillée (CPCV + DSR)

**Date** : 2026-05-12
**Mode** : Architect (spécification seulement — implémentation en mode Code)
**Fichier stratégique** : [`docs/step_02_robust_validation_framework.md`](step_02_robust_validation_framework.md)

---

## Vue d'ensemble du plan d'exécution

5 modifications ordonnées. Les mods 1–4 sont atomiques et testables indépendamment. Le mod 5 est l'orchestrateur.

| # | Fichier | Type de changement | Impact |
|---|---|---|---|
| 1 | [`edge_validation.py`](../learning_machine_learning/analysis/edge_validation.py) | Extension — ajout `psr_from_returns()`, `deflated_sharpe_ratio_from_distribution()`, `validate_edge_distribution()` | Rétrocompatible (nouvelles fonctions) |
| 2 | [`training.py`](../learning_machine_learning/model/training.py) | Extension — `train_test_split_purge` accepte `train_mask`/`test_mask` arbitraires | Rétrocompatible (kwargs optionnels) |
| 3 | [`cpcv.py`](../learning_machine_learning/analysis/cpcv.py) | Création — `generate_cpcv_splits()`, `run_cpcv_backtest()`, `aggregate_cpcv_metrics()` | Nouveau module |
| 4 | Tests unitaires [`test_cpcv.py`](../tests/unit/test_cpcv.py) | Création — 8-10 tests sur splits, invariants temporels, agrégation | Nouveau |
| 5 | [`run_validation_cpcv.py`](../run_validation_cpcv.py) | Création — script orchestrateur CPCV 200 splits | Nouveau |

---

## Architecture des données CPCV

```
┌──────────────────────────────────────────────────────────────┐
│ Timeline EURUSD H1 : 2014-01-01 → 2025-12-31                 │
│                                                              │
│ Groupes : G₀  G₁  G₂  G₃  ...  G₂₂  G₂₃                     │
│           ├────┼────┼────┼────┼────┼────┤                    │
│           2 semaines par groupe = 24 groupes sur 12 mois OOS │
│           (la période train peut être plus large)            │
│                                                              │
│ Split i : train = tous les groupes sauf {G_a, G_b, ..., G_f} │
│           test  = {G_a, G_b, ..., G_f} (6 groupes = 3 mois)  │
│           purge = 48h AVANT chaque groupe test ET APRES       │
│                                                              │
│ Entraînement sur train → prédiction sur test → simulate → SR │
└──────────────────────────────────────────────────────────────┘
```

---

## Mod 1 — [`edge_validation.py`](../learning_machine_learning/analysis/edge_validation.py) : Extension DSR distributionnel

### 1a. `psr_from_returns`

```python
def psr_from_returns(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    PSR(SR*) = Φ( (ŜR - SR*) · √(n-1) / √(1 − γ̂₃·ŜR + (γ̂₄−1)/4 · ŜR²) )

    Args:
        returns: Vecteur de returns (trades).
        sr_benchmark: Sharpe benchmark à tester (défaut: 0 pour H₀: SR ≤ 0).

    Returns:
        Probabilité que le vrai Sharpe > sr_benchmark. NaN si n < 2 ou dénominateur nul.
    """
```

**Logique** : Le `validate_edge()` existant calcule déjà un PSR simplifié via `stats.norm.cdf(dsr)`. Mais ce calcul est couplé à `_expected_max_sr`. On extrait le PSR pur (formule de Bailey) en tant que fonction indépendante.

**Différence vs code existant** : Le code existant dans `validate_edge()` calcule `DSR = (observed_sharpe - E[max(SR)]) / SE(max(SR))` puis `PSR = Φ(DSR)`. Ce n'est pas le PSR de Bailey — c'est un DSR simplifié. La fonction `psr_from_returns()` ci-dessus implémente la vraie formule PSR avec skewness et kurtosis.

### 1b. `deflated_sharpe_ratio_from_distribution`

```python
def deflated_sharpe_ratio_from_distribution(
    observed_sr: float,
    sharpe_distribution: np.ndarray,
    ci_level: float = 0.95,
) -> dict:
    """DSR à partir d'une distribution empirique de Sharpe (issue CPCV).

    Implémente López de Prado (2014) §II :
    SR₀* = √Var({SRᵢ}) · ((1−γ)Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))

    Args:
        observed_sr: Sharpe observé sur le split principal (train≤2023, test=2025).
        sharpe_distribution: Array des Sharpe de chaque split CPCV.
        ci_level: Niveau de confiance (défaut: 0.95).

    Returns:
        {
            "dsr": float,           # Deflated Sharpe Ratio
            "sr0_star": float,      # Seuil de Sharpe déflaté
            "e_max_sr": float,      # E[max(SR)] sous H₀
            "var_sr": float,        # Variance des Sharpe CPCV
            "n_splits": int,        # Nombre de splits
            "pct_profitable": float,# % de splits avec Sharpe > 0
            "mean_sr": float,       # E[SR] sur distribution
            "std_sr": float,        # σ[SR] sur distribution
            "min_sr": float,        # Min Sharpe
            "max_sr": float,        # Max Sharpe
            "median_sr": float,     # Médiane Sharpe
        }
    """
```

**Logique** : Utilise la distribution CPCV pour calculer le vrai DSR corrigé pour le nombre de combinaisons testées. C'est la version "moderne" du DSR — celle qui utilise la variance empirique des splits plutôt qu'une estimation EVT théorique.

### 1c. `validate_edge_distribution`

```python
def validate_edge_distribution(
    trades_df: pd.DataFrame,
    backtest_cfg: BacktestConfig,
    sharpe_distribution: np.ndarray | None = None,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> dict:
    """Wrapper combinant validate_edge + DSR distributionnel.

    Si sharpe_distribution est fournie, le DSR est calculé à partir
    de la distribution CPCV plutôt que via n_trials_searched.

    Args:
        trades_df: DataFrame des trades du split principal.
        backtest_cfg: Configuration backtest.
        sharpe_distribution: Distribution Sharpe CPCV (optionnel).
        n_bootstrap: Itérations bootstrap.
        random_state: Graine aléatoire.

    Returns:
        Dict combinant les métriques de validate_edge() + champ "cpcv_dsr".
    """
```

**Logique** : Point d'entrée unique pour le reporting final. Combine la validation standard (breakeven WR, bootstrap, t-test) avec le DSR distributionnel issu du CPCV.

---

## Mod 2 — [`training.py`](../learning_machine_learning/model/training.py) : Généralisation `train_test_split_purge`

### Changement

La signature actuelle :

```python
def train_test_split_purge(
    df: pd.DataFrame,
    train_end_year: int,
    purge_hours: int = 48,
    extra_drop_cols: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
```

Devient :

```python
def train_test_split_purge(
    df: pd.DataFrame,
    train_end_year: int | None = None,
    purge_hours: int = 48,
    extra_drop_cols: frozenset[str] | None = None,
    *,
    train_mask: pd.Series | None = None,
    test_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
```

**Contrat** :
- Si `train_end_year` est fourni → comportement actuel inchangé (split par année).
- Si `train_mask` ET `test_mask` sont fournis → split arbitraire. `train_end_year` ignoré.
- Si aucun n'est fourni → `ValueError`.
- `test_mask` n'est utilisé que pour le logging (la fonction ne retourne que X_train, y_train, X_cols).

**Appel existant** : `train_test_split_purge(ml_data, train_end_year=2023, purge_hours=48)` → inchangé.

**Appel CPCV** : `train_test_split_purge(ml_data, train_mask=train_idx_bool, test_mask=test_idx_bool, purge_hours=48)`.

### Aucune modification de `walk_forward_train` ni de `train_model`/`train_regressor`.

---

## Mod 3 — [`cpcv.py`](../learning_machine_learning/analysis/cpcv.py) : Module CPCV

### 3a. `generate_cpcv_splits`

```python
def generate_cpcv_splits(
    index: pd.DatetimeIndex,
    n_groups: int = 24,
    k_test: int = 6,
    purge_hours: int = 48,
    n_samples: int = 200,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Génère des splits CPCV avec purge bidirectionnelle.

    Partitionne l'index en `n_groups` groupes contigus. Pour chaque combinaison
    de `k_test` groupes-test, applique une purge de `purge_hours` avant et après
    chaque groupe test, puis yield (train_indices, test_indices).

    Args:
        index: DatetimeIndex trié chronologiquement.
        n_groups: Nombre de groupes (défaut: 24 pour ~2 semaines/groupe sur 12 mois).
        k_test: Nombre de groupes de test par split (défaut: 6 = ~3 mois).
        purge_hours: Heures de purge bidirectionnelle autour de chaque groupe test.
        n_samples: Nombre de combinaisons à échantillonner (défaut: 200).
                   Si None ou > C(n_groups, k_test) → toutes les combinaisons.
        random_state: Graine pour l'échantillonnage des combinaisons.

    Yields:
        Tuple (train_indices: np.ndarray, test_indices: np.ndarray).
        Les indices sont des positions entières dans l'index.

    Invariants garantis :
        - max(train_indices) + purge_hours < min(test_indices)  [purge avant test]
        - max(test_indices) + purge_hours < min(train_after_test) [purge après test]
        - Aucun chevauchement train/test
    """
```

**Algorithme** :
1. Découper `index` en `n_groups` groupes de taille égale.
2. Pour chaque groupe, calculer ses bornes temporelles.
3. Générer toutes les combinaisons `C(n_groups, k_test)`.
4. Si `n_samples` < total → échantillonner aléatoirement avec `random_state`.
5. Pour chaque combinaison de groupes-test :
   - Déterminer les barres appartenant à chaque groupe-test.
   - Appliquer purge : exclure les barres dans `[group_start − purge_hours, group_start)` ET `(group_end, group_end + purge_hours]`.
   - Les barres restantes (hors test + hors purge) = train.
   - Yield (train_indices, test_indices).

### 3b. `run_cpcv_backtest`

```python
def run_cpcv_backtest(
    ml_data: pd.DataFrame,
    ohlcv_h1: pd.DataFrame,
    splits: Iterator[tuple[np.ndarray, np.ndarray]],
    model_factory: ModelFactory,
    backtest_cfg: BacktestConfig,
    instrument_cfg: InstrumentConfig,
    X_cols: list[str],
    target_mode: TargetMode,
    confidence_threshold: float = 0.33,
    continuous_signal_threshold: float = 0.0005,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Exécute le pipeline complet sur chaque split CPCV.

    Pour chaque split (train_idx, test_idx) :
    1. Extrait X_train, y_train, X_test depuis ml_data.
    2. Entraîne le modèle via model_factory.
    3. Prédit sur test_idx.
    4. Simule les trades (classifieur ou régression).
    5. Calcule les métriques.

    Args:
        ml_data: DataFrame ML-ready complet, indexé par Time.
        ohlcv_h1: DataFrame OHLC H1 (nécessaire au simulateur).
        splits: Générateur de (train_indices, test_indices).
        model_factory: Callable (X_train, y_train) -> modèle.
        backtest_cfg: Configuration backtest.
        instrument_cfg: Configuration instrument.
        X_cols: Colonnes de features.
        target_mode: Mode de cible.
        confidence_threshold: Seuil classifieur.
        continuous_signal_threshold: Seuil régression.
        n_jobs: Nombre de jobs joblib (-1 = tous les CPUs).

    Returns:
        DataFrame avec colonnes :
        - split_id, train_start, train_end, test_start, test_end
        - n_train, n_test, n_trades
        - sharpe, sharpe_per_trade, profit_net, win_rate
        - dd, total_return_pct
    """
```

**Logique métier par split** :

```
1. train_data = ml_data.iloc[train_idx]
2. X_train = train_data[X_cols]; y_train = train_data["Target"]
3. model = model_factory(X_train, y_train)
4. X_test = ml_data.iloc[test_idx][X_cols]
5. preds = model.predict(X_test)  [ou predict_proba selon target_mode]
6. Construire un mini-DataFrame de prédictions avec OHLC
7. Simuler les trades → trades_df
8. compute_metrics() → sharpe, wr, etc.
```

**Parallélisation** : `joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_run_one_split)(...) for split in splits)`

**Filtres** : Appliquer les mêmes filtres que `BasePipeline.run_backtest()` (MomentumFilter, VolFilter, SessionFilter) via `FilterPipeline`. Les colonnes de filtre (`Dist_SMA200_D1`, `ATR_Norm`, `RSI_D1_delta`) doivent être présentes dans `ml_data`.

### 3c. `aggregate_cpcv_metrics`

```python
def aggregate_cpcv_metrics(
    results_df: pd.DataFrame,
) -> dict:
    """Agrège les métriques de tous les splits CPCV.

    Args:
        results_df: DataFrame produit par run_cpcv_backtest().

    Returns:
        {
            "n_splits": int,
            "n_splits_valid": int,    # splits avec au moins 1 trade
            "pct_profitable": float,  # % splits avec Sharpe > 0
            "sharpe": {
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
                "median": float,
                "ci_95_lower": float,
                "ci_95_upper": float,
            },
            "n_trades": {
                "mean": float,
                "std": float,
                "total": int,
            },
            "coverage": dict[str, int],  # mois → nombre de splits où il est test
        }
    """
```

---

## Mod 4 — Tests unitaires [`test_cpcv.py`](../tests/unit/test_cpcv.py)

### 10 tests planifiés

| # | Test | Classe | Description |
|---|---|---|---|
| 1 | `test_generate_splits_no_overlap` | `TestGenerateCpcvSplits` | Vérifie max(train) + purge < min(test) pour chaque split |
| 2 | `test_generate_splits_no_future_leak` | `TestGenerateCpcvSplits` | Vérifie qu'aucun idx train n'est postérieur au premier idx test |
| 3 | `test_generate_splits_purge_bidirectional` | `TestGenerateCpcvSplits` | Vérifie la purge APRÈS test aussi |
| 4 | `test_generate_splits_n_samples` | `TestGenerateCpcvSplits` | Vérifie que n_samples est respecté |
| 5 | `test_psr_from_returns_positive` | `TestPsrFromReturns` | PSR > 0.5 pour returns positifs |
| 6 | `test_psr_from_returns_zero` | `TestPsrFromReturns` | PSR ≈ 0.5 pour returns centrés sur 0 |
| 7 | `test_deflated_sr_distribution` | `TestDeflatedSrDistribution` | DSR > 0 pour distribution profitable |
| 8 | `test_deflated_sr_distribution_negative` | `TestDeflatedSrDistribution` | DSR < 0 pour distribution non-profitable |
| 9 | `test_aggregate_cpcv_metrics` | `TestAggregateCpcv` | Vérifie la structure du dict retourné |
| 10 | `test_generate_splits_coverage` | `TestGenerateCpcvSplits` | Chaque mois apparaît au moins 5 fois comme test |

### Fixtures synthétiques

```python
@pytest.fixture
def synthetic_h1_index() -> pd.DatetimeIndex:
    """Index H1 sur 18 mois (assez pour 24 groupes + purge)."""
    return pd.date_range("2023-01-01", "2024-06-30", freq="1h", tz="UTC")

@pytest.fixture
def mock_results_df() -> pd.DataFrame:
    """DataFrame simulant run_cpcv_backtest() output avec 200 splits."""
```

---

## Mod 5 — [`run_validation_cpcv.py`](../run_validation_cpcv.py) : Script orchestrateur

### Spécification

Script standalone exécutable via `python run_validation_cpcv.py`. Il :

1. **Charge les données** via `EurUsdPipeline.load_data()` + `build_features()`.
2. **Détermine la période OOS** : 2024-01-01 → 2025-12-31 (2 ans, 24 mois, → 48 groupes de 2 semaines). La période train est tout ce qui précède (2014–2023).
3. **Génère les splits CPCV** via `generate_cpcv_splits()` avec `n_groups=48, k_test=12` (6 mois de test), `n_samples=200`.
4. **Construit la `model_factory`** selon `target_mode` (RF classifieur ou GBM regressor).
5. **Exécute `run_cpcv_backtest()`** avec `n_jobs=-1`.
6. **Agrège** via `aggregate_cpcv_metrics()`.
7. **Calcule le DSR distributionnel** via `deflated_sharpe_ratio_from_distribution()`.
8. **Exécute `validate_edge_distribution()`** sur le split principal (train≤2023, test=2025).
9. **Sauvegarde** :
   - `predictions/cpcv_results.csv` : DataFrame complet des 200 splits
   - `predictions/cpcv_report.md` : rapport Markdown structuré
10. **Loggue** les métriques clés et le verdict GO/NO-GO.

### Structure du rapport Markdown

```markdown
# Rapport CPCV — EURUSD H1 — {date}

## Configuration
- n_groups: 48, k_test: 12, n_samples: 200
- Model: RandomForestClassifier (500 arbres, max_depth=6)
- Target: {target_mode}, TP={tp_pips}, SL={sl_pips}
- Friction: commission={commission_pips}, slippage={slippage_pips}

## Métriques CPCV (distribution sur 200 splits)

| Métrique | Valeur |
|---|---|
| E[Sharpe] | ... |
| σ[Sharpe] | ... |
| Sharpe median | ... |
| Sharpe min / max | ... / ... |
| % splits profitables | ...% |
| DSR | ... |
| SR₀* | ... |
| Nombre total de trades | ... |
| Trades/split (moyenne) | ... |

## Split principal (train≤2023, test=2025)

| Métrique | Valeur |
|---|---|
| Sharpe | ... |
| p(Sharpe>0) bootstrap | ... |
| DSR distributionnel | ... |
| Breakeven WR / WR observé | ...% / ...% |
| t-statistique | ... (p=...) |

## Verdict
- [ ] DSR > 0
- [ ] % profitables > 60%
- [ ] **GO / NO-GO Phase 2-3**
```

### Paramètres par défaut

```python
N_GROUPS = 48          # 2 semaines par groupe sur 24 mois OOS
K_TEST = 12            # 6 mois de test par split
N_SAMPLES = 200        # 200 combinaisons aléatoires
PURGE_HOURS = 48       # Cohérent avec ModelConfig.purge_hours
TARGET_MODE = "triple_barrier"  # Mode par défaut (baseline v15)
```

---

## Vérifications anti-leak obligatoires

Chaque split CPCV doit passer ces assertions (dans `generate_cpcv_splits`) :

```python
# 1. Index trié
assert index.is_monotonic_increasing

# 2. Pas de chevauchement train/test
assert len(np.intersect1d(train_idx, test_idx)) == 0

# 3. Purge avant test : max(train) + purge_hours < min(test)
assert index[train_idx].max() + timedelta(hours=purge_hours) < index[test_idx].min()

# 4. Purge après test : max(test) + purge_hours < min(train_after_test)
# (seulement si du train existe après le test)
if train_after_test_exists:
    assert index[test_idx].max() + timedelta(hours=purge_hours) < index[train_after_test_idx].min()
```

---

## Dépendances entre mods

```
Mod 1 (edge_validation.py) ── indépendant
Mod 2 (training.py)         ── indépendant
Mod 3 (cpcv.py)             ── dépend de Mod 1, Mod 2
Mod 4 (tests)               ── dépend de Mod 1, Mod 3
Mod 5 (run_validation)      ── dépend de Mod 1, Mod 2, Mod 3
```

**Ordre d'implémentation recommandé** : Mod 2 → Mod 1 → Mod 3 → Mod 4 → Mod 5

---

## Points d'attention

- **`edge_validation.py` existe déjà** avec `validate_edge()`. Ne pas casser l'existant — uniquement ajouter des fonctions. Les tests `test_edge_validation.py` doivent continuer à passer.
- **`train_test_split_purge`** est appelée dans `BasePipeline.train_model()` avec `train_end_year`. Le comportement existant doit être préservé.
- **Le simulateur a besoin de colonnes OHLC** (High, Low, Close) + `Spread`. Ces colonnes doivent être disponibles dans le sous-DataFrame de test pour chaque split.
- **Les filtres** (`FilterPipeline`) nécessitent `Dist_SMA200_D1`, `ATR_Norm`, `RSI_D1_delta`. Ces colonnes sont déjà dans `ml_data` (sauf `ATR_Norm` qui est dans `_FILTER_ONLY_COLS` — pour le CPCV on les garde dans les features).
- **Pas de parallélisation nested** : `joblib.Parallel` pour les splits CPCV, mais `RandomForestClassifier(n_jobs=1)` à l'intérieur pour éviter le fork-bomb.
- **Cache disque** : optionnel, sauvegarder les modèles dans `results/cpcv_models/` si `--cache-models` est passé.
