# Pivot v4 — A7 : Sélection du modèle ML (RF vs HistGBM vs Stacking)

## Préalable obligatoire (à lire dans l'ordre)
1. [00_README.md](00_README.md)
2. [A5_feature_generation.md](A5_feature_generation.md), [A6_feature_ranking.md](A6_feature_ranking.md) — **✅ Terminés**
3. [app/config/features_selected.py](../../app/config/features_selected.py) — top 15 figé par actif
4. [app/backtest/cpcv.py](../../app/backtest/cpcv.py) — CPCV existant
5. [../00_constitution.md](../00_constitution.md) — règles 9, 14

## Objectif
Sélectionner le **meilleur modèle de méta-labeling** parmi 3 candidats via CPCV (Combinatorial Purged Cross-Validation) **sur train ≤ 2022 uniquement**. Le modèle retenu sera figé pour A8 (tuning hyperparams) et B1+ (test OOS).

> **Principe** : la sélection de modèle est une analyse de comparaison train. **0 n_trial** car aucune lecture du test set 2024+.

## Type d'opération
🔧 **Sélection de modèle train-only** — **0 n_trial consommé**.

## 3 candidats à comparer

| Candidat | Description | Hyperparams initiaux | Force | Faiblesse |
|---|---|---|---|---|
| **A. RandomForest** | Baseline historique (v2 H05 +8.84) | n_estim=200, max_depth=4, min_samples_leaf=10, class_weight=balanced | Robuste, interpretable | Plateau perf sur petits datasets |
| **B. HistGradientBoosting** | Sklearn native, équivalent LightGBM | max_iter=200, max_depth=5, learning_rate=0.05, l2=1.0 | Souvent meilleur, gère NaN | Sur-ajuste plus facilement |
| **C. Stacking** | Meta-learner sur RF + HGBM + LR | level0: A + B, level1: LogReg avec calibration isotonique | Combine forces des 2 | Plus lent, plus complexe |

> Pas de XGBoost / LightGBM externes (dépendances lourdes). HistGradientBoosting de sklearn est équivalent.

## Definition of Done (testable)

- [ ] `app/models/candidates.py` (NOUVEAU) contient :
  - `build_rf() -> RandomForestClassifier`
  - `build_hgbm() -> HistGradientBoostingClassifier`
  - `build_stacking() -> StackingClassifier`
- [ ] `app/models/cpcv_evaluation.py` (NOUVEAU) contient `evaluate_model_cpcv(model, X, y, n_splits=5, embargo_pct=0.01) -> CPCVResult` :
  - Retourne Sharpe filtré moyen + std + WR + n_kept
  - Utilise un seuil 0.50 fixe pour A7 (calibration en A8)
- [ ] `scripts/run_a7_model_selection.py` :
  - Pour chaque actif/TF dans `FEATURES_SELECTED`
  - Sur train ≤ 2022, top 15 features
  - Évalue les 3 candidats via CPCV
  - Sélectionne le meilleur Sharpe moyen avec std/mean ratio < 1.0
- [ ] `app/config/model_selected.py` (NOUVEAU, FROZEN) :
  ```python
  MODEL_SELECTED: dict[tuple[str, str], str] = {
      ("US30", "D1"): "hgbm",
      ("EURUSD", "H4"): "stacking",
      ...
  }
  ```
- [ ] `docs/model_selection_v4.md` : rapport comparatif détaillé
- [ ] `tests/unit/test_model_selection.py` : ≥ 6 tests :
  - Chaque candidat fit et predict sur données synthétiques
  - CPCV ne fait pas de leak temporel
  - Embargo respecté
  - Sharpe filtré = baseline si modèle aléatoire
  - Stacking produit des proba ∈ [0, 1]
  - Test sur dataset déséquilibré (10/90) : class_weight balanced fonctionne
- [ ] `predictions/model_selection_v4.json` : résultats détaillés
- [ ] `rtk make verify` → 0 erreur
- [ ] `JOURNAL.md` : entrée A7 avec modèles retenus par actif

## NE PAS FAIRE

- ❌ Ne PAS utiliser le test set ≥ 2024.
- ❌ Ne PAS tuner les hyperparams ici (= A8). Hyperparams = valeurs raisonnables par défaut.
- ❌ Ne PAS dépendre de XGBoost/LightGBM externes.
- ❌ Ne PAS sélectionner un modèle avec std/mean > 1.0 (instable).
- ❌ Ne PAS introduire de modèle deep learning (LSTM, Transformer). Données train ~ 1000 samples = pas assez.
- ❌ Ne PAS incrémenter `n_trials`.

## Étapes détaillées

### Étape 1 — `app/models/candidates.py`

```python
"""Candidats de modèles pour méta-labeling (pivot v4 A7)."""
from __future__ import annotations

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


RANDOM_STATE = 42


def build_rf(seed: int = RANDOM_STATE) -> RandomForestClassifier:
    """Random Forest baseline (v2 H05)."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def build_hgbm(seed: int = RANDOM_STATE) -> HistGradientBoostingClassifier:
    """HistGradientBoosting (équivalent sklearn de LightGBM)."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
        # early_stopping désactivé pour avoir une comparaison déterministe entre folds
        early_stopping=False,
    )


def build_stacking(seed: int = RANDOM_STATE) -> StackingClassifier:
    """Stacking RF + HGBM → meta-learner LogReg avec calibration isotonique."""
    rf = build_rf(seed)
    hgbm = build_hgbm(seed)
    base_estimators = [("rf", rf), ("hgbm", hgbm)]
    meta = LogisticRegression(
        class_weight="balanced", random_state=seed, max_iter=1000,
    )
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    # Calibration isotonique en surcouche → proba mieux calibrées
    return CalibratedClassifierCV(stacking, method="isotonic", cv=3)


CANDIDATES: dict[str, callable] = {
    "rf": build_rf,
    "hgbm": build_hgbm,
    "stacking": build_stacking,
}
```

### Étape 2 — `app/models/cpcv_evaluation.py`

```python
"""Évaluation CPCV d'un modèle de méta-labeling sur train uniquement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class CPCVResult:
    model_name: str
    sharpe_mean: float
    sharpe_std: float
    sharpe_ratio_stability: float  # std / |mean|
    wr_mean: float
    n_kept_mean: float
    fold_sharpes: list[float]
    fold_wrs: list[float]


def _purged_kfold_indices(
    n: int, k: int = 5, embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged k-fold splits with embargo (López de Prado)."""
    embargo = max(1, int(n * embargo_pct))
    fold_size = n // k
    splits = []
    for i in range(k):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        test_idx = np.arange(test_start, test_end)
        train_mask = np.ones(n, dtype=bool)
        # Purge : exclude embargo around test
        purge_start = max(0, test_start - embargo)
        purge_end = min(n, test_end + embargo)
        train_mask[purge_start:purge_end] = False
        train_idx = np.where(train_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


def _compute_sharpe_per_trade(
    proba: np.ndarray,
    pnl: np.ndarray,
    threshold: float = 0.50,
) -> tuple[float, float, int]:
    """Sharpe per-trade × √n_trades annualized, sur les trades filtrés."""
    keep = proba > threshold
    if keep.sum() < 5:
        return 0.0, 0.0, int(keep.sum())
    pnl_kept = pnl[keep]
    wr = float((pnl_kept > 0).mean())
    if pnl_kept.std() == 0:
        return 0.0, wr, int(keep.sum())
    sr_per_trade = pnl_kept.mean() / pnl_kept.std()
    # Annualisation approximative : √n_kept (suppose trades indépendants)
    sr_annualized = sr_per_trade * np.sqrt(keep.sum())
    return float(sr_annualized), wr, int(keep.sum())


def evaluate_model_cpcv(
    model_builder: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    model_name: str,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    threshold: float = 0.50,
    seed: int = 42,
) -> CPCVResult:
    """Évalue un modèle via CPCV. Retourne le Sharpe filtré moyen + std.

    Args:
        model_builder: fonction qui retourne un modèle sklearn-like (fit/predict_proba).
        X: features train.
        y: cible binaire train.
        pnl: PnL brut par trade train (pour calcul Sharpe filtré).
        model_name: identifiant pour log.
        n_splits, embargo_pct: paramètres CPCV.
        threshold: seuil de filtrage (fixé à 0.50 pour A7).
    """
    n = len(X)
    splits = _purged_kfold_indices(n, k=n_splits, embargo_pct=embargo_pct)
    fold_sharpes = []
    fold_wrs = []
    fold_n_kept = []

    for train_idx, test_idx in splits:
        if len(train_idx) < 30 or len(test_idx) < 10:
            continue
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        pnl_te = pnl.iloc[test_idx].values

        model = model_builder(seed)
        model.fit(X_tr.values, y_tr.values)
        proba = model.predict_proba(X_te.values)[:, 1]
        sr, wr, n_kept = _compute_sharpe_per_trade(proba, pnl_te, threshold)
        fold_sharpes.append(sr)
        fold_wrs.append(wr)
        fold_n_kept.append(n_kept)

    if not fold_sharpes:
        return CPCVResult(model_name, 0.0, 0.0, float("inf"), 0.0, 0.0, [], [])

    sr_mean = float(np.mean(fold_sharpes))
    sr_std = float(np.std(fold_sharpes))
    stability = sr_std / (abs(sr_mean) + 1e-9)

    return CPCVResult(
        model_name=model_name,
        sharpe_mean=sr_mean,
        sharpe_std=sr_std,
        sharpe_ratio_stability=stability,
        wr_mean=float(np.mean(fold_wrs)),
        n_kept_mean=float(np.mean(fold_n_kept)),
        fold_sharpes=fold_sharpes,
        fold_wrs=fold_wrs,
    )
```

### Étape 3 — `scripts/run_a7_model_selection.py`

```python
"""Pivot v4 A7 — Sélection de modèle via CPCV train uniquement.

⚠️ Aucune lecture du test set ≥ 2024.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.seeds import set_global_seeds
from app.data.loader import load_asset
from app.config.instruments import ASSET_CONFIGS
from app.config.features_selected import FEATURES_SELECTED
from app.features.superset import build_superset
from app.models.candidates import CANDIDATES
from app.models.cpcv_evaluation import evaluate_model_cpcv
from app.strategies.donchian import DonchianBreakout
from app.backtest.deterministic import run_deterministic_backtest


CUTOFF = pd.Timestamp("2022-12-31 23:59:59", tz="UTC")


STRAT_MAP = {
    ("US30", "D1"): (DonchianBreakout, {"N": 20, "M": 20}),
    ("XAUUSD", "D1"): (DonchianBreakout, {"N": 100, "M": 20}),
    # EURUSD H4 : importer MeanReversionRSIBB depuis B2 quand prêt
}


def evaluate_one_asset(asset: str, tf: str) -> dict:
    cfg = ASSET_CONFIGS[asset]
    key = (asset, tf)
    selected_features = FEATURES_SELECTED.get(key)
    if not selected_features:
        return {"error": f"No selected features for {asset} {tf}"}

    df = load_asset(asset, tf)
    df_train = df.loc[df.index <= CUTOFF]
    if (asset, tf) not in STRAT_MAP:
        return {"error": f"No strategy mapped for {asset} {tf}"}
    strat_cls, strat_kwargs = STRAT_MAP[(asset, tf)]
    strat = strat_cls(**strat_kwargs)

    trades = run_deterministic_backtest(df_train, strat, cfg)
    if trades.empty or len(trades) < 80:
        return {"error": f"Too few trades on train: {len(trades)}"}

    X_full = build_superset(df_train, asset=asset)
    X = X_full.loc[trades.index, list(selected_features)].dropna()
    y = (trades["Pips_Bruts"] > 0).astype(int).loc[X.index]
    pnl = trades["Pips_Bruts"].loc[X.index]

    if len(X) < 80:
        return {"error": f"Too few samples after align: {len(X)}"}

    results = {}
    for name, builder in CANDIDATES.items():
        print(f"  CPCV {name}...")
        r = evaluate_model_cpcv(
            model_builder=builder,
            X=X, y=y, pnl=pnl,
            model_name=name,
            n_splits=5, embargo_pct=0.01,
            threshold=0.50, seed=42,
        )
        results[name] = {
            "sharpe_mean": r.sharpe_mean,
            "sharpe_std": r.sharpe_std,
            "stability": r.sharpe_ratio_stability,
            "wr_mean": r.wr_mean,
            "n_kept_mean": r.n_kept_mean,
            "fold_sharpes": r.fold_sharpes,
        }

    # Sélectionner : meilleur Sharpe moyen avec stabilité < 1.0
    candidates = {n: r for n, r in results.items() if r["stability"] < 1.0}
    if not candidates:
        # Fallback : prendre le meilleur Sharpe peu importe la stabilité
        best = max(results.items(), key=lambda kv: kv[1]["sharpe_mean"])
    else:
        best = max(candidates.items(), key=lambda kv: kv[1]["sharpe_mean"])

    return {
        "asset": asset,
        "tf": tf,
        "n_trades_train": len(X),
        "n_features": len(selected_features),
        "winner_rate_train": float(y.mean()),
        "results_per_model": results,
        "best_model": best[0],
        "best_sharpe_mean": best[1]["sharpe_mean"],
    }


def main() -> int:
    set_global_seeds()
    out = {}
    for (asset, tf) in FEATURES_SELECTED.keys():
        key = f"{asset}_{tf}"
        print(f"Evaluating {key}...")
        out[key] = evaluate_one_asset(asset, tf)

    out_path = Path("predictions/model_selection_v4.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False), encoding="utf-8")

    # Génère app/config/model_selected.py
    lines = []
    for (asset, tf), entry_key in [((a, t), f"{a}_{t}") for (a, t) in FEATURES_SELECTED]:
        if entry_key in out and "best_model" in out[entry_key]:
            lines.append(f'    ("{asset}", "{tf}"): "{out[entry_key]["best_model"]}",')
    content = '"""FROZEN après pivot v4 A7. NE PAS MODIFIER sans nouveau pivot."""\n' \
              'from __future__ import annotations\n\n' \
              'MODEL_SELECTED: dict[tuple[str, str], str] = {\n'
    content += "\n".join(lines) + "\n}\n"
    Path("app/config/model_selected.py").write_text(content, encoding="utf-8")

    print(f"Modèles sélectionnés sauvegardés dans {out_path} et app/config/model_selected.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Étape 4 — Tests `tests/unit/test_model_selection.py`

```python
"""Tests sélection de modèle pivot v4 A7."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.candidates import build_rf, build_hgbm, build_stacking, CANDIDATES
from app.models.cpcv_evaluation import (
    evaluate_model_cpcv, _purged_kfold_indices, _compute_sharpe_per_trade,
)


def _make_dataset(n: int = 500, n_features: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    logit = X.iloc[:, :5].sum(axis=1) + rng.normal(0, 0.3, n)
    y = pd.Series((logit > 0).astype(int))
    pnl = pd.Series(np.where(y == 1, rng.uniform(50, 200, n), rng.uniform(-150, -50, n)))
    return X, y, pnl


def test_rf_fits_and_predicts():
    X, y, _ = _make_dataset()
    rf = build_rf()
    rf.fit(X.values, y.values)
    proba = rf.predict_proba(X.values)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_hgbm_fits_and_predicts():
    X, y, _ = _make_dataset()
    hgbm = build_hgbm()
    hgbm.fit(X.values, y.values)
    proba = hgbm.predict_proba(X.values)
    assert proba.shape == (len(X), 2)


def test_stacking_fits_and_predicts():
    X, y, _ = _make_dataset(n=300)
    st = build_stacking()
    st.fit(X.values, y.values)
    proba = st.predict_proba(X.values)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_purged_kfold_embargo():
    splits = _purged_kfold_indices(n=1000, k=5, embargo_pct=0.02)
    for train_idx, test_idx in splits:
        # Embargo : aucun train index dans [test_start - embargo, test_end + embargo]
        embargo = 20
        test_min, test_max = test_idx.min(), test_idx.max()
        purge_min = max(0, test_min - embargo)
        purge_max = min(1000, test_max + embargo)
        assert not np.any((train_idx >= purge_min) & (train_idx <= purge_max))


def test_compute_sharpe_per_trade_kept_too_few():
    proba = np.array([0.4, 0.45, 0.48])
    pnl = np.array([100, -100, 50])
    sr, wr, n_kept = _compute_sharpe_per_trade(proba, pnl, threshold=0.50)
    assert sr == 0.0
    assert n_kept == 0


def test_cpcv_runs_on_each_candidate():
    X, y, pnl = _make_dataset(n=500)
    for name, builder in CANDIDATES.items():
        r = evaluate_model_cpcv(
            model_builder=builder, X=X, y=y, pnl=pnl,
            model_name=name, n_splits=3,
        )
        assert r.model_name == name
        # Au moins quelques folds ont produit un résultat
        assert len(r.fold_sharpes) >= 1
```

### Étape 5 — Exécution (sur demande utilisateur)

```bash
rtk python scripts/run_a7_model_selection.py
```

Sortie attendue (~5-15 min selon CPU) :
```
Evaluating US30_D1...
  CPCV rf...
  CPCV hgbm...
  CPCV stacking...
Evaluating EURUSD_H4...
  CPCV rf...
  ...
Modèles sélectionnés sauvegardés dans predictions/model_selection_v4.json et app/config/model_selected.py
```

### Étape 6 — Rapport `docs/model_selection_v4.md`

```markdown
# Model Selection v4 (pivot v4 A7)

**Date** : YYYY-MM-DD
**Périmètre** : train ≤ 2022-12-31 UNIQUEMENT
**n_trials** : inchangé
**Méthode** : CPCV 5 folds × embargo 1 %, seuil fixe 0.50

## Résultats par actif

### US30 D1

| Modèle | Sharpe moyen | Sharpe std | Stability | WR moyen | n_kept |
|---|---|---|---|---|---|
| RF | 1.45 | 0.62 | 0.43 | 0.52 | 18 |
| HGBM | 1.78 | 0.71 | 0.40 | 0.55 | 22 |
| Stacking | 1.62 | 0.85 | 0.52 | 0.54 | 20 |

**Modèle retenu** : HGBM (Sharpe 1.78, stability 0.40 < 1.0).

### EURUSD H4
...

### XAUUSD D1
...

## Interprétation

- HGBM a tendance à dominer sur les actifs avec >500 trades train (capacité non-linéaire mieux exploitée).
- RF reste compétitif si stability HGBM > 0.7 (HGBM overfit, RF plus régularisé par nature).
- Stacking apporte rarement +0.2 Sharpe par rapport au meilleur des deux → coût compute pas justifié sauf cas particulier.

## Décision de gel

Les modèles retenus par actif sont FIGÉS dans `app/config/model_selected.py`.
**Aucun changement jusqu'à fin Phase B.**

## Limites

- Seuil 0.50 fixe pour A7. Calibration en A8 peut décaler Sharpe ± 0.2.
- n_splits=5 = compromis vitesse/stabilité. 10 plus rigoureux mais 2× plus lent.
- Le Sharpe per-trade × √n est une approximation ; B1+ utilisera le vrai Sharpe walk-forward.
```

## Tests unitaires associés

6 tests dans `tests/unit/test_model_selection.py` (Étape 4).

## Logging obligatoire

```markdown
## YYYY-MM-DD — Pivot v4 A7 : Sélection de modèle (RF vs HGBM vs Stacking)

- **Statut** : ✅ Terminé
- **Type** : Sélection train (0 n_trial)
- **Fichiers créés** : `app/models/candidates.py`, `app/models/cpcv_evaluation.py`, `scripts/run_a7_model_selection.py`, `app/config/model_selected.py`, `tests/unit/test_model_selection.py`, `docs/model_selection_v4.md`, `predictions/model_selection_v4.json`
- **Modèles retenus par actif** :
  - US30 D1 : ?
  - EURUSD H4 : ?
  - XAUUSD D1 : ?
- **Tests** : 6/6 passed
- **make verify** : ✅ passé
- **Prochaine étape** : A8 — Tuning hyperparams via nested CPCV
```

## Critères go/no-go

- **GO Phase A8** si :
  - `app/config/model_selected.py` contient ≥ 1 modèle par actif
  - Pour chaque actif retenu, Sharpe moyen ≥ 0.5 (sinon, edge train inexistant → A7 NO-GO partiel)
  - Stability < 1.0 sur ≥ 1 actif
- **NO-GO, revenir à** :
  - Si tous les modèles ont Sharpe moyen ≤ 0.3 → revenir A6 (top features insuffisantes) ou abandonner l'actif
  - Si tous les modèles ont stability > 1.5 → besoin de plus de données train ou cible mal posée

## Annexes

### A1 — Pourquoi pas LightGBM / XGBoost / CatBoost externes

- Sklearn HistGradientBoosting depuis 1.3 est dans le top 3 des perfs sur tabular (cf. benchmark Olive 2024).
- Évite une dépendance lourde (~ 50 MB).
- Reproductibilité : moins de versions incompatibles.

Si après A7 on veut absolument LightGBM, c'est +1 hypothèse (nouveau modèle = nouveau backtest OOS).

### A2 — Pourquoi pas de deep learning (LSTM, Transformer)

- ~ 500-1000 samples train = trop peu pour deep learning (besoin > 10 k).
- Tabular data → trees > NN largement.
- LSTM sensible aux look-ahead leaks subtils.

À considérer pour V5 si on accumule 5+ ans de live trading.

### A3 — Pourquoi calibration isotonique sur Stacking

Le `StackingClassifier` produit des proba parfois mal calibrées (proche de 0 ou 1 par overconfidence). `CalibratedClassifierCV(method="isotonic", cv=3)` corrige ça → utile pour la **calibration de seuil** en A8.

### A4 — Pourquoi n_splits=5 et pas 10

- 5 folds = 100/5 = 20 % du train en test par fold → balanced
- 10 folds = 10 % par fold → moins de bruit par fold mais 2× plus lent
- Pour 500-1000 samples train, 5 folds offre 80-200 samples par train_fold → suffisant pour fit RF/HGBM

Si Phase A8 montre des résultats instables → revenir et augmenter à 8 ou 10.

### A5 — Pourquoi class_weight=balanced obligatoire

Les trades Donchian/MeanRev ont typiquement WR train = 40-55 %. C'est proche de 50/50 mais pas exactement → `class_weight=balanced` pondère inversement à la fréquence. Évite que le modèle tende à toujours prédire la classe majoritaire (cf. leçon H1 pivot XAUUSD).

### A6 — Pourquoi le Stacking peut être pire que ses bases

Stacking = méta-modèle qui apprend à pondérer base learners sur les **mêmes données** (via cv interne). Si RF et HGBM sont déjà bien calibrés, le méta-learner peut sur-apprendre des artefacts du cv interne.

En pratique : Stacking gagne quand RF et HGBM ont des erreurs **systématiquement différentes** (low correlation des résiduels). Si les 2 commettent les mêmes erreurs → stacking n'aide pas.

### A7 — Pourquoi le seuil 0.50 ici (pas optimisé)

A7 = sélection de modèle, pas de seuil. Si on tune le seuil ici, on pollue le critère de sélection. A8 tune le seuil après que le modèle soit choisi.

Le seuil 0.50 = "le modèle prédit positif" = neutre. Toute différence Sharpe entre modèles à ce seuil reflète la qualité intrinsèque du modèle.

## Fin du prompt A7.
**Suivant** : [A8_hyperparameter_tuning.md](A8_hyperparameter_tuning.md)
