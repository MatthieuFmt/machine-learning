---
name: model-eval-time-series
description: Évaluation de modèles ML sur séries temporelles — TimeSeriesSplit, purge, métriques financières.
---

# Model Evaluation Time Series — Métriques Fiables en Finance

## Règle cardinale

**Les métriques de classification standard (accuracy globale) sont trompeuses en finance. Le modèle doit être évalué avec TimeSeriesSplit + purge, et les métriques doivent inclure des grandeurs financières (nombre de trades théoriques, profit factor implicite, win_rate).**

## Instructions

### 1. Split d'évaluation obligatoire

```python
# ✅ CORRECT : TimeSeriesSplit avec purge
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=48)  # gap = purge_hours

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # Toujours vérifier l'ordre chronologique
    assert X_train.index.max() < X_val.index.min()

# ❌ INTERDIT : KFold, StratifiedKFold, GroupKFold sur séries temporelles
```

### 2. Métriques minimales à calculer

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

def evaluate_timeseries_model(model, X_val, y_val, probas_val, class_map):
    preds = model.predict(X_val)

    metrics = {
        # Métriques standard
        "accuracy": accuracy_score(y_val, preds),
        "f1_macro": f1_score(y_val, preds, average="macro"),
        "f1_weighted": f1_score(y_val, preds, average="weighted"),
        "classification_report": classification_report(y_val, preds),

        # Métriques par classe
        "confusion_matrix": confusion_matrix(y_val, preds).tolist(),

        # Métriques financières
        "n_signaux": int((preds != 0).sum()),
        "pct_signaux": round(float((preds != 0).mean() * 100), 1),
        "precision_long": precision_by_class(y_val, preds, 1),
        "precision_short": precision_by_class(y_val, preds, -1),

        # Calibration
        "log_loss": log_loss(y_val, probas_val),
    }
    return metrics
```

### 3. Feature importance — deux méthodes obligatoires

```python
# Méthode 1 : Impureté (Gini) — rapide, intégrée à sklearn
fi_gini = pd.DataFrame({
    "feature": X_cols,
    "importance_%": model.feature_importances_ * 100,
}).sort_values("importance_%", ascending=False)

# Méthode 2 : Permutation — plus fiable, indépendante du modèle
from sklearn.inspection import permutation_importance
perm = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42)
fi_perm = pd.DataFrame({
    "feature": X_cols,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

# ✅ Toujours reporter les DEUX. Si elles divergent fortement → overfitting.
```

### 4. Probabilités calibrées

```python
# Les probas RandomForest brutes sont MAL calibrées (tendance à pousser vers 0/1)
# → Toujours vérifier avec un reliability diagram (sklearn.calibration.calibration_curve)

from sklearn.calibration import CalibratedClassifierCV

# Calibration sur le validation set
calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)
probas_cal = calibrated.predict_proba(X_test)
```

### 5. Walk-forward validation

```python
# Réentraîner périodiquement (ex: tous les 6 mois) pour mesurer la stabilité
def walk_forward_validation(df, model_cls, train_window_months=36, test_window_months=6):
    results = []
    start = df.index.min()
    end = df.index.max()

    current = start + pd.DateOffset(months=train_window_months)
    while current + pd.DateOffset(months=test_window_months) <= end:
        train = df[(df.index >= start) & (df.index < current)]
        test = df[(df.index >= current) & (df.index < current + pd.DateOffset(months=test_window_months))]

        model = model_cls()
        model.fit(train[X_cols], train["Target"])
        preds = model.predict(test[X_cols])
        acc = accuracy_score(test["Target"], preds)

        results.append({"cutoff": current, "accuracy": acc, "n_train": len(train), "n_test": len(test)})
        current += pd.DateOffset(months=test_window_months)

    return pd.DataFrame(results)
```

### 6. Overfitting detection

```python
# Signes d'overfitting :
# 1. Accuracy train >> accuracy test (écart > 15%)
# 2. Feature importance Gini >> Permutation (features instables)
# 3. Performance qui se dégrade fortement d'une année OOS à l'autre
# 4. Peu de features avec importance > 0 en permutation

# Toujours logger l'écart train/OOS :
logger.info("Accuracy: train=%.2f, val=%d=%.2f, test=%d=%.2f",
            train_acc, val_year, val_acc, test_year, test_acc)
if train_acc - test_acc > 0.15:
    logger.warning("OVERFITTING DETECTÉ: écart train/test = %.2f", train_acc - test_acc)
```

## Checklist finale

1. Split avec `gap >= max(window, 48)` heures
2. Métriques financières (n_signaux, pct_signaux) en plus des métriques ML
3. Les DEUX feature importances (Gini + Permutation)
4. Vérification de calibration (reliability diagram ou log-loss)
5. Écart train/OOS < 15%
6. Walk-forward validation si > 3 ans de données
