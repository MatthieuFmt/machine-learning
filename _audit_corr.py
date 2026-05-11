"""Test de corrélation features → rendement forward (détection look-ahead)."""
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier

from config import FILE_ML_READY, FILE_EURUSD_H1_CLEAN, RF_PARAMS, TRAIN_END_YEAR, PURGE_HOURS

df = pd.read_csv(FILE_ML_READY, index_col="Time", parse_dates=True)
prices = pd.read_csv(FILE_EURUSD_H1_CLEAN, index_col="Time", parse_dates=True)

train_cutoff = pd.to_datetime(f"{TRAIN_END_YEAR + 1}-01-01") - timedelta(hours=PURGE_HOURS)
train = df[df.index < train_cutoff]
test_2024 = df[(df.index >= "2024-01-01") & (df.index < "2025-01-01")]
test_2025 = df[(df.index >= "2025-01-01") & (df.index < "2026-01-01")]

X_cols = [c for c in df.columns if c not in ["Target", "Spread"]]

print("=== 1. OOB Score vs Test Accuracy ===")
rf = RandomForestClassifier(
    oob_score=True,
    random_state=RF_PARAMS["random_state"],
    n_estimators=RF_PARAMS["n_estimators"],
    max_depth=RF_PARAMS["max_depth"],
    min_samples_leaf=RF_PARAMS["min_samples_leaf"],
    class_weight=RF_PARAMS["class_weight"],
    n_jobs=RF_PARAMS["n_jobs"],
)
rf.fit(train[X_cols], train["Target"])
acc_2024 = rf.score(test_2024[X_cols], test_2024["Target"])
acc_2025 = rf.score(test_2025[X_cols], test_2025["Target"])
print(f"OOB score (train <= {TRAIN_END_YEAR}): {rf.oob_score_:.4f}")
print(f"Test accuracy 2024: {acc_2024:.4f}")
print(f"Test accuracy 2025: {acc_2025:.4f}")
print(f"Delta OOB -> 2024: {rf.oob_score_ - acc_2024:+.4f}")
print(f"Delta OOB -> 2025: {rf.oob_score_ - acc_2025:+.4f}")

print()
print("=== 2. Correlation features -> forward return ===")
closes = prices["Close"]
for horizon in [1, 6, 12, 24]:
    fwd_ret = closes.shift(-horizon) / closes - 1.0
    common_idx = train.index.intersection(fwd_ret.dropna().index)
    fwd = fwd_ret.loc[common_idx]
    corrs = {}
    for col in X_cols:
        feat = train[col].loc[common_idx]
        corrs[col] = np.corrcoef(feat, fwd)[0, 1]
    top5 = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"  Horizon {horizon:2d}h -- top5 |corr|:")
    for name, corr in top5:
        print(f"    {name:25s} {corr:+.4f}")