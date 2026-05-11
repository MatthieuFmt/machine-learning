"""Package model — entraînement, évaluation, prédiction RandomForest."""

from learning_machine_learning.model.training import train_test_split_purge, train_model
from learning_machine_learning.model.evaluation import (
    evaluate_model,
    feature_importance_impurity,
    feature_importance_permutation,
)
from learning_machine_learning.model.prediction import predict_oos

__all__ = [
    "train_test_split_purge",
    "train_model",
    "evaluate_model",
    "feature_importance_impurity",
    "feature_importance_permutation",
    "predict_oos",
]
