"""Package principal du pipeline ML/trading multi-actif.

Architecture :
    config/   — dataclasses de configuration par domaine
    core/     — types, logging, exceptions
    data/     — ingestion, nettoyage, validation
    features/ — feature engineering, triple barrier, merge multi-TF
    model/    — entraînement RandomForest, évaluation, prédiction
    backtest/ — simulation stateful, filtres, métriques, sizing, reporting
    analysis/ — diagnostics post-backtest, audit look-ahead
    pipelines/— orchestrateurs par instrument (EurUsdPipeline, BtcUsdPipeline)
"""
