"""FROZEN après pivot v4 A8 (3 entrées) + C4 (extension multi-actifs).

NE PAS MODIFIER MANUELLEMENT. Seules les phases A8 / C4 peuvent y ajouter.
"""
from __future__ import annotations

HYPERPARAMS_TUNED: dict[tuple[str, str], dict] = {
    ('US30', 'D1'): {
        'model': 'rf',
        'params': {'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 100},
        'threshold': 0.55,
        'expected_sharpe_outer': 1.913,
        'expected_wr': 0.575,
    },
    ('EURUSD', 'H4'): {
        'model': 'rf',
        'params': {'max_depth': 6, 'min_samples_leaf': 10, 'n_estimators': 100},
        'threshold': 0.55,
        'expected_sharpe_outer': 0.592,
        'expected_wr': 0.515,
    },
    ('XAUUSD', 'D1'): {
        'model': 'stacking',
        'params': {},
        'threshold': 0.5,
        'expected_sharpe_outer': 0.0,
        'expected_wr': 0.0,
    },
    ('ETHUSD', 'D1'): {
        'model': 'hgbm',
        'params': {'learning_rate': 0.05, 'max_depth': 3, 'max_leaf_nodes': 15, 'min_samples_leaf': 20},
        'threshold': 0.5,
        'expected_sharpe_outer': 1.7001390048285607,
        'expected_wr': 0.6504095904095905,
    },
    ('ETHUSD', 'H4'): {
        'model': 'hgbm',
        'params': {'learning_rate': 0.05, 'max_depth': 3, 'max_leaf_nodes': 15, 'min_samples_leaf': 50},
        'threshold': 0.6,
        'expected_sharpe_outer': 0.3875050021743681,
        'expected_wr': 0.44386911048980016,
    },
    ('ETHUSD', 'H1'): {
        'model': 'hgbm',
        'params': {'learning_rate': 0.05, 'max_depth': 6, 'max_leaf_nodes': 15, 'min_samples_leaf': 50},
        'threshold': 0.5,
        'expected_sharpe_outer': 1.8118526330187188,
        'expected_wr': 0.4738120606533406,
    },
    ('EURUSD', 'D1'): {
        'model': 'stacking',
        'params': {},
        'threshold': 0.5,
        'expected_sharpe_outer': 0.0,
        'expected_wr': 0.0,
    },
    ('GBPUSD', 'D1'): {
        'model': 'rf',
        'params': {'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 200},
        'threshold': 0.5,
        'expected_sharpe_outer': 7.81755903584857,
        'expected_wr': 0.7926498217344192,
    },
    ('GBPUSD', 'H4'): {
        'model': 'rf',
        'params': {'max_depth': 10, 'min_samples_leaf': 10, 'n_estimators': 100},
        'threshold': 0.5,
        'expected_sharpe_outer': 3.4505556892418285,
        'expected_wr': 0.53303663003663,
    },
    ('USDCHF', 'D1'): {
        'model': 'stacking',
        'params': {},
        'threshold': 0.5,
        'expected_sharpe_outer': 0.0,
        'expected_wr': 0.0,
    },
    ('USDCHF', 'H4'): {
        'model': 'rf',
        'params': {'max_depth': 3, 'min_samples_leaf': 10, 'n_estimators': 100},
        'threshold': 0.6,
        'expected_sharpe_outer': 1.1663129642828287,
        'expected_wr': 0.2979013739883305,
    },
}
