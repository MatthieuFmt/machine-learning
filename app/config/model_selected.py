"""FROZEN après pivot v4 A7 (3 entrées) + C3 (extension multi-actifs).

NE PAS MODIFIER MANUELLEMENT. Seules les phases A7 / C3 peuvent y ajouter.
"""
from __future__ import annotations

MODEL_SELECTED: dict[tuple[str, str], str] = {
    ('US30', 'D1'): 'rf',
    ('EURUSD', 'H4'): 'rf',
    ('XAUUSD', 'D1'): 'stacking',
    ('BTCUSD', 'D1'): 'hgbm',
    ('ETHUSD', 'D1'): 'hgbm',
    ('ETHUSD', 'H4'): 'hgbm',
    ('ETHUSD', 'H1'): 'hgbm',
    ('EURUSD', 'D1'): 'stacking',
    ('GBPUSD', 'D1'): 'rf',
    ('GBPUSD', 'H4'): 'rf',
    ('USDCHF', 'D1'): 'stacking',
    ('USDCHF', 'H4'): 'rf',
}
