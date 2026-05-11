"""Features macro — rendements des instruments correles.

Chaque instrument macro (XAUUSD, USDCHF) produit une colonne de log-return.
Ces returns sont fusionnes sur H1 via merge_asof dans le merger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_macro_return(close: pd.Series, name: str) -> pd.DataFrame:
    """Calcule le log-return d'un instrument macro.

    Args:
        close: Serie Close de l'instrument macro.
        name: Nom de la colonne de sortie (ex: 'XAU_Return').

    Returns:
        DataFrame avec une colonne nommee `name`.
    """
    result = pd.DataFrame(index=close.index)
    result[name] = np.log(close / close.shift(1))
    return result
