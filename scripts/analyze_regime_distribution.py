#!/usr/bin/env python3
"""Analyse de la distribution des régimes (trend / range / vol_high) — Phase F4.

Pour chaque actif disponible dans data/raw/ avec un timeframe D1 valide :
    1. Charge le CSV via app.data.loader.load_asset.
    2. Applique app.features.regime.detect_regime (paramètres par défaut MVP).
    3. Calcule la part de chaque régime (% des barres non-warmup).
    4. Écrit data/analysis/regime_distribution.csv et imprime un tableau Markdown
       réutilisable dans docs/regime_analysis.md.

Usage :
    python scripts/analyze_regime_distribution.py
    python scripts/analyze_regime_distribution.py --tf H4
    python scripts/analyze_regime_distribution.py --assets EURUSD GBPUSD US30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.exceptions import DataValidationError
from app.core.logging import get_logger
from app.data.loader import load_asset
from app.features.regime import detect_regime

logger = get_logger(__name__)

DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_OUT = Path("data/analysis/regime_distribution.csv")


def list_assets_with_tf(data_root: Path, tf: str) -> list[str]:
    """Retourne la liste des dossiers d'actifs contenant un CSV `*_<tf>.csv`."""
    if not data_root.exists():
        return []
    assets: list[str] = []
    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        if list(sub.glob(f"*_{tf}.csv")):
            assets.append(sub.name)
    return assets


def analyze_asset(asset: str, tf: str, data_root: Path) -> dict[str, object] | None:
    """Charge un actif/TF et calcule la distribution des régimes.

    Returns:
        Dict avec asset, tf, n_bars, n_classified, pct_trend, pct_range,
        pct_vol_high, start, end. None si échec de validation.
    """
    try:
        df = load_asset(asset, tf, data_root=data_root)
    except (DataValidationError, FileNotFoundError) as exc:
        logger.warning("skip_asset", extra={"context": {"asset": asset, "tf": tf, "err": str(exc)}})
        return None

    regime = detect_regime(df)
    classified = regime.dropna()
    n_total = len(regime)
    n_class = len(classified)
    if n_class == 0:
        return None

    counts = classified.value_counts()
    pct = {label: float(counts.get(label, 0)) / n_class for label in ("trend", "range", "vol_high")}

    return {
        "asset": asset,
        "tf": tf,
        "n_bars": n_total,
        "n_classified": n_class,
        "pct_trend": round(pct["trend"], 4),
        "pct_range": round(pct["range"], 4),
        "pct_vol_high": round(pct["vol_high"], 4),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
    }


def print_markdown_table(rows: list[dict[str, object]]) -> None:
    """Imprime un tableau Markdown trié par actif."""
    if not rows:
        print("Aucun actif analysé.")
        return
    print()
    print(f"| Actif | TF | Barres | Trend % | Range % | Vol_high % | Période |")
    print(f"|---|---|---:|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda x: str(x["asset"])):
        print(
            f"| {r['asset']} | {r['tf']} | {r['n_classified']} "
            f"| {float(r['pct_trend']) * 100:.1f} "
            f"| {float(r['pct_range']) * 100:.1f} "
            f"| {float(r['pct_vol_high']) * 100:.1f} "
            f"| {r['start']} → {r['end']} |"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf", default="D1", choices=["D1", "H4", "H1", "W1"])
    parser.add_argument("--assets", nargs="*", default=None,
                        help="Sous-liste d'actifs (défaut : tous ceux trouvés)")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), type=Path)
    parser.add_argument("--out", default=str(DEFAULT_OUT), type=Path)
    args = parser.parse_args()

    if args.assets:
        assets = args.assets
    else:
        assets = list_assets_with_tf(args.data_root, args.tf)

    if not assets:
        print(f"Aucun actif trouvé dans {args.data_root} pour tf={args.tf}.")
        return 1

    print(f"[F4] Analyse {len(assets)} actifs sur {args.tf}…")
    rows: list[dict[str, object]] = []
    for asset in assets:
        row = analyze_asset(asset, args.tf, args.data_root)
        if row is not None:
            rows.append(row)
            print(
                f"  ✓ {asset:<10} n={row['n_classified']:>6}  "
                f"trend={float(row['pct_trend']) * 100:5.1f}%  "
                f"range={float(row['pct_range']) * 100:5.1f}%  "
                f"vol_high={float(row['pct_vol_high']) * 100:5.1f}%"
            )
        else:
            print(f"  ✗ {asset:<10} (skip)")

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nCSV écrit : {out_path}")

    print_markdown_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
