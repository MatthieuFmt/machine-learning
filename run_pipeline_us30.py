"""Script de lancement — US30 D1 V2 Hypothesis 01.

Usage :
    python run_pipeline_us30.py

Protocole anti-snooping : UN SEUL REGARD. Ne pas modifier après exécution.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH si nécessaire
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def setup_logging() -> None:
    """Configure le logging structuré."""
    from app.core.logging import setup_logging as _setup

    _setup(level=logging.INFO)


def main() -> int:
    setup_logging()

    from app.core.logging import get_logger
    from app.pipelines.us30 import Us30Pipeline

    logger = get_logger("run_pipeline_us30")

    logger.info("=" * 60)
    logger.info("US30 D1 — V2 Hypothesis 01")
    logger.info("Train ≤ 2022 | Val 2023 | Test 2024-2025")
    logger.info("=" * 60)

    # ── Vérification préalable des données ──
    d1_path = Path("cleaned-data/USA30IDXUSD_D1_cleaned.csv")
    h4_path = Path("cleaned-data/USA30IDXUSD_H4_cleaned.csv")

    if not d1_path.exists():
        logger.warning(
            "%s absent — tentative de génération via inspect_us30_csv.py",
            d1_path,
        )
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/inspect_us30_csv.py"],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            logger.error(
                "Données US30 indisponibles. Fallback suggéré : "
                "XAUUSD H4 ou GER30 D1. Vérifier data/ ou fournir les CSV."
            )
            return 1

    if not d1_path.exists():
        logger.error(
            "Données US30 indisponibles. Fallback suggéré : "
            "XAUUSD H4 ou GER30 D1. Vérifier data/ ou fournir les CSV."
        )
        return 1

    # ── Pipeline ──
    pipeline = Us30Pipeline()

    try:
        results = pipeline.run()
    except FileNotFoundError as exc:
        logger.error("Échec chargement données : %s", exc)
        logger.error(
            "Exécuter d'abord : python scripts/inspect_us30_csv.py"
        )
        return 1
    except Exception as exc:
        logger.error("Échec pipeline : %s", exc, exc_info=True)
        return 1

    # ── Rapport ──
    output_path = "predictions/us30_metrics_v2_01.json"
    pipeline.save_report(
        metrics=results["metrics"],
        trades=results["trades"],
        path=output_path,
    )

    # ── Affichage Sharpe OOS ──
    with open(output_path) as f:
        report = json.load(f)

    sharpe = report.get("sharpe", 0.0)
    print()
    print("=" * 60)
    print(f"Sharpe 2024-2025 : {sharpe:.3f}")
    print("=" * 60)

    # Métriques par année
    for year_str, m in report.get("metrics", {}).items():
        print(
            f"  {year_str} : Sharpe={m.get('sharpe', 0):.3f}, "
            f"WR={m.get('win_rate', 0):.1f}%, "
            f"Trades={m.get('trades', 0)}, "
            f"PnL={m.get('profit_net', 0):.1f} pips"
        )

    # GO / NO-GO
    breakeven_wr = 37.0  # ≈ 37% avec TP=200/SL=100/friction=8
    test_metrics = {}
    for year_str, m in report.get("metrics", {}).items():
        if int(year_str) >= 2024:
            test_metrics[year_str] = m

    if test_metrics:
        # Moyenne pondérée des WR sur test
        total_trades = sum(m["trades"] for m in test_metrics.values())
        if total_trades > 0:
            avg_wr = sum(
                m["win_rate"] * m["trades"] for m in test_metrics.values()
            ) / total_trades
        else:
            avg_wr = 0.0

        print()
        if sharpe > 0 and avg_wr > breakeven_wr:
            print(">>> GO : Sharpe > 0 ET WR > breakeven → CPCV")
        elif sharpe > 0:
            print(
                f">>> ATTENTION : Sharpe > 0 mais WR={avg_wr:.1f}% <= "
                f"breakeven WR={breakeven_wr:.1f}%"
            )
        else:
            print(">>> NO-GO : Sharpe <= 0 -> passer a v2-02 (XAUUSD H4)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
