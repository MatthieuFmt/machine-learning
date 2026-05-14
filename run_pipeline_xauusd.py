"""Script de lancement — XAUUSD H4 V2 Hypothesis 02.

Usage :
    python run_pipeline_xauusd.py

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
    from app.pipelines.xauusd import XauUsdPipeline

    logger = get_logger("run_pipeline_xauusd")

    logger.info("=" * 60)
    logger.info("XAUUSD H4 — V2 Hypothesis 02")
    logger.info("Train ≤ 2022 | Val 2023 | Test 2024-2025")
    logger.info("=" * 60)

    # ── Vérification préalable des données ──
    h4_path = Path("cleaned-data/XAUUSD_H4_cleaned.csv")
    d1_path = Path("cleaned-data/XAUUSD_D1_cleaned.csv")

    if not h4_path.exists() or not d1_path.exists():
        logger.warning(
            "Données cleaned absentes — tentative de génération via "
            "inspect_xauusd_csv.py"
        )
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/inspect_xauusd_csv.py"],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            logger.error(
                "Données XAUUSD indisponibles. Vérifier data/ ou fournir "
                "XAUUSD_H4.csv et XAUUSD_D1.csv."
            )
            return 1

    if not h4_path.exists():
        logger.error(
            "Données XAUUSD H4 indisponibles. Vérifier data/ ou fournir "
            "XAUUSD_H4.csv."
        )
        return 1

    # ── Pipeline ──
    pipeline = XauUsdPipeline()

    try:
        results = pipeline.run()
    except FileNotFoundError as exc:
        logger.error("Échec chargement données : %s", exc)
        logger.error(
            "Exécuter d'abord : python scripts/inspect_xauusd_csv.py"
        )
        return 1
    except Exception as exc:
        logger.error("Échec pipeline : %s", exc, exc_info=True)
        return 1

    # ── Rapport ──
    output_path = "predictions/xauusd_metrics_v2_02.json"
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
    # Breakeven WR ≈ 37% avec TP=300/SL=150/friction=35 (commission=25 + slippage=10)
    breakeven_wr = 37.0
    test_metrics = {}
    for year_str, m in report.get("metrics", {}).items():
        if int(year_str) >= 2024:
            test_metrics[year_str] = m

    if test_metrics:
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
            print(">>> NO-GO : Sharpe <= 0 -> passer a H03")

    return 0


if __name__ == "__main__":
    sys.exit(main())
