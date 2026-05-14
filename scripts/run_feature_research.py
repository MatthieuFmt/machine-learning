"""Script CLI de recherche de features (Prompt 04).

Usage :
    python scripts/run_feature_research.py --asset US30 --tf D1 --horizon 5 --n-top 15
"""

from __future__ import annotations

import argparse
import sys

from app.core.logging import get_logger, setup_logging
from app.core.seeds import set_global_seeds
from app.features.research import rank_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank features by predictive power (Prompt 04 — Feature Research Harness)",
    )
    parser.add_argument(
        "--asset", required=True,
        help="Asset name in data/raw/ (ex: US30, XAUUSD)",
    )
    parser.add_argument(
        "--tf", required=True,
        choices=["D1", "H4", "H1", "M15", "M5"],
        help="Timeframe",
    )
    parser.add_argument(
        "--horizon", type=int, default=5,
        help="Forward return horizon in bars (default: 5)",
    )
    parser.add_argument(
        "--n-top", type=int, default=20,
        help="Number of top features to display (default: 20)",
    )
    parser.add_argument(
        "--train-end", default="2022-12-31",
        help="Train cutoff date YYYY-MM-DD (default: 2022-12-31)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds()
    setup_logging()
    logger = get_logger(__name__)

    try:
        result = rank_features(
            asset=args.asset,
            tf=args.tf,
            target_horizon=args.horizon,
            n_top=args.n_top,
            train_end=args.train_end,
        )
    except Exception as exc:
        logger.error("rank_features_failed", extra={"context": {
            "asset": args.asset,
            "tf": args.tf,
            "error": str(exc),
        }})
        sys.exit(1)

    # Affichage console
    print(f"\nTop {len(result)} features for {args.asset} {args.tf} "
          f"(horizon={args.horizon}, train_end={args.train_end}):")
    print("-" * 70)
    print(
        result[["feature", "composite_rank", "mutual_info", "abs_corr"]]
        .to_string(index=False)
    )

    logger.info("cli_done", extra={"context": {
        "asset": args.asset,
        "tf": args.tf,
        "n_features_ranked": len(result),
    }})


if __name__ == "__main__":
    main()
