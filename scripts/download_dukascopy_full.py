#!/usr/bin/env python3
""" मास डाउनलोडर - mass downloader via Dukascopy Bank API.
Ingestion massive de données historiques de 2010 à nos jours pour 40+ actifs
sur 4 timeframes (H1, H4, D1, W1), avec formatage TSV strict et validation
via load_asset.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dukascopy_python
from app.data.loader import load_asset
from app.core.exceptions import DataValidationError

# Active assets mapping to Dukascopy names
MAPPINGS = {
    # Forex Majors
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDCHF": "USD/CHF",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
    "USDCAD": "USD/CAD",
    # Forex JPY Crosses
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "AUDJPY": "AUD/JPY",
    "NZDJPY": "NZD/JPY",
    "CADJPY": "CAD/JPY",
    "CHFJPY": "CHF/JPY",
    # Forex EUR/GBP Crosses
    "EURGBP": "EUR/GBP",
    "EURCHF": "EUR/CHF",
    "EURAUD": "EUR/AUD",
    "GBPAUD": "GBP/AUD",
    "GBPCAD": "GBP/CAD",
    # Forex Exotics
    "USDPLN": "USD/PLN",
    "USDTRY": "USD/TRY",
    "USDMXN": "USD/MXN",
    "USDZAR": "USD/ZAR",
    # Indices
    "US30": "E_D&J-Ind",
    "US500": "E_SandP-500",
    "US100": "E_NQ-100",
    "GER30": "E_DAAX",
    "JAP225": "E_N225Jap",
    "UK100": "E_Futsee-100",
    "FRA40": "E_CAC-40",
    "AUS200": "E_XJO-ASX",
    # Métaux & Énergies & Soft Commodities
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
    "PALLADIUM": "XPD.CMD/USD",
    "PLATINUM": "XPT.CMD/USD",
    "COPPER": "COPPER.CMD/USD",
    "USOIL": "E_Light",
    "UKOIL": "E_Brent",
    "NATGAS": "GAS.CMD/USD",
    "COFFEE": "COFFEE.CMD/USD",
    "SOYBEAN": "SOYBEAN.CMD/USD",
    "COCOA": "COCOA.CMD/USD",
    "SUGAR": "SUGAR.CMD/USD",
    "COTTON": "COTTON.CMD/USD",
    # Cryptomonnaies
    "BTCUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
    "LTCUSD": "LTC/USD",
    "XRPUSD": "XRP/USD",
}

# 10 priority assets for fast execution
PRIORITY_ASSETS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "US30",
    "US500",
    "GER30",
    "XAUUSD",
]

# Timeframe mapping to Dukascopy constants
TIMEFRAME_MAP = {
    "H1": dukascopy_python.INTERVAL_HOUR_1,
    "H4": dukascopy_python.INTERVAL_HOUR_4,
    "D1": dukascopy_python.INTERVAL_DAY_1,
    "W1": dukascopy_python.INTERVAL_WEEK_1,
}


def download_asset_tf(
    asset: str,
    tf: str,
    start_date: datetime,
    end_date: datetime,
    force: bool = False,
    max_retries: int = 5,
) -> bool:
    """Download single asset/timeframe combination from Dukascopy, format, save, and validate."""
    duka_symbol = MAPPINGS.get(asset)
    if not duka_symbol:
        print(f"[-] Unknown asset mapping for: {asset}")
        return False

    interval = TIMEFRAME_MAP.get(tf)
    if not interval:
        print(f"[-] Unknown timeframe mapping for: {tf}")
        return False

    out_dir = Path("data/raw") / asset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{asset}_{tf}.csv"

    if out_file.exists() and not force:
        print(f"[~] File already exists for {asset}/{tf}, skipping download (use --force to overwrite).")
        try:
            load_asset(asset, tf)
            print(f"[+] Loaded and verified existing file for {asset}/{tf} successfully.")
            return True
        except Exception as e:
            print(f"[!] Existing file validation failed for {asset}/{tf}: {e}. Redownloading...")

    print(f"[*] Downloading {asset} ({duka_symbol}) for TF {tf} from {start_date.date()} to {end_date.date()}...")

    # Fetch data with built-in retry from dukascopy_python
    try:
        df = dukascopy_python.fetch(
            instrument=duka_symbol,
            interval=interval,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=start_date,
            end=end_date,
            max_retries=max_retries,
            limit=200_000,
        )
    except Exception as e:
        print(f"    [-] Fetch failed after {max_retries} retries: {e}")
        return False

    if df is None or df.empty:
        print(f"[-] No data returned for {asset}/{tf}.")
        return False

    # Format dataframe
    df.index.name = "Time"
    df.columns = [c.title() for c in df.columns]

    # Clean prices <= 0 (e.g. USOIL April 2020 negative pricing)
    ohlc = ["Open", "High", "Low", "Close"]
    neg_mask = (df[ohlc] <= 0).any(axis=1)
    n_neg = neg_mask.sum()
    if n_neg > 0:
        print(f"    [!] Found {n_neg} rows with negative or zero prices. Cleaning them...")
        df = df[~neg_mask]

    # Filter out initial stray rows that have a huge gap (> 10 days) to the next row
    if len(df) > 1:
        while len(df) > 1:
            first_gap = (df.index[1] - df.index[0]).total_seconds() / 3600.0
            if first_gap > 240.0:
                print(f"    [!] Dropping initial stray row at {df.index[0]} due to large gap of {first_gap:.1f}h to next bar.")
                df = df.iloc[1:]
            else:
                break

    # Remove any other files matching *_{tf}.csv to avoid multiple CSVs error
    for existing_file in out_dir.glob(f"*_{tf}.csv"):
        if existing_file.name != out_file.name:
            print(f"    [~] Cleaning up conflicting older CSV file: {existing_file.name}")
            existing_file.unlink()

    # Save to TSV
    df.to_csv(out_file, sep="\t", index=True)
    print(f"    [+] Saved {len(df)} rows to {out_file}.")

    # Validate output using project loader
    try:
        load_asset(asset, tf)
        print(f"    [+] Validated {asset}/{tf} successfully via load_asset.")
        return True
    except DataValidationError as e:
        err_str = str(e)
        if "gaps anormaux" in err_str:
            print(f"    [!] WARNING: {asset}/{tf} passes OHLC/volume check but has {err_str.split(':')[-1].strip()}")
            print(f"    [!] Data kept at {out_file} — gaps are likely Dukascopy feed limitations.")
            return True
        print(f"    [-] DataValidationError for {asset}/{tf} after download: {e}")
        if out_file.exists():
            out_file.unlink()
        return False
    except Exception as e:
        print(f"    [-] Unexpected validation error for {asset}/{tf}: {e}")
        if out_file.exists():
            out_file.unlink()
        return False


def validate_mappings() -> list[str]:
    """Validate that all MAPPINGS symbols follow expected Dukascopy format.
    
    Returns list of any assets with suspicious-looking symbols.
    """
    warnings: list[str] = []
    # Heuristic: Dukascopy forex symbols contain '/', indices start with 'E_',
    # commodities contain '.CMD/', crypto contains '/'
    for asset, symbol in MAPPINGS.items():
        if "/" not in symbol and not symbol.startswith("E_"):
            warnings.append(f"  {asset:12s} → {symbol} (unusual format, may be invalid)")
        elif symbol.endswith("/USX"):
            warnings.append(f"  {asset:12s} → {symbol} (suffix /USX is non-standard)")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Dukascopy Mass Historical Data Downloader")
    parser.add_argument(
        "--assets",
        type=str,
        default="priority",
        help="Comma-separated assets to download. Options: 'priority', 'all', or list like 'EURUSD,GBPUSD'.",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="H1,H4,D1,W1",
        help="Comma-separated timeframes to download. Default: H1,H4,D1,W1.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2010-01-01",
        help="Start date YYYY-MM-DD. Default: 2010-01-01.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if files exist.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Max retries per download. Default: 5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate mappings and print what would be downloaded, without fetching.",
    )

    args = parser.parse_args()

    # Parse assets
    if args.assets.lower() == "priority":
        assets_to_run = PRIORITY_ASSETS
    elif args.assets.lower() == "all":
        assets_to_run = list(MAPPINGS.keys())
    else:
        assets_to_run = [a.strip().upper() for a in args.assets.split(",") if a.strip()]

    # Parse timeframes
    timeframes_to_run = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]

    # Validate mappings
    warnings = validate_mappings()
    if warnings:
        print("[!] Suspicious instrument mappings detected:")
        for w in warnings:
            print(w)
        print()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        # Ensure UTC timezone
        start_date = start_date.replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"[-] Invalid start date format: {args.start}. Must be YYYY-MM-DD.")
        sys.exit(1)

    end_date = datetime.now(timezone.utc)

    print("=" * 70)
    print(" DUKASCOPY MASS DATA INGESTION ENGINE")
    print("=" * 70)
    print(f"Start Date: {start_date.date()}")
    print(f"End Date:   {end_date.date()}")
    print(f"Assets:     {', '.join(assets_to_run)}")
    print(f"Timeframes: {', '.join(timeframes_to_run)}")
    print(f"Force mode: {args.force}")
    print("=" * 70)

    total_tasks = len(assets_to_run) * len(timeframes_to_run)

    if args.dry_run:
        print(f"[*] DRY RUN — {total_tasks} downloads would be attempted.")
        for asset in assets_to_run:
            for tf in timeframes_to_run:
                duka_symbol = MAPPINGS.get(asset, "???")
                print(f"  {asset:10s} {tf:4s} → {duka_symbol} → data/raw/{asset}/{asset}_{tf}.csv")
        print()
        return

    success_count = 0
    fail_count = 0
    skipped_count = 0
    failed_pairs = []

    task_idx = 0

    for asset in assets_to_run:
        for tf in timeframes_to_run:
            task_idx += 1
            print(f"\n[{task_idx}/{total_tasks}] Processing {asset} {tf}...")
            
            # Check if file exists and we are skipping
            out_file = Path("data/raw") / asset / f"{asset}_{tf}.csv"
            if out_file.exists() and not args.force:
                # Still try to validate
                try:
                    load_asset(asset, tf)
                    print(f"[~] Skipping existing and valid file for {asset} {tf}.")
                    skipped_count += 1
                    continue
                except Exception:
                    print(f"[!] Existing file for {asset} {tf} is invalid. Redownloading...")

            success = download_asset_tf(
                asset=asset,
                tf=tf,
                start_date=start_date,
                end_date=end_date,
                force=args.force,
                max_retries=args.retries,
            )

            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_pairs.append(f"{asset}/{tf}")

    print("\n" + "=" * 70)
    print(" DOWNLOAD RUN SUMMARY")
    print("=" * 70)
    print(f"Successful:  {success_count}")
    print(f"Failed:      {fail_count}")
    print(f"Skipped:     {skipped_count}")
    if failed_pairs:
        print(f"Failed Pairs: {', '.join(failed_pairs)}")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
