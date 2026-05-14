#!/usr/bin/env python3

"""Download 5-minute Coinbase spot candles for BTC and ETH."""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"
PRODUCTS = {
    "BTC": "BTC-USD",
    #"ETH": "ETH-USD",
}
GRANULARITY_SECONDS = 300
MAX_CANDLES_PER_REQUEST = 300
REQUEST_SLEEP_SECONDS = 0.20
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 6
CSV_COLUMNS = (
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "product_id",
    "source",
)


# -----------------------------------------------------------------------------
# Configuration
# Edit these values before running the script.
# -----------------------------------------------------------------------------
ASSETS = ("BTC")
YEARS_BACK = 10
START = None
END = None
SLEEP_SECONDS = REQUEST_SLEEP_SECONDS
OVERWRITE = False


def parse_utc_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SystemExit(f"Unsupported timestamp format: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def subtract_years(timestamp: datetime, years: int) -> datetime:
    try:
        return timestamp.replace(year=timestamp.year - years)
    except ValueError:
        return timestamp.replace(month=2, day=28, year=timestamp.year - years)


def compute_window() -> tuple[datetime, datetime]:
    end_utc = parse_utc_timestamp(END) if END else datetime.now(UTC)
    if START:
        start_utc = parse_utc_timestamp(START)
    else:
        start_utc = subtract_years(end_utc, YEARS_BACK)

    if start_utc >= end_utc:
        raise SystemExit("START must be earlier than END")

    return start_utc, end_utc


def iter_request_windows(start_utc: datetime, end_utc: datetime) -> Iterable[tuple[datetime, datetime]]:
    chunk_span = timedelta(seconds=GRANULARITY_SECONDS * MAX_CANDLES_PER_REQUEST)
    current = start_utc
    while current < end_utc:
        chunk_end = min(current + chunk_span, end_utc)
        yield current, chunk_end
        current = chunk_end


def make_request(product_id: str, start_utc: datetime, end_utc: datetime) -> list[list[object]]:
    params = urlencode(
        {
            "start": start_utc.isoformat().replace("+00:00", "Z"),
            "end": end_utc.isoformat().replace("+00:00", "Z"),
            "granularity": str(GRANULARITY_SECONDS),
        }
    )
    url = BASE_URL.format(product_id=product_id) + "?" + params
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "master-thesis-coinbase-downloader/1.0",
        },
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected Coinbase response for {product_id}: {payload!r}")
            return payload
        except HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES:
                backoff = min(2 ** (attempt - 1), 30)
                print(
                    f"[retry] {product_id} {start_utc.isoformat()} -> {end_utc.isoformat()} "
                    f"HTTP {exc.code}; sleeping {backoff}s",
                    file=sys.stderr,
                )
                time.sleep(backoff)
                continue
            raise
        except URLError:
            if attempt < MAX_RETRIES:
                backoff = min(2 ** (attempt - 1), 30)
                time.sleep(backoff)
                continue
            raise

    raise RuntimeError(f"Failed to fetch candles for {product_id}")


def normalize_rows(product_id: str, payload: list[list[object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, list) or len(item) < 6:
            continue
        timestamp = datetime.fromtimestamp(int(item[0]), tz=UTC)
        rows.append(
            {
                "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "low": item[1],
                "high": item[2],
                "open": item[3],
                "close": item[4],
                "volume": item[5],
                "product_id": product_id,
                "source": "coinbase_exchange_spot",
            }
        )
    rows.sort(key=lambda row: row["timestamp_utc"])
    return rows


def collect_candles(
    product_id: str,
    start_utc: datetime,
    end_utc: datetime,
    sleep_seconds: float,
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    windows = list(iter_request_windows(start_utc, end_utc))
    total_windows = len(windows)

    for index, (window_start, window_end) in enumerate(windows, start=1):
        print(
            f"[{product_id}] request {index}/{total_windows}: "
            f"{window_start.isoformat()} -> {window_end.isoformat()}",
            file=sys.stderr,
        )
        payload = make_request(product_id, window_start, window_end)
        rows = normalize_rows(product_id, payload)
        for row in rows:
            merged[str(row["timestamp_utc"])] = row
        if sleep_seconds > 0 and index < total_windows:
            time.sleep(sleep_seconds)

    ordered = [merged[key] for key in sorted(merged)]
    return ordered


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    *,
    asset: str,
    product_id: str,
    start_utc: datetime,
    end_utc: datetime,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    timestamps = [str(row["timestamp_utc"]) for row in rows]
    metadata = {
        "asset": asset,
        "product_id": product_id,
        "source": "coinbase_exchange_spot",
        "granularity_seconds": GRANULARITY_SECONDS,
        "requested_start_utc": start_utc.isoformat().replace("+00:00", "Z"),
        "requested_end_utc": end_utc.isoformat().replace("+00:00", "Z"),
        "observed_start_utc": timestamps[0] if timestamps else None,
        "observed_end_utc": timestamps[-1] if timestamps else None,
        "row_count": len(rows),
        "downloaded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def ensure_writable_target(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing file: {path}\n"
            "Re-run with --overwrite to replace it."
        )


def main() -> None:
    assets = ASSETS or tuple(sorted(PRODUCTS))
    invalid_assets = [asset for asset in assets if asset not in PRODUCTS]
    if invalid_assets:
        raise SystemExit(f"Unsupported assets in ASSETS: {invalid_assets}")

    start_utc, end_utc = compute_window()

    for asset in assets:
        product_id = PRODUCTS[asset]
        asset_dir = SCRIPT_DIR / asset
        csv_path = asset_dir / f"{asset}_USD_coinbase_spot_5min.csv"
        metadata_path = asset_dir / f"{asset}_USD_coinbase_spot_5min.metadata.json"

        ensure_writable_target(csv_path, OVERWRITE)
        ensure_writable_target(metadata_path, OVERWRITE)

        rows = collect_candles(
            product_id=product_id,
            start_utc=start_utc,
            end_utc=end_utc,
            sleep_seconds=SLEEP_SECONDS,
        )
        write_csv(rows, csv_path)
        write_metadata(
            asset=asset,
            product_id=product_id,
            start_utc=start_utc,
            end_utc=end_utc,
            rows=rows,
            output_path=metadata_path,
        )
        print(
            f"[done] {asset} -> {csv_path} ({len(rows)} rows)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
