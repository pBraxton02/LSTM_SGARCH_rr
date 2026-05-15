#!/usr/bin/env python3

"""Preprocess BTC Coinbase spot candles for LSTM/SGARCH experiments."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "BTC" / "BTC_USD_coinbase_spot_5min.csv"
DEFAULT_OUTPUT = (
    SCRIPT_DIR / "data" / "BTC" / "BTC_USD_coinbase_spot_daily_features.csv"
)

REQUIRED_COLUMNS = {
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "product_id",
    "source",
}
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMNS = {"target_log_return_next", "target_direction_next"}
NON_FEATURE_COLUMNS = {
    "timestamp_utc",
    "product_id",
    "source",
    "split",
    "observations",
    "expected_observations",
    "coverage_ratio",
    *TARGET_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing pipeline.

    params:
        None

    returns:
        argparse.Namespace with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess BTC 5-minute candles into model-ready features."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Defaults to OUTPUT with .metadata.json suffix.",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=None,
        help="Defaults to OUTPUT with .scaler.pkl suffix when scaling is enabled.",
    )
    parser.add_argument(
        "--interval",
        default="1D",
        help="Fixed pandas interval for resampling, e.g. 1h or 1D.",
    )
    parser.add_argument(
        "--base-frequency",
        default="5min",
        help="Frequency of the raw candles, used for coverage checks.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="Drop resampled periods with less raw-candle coverage than this ratio.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="+",
        default=[7, 14, 30],
        help="Rolling feature windows measured in resampled periods.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[1, 2, 3, 7],
        help="Lagged feature windows measured in resampled periods.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=1,
        help="Number of periods ahead for the target return.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Chronological train fraction used to fit scalers.",
    )
    parser.add_argument(
        "--scale",
        choices=["none", "standard", "minmax"],
        default="standard",
        help="Scale numeric model features after feature engineering.",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="Keep rows with feature/target NaNs instead of dropping them.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO 8601 string with a Z suffix.

    params:
        None

    returns:
        UTC timestamp string in the format YYYY-MM-DDTHH:MM:SS.ffffffZ.
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metadata_path_for(output_path: Path) -> Path:
    """
    Derive the metadata JSON path from the output CSV path.

    params:
        output_path: Path to the output CSV file.

    returns:
        Path with the suffix replaced by .metadata.json.
    """
    return output_path.with_suffix(".metadata.json")


def scaler_path_for(output_path: Path) -> Path:
    """
    Derive the scaler pickle path from the output CSV path.

    params:
        output_path: Path to the output CSV file.

    returns:
        Path with the suffix replaced by .scaler.pkl.
    """
    return output_path.with_suffix(".scaler.pkl")


def timedelta_ratio(numerator: str, denominator: str) -> int:
    """
    Return the integer ratio of two pandas-parseable timedelta strings.

    params:
        numerator: The larger timedelta string (e.g. "1D").
        denominator: The smaller timedelta string (e.g. "5min").

    returns:
        Integer ratio of numerator to denominator.

    raises:
        SystemExit if either value is non-positive or the ratio is not a whole number.
    """
    numerator_delta = pd.Timedelta(numerator)
    denominator_delta = pd.Timedelta(denominator)
    if numerator_delta <= pd.Timedelta(0):
        raise SystemExit("--interval must be positive")
    if denominator_delta <= pd.Timedelta(0):
        raise SystemExit("--base-frequency must be positive")
    ratio = numerator_delta / denominator_delta
    if ratio < 1 or not float(ratio).is_integer():
        raise SystemExit(
            "--interval must be an integer multiple of --base-frequency. "
            "Use fixed intervals such as 1h, 4h, or 1D."
        )
    return int(ratio)


def load_raw_candles(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load, validate, and clean raw OHLCV candles from a CSV file.

    params:
        path: Path to the raw candles CSV file.

    returns:
        Tuple of (cleaned DataFrame indexed by timestamp_utc, profiling metadata dict).

    raises:
        SystemExit if required columns are missing, close prices are non-positive,
        or OHLC values are internally inconsistent.
    """
    df = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise SystemExit(f"Missing required columns in {path}: {missing_columns}")

    raw_rows = len(df)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    for column in OHLCV_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_values = df[list(REQUIRED_COLUMNS)].isna().sum().to_dict()
    numeric_na_rows = int(df[OHLCV_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=["timestamp_utc", *OHLCV_COLUMNS])
    df = df.sort_values("timestamp_utc")

    duplicate_timestamps = int(df["timestamp_utc"].duplicated(keep="last").sum())
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="last")

    invalid_close_rows = int((df["close"] <= 0).sum())
    if invalid_close_rows:
        raise SystemExit(f"Found {invalid_close_rows} rows with non-positive close.")

    invalid_ohlc_rows = int(
        (
            (df["low"] > df[["open", "close"]].min(axis=1))
            | (df["high"] < df[["open", "close"]].max(axis=1))
        ).sum()
    )
    if invalid_ohlc_rows:
        raise SystemExit(f"Found {invalid_ohlc_rows} rows with inconsistent OHLC.")

    df = df.set_index("timestamp_utc")
    df.index.name = "timestamp_utc"

    profile = {
        "raw_rows": raw_rows,
        "rows_after_cleaning": int(len(df)),
        "dropped_numeric_na_rows": numeric_na_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_values": {key: int(value) for key, value in missing_values.items()},
        "observed_start_utc": df.index.min().isoformat().replace("+00:00", "Z"),
        "observed_end_utc": df.index.max().isoformat().replace("+00:00", "Z"),
    }
    return df, profile


def add_gap_profile(
    metadata: dict[str, Any], df: pd.DataFrame, base_frequency: str
) -> None:
    """
    Augment metadata with gap statistics computed from the raw candle index.

    params:
        metadata: Dict to update in-place with gap statistics.
        df: DataFrame with a DatetimeIndex at base_frequency cadence.
        base_frequency: Expected candle frequency string (e.g. "5min").

    returns:
        None
    """
    expected_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=base_frequency,
        tz="UTC",
    )
    gaps = df.index.to_series().diff().dropna().value_counts().head(10)
    metadata["expected_base_frequency_rows"] = int(len(expected_index))
    metadata["missing_base_frequency_rows"] = int(len(expected_index) - len(df.index))
    metadata["largest_gap_seconds"] = (
        int(df.index.to_series().diff().dropna().max().total_seconds())
        if len(df) > 1
        else 0
    )
    top_gap_seconds: dict[str, int] = {}
    for delta, count in gaps.items():
        if isinstance(delta, pd.Timedelta):
            top_gap_seconds[str(int(delta.total_seconds()))] = int(count)
    metadata["top_gap_seconds"] = top_gap_seconds


def resample_candles(
    df: pd.DataFrame,
    interval: str,
    base_frequency: str,
    min_coverage: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Resample raw candles to a coarser interval and filter by coverage.

    params:
        df: Cleaned raw candles DataFrame indexed by timestamp_utc.
        interval: Target resampling interval (e.g. "1D", "1h").
        base_frequency: Frequency of the raw candles (e.g. "5min").
        min_coverage: Minimum fraction of expected raw candles required to keep a period.

    returns:
        Tuple of (resampled DataFrame, resampling profile metadata dict).
    """
    expected_observations = timedelta_ratio(interval, base_frequency)
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "product_id": "first",
        "source": "first",
    }
    # Pandas accepts column->aggregation mappings here; cast for stub compatibility.
    resampled = df.resample(interval, closed="left", label="left").agg(
        cast(Any, aggregations)
    )
    observations = df["close"].resample(interval, closed="left", label="left").count()
    resampled["observations"] = observations.astype("int64")
    resampled["expected_observations"] = expected_observations
    resampled["coverage_ratio"] = (
        resampled["observations"] / resampled["expected_observations"]
    )

    before_drop = len(resampled)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled = resampled[resampled["coverage_ratio"] >= min_coverage]

    profile = {
        "interval": interval,
        "base_frequency": base_frequency,
        "expected_observations_per_period": expected_observations,
        "min_coverage": min_coverage,
        "resampled_rows_before_coverage_filter": int(before_drop),
        "resampled_rows_after_coverage_filter": int(len(resampled)),
        "dropped_resampled_rows": int(before_drop - len(resampled)),
    }
    return resampled, profile


def add_features(
    df: pd.DataFrame,
    rolling_windows: list[int],
    lags: list[int],
    forecast_horizon: int,
) -> pd.DataFrame:
    """
    Engineer model features and target columns from resampled OHLCV data.

    params:
        df: Resampled OHLCV DataFrame indexed by timestamp_utc.
        rolling_windows: List of window sizes (in periods) for rolling statistics.
        lags: List of lag sizes (in periods) for lagged return and volume features.
        forecast_horizon: Number of periods ahead for the target log return.

    returns:
        DataFrame with all original columns plus engineered features and targets.
    """
    featured = df.copy()
    featured["log_close"] = np.log(featured["close"])
    featured["log_return"] = featured["log_close"].diff()
    featured["simple_return"] = featured["close"].pct_change(fill_method=None)
    featured["close_open_log_return"] = np.log(featured["close"] / featured["open"])
    featured["high_low_range"] = (featured["high"] - featured["low"]) / featured[
        "close"
    ]
    featured["volume_log"] = np.log1p(featured["volume"])

    for window in rolling_windows:
        featured[f"return_mean_{window}"] = (
            featured["log_return"].rolling(window).mean()
        )
        featured[f"return_volatility_{window}"] = (
            featured["log_return"].rolling(window).std()
        )
        featured[f"close_sma_{window}"] = featured["close"].rolling(window).mean()
        featured[f"close_ema_{window}"] = (
            featured["close"].ewm(span=window, adjust=False).mean()
        )
        featured[f"volume_log_mean_{window}"] = (
            featured["volume_log"].rolling(window).mean()
        )

    for lag in lags:
        featured[f"log_return_lag_{lag}"] = featured["log_return"].shift(lag)
        featured[f"volume_log_lag_{lag}"] = featured["volume_log"].shift(lag)

    target = featured["log_return"].shift(-forecast_horizon)
    featured["target_log_return_next"] = target
    featured["target_direction_next"] = np.where(target.isna(), np.nan, target > 0)

    return featured


def assign_chronological_split(df: pd.DataFrame, train_size: float) -> pd.DataFrame:
    """
    Label rows as "train" or "test" based on chronological order.

    params:
        df: Feature DataFrame sorted chronologically.
        train_size: Fraction of rows assigned to the train split (0 < train_size < 1).

    returns:
        DataFrame with an added "split" column containing "train" or "test" labels.

    raises:
        SystemExit if train_size is not strictly between 0 and 1.
    """
    if not 0 < train_size < 1:
        raise SystemExit("--train-size must be between 0 and 1.")
    split_index = int(len(df) * train_size)
    split = np.where(np.arange(len(df)) < split_index, "train", "test")
    result = df.copy()
    result["split"] = split
    return result


def feature_columns_for_scaling(df: pd.DataFrame) -> list[str]:
    """
    Identify numeric columns that should be scaled.

    params:
        df: Feature DataFrame including metadata and target columns.

    returns:
        List of column names that are numeric and not in NON_FEATURE_COLUMNS.
    """
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns
    return [column for column in numeric_columns if column not in NON_FEATURE_COLUMNS]


def scale_features(
    df: pd.DataFrame,
    method: str,
    train_size: float,
) -> tuple[pd.DataFrame, dict[str, Any], Any | None]:
    """
    Scale numeric model features using a scaler fitted on the train split only.

    params:
        df: Feature DataFrame before scaling.
        method: Scaling method — one of "none", "standard", or "minmax".
        train_size: Fraction of rows used as the train split for fitting the scaler.

    returns:
        Tuple of (scaled DataFrame, scaling metadata dict, fitted scaler or None).
    """
    df = assign_chronological_split(df, train_size)
    feature_columns = feature_columns_for_scaling(df)
    if method == "none":
        return df, {"scale": "none", "scaled_columns": []}, None

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    train_mask = df["split"] == "train"
    scaler.fit(df.loc[train_mask, feature_columns])

    scaled = df.copy()
    scaled.loc[:, feature_columns] = scaler.transform(scaled[feature_columns])
    metadata = {
        "scale": method,
        "scaled_columns": feature_columns,
        "scaler_fit_rows": int(train_mask.sum()),
    }
    return scaled, metadata, scaler


def write_outputs(
    df: pd.DataFrame,
    metadata: dict[str, Any],
    scaler: Any | None,
    output_path: Path,
    metadata_output_path: Path,
    scaler_output_path: Path | None,
) -> None:
    """
    Write the processed DataFrame, metadata JSON, and optional scaler pickle to disk.

    params:
        df: Processed feature DataFrame to save as CSV.
        metadata: Provenance and statistics dict to save as JSON.
        scaler: Fitted scaler object to serialise, or None if scaling was skipped.
        output_path: Destination path for the output CSV.
        metadata_output_path: Destination path for the metadata JSON.
        scaler_output_path: Destination path for the scaler pickle, or None.

    returns:
        None
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    if scaler_output_path is not None:
        scaler_output_path.parent.mkdir(parents=True, exist_ok=True)

    df.reset_index().to_csv(output_path, index=False)
    metadata_output_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    if scaler is not None and scaler_output_path is not None:
        with scaler_output_path.open("wb") as handle:
            pickle.dump(scaler, handle)


def main() -> None:
    """
    Entry point: orchestrate argument parsing, data loading, feature engineering,
    scaling, and output writing.

    params:
        None

    returns:
        None
    """
    args = parse_args()
    metadata_output = args.metadata_output or metadata_path_for(args.output)
    scaler_output = args.scaler_output or scaler_path_for(args.output)

    raw, metadata = load_raw_candles(args.input)
    add_gap_profile(metadata, raw, args.base_frequency)

    resampled, resample_metadata = resample_candles(
        raw,
        interval=args.interval,
        base_frequency=args.base_frequency,
        min_coverage=args.min_coverage,
    )
    metadata.update(resample_metadata)

    featured = add_features(
        resampled,
        rolling_windows=args.rolling_windows,
        lags=args.lags,
        forecast_horizon=args.forecast_horizon,
    )

    rows_before_na_drop = len(featured)
    if not args.keep_na:
        featured = featured.dropna()
    metadata["rows_before_feature_na_drop"] = int(rows_before_na_drop)
    metadata["rows_after_feature_na_drop"] = int(len(featured))
    metadata["dropped_feature_na_rows"] = int(rows_before_na_drop - len(featured))
    metadata["rolling_windows"] = args.rolling_windows
    metadata["lags"] = args.lags
    metadata["forecast_horizon"] = args.forecast_horizon

    processed, scale_metadata, scaler = scale_features(
        featured,
        method=args.scale,
        train_size=args.train_size,
    )
    metadata.update(scale_metadata)
    metadata["train_size"] = args.train_size
    metadata["output_rows"] = int(len(processed))
    metadata["output_columns"] = list(processed.reset_index().columns)
    metadata["created_at_utc"] = utc_now_iso()

    write_outputs(
        processed,
        metadata=metadata,
        scaler=scaler,
        output_path=args.output,
        metadata_output_path=metadata_output,
        scaler_output_path=scaler_output if scaler is not None else None,
    )
    print(f"[done] wrote {args.output} ({len(processed)} rows)")
    print(f"[done] wrote {metadata_output}")
    if scaler is not None:
        print(f"[done] wrote {scaler_output}")


if __name__ == "__main__":
    main()
