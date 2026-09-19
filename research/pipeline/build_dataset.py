"""Download and lock real market data: python -m research.pipeline.build_dataset.

Raw timestamps are UTC bar opens (timezone-naive datetime64[ns], per SPEC).
Sample timestamps are UTC bar closes: the full final input candle is available.
No interpolation, synthetic fallback, or combining candles from different venues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import ccxt
import numpy as np
import pandas as pd
from PIL import Image
import yaml

from research.pipeline.render_market import render
from src.data.market_data import DEFAULT_EXCHANGE_FALLBACKS


FIELDS = ["open", "high", "low", "close", "volume"]
BAR = pd.Timedelta(minutes=5)
BAR_MS = 300_000
ROOT = Path(__file__).resolve().parents[1]


def validate_config(config: dict) -> None:
    expected = {"symbol": "BTC/USDT", "interval": "5m", "window_bars": 48,
                "horizon_bars": 6, "sample_stride": 6, "history_years": 3,
                "image_size": [64, 64], "split": {"train": 0.6, "val": 0.2, "test": 0.2}}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"SPEC requires {key}={value!r}, got {config.get(key)!r}")


def clean_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Sort/deduplicate valid candles, then reject any missing 5-minute interval."""
    frame = raw[["timestamp", *FIELDS]].copy()
    frame["timestamp"] = (pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
                          .dt.tz_localize(None).astype("datetime64[ns]"))
    frame[FIELDS] = frame[FIELDS].apply(pd.to_numeric, errors="coerce")
    values = frame[FIELDS].to_numpy()
    valid = (frame.timestamp.notna().to_numpy() & np.isfinite(values).all(axis=1)
             & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0)
             & (values[:, 2] <= np.minimum(values[:, 0], values[:, 3]))
             & (values[:, 1] >= np.maximum(values[:, 0], values[:, 3])))
    frame = frame.loc[valid].drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError("no valid OHLCV candles")
    if frame.timestamp.duplicated().any():
        raise ValueError("conflicting OHLCV candles at the same timestamp")
    if (frame.timestamp.astype("int64") % BAR.value != 0).any():
        raise ValueError("timestamps are not aligned to the 5m grid")
    gaps = frame.timestamp.diff().iloc[1:] != BAR
    if gaps.any():
        i = gaps[gaps].index[0]
        raise ValueError(f"5m continuity failure: {frame.timestamp.iloc[i - 1]} -> {frame.timestamp.iloc[i]}")
    return frame


def fetch_ohlcv(config: dict, start: pd.Timestamp, end: pd.Timestamp,
                exchange_ids: list[str] | None = None) -> tuple[pd.DataFrame, str]:
    """Fetch the complete [start, end) interval, falling back between public venues."""
    validate_config(config)
    start_ms, end_ms = start.value // 1_000_000, end.value // 1_000_000
    errors = []
    for exchange_id in exchange_ids or DEFAULT_EXCHANGE_FALLBACKS:
        exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True, "timeout": 20_000,
            "options": {"defaultType": "spot", "fetchMarkets": {"types": ["spot"]}},
        })
        rows, cursor, page = [], start_ms, 0
        print(f"Fetching {exchange_id} BTC/USDT 5m: {start} .. {end} (end exclusive)", flush=True)
        try:
            while cursor < end_ms:
                for attempt in range(3):
                    try:
                        chunk = exchange.fetch_ohlcv(config["symbol"], timeframe=config["interval"],
                                                     since=int(cursor), limit=1000)
                        break
                    except ccxt.NetworkError:
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
                chunk = sorted((row for row in chunk if cursor <= row[0] < end_ms), key=lambda row: row[0])
                if not chunk or chunk[0][0] != cursor:
                    raise ValueError(f"missing history at {pd.to_datetime(cursor, unit='ms', utc=True)}")
                rows.extend(chunk)
                cursor = chunk[-1][0] + BAR_MS
                page += 1
                if page % 25 == 0 or cursor >= end_ms:
                    print(f"  {exchange_id}: {len(rows):,} candles ({100 * (cursor - start_ms) / (end_ms - start_ms):.1f}%)", flush=True)
            raw = pd.DataFrame(rows, columns=["timestamp", *FIELDS])
            raw["timestamp"] = pd.to_datetime(raw.timestamp, unit="ms", utc=True)
            raw = clean_ohlcv(raw)
            expected = (end_ms - start_ms) // BAR_MS
            if len(raw) != expected:
                raise ValueError(f"incomplete history: {len(raw)} candles, expected {expected}")
            return raw, exchange_id
        except (ccxt.BaseError, ValueError) as exc:
            detail = f"{exchange_id}: {type(exc).__name__}: {exc}"
            errors.append(detail)
            print(detail, flush=True)
    raise RuntimeError("No exchange provided complete real OHLCV data; stopping. " + "; ".join(errors))


def build_samples(raw: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Split raw time first; every input and label must fit inside its own split."""
    validate_config(config)
    n = len(raw)
    window, horizon, stride = (config[k] for k in ("window_bars", "horizon_bars", "sample_stride"))
    embargo = window + horizon
    first_cut, second_cut = int(n * 0.6), int(n * 0.8)
    ranges = {"train": (0, first_cut), "val": (first_cut + embargo, second_cut),
              "test": (second_cut + embargo, n)}
    candidates = np.arange(window - 1, n - horizon, stride, dtype=np.int64)
    closes = raw.close.to_numpy()
    parts, positions, splits = [], [], {}
    next_id = 0
    for name, (start, stop) in ranges.items():
        ends = candidates[(candidates - window + 1 >= start) & (candidates + horizon < stop)]
        if not len(ends):
            raise ValueError(f"not enough candles for {name} after purging and embargo")
        ids = np.arange(next_id, next_id + len(ends), dtype=np.int64)
        returns = closes[ends + horizon] / closes[ends] - 1
        part = pd.DataFrame({"sample_id": ids, "timestamp": raw.timestamp.iloc[ends].to_numpy() + BAR,
                             "image_idx": ids, "future_return": returns,
                             "label": (returns > 0).astype(np.int8), "split": name})
        parts.append(part)
        positions.append(ends)
        splits[name] = {"sample_id_range": [int(ids[0]), int(ids[-1])], "count": len(ids),
                        "raw_bar_range": [start, stop], "up_ratio": float(part.label.mean()),
                        "first_timestamp": part.timestamp.iloc[0].isoformat(),
                        "last_timestamp": part.timestamp.iloc[-1].isoformat()}
        next_id += len(ids)
    metadata = {"embargo_bars": embargo,
                "embargo_raw_bar_ranges": [[first_cut, first_cut + embargo], [second_cut, second_cut + embargo]],
                "sample_id_range_convention": "inclusive", "raw_bar_range_convention": "start inclusive, end exclusive",
                "splits": splits}
    return pd.concat(parts, ignore_index=True), np.concatenate(positions), metadata


def write_dataset(raw: pd.DataFrame, config: dict, output_dir: Path, *, exchange_id: str) -> dict:
    """Write the complete version via a staging directory; never replace locked files."""
    output_dir = Path(output_dir)
    artifacts = ["raw_ohlcv.parquet", "samples.parquet", "images.npy", "splits.json", "data_hash.txt", "audit"]
    if any((output_dir / name).exists() for name in artifacts):
        raise FileExistsError(f"dataset already exists in {output_dir}; choose a new output directory")
    raw = clean_ohlcv(raw)
    samples, positions, metadata = build_samples(raw, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=output_dir) as staging:
        stage = Path(staging)
        raw.to_parquet(stage / "raw_ohlcv.parquet", index=False)
        with (stage / "raw_ohlcv.parquet").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        (stage / "data_hash.txt").write_text(digest + "\n")
        samples.to_parquet(stage / "samples.parquet", index=False)
        images = np.lib.format.open_memmap(stage / "images.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(samples), 64, 64, 3))
        bars = raw[FIELDS].to_numpy(dtype=np.float64)
        for i, t in enumerate(positions):
            images[i] = render(bars[t - 47:t + 1])
            if (i + 1) % 10_000 == 0:
                print(f"Rendered {i + 1:,}/{len(samples):,} samples", flush=True)
        images.flush()
        audit_ids = np.sort(np.random.default_rng(0).choice(len(samples), size=min(100, len(samples)), replace=False))
        (stage / "audit").mkdir()
        for i in audit_ids:
            Image.fromarray(images[i]).save(stage / "audit" / f"sample_{i:06d}.png")
        del images
        metadata.update({"exchange": exchange_id, "symbol": config["symbol"], "interval": config["interval"],
                         "history_years": config["history_years"], "window_bars": 48, "horizon_bars": 6,
                         "sample_stride": 6, "image_size": [64, 64], "timezone": "UTC",
                         "raw_timestamp": "bar open", "sample_timestamp": "last input bar close",
                         "raw_bars": len(raw), "raw_start": raw.timestamp.iloc[0].isoformat(),
                         "raw_end_exclusive": (raw.timestamp.iloc[-1] + BAR).isoformat(),
                         "raw_sha256": digest, "sample_count": len(samples),
                         "up_ratio": float(samples.label.mean()), "test_at_least_10000": metadata["splits"]["test"]["count"] >= 10_000,
                         "audit_seed": 0, "audit_sample_ids": audit_ids.tolist()})
        (stage / "splits.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Publish the manifest last: its presence marks a completed dataset.
        for name in artifacts:
            if name != "splits.json":
                (stage / name).rename(output_dir / name)
        (stage / "splits.json").rename(output_dir / "splits.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/experiment.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    if (args.output_dir / "splits.json").exists():
        parser.error("dataset is already locked; use a different --output-dir")
    end = pd.Timestamp.now(tz="UTC").floor("5min")
    start = end - pd.DateOffset(years=config["history_years"])
    raw, exchange_id = fetch_ohlcv(config, start, end)
    metadata = write_dataset(raw, config, args.output_dir, exchange_id=exchange_id)
    print(json.dumps({"output_dir": str(args.output_dir), "sample_count": metadata["sample_count"],
                      "splits": metadata["splits"], "test_at_least_10000": metadata["test_at_least_10000"],
                      "up_ratio": metadata["up_ratio"]}, indent=2))


if __name__ == "__main__":
    main()
