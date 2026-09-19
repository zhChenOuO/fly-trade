"""Rebuild the image cache from the locked OHLCV and sample index files."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

RESEARCH_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESEARCH_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from research.pipeline.render_market import render

WINDOW_BARS = 48
BAR_INTERVAL = pd.Timedelta(minutes=5)
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
AUDIT_NAME = re.compile(r"sample_(\d+)\.png$")


def window_for_sample(raw: pd.DataFrame, sample_timestamp) -> np.ndarray:
    """Return the same 48 bars selected by run_experiment.get_windows()."""
    decision_bar = pd.Timestamp(sample_timestamp) - BAR_INTERVAL
    end = int(raw["timestamp"].searchsorted(decision_bar))
    start = end - WINDOW_BARS + 1
    if start < 0 or end >= len(raw):
        raise ValueError(
            f"sample timestamp {sample_timestamp!r} has no complete 48-bar window"
        )
    window = raw.iloc[start : end + 1][OHLCV_COLUMNS].to_numpy()
    if window.shape != (WINDOW_BARS, len(OHLCV_COLUMNS)):
        raise ValueError(
            f"sample timestamp {sample_timestamp!r} produced window {window.shape}"
        )
    return window


def rebuild_images(
    raw_path: Path,
    samples_path: Path,
    audit_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Render all samples by image_idx and publish only after audit equality."""
    samples = pd.read_parquet(
        samples_path, columns=["sample_id", "timestamp", "image_idx"]
    )
    raw = pd.read_parquet(raw_path, columns=["timestamp", *OHLCV_COLUMNS])

    if not raw["timestamp"].is_monotonic_increasing:
        raise ValueError("raw OHLCV timestamps must be increasing for searchsorted")
    if samples["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    if samples["image_idx"].duplicated().any():
        raise ValueError("image_idx values must be unique")

    ordered = samples.sort_values("image_idx").reset_index(drop=True)
    image_indices = ordered["image_idx"].to_numpy(dtype=np.int64)
    expected_indices = np.arange(len(ordered), dtype=np.int64)
    if not np.array_equal(image_indices, expected_indices):
        raise ValueError("image_idx values must cover 0..N-1 without gaps")

    audit_files = sorted(audit_dir.glob("sample_*.png"))
    if len(audit_files) != 100:
        raise ValueError(f"expected 100 audit PNGs, found {len(audit_files)}")
    image_idx_by_sample_id = dict(
        zip(
            ordered["sample_id"].to_numpy(dtype=np.int64),
            image_indices,
            strict=True,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temp_path = Path(temp_name)

    try:
        images = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.uint8,
            shape=(len(ordered), 64, 64, 3),
        )
        for row in ordered.itertuples(index=False):
            images[int(row.image_idx)] = render(window_for_sample(raw, row.timestamp))
        images.flush()

        mismatches: list[str] = []
        for audit_path in audit_files:
            match = AUDIT_NAME.fullmatch(audit_path.name)
            if match is None:
                raise ValueError(f"unexpected audit filename: {audit_path.name}")
            sample_id = int(match.group(1))
            if sample_id not in image_idx_by_sample_id:
                raise ValueError(f"audit sample_id {sample_id} is missing from samples")
            image_idx = image_idx_by_sample_id[sample_id]
            with Image.open(audit_path) as audit_image:
                audit_pixels = np.asarray(audit_image.convert("RGB"), dtype=np.uint8)
            if audit_pixels.shape != (64, 64, 3):
                raise ValueError(
                    f"audit image {audit_path.name} has shape {audit_pixels.shape}"
                )
            if not np.array_equal(images[image_idx], audit_pixels):
                mismatches.append(audit_path.name)

        if mismatches:
            raise RuntimeError(
                "audit PNG pixel mismatch: " + ", ".join(mismatches[:10])
            )

        del images
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return {
        "path": str(output_path),
        "shape": [len(ordered), 64, 64, 3],
        "dtype": "uint8",
        "audit_images": len(audit_files),
        "audit_mismatches": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RESEARCH_DIR / "data/raw_ohlcv.parquet")
    parser.add_argument("--samples", type=Path, default=RESEARCH_DIR / "data/samples.parquet")
    parser.add_argument("--audit-dir", type=Path, default=RESEARCH_DIR / "data/audit")
    parser.add_argument("--output", type=Path, default=RESEARCH_DIR / "data/images.npy")
    args = parser.parse_args()

    result = rebuild_images(args.raw, args.samples, args.audit_dir, args.output)
    print(
        f"Rebuilt {result['path']}: shape={result['shape']}, "
        f"dtype={result['dtype']}, audit={result['audit_images']}/100 exact"
    )


if __name__ == "__main__":
    main()
