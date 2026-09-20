"""Build Research v2 multi-horizon labels and manifest.

Execution:
    python -m research.pipeline.build_labels_v2

Reads existing:
    research/data/raw_ohlcv.parquet
    research/data/samples.parquet
    research/data/splits.json
    research/config/experiment_v2.yaml

Outputs:
    research/data/labels_v2.parquet
    research/data/manifest_v2.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import yaml

import numpy as np
import pandas as pd

from research.pipeline.labels import (
    compute_cost_threshold,
    compute_labels,
)

BAR = pd.Timedelta(minutes=5)
ROOT = Path(__file__).resolve().parents[1]


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA256 hex digest of a file."""
    path = Path(path)
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def get_git_commit() -> str:
    """Get current git HEAD commit hash, or 'UNKNOWN' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return out
    except Exception:
        return "UNKNOWN"


def build_labels_v2(
    data_dir: Path | str = ROOT / "data",
    config_path: Path | str = ROOT / "config/experiment_v2.yaml",
) -> tuple[pd.DataFrame, dict]:
    """Load inputs, compute Research v2 labels, write parquet and manifest.

    Parameters
    ----------
    data_dir: Path or str
        Directory containing raw_ohlcv.parquet, samples.parquet, splits.json.
    config_path: Path or str
        Path to experiment_v2.yaml.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (labels_df, manifest_dict)
    """
    data_dir = Path(data_dir)
    config_path = Path(config_path)

    raw_path = data_dir / "raw_ohlcv.parquet"
    samples_path = data_dir / "samples.parquet"
    splits_path = data_dir / "splits.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing {samples_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    # 1. Load config and inputs
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw = pd.read_parquet(raw_path)
    samples = pd.read_parquet(samples_path)
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_info = json.load(f)

    # 2. Derive cost threshold from config
    cost_cfg = config["cost"]
    cost_fee = float(cost_cfg["cost_fee_per_side"])
    cost_slip = float(cost_cfg["cost_slippage"])
    cost_threshold = compute_cost_threshold(cost_fee, cost_slip)

    # 3. Align positions and map split names
    # In v2, the old 'test' is downgraded to 'dev_test_v1'
    positions = np.asarray(
        raw.timestamp.searchsorted(samples.timestamp - BAR),
        dtype=np.int64,
    )

    mapped_split = samples["split"].replace({"test": "dev_test_v1"}).to_numpy()

    # Split ranges in raw bar indices: {split_name: [start, stop)}
    split_ranges = {
        "train": splits_info["splits"]["train"]["raw_bar_range"],
        "val": splits_info["splits"]["val"]["raw_bar_range"],
        "dev_test_v1": splits_info["splits"]["test"]["raw_bar_range"],
    }

    horizons = config["data"]["prediction_horizons"]  # [1, 3, 6, 12]

    # 4. Pure label computation with split boundary purge
    labels_computed = compute_labels(
        raw=raw,
        positions=positions,
        split_names=mapped_split,
        split_ranges=split_ranges,
        cost_threshold=cost_threshold,
        horizons=horizons,
    )

    # 5. Assemble labels_v2 DataFrame
    labels_df = pd.DataFrame({
        "sample_id": samples["sample_id"].to_numpy(dtype=np.int64),
        "timestamp": samples["timestamp"].to_numpy(),
        "split": mapped_split,
        **{col: labels_computed[col].to_numpy() for col in labels_computed.columns},
    })

    # 6. Compute boundary NaN statistics per split
    splits_manifest = {}
    boundary_nan_total = 0

    for s_name in ["train", "val", "dev_test_v1"]:
        part = labels_df[labels_df["split"] == s_name]
        orig_s_key = "test" if s_name == "dev_test_v1" else s_name
        orig_split_info = splits_info["splits"][orig_s_key]

        nan_counts = {
            col: int(part[col].isna().sum())
            for col in labels_computed.columns
        }
        boundary_nan_total += nan_counts.get("future_return_12", 0)

        splits_manifest[s_name] = {
            "role": "development_test_v1" if s_name == "dev_test_v1" else s_name,
            "count": len(part),
            "sample_id_range": [int(part["sample_id"].iloc[0]), int(part["sample_id"].iloc[-1])],
            "first_timestamp": part["timestamp"].iloc[0].isoformat(),
            "last_timestamp": part["timestamp"].iloc[-1].isoformat(),
            "raw_bar_range": orig_split_info["raw_bar_range"],
            "boundary_nan_counts": nan_counts,
        }

    splits_manifest["holdout"] = {
        "status": "PENDING_FUTURE_COLLECTION",
        "role": "future_sealed_holdout",
        "note": "Awaiting future BTC/USDT spot data collection; strictly prohibited to use dev_test_v1.",
    }

    # 7. Write labels_v2.parquet via atomic staging
    labels_output_path = data_dir / "labels_v2.parquet"
    manifest_output_path = data_dir / "manifest_v2.json"

    with tempfile.TemporaryDirectory(prefix=".labels-v2-", dir=data_dir) as staging_dir:
        stage_path = Path(staging_dir)
        stage_labels_path = stage_path / "labels_v2.parquet"
        labels_df.to_parquet(stage_labels_path, index=False)

        labels_sha256 = compute_file_sha256(stage_labels_path)
        exp_v2_sha256 = compute_file_sha256(config_path)
        raw_sha256 = compute_file_sha256(raw_path)

        # 8. Compute distribution summaries
        valid_actions = labels_df["action"].dropna()
        action_counts = {k: int(v) for k, v in valid_actions.value_counts().to_dict().items()}
        action_ratios = {k: float(v) for k, v in valid_actions.value_counts(normalize=True).to_dict().items()}

        stats_summary = {}
        for col in ["future_return_1", "future_return_3", "future_return_6", "future_return_12",
                    "future_volatility_6", "maximum_favorable_excursion_6", "maximum_adverse_excursion_6"]:
            vals = labels_df[col].dropna()
            stats_summary[col] = {
                "count": int(len(vals)),
                "nan_count": int(labels_df[col].isna().sum()),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "median": float(vals.median()),
                "max": float(vals.max()),
            }

        manifest = {
            "manifest_version": "2.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "experiment_config": {
                "path": str(config_path.relative_to(ROOT.parent)),
                "sha256": exp_v2_sha256,
            },
            "data_period": {
                "symbol": config["data"]["symbol"],
                "interval": config["data"]["interval"],
                "raw_bars": len(raw),
                "raw_start": raw["timestamp"].iloc[0].isoformat(),
                "raw_end_exclusive": (raw["timestamp"].iloc[-1] + BAR).isoformat(),
                "raw_sha256": raw_sha256,
            },
            "splits": splits_manifest,
            "governance": {
                "old_test_role": config["governance"]["old_test_role"],
                "old_test_result_status": config["governance"]["old_test_result_status"],
                "old_test_highest_level": config["governance"]["old_test_highest_level"],
                "sealed_holdout_access": config["governance"]["sealed_holdout_access"],
            },
            "cost_settings": {
                "cost_fee_per_side": cost_fee,
                "cost_slippage": cost_slip,
                "cost_threshold": cost_threshold,
                "derivation": "cost_threshold = 2 * cost_fee_per_side + cost_slippage",
            },
            "purge_settings": {
                "input_window_bars": config["data"]["input_window_bars"],
                "primary_horizon": config["data"]["primary_horizon"],
                "max_horizon": max(horizons),
                "purge_bars_rule": f"window_bars ({config['data']['input_window_bars']}) + max_horizon ({max(horizons)}) = {config['data']['purge_bars']} bars",
                "embargo_bars_in_splits": splits_info.get("embargo_bars", 54),
                "boundary_nan_rule": "Samples where t + h >= split_stop are marked NaN without filling across splits",
                "boundary_nan_total": boundary_nan_total,
            },
            "labels_v2_parquet": {
                "path": str(labels_output_path.relative_to(ROOT.parent)),
                "sha256": labels_sha256,
                "total_samples": len(labels_df),
                "columns": list(labels_df.columns),
                "action_counts": action_counts,
                "action_ratios": action_ratios,
                "summary_statistics": stats_summary,
            },
        }

        stage_manifest_path = stage_path / "manifest_v2.json"
        stage_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        # Atomic move to destination
        stage_labels_path.rename(labels_output_path)
        stage_manifest_path.rename(manifest_output_path)

    return labels_df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/experiment_v2.yaml",
        help="Path to experiment_v2.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory containing market data parquet/json files",
    )
    args = parser.parse_args()

    print(f"Building Research v2 labels using config {args.config}...")
    labels_df, manifest = build_labels_v2(data_dir=args.data_dir, config_path=args.config)
    print(f"Successfully generated {len(labels_df):,} samples in {args.data_dir / 'labels_v2.parquet'}")
    print(f"Manifest written to {args.data_dir / 'manifest_v2.json'}")
    print("Action distribution:")
    for act, cnt in manifest["labels_v2_parquet"]["action_counts"].items():
        pct = manifest["labels_v2_parquet"]["action_ratios"][act]
        print(f"  {act:<5s}: {cnt:>6d} ({pct:.2%})")


if __name__ == "__main__":
    main()
