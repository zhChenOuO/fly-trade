"""Run the frozen Phase 1a Level 1-2 input-dependence experiment on FlyWire."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from research.pipeline.action_decoder import decode  # noqa: E402
from research.pipeline.baselines import matched_random  # noqa: E402
from research.pipeline.encoding_v2 import calibrate_baseline  # noqa: E402
from research.pipeline.fly_simulator import FlySimulator  # noqa: E402
from research.pipeline.flywire_graph import load_flywire_graph  # noqa: E402
from research.pipeline.render_market import render  # noqa: E402
from research.pipeline.retina import retina_encoder  # noqa: E402
from research.pipeline.statistics import delta_margin_effect, evaluate_levels  # noqa: E402
from research.run_experiment import get_windows  # noqa: E402
from research.v2_health import nuisance, r2  # noqa: E402

CFG = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
DATA = ROOT / "data"
OUT = ROOT / "outputs/v3/phase1a_remote"
BASELINE_PATH = ROOT / "outputs/v3/baseline_remote.json"
DYNAMICS_PATH = ROOT / "outputs/v3/dynamics_remote.json"
MEAN_IMAGE_PATH = ROOT / "outputs/v3/train_mean_image.npy"
GRAPH_CACHE_PATH = DATA / "flywire/graph_cache.npz"
WINDOW_BARS = int(CFG["window_bars"])
SIM_CHUNK = 256
REPEAT_INPUTS = 200
REPEAT_COUNT = 100
CALIBRATION_COUNT = 2000
PROGRESS_CHUNKS = 20
EXPECTED_SEEDS = 30
EXPECTED_PERMUTATIONS = 1000
EXPECTED_BOOTSTRAPS = 2000
# Empirical upper envelope: 4.76837158203125e-7 was the largest absolute
# score delta across 30 fixed-seed, 32-image reruns on the target RTX 5070.
# The standard-deviation cap keeps this tolerance scale-aware and never above
# 1e-5 of the observed score spread.
REPRO_SCORE_ABS_TOL = 4.76837158203125e-7
REPRO_SCORE_SD_FRACTION = 1e-5
DECISION_COLUMNS = [
    "sample_id", "seed", "group", "buy_score", "sell_score", "action", "latency_ms"
]

_LOG_PATH: Path | None = None
_PEAK_DEVICE_USED_BYTES = 0


def _log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def _check_vram_limit() -> None:
    global _PEAK_DEVICE_USED_BYTES
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = int(total_bytes - free_bytes)
    _PEAK_DEVICE_USED_BYTES = max(_PEAK_DEVICE_USED_BYTES, used_bytes)
    if used_bytes > int(total_bytes * 0.8):
        raise MemoryError(
            f"CUDA device use {used_bytes / total_bytes:.1%} exceeded the frozen 80% VRAM limit"
        )


def preflight_summary(buy_score: np.ndarray, sell_score: np.ndarray) -> dict:
    """Check finite, input-varying 20-image outputs and the frozen 5% floor."""
    buy = np.asarray(buy_score, dtype=np.float64)
    sell = np.asarray(sell_score, dtype=np.float64)
    if buy.ndim != 1 or sell.shape != buy.shape or not buy.size:
        raise ValueError("preflight scores must be equal, nonempty vectors")
    finite = bool(np.isfinite(buy).all() and np.isfinite(sell).all())
    if finite:
        margin = buy - sell
        actions = decode(buy, sell)
        buy_ratio = float(actions.mean())
        minority = float(min(buy_ratio, 1.0 - buy_ratio))
        margin_range = float(np.ptp(margin))
    else:
        buy_ratio = minority = margin_range = None
    passed = bool(finite and minority >= 0.05 and margin_range > 0.0)
    return {
        "passed": passed,
        "n_images": int(len(buy)),
        "finite_scores": finite,
        "buy_ratio": buy_ratio,
        "minority_action_ratio": minority,
        "minority_threshold": 0.05,
        "margin_range": margin_range,
        "input_dependent_margin": bool(margin_range is not None and margin_range > 0.0),
        "noise_std": 0.0,
    }


def reproducibility_summary(
    first_buy: np.ndarray,
    first_sell: np.ndarray,
    second_buy: np.ndarray,
    second_sell: np.ndarray,
) -> dict:
    """Measure fixed-seed score drift and constrain any action flips to tiny margins."""
    first_buy = np.asarray(first_buy, dtype=np.float64)
    first_sell = np.asarray(first_sell, dtype=np.float64)
    second_buy = np.asarray(second_buy, dtype=np.float64)
    second_sell = np.asarray(second_sell, dtype=np.float64)
    if (
        first_buy.ndim != 1
        or not first_buy.size
        or first_sell.shape != first_buy.shape
        or second_buy.shape != first_buy.shape
        or second_sell.shape != first_buy.shape
    ):
        raise ValueError("reproducibility scores must be equal, nonempty vectors")
    score_arrays = (first_buy, first_sell, second_buy, second_sell)
    if not all(np.isfinite(scores).all() for scores in score_arrays):
        raise FloatingPointError("reproducibility scores must be finite")

    reference_scores = np.concatenate([first_buy, first_sell])
    score_sd = float(np.std(reference_scores))
    score_tolerance = min(REPRO_SCORE_ABS_TOL, REPRO_SCORE_SD_FRACTION * score_sd)
    absolute_diffs = np.concatenate(
        [np.abs(first_buy - second_buy), np.abs(first_sell - second_sell)]
    )
    reference_values = np.concatenate([first_buy, first_sell])
    repeated_values = np.concatenate([second_buy, second_sell])
    relative_denominators = np.maximum(
        np.maximum(np.abs(reference_values), np.abs(repeated_values)),
        np.finfo(np.float64).tiny,
    )
    relative_diffs = absolute_diffs / relative_denominators

    first_actions = decode(first_buy, first_sell)
    second_actions = decode(second_buy, second_sell)
    flips = first_actions != second_actions
    reference_margins = first_buy - first_sell
    repeated_margins = second_buy - second_sell
    flipped_samples = [
        {
            "offset": int(index),
            "reference_margin": float(reference_margins[index]),
            "repeat_margin": float(repeated_margins[index]),
            "abs_reference_margin": float(abs(reference_margins[index])),
            "reference_action": int(first_actions[index]),
            "repeat_action": int(second_actions[index]),
        }
        for index in np.flatnonzero(flips)
    ]
    allowed_flips = flips & (np.abs(reference_margins) < score_tolerance)
    scores_within_tolerance = bool(np.all(absolute_diffs <= score_tolerance))
    disallowed_action_flips = int(np.count_nonzero(flips & ~allowed_flips))
    actions_within_tolerance = disallowed_action_flips == 0

    return {
        "passed": bool(scores_within_tolerance and actions_within_tolerance),
        "scores_exact": bool(
            np.array_equal(first_buy, second_buy)
            and np.array_equal(first_sell, second_sell)
        ),
        "actions_exact": bool(np.array_equal(first_actions, second_actions)),
        "scores_within_tolerance": scores_within_tolerance,
        "actions_within_tolerance": actions_within_tolerance,
        "score_std_reference": score_sd,
        "score_tolerance": score_tolerance,
        "max_abs_score_diff": float(absolute_diffs.max(initial=0.0)),
        "max_rel_score_diff_symmetric_denominator": float(relative_diffs.max(initial=0.0)),
        "max_abs_diff_over_score_std": (
            float(absolute_diffs.max(initial=0.0) / score_sd) if score_sd > 0 else None
        ),
        "actions_flipped": int(np.count_nonzero(flips)),
        "allowed_action_flips": int(np.count_nonzero(allowed_flips)),
        "disallowed_action_flips": disallowed_action_flips,
        "max_abs_reference_margin_on_flips": (
            max(item["abs_reference_margin"] for item in flipped_samples)
            if flipped_samples
            else None
        ),
        "flipped_samples": flipped_samples,
    }


def _market_state_components(
    samples: pd.DataFrame,
    raw: pd.DataFrame,
    chunk_size: int = 512,
) -> tuple[pd.Series, np.ndarray]:
    """Return six past-only states and their Train-only position cutpoints."""
    required = {"sample_id", "timestamp", "split"}
    if not required.issubset(samples.columns):
        raise ValueError(f"samples are missing label-free state columns: {sorted(required - set(samples.columns))}")
    if not samples.split.isin(["train", "val", "test"]).all():
        raise ValueError("unknown split in label-free sample table")

    positions = np.empty(len(samples), dtype=np.float64)
    positive_return = np.empty(len(samples), dtype=bool)
    for start in range(0, len(samples), chunk_size):
        stop = min(start + chunk_size, len(samples))
        windows = get_windows(samples.iloc[start:stop], raw)
        close = windows[:, :, 3]
        low = windows[:, :, 2].min(axis=1)
        high = windows[:, :, 1].max(axis=1)
        denominator = high - low
        positions[start:stop] = np.divide(
            close[:, -1] - low,
            denominator,
            out=np.zeros(stop - start, dtype=np.float64),
            where=denominator > 0,
        )
        positive_return[start:stop] = close[:, -1] > close[:, 0]

    train_mask = samples.split.to_numpy() == "train"
    if not train_mask.any():
        raise ValueError("Train samples are required to freeze market-state cutpoints")
    cuts = np.quantile(positions[train_mask], [1.0 / 3.0, 2.0 / 3.0])
    position_bin = np.digitize(positions, cuts)
    states = positive_return.astype(np.int8) * 3 + position_bin.astype(np.int8)
    return pd.Series(states, index=samples.sample_id.to_numpy(), name="market_state"), cuts


def market_state(samples: pd.DataFrame, raw: pd.DataFrame, chunk_size: int = 512) -> pd.Series:
    """Six past-only states: 48-bar direction x Train-binned close position."""
    return _market_state_components(samples, raw, chunk_size)[0]


def verify_render_alignment(
    samples: pd.DataFrame,
    raw: pd.DataFrame,
    images: np.ndarray,
    *,
    chunk_size: int = 256,
    progress=None,
) -> int:
    """Re-render every image from the locked decision bars and assert exact equality."""
    needed = {"sample_id", "image_idx", "timestamp"}
    if not needed.issubset(samples.columns):
        raise ValueError(f"samples are missing alignment columns: {sorted(needed - set(samples.columns))}")
    image_idx = samples.image_idx.to_numpy(dtype=np.int64)
    if len(samples) != len(images) or not np.array_equal(np.sort(image_idx), np.arange(len(images))):
        raise ValueError("sample image_idx must map one-to-one onto images.npy")
    if images.ndim != 4 or images.shape[1:] != (64, 64, 3):
        raise ValueError(f"images.npy must have shape (N,64,64,3), got {images.shape}")

    for start in range(0, len(samples), chunk_size):
        stop = min(start + chunk_size, len(samples))
        sub = samples.iloc[start:stop]
        windows = get_windows(sub, raw)
        rebuilt = np.stack([render(window) for window in windows])
        stored = np.asarray(images[sub.image_idx.to_numpy(dtype=np.int64)])
        if not np.array_equal(rebuilt, stored):
            mismatch = np.flatnonzero(np.any(rebuilt != stored, axis=(1, 2, 3)))
            ids = sub.sample_id.to_numpy()[mismatch[:10]].tolist()
            raise AssertionError(f"re-rendered images differ from images.npy at sample_id={ids}")
        if progress is not None and (stop == len(samples) or stop % 5000 < chunk_size):
            progress(f"image alignment verified {stop:,}/{len(samples):,}")
    return len(samples)


def variant_images(
    samples: pd.DataFrame,
    raw: pd.DataFrame,
    images: np.ndarray,
    *,
    n: int = REPEAT_INPUTS,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Reproduce the frozen run_experiment flip/shuffle/mask/black/mean variants."""
    val = samples.loc[samples.split == "val"]
    if n < 1 or n > len(val):
        raise ValueError(f"need {n} Val samples, found {len(val)}")
    sub = val.sample(n, random_state=seed)
    windows = get_windows(sub, raw)

    def flip_price(window):
        mid = window[:, 1].max() + window[:, 2].min()
        flipped = window.copy()
        flipped[:, 0] = mid - window[:, 0]
        flipped[:, 3] = mid - window[:, 3]
        flipped[:, 1] = mid - window[:, 2]
        flipped[:, 2] = mid - window[:, 1]
        return flipped

    def mask_recent25(window):
        masked = window.copy()
        masked[-WINDOW_BARS // 4 :] = masked[: WINDOW_BARS - WINDOW_BARS // 4].mean(axis=0)
        return masked

    rng = np.random.default_rng(seed)
    original = np.stack([render(window) for window in windows])
    stored = np.asarray(images[sub.image_idx.to_numpy(dtype=np.int64)])
    if not np.array_equal(original, stored):
        raise AssertionError("variant original re-render differs from images.npy")
    variants = {
        "original": original,
        "flip_price": np.stack([render(flip_price(window)) for window in windows]),
        "shuffle_time": np.stack([render(window[rng.permutation(WINDOW_BARS)]) for window in windows]),
        "mask_recent25": np.stack([render(mask_recent25(window)) for window in windows]),
    }
    variants["black"] = np.zeros_like(original)
    mean_image = original.mean(axis=0, keepdims=True).round().astype(np.uint8)
    variants["mean_image"] = np.repeat(mean_image, n, axis=0)
    return variants


def _new_simulator(graph, encoder, baseline, dynamics, seed: int, noise_std: float) -> FlySimulator:
    return FlySimulator(
        graph,
        seed=seed,
        steps=int(dynamics["steps"]),
        gain=float(dynamics["gain"]),
        leak=float(dynamics["leak"]),
        noise_std=float(noise_std),
        encoder=encoder,
        baseline=baseline,
        direct_currents=True,
        backend="torch",
    )


def _simulate_chunks(
    simulator: FlySimulator,
    input_images: np.ndarray,
    *,
    category: str,
    tracker: dict,
    description: str,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    buy_parts, sell_parts = [], []
    started = time.perf_counter()
    total = len(input_images)
    for start in range(0, total, SIM_CHUNK):
        stop = min(start + SIM_CHUNK, total)
        output = simulator.run(input_images[start:stop])
        _check_vram_limit()
        buy_parts.append(output["buy_score"])
        sell_parts.append(output["sell_score"])
        if progress and ((stop // SIM_CHUNK) % PROGRESS_CHUNKS == 0 or stop == total):
            _log(f"{description}: {stop:,}/{total:,} images")
    buy = np.concatenate(buy_parts) if buy_parts else np.empty(0, dtype=np.float64)
    sell = np.concatenate(sell_parts) if sell_parts else np.empty(0, dtype=np.float64)
    elapsed = time.perf_counter() - started
    if not np.isfinite(buy).all() or not np.isfinite(sell).all():
        raise FloatingPointError(f"non-finite simulator output in {description}")
    item = tracker.setdefault(category, {"images": 0, "seconds": 0.0})
    item["images"] += total
    item["seconds"] += elapsed
    return buy, sell, elapsed / max(total, 1) * 1000.0


def _decision_rows(sample_ids, seed, group, buy, sell, latency_ms) -> pd.DataFrame:
    actions = decode(buy, sell).astype(np.int8)
    return pd.DataFrame(
        {
            "sample_id": np.asarray(sample_ids, dtype=np.int64),
            "seed": np.full(len(sample_ids), seed, dtype=np.int16),
            "group": group,
            "buy_score": np.asarray(buy, dtype=np.float64),
            "sell_score": np.asarray(sell, dtype=np.float64),
            "action": actions,
            "latency_ms": np.full(len(sample_ids), latency_ms, dtype=np.float64),
        },
        columns=DECISION_COLUMNS,
    )


def _runtime_summary(tracker: dict) -> dict:
    total_images = int(sum(item["images"] for item in tracker.values()))
    total_seconds = float(sum(item["seconds"] for item in tracker.values()))
    return {
        "simulated_images": total_images,
        "simulator_seconds": total_seconds,
        "throughput_images_per_second": total_images / total_seconds if total_seconds else 0.0,
        "by_stage": tracker,
    }


def _run_phase1a() -> tuple[dict, dict, dict]:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    health_path = OUT / "health.json"
    health = {"status": "running", "checks": {}}
    global _PEAK_DEVICE_USED_BYTES
    _PEAK_DEVICE_USED_BYTES = 0
    tracker: dict = {}
    _log(f"Phase 1a start; output={OUT}")

    try:
        if (
            int(CFG["window_bars"]) != 48
            or int(CFG["seeds"]) != EXPECTED_SEEDS
            or int(CFG["permutations"]) != EXPECTED_PERMUTATIONS
            or int(CFG["bootstrap_samples"]) != EXPECTED_BOOTSTRAPS
        ):
            raise ValueError("experiment config differs from the frozen Phase 1a sample/resampling counts")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; Phase 1a requires the frozen torch backend")
        if not DYNAMICS_PATH.is_file() or not MEAN_IMAGE_PATH.is_file() or not GRAPH_CACHE_PATH.is_file():
            raise FileNotFoundError("required frozen Phase 0 dynamics, train mean image, or FlyWire graph cache is missing")

        dynamics = json.loads(DYNAMICS_PATH.read_text())
        if dynamics.get("graph") != "flywire" or int(dynamics.get("steps", 0)) != 32:
            raise ValueError("Phase 0 dynamics do not match the frozen FlyWire configuration")
        if float(dynamics["leak"]) != 0.5 or float(dynamics["noise_std_robustness"]) <= 0:
            raise ValueError("Phase 0 leak/noise parameters do not match the frozen configuration")
        mean_image = np.load(MEAN_IMAGE_PATH)
        if mean_image.shape != (64, 64, 3) or not np.isfinite(mean_image).all():
            raise ValueError("frozen train_mean_image.npy is invalid")

        device = torch.cuda.get_device_properties(0)
        health["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device": device.name,
            "total_vram_bytes": int(device.total_memory),
            "vram_fraction_limit": 0.8,
        }
        health["phase0_spec_verdict"] = "GO"
        health["phase0_artifact_note"] = (
            "phase0_report_remote.json retains the earlier STOP generated before the frozen SPEC_v3 Amendment 2; "
            "the amended SPEC_v3 verdict is GO because PC-2 is report-only."
        )
        _log(
            f"GPU={device.name}; VRAM={device.total_memory / 1024**3:.2f} GiB; "
            f"torch={torch.__version__}; CUDA={torch.version.cuda}"
        )

        samples = pd.read_parquet(
            DATA / "samples.parquet",
            columns=["sample_id", "timestamp", "image_idx", "split"],
        ).sort_values("sample_id").reset_index(drop=True)
        raw = pd.read_parquet(DATA / "raw_ohlcv.parquet")
        images = np.load(DATA / "images.npy", mmap_mode="r")
        split_metadata = json.loads((DATA / "splits.json").read_text())
        test_status = str(split_metadata.get("test_status", ""))
        if not test_status.startswith("development_test_v1"):
            raise ValueError("test split must remain marked development_test_v1 for Phase 1a")
        if not raw.timestamp.is_monotonic_increasing or raw.timestamp.duplicated().any():
            raise ValueError("raw OHLCV timestamps must be strictly increasing")
        if not samples.split.isin(["train", "val", "test"]).all():
            raise ValueError("samples contains an unknown split")
        train = samples.loc[samples.split == "train"].reset_index(drop=True)
        val = samples.loc[samples.split == "val"].reset_index(drop=True)
        selected = samples.loc[samples.split.isin(["val", "test"])].sort_values("sample_id").reset_index(drop=True)
        if len(train) < CALIBRATION_COUNT or len(val) < max(REPEAT_INPUTS, 20) or selected.empty:
            raise ValueError("dataset is too small for the frozen Phase 1a sample counts")
        _log(f"label-free data loaded: train={len(train):,}, val={len(val):,}, test={int((samples.split == 'test').sum()):,}")

        graph = load_flywire_graph(cache_path=GRAPH_CACHE_PATH)
        retina = retina_encoder(graph, mean_image)
        torch.cuda.reset_peak_memory_stats()
        sample_positions = np.linspace(0, len(train) - 1, CALIBRATION_COUNT, dtype=np.int64)
        calibration = train.iloc[sample_positions]
        calibration_images = np.asarray(images[calibration.image_idx.to_numpy(dtype=np.int64)])
        calibration_started = time.perf_counter()
        baseline = calibrate_baseline(
            lambda _: _new_simulator(graph, retina, None, dynamics, 0, 0.0),
            calibration_images,
            chunk=SIM_CHUNK,
        )
        _check_vram_limit()
        calibration_seconds = time.perf_counter() - calibration_started
        if not all(np.isfinite(value) for value in baseline.values()) or min(baseline["buy_std"], baseline["sell_std"]) <= 0:
            raise FloatingPointError("Train z-score baseline is non-finite or has zero standard deviation")
        calibration_sim = _new_simulator(graph, retina, baseline, dynamics, 0, 0.0)
        train_buy, train_sell, train_margin_latency = _simulate_chunks(
            calibration_sim,
            calibration_images,
            category="train_margin_sd",
            tracker=tracker,
            description="Train margin SD",
        )
        train_margin_sd = float(np.std(train_buy - train_sell))
        if not np.isfinite(train_margin_sd) or train_margin_sd <= 0:
            raise FloatingPointError("Train margin standard deviation must be finite and positive")
        tracker["baseline_calibration"] = {"images": CALIBRATION_COUNT, "seconds": calibration_seconds}
        baseline_record = {
            **baseline,
            "train_margin_sd": train_margin_sd,
            "calibration_count": CALIBRATION_COUNT,
            "calibration_sample_id_first": int(calibration.sample_id.iloc[0]),
            "calibration_sample_id_last": int(calibration.sample_id.iloc[-1]),
            "selection": "2000 equally spaced rows from sample_id-sorted Train split",
            "dynamics": {
                "gain": float(dynamics["gain"]),
                "leak": float(dynamics["leak"]),
                "steps": int(dynamics["steps"]),
                "noise_std_calibration": 0.0,
                "noise_std_robustness": float(dynamics["noise_std_robustness"]),
            },
            "train_mean_image_sha256": hashlib.sha256(MEAN_IMAGE_PATH.read_bytes()).hexdigest(),
            "backend": "torch",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        _write_json(BASELINE_PATH, baseline_record)
        health["checks"]["train_baseline"] = {
            "passed": True,
            "count": CALIBRATION_COUNT,
            "seconds": calibration_seconds,
            "train_margin_sd": train_margin_sd,
        }
        _write_json(health_path, health)
        _log(f"Train baseline frozen from {CALIBRATION_COUNT} spaced images; margin_sd={train_margin_sd:.8f}")

        val20 = val.iloc[:20]
        preflight_images = np.asarray(images[val20.image_idx.to_numpy(dtype=np.int64)])
        preflight_sim = _new_simulator(graph, retina, baseline, dynamics, 0, 0.0)
        preflight_buy, preflight_sell, _ = _simulate_chunks(
            preflight_sim,
            preflight_images,
            category="preflight20",
            tracker=tracker,
            description="20-image Val preflight",
        )
        preflight = preflight_summary(preflight_buy, preflight_sell)
        health["checks"]["preflight_20_val"] = preflight
        _write_json(health_path, health)
        _log(
            f"20-image Val preflight: passed={preflight['passed']}, "
            f"minority={preflight['minority_action_ratio']}, margin_range={preflight['margin_range']}"
        )
        if not preflight["passed"]:
            raise RuntimeError("20-image Val preflight failed the frozen minority/input-variation check")

        alignment_start = time.perf_counter()
        aligned_count = verify_render_alignment(
            samples,
            raw,
            images,
            chunk_size=SIM_CHUNK,
            progress=_log,
        )
        health["checks"]["full_render_alignment"] = {
            "passed": True,
            "images_checked": aligned_count,
            "seconds": time.perf_counter() - alignment_start,
            "rule": "run_experiment.get_windows: timestamp minus 5 minutes, final 48 raw bars inclusive",
        }
        _write_json(health_path, health)
        _log(f"exact image alignment passed for all {aligned_count:,} samples")

        selected_ids = selected.sample_id.to_numpy(dtype=np.int64)
        selected_images = np.asarray(images[selected.image_idx.to_numpy(dtype=np.int64)])
        val_ids = val.sample_id.to_numpy(dtype=np.int64)
        val_positions = np.flatnonzero(np.isin(selected_ids, val_ids))
        deterministic_images = np.asarray(selected_images[:32])
        deterministic_results = []
        for seed in range(int(CFG["seeds"])):
            first = _simulate_chunks(
                _new_simulator(graph, retina, baseline, dynamics, seed, dynamics["noise_std_robustness"]),
                deterministic_images,
                category="seed_reproducibility",
                tracker=tracker,
                description=f"seed {seed} reproducibility first pass",
            )
            second = _simulate_chunks(
                _new_simulator(graph, retina, baseline, dynamics, seed, dynamics["noise_std_robustness"]),
                deterministic_images,
                category="seed_reproducibility",
                tracker=tracker,
                description=f"seed {seed} reproducibility repeat",
            )
            comparison = reproducibility_summary(first[0], first[1], second[0], second[1])
            deterministic_results.append({"seed": seed, **comparison})
            if not comparison["passed"]:
                health["checks"]["seed_reproducibility"] = {
                    "passed": False,
                    "n_seeds_completed": len(deterministic_results),
                    "images_per_repeat": len(deterministic_images),
                    "score_tolerance_rule": "min(4.76837158203125e-7, 1e-5 * pooled reference score SD)",
                    "results": deterministic_results,
                }
                _write_json(health_path, health)
                _log(
                    "fixed-seed reproducibility failed: "
                    + json.dumps({"seed": seed, **comparison}, allow_nan=False)
                )
                raise RuntimeError(f"fixed-seed A scores/actions exceed the measured reproducibility tolerance for seed {seed}")
        health["checks"]["seed_reproducibility"] = {
            "passed": True,
            "n_seeds": len(deterministic_results),
            "images_per_repeat": len(deterministic_images),
            "score_tolerance_rule": "min(4.76837158203125e-7, 1e-5 * pooled reference score SD)",
            "score_tolerance_max": REPRO_SCORE_ABS_TOL,
            "total_action_flips": sum(result["actions_flipped"] for result in deterministic_results),
            "results": deterministic_results,
        }
        _write_json(health_path, health)
        _log(f"fixed-seed reproducibility passed for all {len(deterministic_results)} seeds")

        all_ids = selected.sample_id.to_numpy(dtype=np.int64)
        seed_count = int(CFG["seeds"])
        seed_values = list(range(seed_count))
        a_by_seed = {}
        a_frames = []
        a_started = time.perf_counter()
        for seed in seed_values:
            _log(f"A fly_intact seed {seed}/{seed_count - 1} start")
            simulator = _new_simulator(graph, retina, baseline, dynamics, seed, dynamics["noise_std_robustness"])
            buy, sell, latency = _simulate_chunks(
                simulator,
                selected_images,
                category="A_fly_intact",
                tracker=tracker,
                description=f"A seed {seed}",
                progress=True,
            )
            actions = decode(buy, sell).astype(np.int8)
            a_by_seed[seed] = {"buy": buy, "sell": sell, "action": actions, "latency_ms": latency}
            a_frames.append(_decision_rows(all_ids, seed, "fly_intact", buy, sell, latency))
            _log(f"A fly_intact seed {seed} complete; {len(all_ids):,} decisions at {1.0 / (latency / 1000.0):.2f} images/s")
        a_seconds = time.perf_counter() - a_started
        a_decisions = pd.concat(a_frames, ignore_index=True)
        val_id_set = set(val_ids.tolist())
        p_buy = float(a_decisions.loc[a_decisions.sample_id.isin(val_id_set), "action"].mean())
        if not np.isfinite(p_buy) or not 0.0 <= p_buy <= 1.0:
            raise FloatingPointError("validation-calibrated matched-random probability is invalid")
        _log(f"A completed in {a_seconds:.1f}s; matched-random BUY probability={p_buy:.8f}")

        c_frames, b_frames, e_frames = [], [], []
        mean_input = np.asarray(mean_image, dtype=np.float32)
        for seed in seed_values:
            a = a_by_seed[seed]
            permutation = np.random.default_rng(10_000 + seed).permutation(len(all_ids))
            c_frames.append(
                _decision_rows(
                    all_ids,
                    seed,
                    "input_shuffled",
                    a["buy"][permutation],
                    a["sell"][permutation],
                    a["latency_ms"],
                )
            )
            b_actions = matched_random(p_buy, len(all_ids), 20_000 + seed)
            b_frames.append(
                pd.DataFrame(
                    {
                        "sample_id": all_ids,
                        "seed": np.full(len(all_ids), seed, dtype=np.int16),
                        "group": "matched_random",
                        "buy_score": np.zeros(len(all_ids), dtype=np.float64),
                        "sell_score": np.zeros(len(all_ids), dtype=np.float64),
                        "action": b_actions.astype(np.int8),
                        "latency_ms": np.zeros(len(all_ids), dtype=np.float64),
                    },
                    columns=DECISION_COLUMNS,
                )
            )
            constant_batch = np.broadcast_to(mean_input, (len(all_ids), *mean_input.shape))
            e_buy, e_sell, e_latency = _simulate_chunks(
                _new_simulator(graph, retina, baseline, dynamics, seed, dynamics["noise_std_robustness"]),
                constant_batch,
                category="E_constant_input",
                tracker=tracker,
                description=f"E seed {seed}",
                progress=True,
            )
            e_actions = decode(e_buy, e_sell).astype(np.int8)
            e_frames.append(_decision_rows(all_ids, seed, "constant_input", e_buy, e_sell, e_latency))
            _log(f"B/C/E controls seed {seed} complete")
        c_decisions = pd.concat(c_frames, ignore_index=True)
        b_decisions = pd.concat(b_frames, ignore_index=True)
        e_decisions = pd.concat(e_frames, ignore_index=True)
        decisions = pd.concat([a_decisions, b_decisions, c_decisions, e_decisions], ignore_index=True)
        decisions = decisions[DECISION_COLUMNS]
        if decisions.duplicated(["sample_id", "seed", "group"]).any():
            raise ValueError("decision keys are not unique")
        decisions_path = OUT / "decisions.parquet"
        decisions.to_parquet(decisions_path, index=False)
        _log(f"saved {len(decisions):,} decision rows to ignored {decisions_path}")

        # The shared 200-Val set and all six registered variants are reused across 100 noise seeds.
        variants = variant_images(samples, raw, images, n=REPEAT_INPUTS, seed=0)
        base_actions = np.empty((REPEAT_INPUTS, REPEAT_COUNT), dtype=np.int8)
        repeat_margins = np.empty((REPEAT_INPUTS, REPEAT_COUNT), dtype=np.float32)
        perturbation_margins = {
            name: np.empty((REPEAT_INPUTS, REPEAT_COUNT), dtype=np.float32)
            for name in variants
            if name != "original"
        }
        perturbation_actions = {
            name: np.empty((REPEAT_INPUTS, REPEAT_COUNT), dtype=np.int8)
            for name in variants
            if name != "original"
        }
        repeats_started = time.perf_counter()
        for repeat in range(REPEAT_COUNT):
            seed = 50_000 + repeat
            for name, variant in variants.items():
                buy, sell, _ = _simulate_chunks(
                    _new_simulator(graph, retina, baseline, dynamics, seed, dynamics["noise_std_robustness"]),
                    variant,
                    category="repeat_and_perturbation",
                    tracker=tracker,
                    description=f"repeat {repeat} {name}",
                )
                margin = (buy - sell).astype(np.float32)
                action = decode(buy, sell).astype(np.int8)
                if name == "original":
                    base_actions[:, repeat] = action
                    repeat_margins[:, repeat] = margin
                else:
                    perturbation_actions[name][:, repeat] = action
                    perturbation_margins[name][:, repeat] = margin
            if (repeat + 1) % 10 == 0 or repeat + 1 == REPEAT_COUNT:
                _log(f"repeat/sensitivity {repeat + 1}/{REPEAT_COUNT} seeds complete")
        repeat_seconds = time.perf_counter() - repeats_started
        baseline_repeat_margin = repeat_margins.mean(axis=1)
        sensitivity = {
            "n_inputs": REPEAT_INPUTS,
            "n_repeats": REPEAT_COUNT,
            "input_sample_ids": samples.loc[
                samples.split == "val"
            ].sample(REPEAT_INPUTS, random_state=0).sample_id.astype(int).tolist(),
            "train_margin_sd": train_margin_sd,
            "variants": {},
        }
        for name, margins in perturbation_margins.items():
            aggregate_effect = delta_margin_effect(
                baseline_repeat_margin,
                margins.mean(axis=1),
                train_margin_sd,
            )
            per_seed = []
            for repeat in range(REPEAT_COUNT):
                per_seed.append(
                    {
                        "seed": 50_000 + repeat,
                        "margin_effect": float(
                            np.abs(margins[:, repeat] - repeat_margins[:, repeat]).mean() / train_margin_sd
                        ),
                        "action_change_rate": float(
                            (perturbation_actions[name][:, repeat] != base_actions[:, repeat]).mean()
                        ),
                    }
                )
            sensitivity["variants"][name] = {
                "mean_margin_effect": aggregate_effect,
                "mean_action_change_rate": float(
                    (perturbation_actions[name] != base_actions).mean()
                ),
                "per_seed": per_seed,
            }
        _write_json(OUT / "sensitivity.json", sensitivity)
        _log(f"repeat and sensitivity calculations complete in {repeat_seconds:.1f}s")

        # All remaining computations before this point use only inputs and actions.
        val_windows = get_windows(val, raw)
        val_images = np.asarray(images[val.image_idx.to_numpy(dtype=np.int64)])
        nuisance_features = nuisance(val_images, val_windows)
        a_val_margins = np.column_stack(
            [a_by_seed[seed]["buy"][val_positions] - a_by_seed[seed]["sell"][val_positions] for seed in seed_values]
        )
        nuisance_r2 = r2(a_val_margins.mean(axis=1), nuisance_features.to_numpy(dtype=np.float64))
        if not np.isfinite(nuisance_r2):
            raise FloatingPointError("Val nuisance R-squared is non-finite")
        market_states, market_state_cuts = _market_state_components(samples, raw)
        _log(f"label-free market_state and nuisance calculation complete; nuisance_r2={nuisance_r2:.8f}")

        # Labels are loaded only at the final, specified statistics.evaluate_levels boundary.
        evaluation_samples = pd.read_parquet(
            DATA / "samples.parquet",
            columns=["sample_id", "label", "split"],
        )
        evaluation_started = time.perf_counter()
        evaluated = evaluate_levels(
            decisions,
            evaluation_samples,
            repeat_actions=base_actions,
            repeat_margins=repeat_margins,
            perturbation_margins=perturbation_margins,
            train_margin_sd=train_margin_sd,
            market_state=market_states,
            nuisance_r2=nuisance_r2,
            connectome_source="flywire",
            n_permutations=int(CFG["permutations"]),
            n_bootstrap=int(CFG["bootstrap_samples"]),
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        level2_effects = evaluated["level_2"]["criteria"]["seed_direction"].get("effects_by_seed", [])
        per_seed = []
        for index, seed in enumerate(seed_values):
            val_mask = np.isin(all_ids, val_ids)
            b_val = b_decisions.loc[(b_decisions.seed == seed) & b_decisions.sample_id.isin(val_id_set), "action"].to_numpy()
            c_val = c_decisions.loc[(c_decisions.seed == seed) & c_decisions.sample_id.isin(val_id_set), "action"].to_numpy()
            e_val = e_decisions.loc[(e_decisions.seed == seed) & e_decisions.sample_id.isin(val_id_set), "action"].to_numpy()
            per_seed.append(
                {
                    "seed": seed,
                    "A_val_buy_ratio": float(a_by_seed[seed]["action"][val_mask].mean()),
                    "A_val_mean_margin": float(a_val_margins[:, index].mean()),
                    "B_val_buy_ratio": float(b_val.mean()),
                    "C_val_buy_ratio": float(c_val.mean()),
                    "E_val_buy_ratio": float(e_val.mean()),
                    "level_2_input_shuffle_effect": float(level2_effects[index]) if index < len(level2_effects) else None,
                }
            )
        runtime = _runtime_summary(tracker)
        runtime["evaluate_levels_seconds"] = evaluation_seconds
        runtime["total_seconds"] = time.perf_counter() - started
        runtime["decision_A_seconds"] = a_seconds
        runtime["repeat_sensitivity_seconds"] = repeat_seconds
        runtime["train_margin_latency_ms"] = train_margin_latency
        metrics = {
            "spec": "SPEC_v3 Phase 1a, frozen Level 1-2 thresholds",
            "data_status": "development; test split is development_test_v1, not sealed holdout evidence",
            "test_split_status": test_status,
            "groups": ["fly_intact", "matched_random", "input_shuffled", "constant_input"],
            "seeds": seed_values,
            "thresholds": evaluated["thresholds"],
            "global_gate": evaluated["global_gate"],
            "validation_buy_ratio": evaluated["validation_buy_ratio"],
            "sample_counts": {
                "train": len(train),
                "val": len(val),
                "test_development": int((samples.split == "test").sum()),
                "A_samples_per_seed": len(all_ids),
            },
            "market_state": {
                "states": 6,
                "train_position_tercile_cutpoints": market_state_cuts.tolist(),
                "counts": {str(key): int(value) for key, value in market_states.value_counts().sort_index().items()},
                "rule": "sign(close[-1]/close[0]-1) x Train-tercile of last close position in [min low,max high]",
            },
            "nuisance": {
                "r2": float(nuisance_r2),
                "features": list(nuisance_features.columns),
                "margin": "per-sample mean across A seeds 0..29 on Val",
            },
            "repeat_summary": {
                "consistency": float(np.maximum(base_actions.mean(axis=1), 1.0 - base_actions.mean(axis=1)).mean()),
            },
            "level_1": evaluated["level_1"],
            "level_2": evaluated["level_2"],
            "level_3": {
                "passed": bool(evaluated["level_3"]["passed"]),
                "reason": "degree_scramble (D) is reserved for Phase 1b and is absent from this Phase 1a run.",
            },
            "level_4": {
                "passed": bool(evaluated["level_4"]["passed"] and test_status.startswith("sealed_holdout_v2")),
                "reason": "the test split is development_test_v1, not sealed_holdout_v2; no Level 4 evidence is claimed.",
            },
            "per_seed_effects": per_seed,
            "runtime": runtime,
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "total_vram_bytes": int(device.total_memory),
                "peak_device_used_bytes": int(_PEAK_DEVICE_USED_BYTES),
                "peak_device_used_fraction": float(_PEAK_DEVICE_USED_BYTES / device.total_memory),
                "reserved_fraction": float(torch.cuda.max_memory_reserved() / device.total_memory),
                "batch_policy": "TorchReservoir dynamically caps batches to <=80% total VRAM",
            },
        }
        _write_json(OUT / "metrics.json", metrics)
        _log(
            f"statistics complete in {evaluation_seconds:.1f}s; "
            f"Level 1={evaluated['level_1']['passed']}, Level 2={evaluated['level_2']['passed']}"
        )

        health["status"] = "completed"
        health["checks"]["preflight_20_val"]["passed"] = True
        health["checks"]["full_render_alignment"]["passed"] = True
        health["checks"]["seed_reproducibility"]["passed"] = True
        health["checks"]["phase1a_run"] = {
            "passed": True,
            "level_1_passed": bool(evaluated["level_1"]["passed"]),
            "level_2_passed": bool(evaluated["level_2"]["passed"]),
            "level_2_gate_failure_is_experimental_result": not bool(evaluated["level_2"]["passed"]),
        }
        health["runtime"] = runtime
        health["memory"] = metrics["memory"]
        _write_json(health_path, health)
        _log(f"Phase 1a complete in {runtime['total_seconds']:.1f}s")
        return metrics, sensitivity, health
    except Exception as error:
        health["status"] = "failed"
        health["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(health_path, health)
        _log(f"FATAL {type(error).__name__}: {error}")
        raise


def main() -> None:
    global _LOG_PATH
    OUT.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = OUT / "run.log"
    _LOG_PATH.write_text("", encoding="utf-8")
    _run_phase1a()


if __name__ == "__main__":
    main()
