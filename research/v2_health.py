"""input_encoding_v2 健康檢查與編碼選擇。規則見 outputs/v2/selection_rule.md(預先註冊)。
只用 Train/Val 圖片與 OHLCV,不讀 label / future_return / test。用法: cd flywire && .venv/bin/python research/v2_health.py
"""
from __future__ import annotations

import hashlib, json, sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.run_experiment import CFG, ROOT, W, get_windows, graph, load, variant_images  # noqa: E402
from research.pipeline.encoding_v2 import MODES, calibrate_baseline, encoder  # noqa: E402
from research.pipeline.fly_simulator import FlySimulator  # noqa: E402

OUT = ROOT / "outputs/v2"
N_VAL, N_REP_IN, N_REP, N_CAL = 1000, 200, 10, 2000
rng = np.random.default_rng(0)


def margins(sim, imgs, chunk=500):
    o = [sim.run(imgs[i:i + chunk]) for i in range(0, len(imgs), chunk)]
    return np.concatenate([x["buy_score"] for x in o]) - np.concatenate([x["sell_score"] for x in o])


def nuisance(imgs, wins):
    x = imgs.astype(np.float64) / 255.0
    close, ret = wins[:, :, 3], np.diff(np.log(wins[:, :, 3]), axis=1)
    return pd.DataFrame({
        "mean_brightness": x.mean(axis=(1, 2, 3)), "nonzero_pixel_ratio": (imgs.sum(-1) > 0).mean(axis=(1, 2)),
        "total_pixel_energy": (x ** 2).sum(axis=(1, 2, 3)),
        "price_vertical_range": (wins[:, :, 1].max(1) - wins[:, :, 2].min(1)) / close[:, -1],
        "n_up_candles": (wins[:, :, 3] >= wins[:, :, 0]).sum(1), "n_down_candles": (wins[:, :, 3] < wins[:, :, 0]).sum(1),
        "volatility": ret.std(axis=1)})


def r2(y, X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.column_stack([np.ones(len(y)), X])
    res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - res.var() / y.var())


def main():
    samples, images = load()
    raw = pd.read_parquet(ROOT / "data/raw_ohlcv.parquet")
    tr = samples[samples.split == "train"]
    mean_img = np.asarray(images[tr.image_idx.to_numpy()[::5]]).astype(np.float64).mean(axis=0)
    np.save(OUT / "train_mean_image.npy", mean_img)
    cal = np.asarray(images[tr.image_idx.to_numpy()[:: max(len(tr) // N_CAL, 1)][:N_CAL]])
    val = samples[samples.split == "val"].sample(N_VAL, random_state=0).sort_values("sample_id")
    val_imgs, val_wins = np.asarray(images[val.image_idx.to_numpy()]), get_windows(val, raw)
    variants = variant_images(samples, raw, 200, 0)
    nui = nuisance(val_imgs, val_wins)
    g = graph()
    report = {}
    for mode in MODES:
        enc = encoder(mode, mean_img)
        mk = lambda bl, seed=0: FlySimulator(g, seed=seed, **CFG["sim"], encoder=enc, baseline=bl)
        bl = calibrate_baseline(lambda _: mk(None), cal)
        sd_train = float(margins(mk(bl), cal).std())
        mv = margins(mk(bl), val_imgs)
        rep = np.stack([margins(mk(bl, 100 + r), val_imgs[:N_REP_IN]) for r in range(N_REP)], axis=1)
        within, between = rep.var(axis=1, ddof=1).mean(), rep.mean(axis=1).var(ddof=1)
        vm = {k: margins(mk(bl), v) for k, v in variants.items()}
        pert = {k: float(np.abs(vm[k] - vm["original"]).mean() / sd_train) for k in ("flip_price", "shuffle_time", "mask_recent25", "black", "mean_image")}
        mean_m = margins(mk(bl, 7), np.repeat(np.clip(mean_img.round(), 0, 255).astype(np.uint8)[None], N_VAL, axis=0))
        feats = nui.to_numpy()
        gates = {
            "G1_minority_ratio": float(min((mv > 0).mean(), (mv <= 0).mean())),
            "G2_finite_nonconst": bool(np.isfinite(mv).all() and mv.std() > 0),
            "G3_between_within": float(between / within),
            "G4_perturbation": max(pert["flip_price"], pert["shuffle_time"], pert["mask_recent25"]),
            "G5_ks_p_vs_mean_image": float(ks_2samp(mv, mean_m).pvalue),
        }
        passed = (gates["G1_minority_ratio"] >= 0.05 and gates["G2_finite_nonconst"] and gates["G3_between_within"] >= 3
                  and gates["G4_perturbation"] >= 0.2 and gates["G5_ks_p_vs_mean_image"] < 0.05)
        report[mode] = {"passed": bool(passed), "gates": gates, "perturbation_effect": pert, "sd_train_margin": sd_train,
                        "margin_val": {"mean": float(mv.mean()), "std": float(mv.std()), "p5_p95": np.percentile(mv, [5, 95]).tolist()},
                        "nuisance_r2_full": r2(mv, feats),
                        "nuisance_r2_shortcut": r2(mv, nui[["mean_brightness", "nonzero_pixel_ratio", "total_pixel_energy"]].to_numpy()),
                        "nuisance_spearman": {c: float(spearmanr(mv, nui[c]).statistic) for c in nui}, "baseline": bl}
        print(f"{mode:11s} pass={passed} minority={gates['G1_minority_ratio']:.3f} b/w={gates['G3_between_within']:.1f} "
              f"pert={gates['G4_perturbation']:.2f} ks_p={gates['G5_ks_p_vs_mean_image']:.3g} "
              f"nuisR2={report[mode]['nuisance_r2_full']:.3f} shortcutR2={report[mode]['nuisance_r2_shortcut']:.3f}")
    ok = [m for m in MODES if report[m]["passed"]]
    chosen = None
    if ok:
        best = min(report[m]["nuisance_r2_full"] for m in ok)
        chosen = next(m for m in MODES if m in ok and report[m]["nuisance_r2_full"] - best < 0.02)  # 平手依 MODES 順序
    report["chosen"] = chosen
    (OUT / "health.json").write_text(json.dumps(report, indent=2))
    if chosen is None:
        print("NO ENCODING PASSED -> 停止,不進入 Level 1-3"); sys.exit(1)
    freeze = {"version": "input_encoding_v2", "mode": chosen, "baseline": report[chosen]["baseline"],
              "train_mean_image_sha256": hashlib.sha256((OUT / "train_mean_image.npy").read_bytes()).hexdigest(),
              "train_range": [tr.timestamp.min().isoformat(), tr.timestamp.max().isoformat()], "n_train": len(tr),
              "experiment_yaml_sha256": hashlib.sha256((ROOT / "config/experiment.yaml").read_bytes()).hexdigest(),
              "decision_rule": "margin = z_buy - z_sell > 0 -> BUY, else SELL (frozen)"}
    (OUT / "encoding_v2.json").write_text(json.dumps(freeze, indent=2))
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
