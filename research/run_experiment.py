"""Phase 2 整合:跑 A-E 五組,輸出 decisions.parquet / metrics.json / sensitivity.json。

只跑 val+test(不訓練,train 只用來算 constant-input 的平均圖與 market_state 的波動分箱)。
用法: cd flywire && .venv/bin/python research/run_experiment.py [--seeds 30] [--pilot]
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from src.connectome.synthetic import make_synthetic_connectome  # noqa: E402
from research.pipeline.action_decoder import decode  # noqa: E402
from research.pipeline.baselines import matched_random_from_validation  # noqa: E402
from research.pipeline.fly_simulator import FlySimulator  # noqa: E402
from research.pipeline.graph_variants import degree_preserved_scramble  # noqa: E402
from research.pipeline.render_market import render  # noqa: E402
from research.pipeline.statistics import evaluate_levels  # noqa: E402

CFG = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
W, H = CFG["window_bars"], CFG["horizon_bars"]


def load():
    samples = pd.read_parquet(ROOT / "data/samples.parquet")
    images = np.load(ROOT / "data/images.npy", mmap_mode="r")
    return samples, images


def graph(seed_cfg=None):
    return make_synthetic_connectome(**CFG["connectome"]["synthetic"])


def sim(g, seed, imgs, chunk=1000):
    """分批模擬:run() 會把整批轉 float64(2 萬張 ≈ 2GB),多 worker 並行時會 OOM。"""
    t = time.perf_counter()
    fly = FlySimulator(g, seed=seed, **CFG["sim"])
    outs = [fly.run(imgs[i:i + chunk]) for i in range(0, len(imgs), chunk)]
    buy, sell = (np.concatenate([o[k] for o in outs]) for k in ("buy_score", "sell_score"))
    return buy, sell, (time.perf_counter() - t) * 1e3 / max(len(imgs), 1)


def run_seed(args):
    """一個 seed: A(intact) / D(scramble) / E(constant)。seed 決定 noise 與 scramble 的隨機性。"""
    seed, ids, img_idx, const_img = args
    imgs = np.load(ROOT / "data/images.npy", mmap_mode="r")[img_idx]
    g = graph()
    rows = {}
    rows["fly_intact"] = sim(g, seed, imgs)
    rows["degree_scramble"] = sim(degree_preserved_scramble(g, seed=seed), seed, imgs)
    rows["constant_input"] = sim(g, seed, np.broadcast_to(const_img, (len(imgs), *const_img.shape)))
    frames = []
    for group, (b, s, lat) in rows.items():
        frames.append(pd.DataFrame({"sample_id": ids, "seed": seed, "group": group, "buy_score": b,
                                    "sell_score": s, "action": decode(b, s).astype(int), "latency_ms": lat}))
    return pd.concat(frames, ignore_index=True)


def market_state(samples, raw):
    """只用「當下與過去」: 48 根報酬符號 x 已實現波動三分位(分位點取自 train)。"""
    close = raw.close.to_numpy()
    end = raw.timestamp.searchsorted(samples.timestamp - pd.Timedelta(minutes=5))  # 決策 bar 的 open time
    idx = end[:, None] - np.arange(W - 1, -1, -1)[None]
    c = close[idx]
    ret = np.sign(c[:, -1] / c[:, 0] - 1)
    vol = np.diff(np.log(c), axis=1).std(axis=1)
    cuts = np.quantile(vol[(samples.split == "train").to_numpy()], [1 / 3, 2 / 3])
    return pd.Series((ret > 0).astype(int) * 3 + np.digitize(vol, cuts), index=samples.sample_id.to_numpy())


def get_windows(sub, raw):
    end = raw.timestamp.searchsorted(sub.timestamp - pd.Timedelta(minutes=5))
    return np.stack([raw.iloc[e - W + 1:e + 1][["open", "high", "low", "close", "volume"]].to_numpy() for e in end])


def variant_images(samples, raw, n=200, seed=0):
    """§5.2 受控變體圖片(val 子樣本): 原圖 / 上下翻轉 / 打亂時間 / 遮蔽最近 25% / 全黑 / 平均圖。"""
    rng = np.random.default_rng(seed)
    sub = samples[samples.split == "val"].sample(n, random_state=seed)
    wins = get_windows(sub, raw)
    def flip(w):
        m = w[:, 1].max() + w[:, 2].min()  # 以窗內價格區間中點鏡射
        o = w.copy(); o[:, 0], o[:, 3] = m - w[:, 0], m - w[:, 3]; o[:, 1], o[:, 2] = m - w[:, 2], m - w[:, 1]
        return o
    def mask(w):
        o = w.copy(); o[-W // 4:] = o[:W - W // 4].mean(axis=0); return o  # 最近 25% 以窗內前段均值取代
    variants = {"original": lambda w: w, "flip_price": flip,
                "shuffle_time": lambda w: w[rng.permutation(W)], "mask_recent25": mask}
    imgs = {k: np.stack([render(f(w)) for w in wins]) for k, f in variants.items()}
    stored = np.asarray(np.load(ROOT / 'data/images.npy', mmap_mode='r')[sub.image_idx.to_numpy()])
    assert (imgs['original'] == stored).all(), 'window 對齊錯誤: 重渲染與 images.npy 不一致'
    imgs["black"] = np.zeros_like(imgs["original"])
    imgs["mean_image"] = np.repeat(imgs["original"].mean(axis=0, keepdims=True).round().astype(np.uint8), n, axis=0)
    return imgs


def sensitivity(samples, raw, g, n=200, seed=0):
    imgs = variant_images(samples, raw, n, seed)
    res = {k: sim(g, seed, v)[:2] for k, v in imgs.items()}
    base = res["original"][0] - res["original"][1]
    out = {}
    for k, (b, s, *_) in res.items():
        d = b - s
        out[k] = {"action_change_rate": float(((d > 0) != (base > 0)).mean()),
                  "mean_abs_score_diff_change": float(np.abs(d - base).mean()), "buy_ratio": float((d > 0).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=CFG["seeds"])
    ap.add_argument("--pilot", action="store_true", help="每 split 只取 500 個樣本,快速檢查流程")
    ap.add_argument("--repeats-inputs", type=int, default=200)
    a = ap.parse_args()

    samples, images = load()
    raw = pd.read_parquet(ROOT / "data/raw_ohlcv.parquet")
    sel = samples[samples.split.isin(["val", "test"])].sort_values("sample_id")
    if a.pilot:
        sel = sel.groupby("split", group_keys=False).head(500)
    ids, img_idx = sel.sample_id.to_numpy(), sel.image_idx.to_numpy()
    tr = samples[samples.split == "train"].image_idx.to_numpy()[::15]
    const_img = np.asarray(images[tr]).mean(axis=0).round().astype(np.uint8)  # E: train 平均圖
    seeds = list(range(a.seeds))
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        parts = list(ex.map(run_seed, [(s, ids, img_idx, const_img) for s in seeds]))
    dec = pd.concat(parts, ignore_index=True)
    print(f"simulated {len(dec):,} rows in {time.time() - t0:.0f}s")

    # C: input-shuffled —— 每個 seed 把 intact 的決策與 sample_id 隨機錯配(同一批 score,只切斷與市場的對應)
    intact = dec[dec.group == "fly_intact"]
    shuf = []
    for s in seeds:
        d = intact[intact.seed == s].reset_index(drop=True).copy()
        perm = np.random.default_rng(10_000 + s).permutation(len(d))
        d.loc[:, ["buy_score", "sell_score", "action", "latency_ms"]] = d.loc[perm, ["buy_score", "sell_score", "action", "latency_ms"]].to_numpy()
        d["group"] = "input_shuffled"; shuf.append(d)
    # B: matched random —— BUY 機率取自 Fly-Intact 的 validation BUY 比例
    sel_samples = samples[samples.sample_id.isin(ids)]
    rnd = []
    for s in seeds:
        act = matched_random_from_validation(dec, sel_samples, len(ids), seed=20_000 + s)
        rnd.append(pd.DataFrame({"sample_id": ids, "seed": s, "group": "matched_random", "buy_score": 0.0,
                                 "sell_score": 0.0, "action": act.astype(int), "latency_ms": 0.0}))
    dec = pd.concat([dec, *shuf, *rnd], ignore_index=True)
    dec.to_parquet(out / "decisions.parquet")

    # §5.1 重複輸入: 200 個 val 輸入 x 100 次(每次不同 noise seed)
    g = graph()
    val_idx = sel[sel.split == "val"].image_idx.to_numpy()[:a.repeats_inputs]
    val_imgs = np.asarray(images[val_idx])
    reps = 100 if not a.pilot else 10
    repeat_actions = np.stack([decode(*sim(g, 50_000 + r, val_imgs)[:2]) for r in range(reps)], axis=1)

    # Level 4 replication: val 期間(早於 test、不重疊)當作另一個時期,split 標成 test 後獨立評估
    vs = sel_samples[sel_samples.split == "val"].assign(split="test")
    vd = dec[dec.sample_id.isin(vs.sample_id) & dec.group.isin(["fly_intact", "matched_random"])]
    ms = market_state(samples, raw)
    levels = evaluate_levels(
        dec, sel_samples, repeat_actions=repeat_actions, market_state=ms.reindex(sel_samples.sample_id),
        connectome_source=CFG["connectome"]["source"], replication_decisions=vd, replication_samples=vs,
        n_permutations=CFG["permutations"] if not a.pilot else 100,
        n_bootstrap=CFG["bootstrap_samples"] if not a.pilot else 100)
    (out / "metrics.json").write_text(json.dumps(levels, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (out / "sensitivity.json").write_text(json.dumps(sensitivity(samples, raw, g), indent=2))
    print("highest_level =", levels["highest_level"])
    for k in ("level_1", "level_2", "level_3", "level_4"):
        print(k, levels[k]["passed"], {c: v["passed"] for c, v in levels[k]["criteria"].items()})


if __name__ == "__main__":
    main()
