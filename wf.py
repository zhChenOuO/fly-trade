"""Walk-forward 評估 —— 調參時唯一該相信的數字。

    python wf.py --config configs/x.yaml --name x --workers 2

流程:
  資料 = [ dev(前 1-holdout_frac) | holdout(最後 holdout_frac,封存) ]
  dev 內做「擴張視窗」walk-forward:第 k 折用 [0, s_k) 訓練、[s_k, s_k+L) 測試,
  所有折的樣本外報酬串起來算一個 Sharpe。每折 x 每個 seed 各訓練一次;
  headline = 多 seed 部位平均(集成)的 OOS Sharpe。

  holdout 只有加 --final 才會碰(用全部 dev 訓練 -> 測 holdout)。調參的人不准用。
  Sharpe 的標準誤 ≈ sqrt((1 + SR²/2) / 年數),務必一起看:OOS 只有 ~3 年時 SE 約 0.8。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from main import load_connectome
from src.backtest.engine import run_backtest_pos, run_buy_and_hold
from src.backtest.position import to_positions
from src.backtest.metrics import periods_per_year, sharpe_ratio, max_drawdown
from src.data.features import build_features
from src.data.market_data import load_ohlcv_cached, make_synthetic_ohlcv
from src.train.evolve import ConnectomeTradingAgent, train_cma


def _job(args):
    cfg, assets, tr_end, te_end, seed = args  # assets: [(name, ts, feats, close)],第 0 個是主資產(BTC)
    m, t = cfg["market"], cfg["train"]
    pcfg = cfg.get("pos")
    ppy = periods_per_year(m["timeframe"])
    t0 = assets[0][1]
    win = []  # 每個資產對應主資產這一折的 [訓練結束, 測試結束) 索引;以「時間戳」對齊,訓練只用 < 測試起點的資料
    for name, ts, f, c in assets:
        win.append((int(np.searchsorted(ts, t0[tr_end])), len(ts) if te_end >= len(t0) else int(np.searchsorted(ts, t0[te_end]))))
    feats, close = assets[0][2], assets[0][3]
    agent = ConnectomeTradingAgent(load_connectome(cfg), n_features=feats.shape[1], motor_window=t["motor_window"])
    extra = [(a[2][:w[0]], a[3][:w[0]]) for a, w in zip(assets[1:], win[1:]) if w[0] >= 500]
    theta, _ = train_cma(
        agent, feats[:tr_end], close[:tr_end], sigma0=t["sigma0"], max_iter=t["max_iter"],
        popsize=t.get("popsize"), seed=seed, verbose=False, fee_bps=m["fee_bps"], ppy=ppy,
        l2=t.get("l2", 0.0), turnover_pen=t.get("turnover_pen", 0.0), expo_floor=t.get("expo_floor", 0.0), extra=extra, pcfg=pcfg,
    )
    # 測試段前面多帶一段 warmup(reservoir 需要暖機),但只計分測試段;只用到過去,不會漏
    store = {}
    for (name, ts, f, c), (sa, eb) in zip(assets, win):
        if eb - sa < 50:
            continue
        w = min(sa, 200); lo = sa - w
        store[name] = (agent.logits(f[lo:eb], theta), c[lo:eb], w)
    train_sharpe = run_backtest_pos(close[:tr_end], to_positions(agent.logits(feats[:tr_end], theta), close[:tr_end], pcfg, ppy),
                                    m["fee_bps"], ppy).sharpe
    return tr_end, seed, store, train_sharpe


def evaluate(res, fl, seeds, asset_names, pcfg, fee_bps, ppy):
    """res[(fold_start, seed)] = store。回傳 {asset: (ens_rets, bh_rets, seed_rets{seed:[..]}, fold_sharpes)}。
    只依賴存下來的 logits,所以可以事後改部位映射/手續費重算(posthoc.py)。"""
    out = {}
    for nm in asset_names:
        ens, bh, seed_rets, fs = [], [], {s: [] for s in seeds}, []
        for (a, b) in fl:
            if nm not in res[(a, seeds[0])]:
                continue
            pos = []
            for s in seeds:
                L, C, w = res[(a, s)][nm]
                pos.append(to_positions(L, C, pcfg, ppy)[w:])
            c = res[(a, seeds[0])][nm][1][res[(a, seeds[0])][nm][2] - 1:]
            pos = np.stack(pos)
            for i, s in enumerate(seeds):
                seed_rets[s].append(run_backtest_pos(c[1:], pos[i], fee_bps, ppy).strategy_returns)
            r = run_backtest_pos(c[1:], pos.mean(axis=0), fee_bps, ppy)
            ens.append(r.strategy_returns); fs.append(r.sharpe)
            bh.append(run_buy_and_hold(c[1:], fee_bps, ppy).strategy_returns)
        if ens:
            out[nm] = (np.concatenate(ens), np.concatenate(bh), {s: np.concatenate(v) for s, v in seed_rets.items()}, fs)
    return out


def folds(n_dev: int, init_frac: float, n_folds: int):
    s0 = int(n_dev * init_frac)
    L = (n_dev - s0) // n_folds
    return [(s0 + k * L, s0 + (k + 1) * L) for k in range(n_folds)]


def load_assets(cfg):
    m = cfg["market"]
    syms = [m["symbol"]] + list(m.get("pool", []))
    assets = []
    for sy in syms:
        df = make_synthetic_ohlcv(n_bars=m["limit"], seed=m.get("seed", 0), sigma=0.01) if m.get("synthetic") else \
            load_ohlcv_cached(sy, m["timeframe"], m["limit"], m.get("exchange_ids"))
        assets.append((sy, df.index.values, build_features(df), df["close"].values))
    return assets


def main(cfg: dict, name: str, workers: int, final: bool):
    m, w = cfg["market"], cfg["wf"]
    pcfg = cfg.get("pos")
    ppy = periods_per_year(m["timeframe"])
    assets = load_assets(cfg)
    n = len(assets[0][1])
    n_dev = int(n * (1 - w["holdout_frac"]))
    fl = [(n_dev, n)] if final else folds(n_dev, w["init_frac"], w["n_folds"])
    seeds = w["seeds"]
    print(f"[{name}] bars={n} dev={n_dev} {'FINAL holdout' if final else f'{len(fl)} folds'} x {len(seeds)} seeds, "
          f"assets={[a[0] for a in assets]}, pos={pcfg}, workers={workers}")

    jobs = [(cfg, assets, a, b, s) for (a, b) in fl for s in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_job, jobs))
    res = {(tr, s): st for tr, s, st, _ in out}
    train_sh = float(np.mean([ts for *_, ts in out]))
    out_dir = Path("outputs/sweeps"); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}{'_FINAL' if final else ''}_logits.pkl", "wb") as f:
        pickle.dump({"res": res, "fl": fl, "seeds": seeds, "assets": [a[0] for a in assets], "cfg": cfg, "train_sharpe_mean": train_sh}, f)

    return summarize(name, final, cfg, res, fl, seeds, [a[0] for a in assets], train_sh, time.time() - t0, save=True)


def summarize(name, final, cfg, res, fl, seeds, names, train_sh, elapsed, save=False, fee_bps=None):
    m = cfg["market"]; pcfg = cfg.get("pos"); ppy = periods_per_year(m["timeframe"])
    fee = m["fee_bps"] if fee_bps is None else fee_bps
    ev = evaluate(res, fl, seeds, names, pcfg, fee, ppy)
    ens, bh, seed_rets, fold_ens_sharpe = ev[names[0]]
    years = len(ens) / ppy
    sh = sharpe_ratio(ens, ppy)
    seed_sh = [sharpe_ratio(seed_rets[s], ppy) for s in seeds]
    eq = np.cumprod(1 + ens)
    summary = {
        "name": name, "final": final,
        "ensemble_oos_sharpe": round(sh, 3),
        "sharpe_se": round(float(np.sqrt((1 + sh**2 / 2) / years)), 3),
        "seed_oos_sharpe_mean": round(float(np.mean(seed_sh)), 3),
        "seed_oos_sharpe_min": round(float(np.min(seed_sh)), 3),
        "buy_hold_sharpe_same_bars": round(sharpe_ratio(bh, ppy), 3),
        "oos_total_return": round(float(eq[-1] - 1), 3),
        "oos_max_drawdown": round(max_drawdown(eq), 3),
        "folds_positive": f"{sum(x > 0 for x in fold_ens_sharpe)}/{len(fold_ens_sharpe)}",
        "fold_sharpes": [round(x, 2) for x in fold_ens_sharpe],
        "train_sharpe_mean": round(train_sh, 3),
        "per_asset_oos": {k: {"ens_sharpe": round(sharpe_ratio(v[0], ppy), 3), "bh_sharpe": round(sharpe_ratio(v[1], ppy), 3),
                              "folds_pos": f"{sum(x > 0 for x in v[3])}/{len(v[3])}"} for k, v in ev.items()},
        "oos_years": round(years, 2), "fee_bps": fee, "elapsed_s": round(elapsed), "cfg": cfg,
    }
    if save:
        out_dir = Path("outputs/sweeps")
        (out_dir / f"{name}{'_FINAL' if final else ''}.json").write_text(json.dumps(summary, indent=1, default=str))
    show = {k: v for k, v in summary.items() if k != "cfg"}
    print(json.dumps(show, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--final", action="store_true", help="碰封存 holdout。只有最終驗證的人可以用。")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.name, a.workers, a.final)
