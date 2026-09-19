"""事後(不重訓)重算:同一批已訓練模型的 logits,換部位映射 / 手續費 / seed 子集。
不是 wf.py 執行,但每個變體都是對 OOS 的一次「看」,REPORT.md 會全數列出以計多重檢定。
    python posthoc.py outputs/sweeps/x_logits.pkl            # 內建預先寫死的網格
    python posthoc.py x_logits.pkl --pos '{"long_only":true}' --fee 10 --seeds 0,1,2
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from wf import summarize
from src.backtest.metrics import periods_per_year

GRID = [  # 預先固定,不看結果調整
    ("base", None, 5), ("long_only", {"long_only": True}, 5),
    ("min_hold6", {"min_hold": 6}, 5), ("min_hold24", {"min_hold": 24}, 5),
    ("margin_q50", "Q50", 5), ("margin_q75", "Q75", 5),
    ("vol0.3", {"vol_target": 0.3}, 5), ("vol0.5", {"vol_target": 0.5}, 5),
    ("LO+mh6", {"long_only": True, "min_hold": 6}, 5),
    ("LO+mh6+vol0.4", {"long_only": True, "min_hold": 6, "vol_target": 0.4}, 5),
    ("base fee10", None, 10), ("base fee20", None, 20),
    ("LO+mh6 fee10", {"long_only": True, "min_hold": 6}, 10), ("LO+mh6 fee20", {"long_only": True, "min_hold": 6}, 20),
]

def row(d, label, pcfg, fee, seeds):
    cfg = json.loads(json.dumps(d["cfg"])); cfg["pos"] = pcfg
    s = summarize(label, False, cfg, d["res"], d["fl"], seeds, d["assets"], d["train_sharpe_mean"], 0, fee_bps=fee)
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pkl"); ap.add_argument("--pos"); ap.add_argument("--fee", type=float, default=5.0)
    ap.add_argument("--seeds"); a = ap.parse_args()
    d = pickle.load(open(a.pkl, "rb")); seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else d["seeds"]
    import io, contextlib
    def quiet(*args):
        with contextlib.redirect_stdout(io.StringIO()):
            return row(d, *args)
    if a.pos is not None:
        s = quiet("custom", json.loads(a.pos), a.fee, seeds)
        print({k: s[k] for k in ["ensemble_oos_sharpe", "sharpe_se", "seed_oos_sharpe_min", "folds_positive", "oos_total_return", "oos_max_drawdown"]}); sys.exit()
    # logit gap 的分位數(只當尺度用)
    gaps = []
    for st in d["res"].values():
        L = st[d["assets"][0]][0]; ss = np.sort(L, 1); gaps.append(ss[:, -1] - ss[:, -2])
    g = np.concatenate(gaps); q = {"Q50": float(np.quantile(g, .5)), "Q75": float(np.quantile(g, .75))}
    print("logit gap quantiles", q)
    print(f"{'variant':16s} seeds  ens_sh   se  seed_min folds  ret    mdd  {'  '.join(x[:3] for x in d['assets'])}")
    for ns in sorted({3, 5, len(d["seeds"])}):
        if ns > len(d["seeds"]): continue
        for label, pc, fee in GRID:
            if ns != 3 and label not in ("base", "LO+mh6", "LO+mh6+vol0.4"): continue
            if pc in ("Q50", "Q75"): pc = {"margin": q[pc]}
            s = quiet(label, pc, fee, d["seeds"][:ns])
            pa = " ".join(f"{v['ens_sharpe']:5.2f}" for v in s["per_asset_oos"].values())
            print(f"{label:16s} {ns:4d} {s['ensemble_oos_sharpe']:7.3f} {s['sharpe_se']:5.2f} {s['seed_oos_sharpe_min']:7.3f} {s['folds_positive']:5s} {s['oos_total_return']:6.2f} {s['oos_max_drawdown']:6.2f}  {pa}")
