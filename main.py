"""端到端執行入口。

用法:
    python main.py --config configs/default.yaml

流程:
    1. 載入連接體(預設用合成資料,設定檔改成 hemibrain/flywire 可換真實資料)
    2. 抓/生成市場資料,做特徵工程
    3. 切訓練集/測試集(避免用測試集資訊訓練,防止過擬合的假象)
    4. 用 CMA-ES 訓練輸入編碼層 + 讀出層
    5. 在測試集上跑回測,印出績效指標
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.connectome.synthetic import make_synthetic_connectome
from src.connectome.build_graph import load_cache
from src.data.market_data import load_ohlcv
from src.data.features import build_features
from src.train.evolve import ConnectomeTradingAgent, train_cma
from src.backtest.engine import run_backtest
from src.backtest.metrics import periods_per_year


def load_connectome(cfg: dict):
    src = cfg["connectome"]["source"]
    if src == "synthetic":
        sc = cfg["connectome"]["synthetic"]
        return make_synthetic_connectome(**sc)
    elif src == "cache":
        return load_cache(cfg["connectome"]["cache_path"])
    elif src == "hemibrain":
        from src.connectome.hemibrain_loader import load_hemibrain_connectome
        return load_hemibrain_connectome(**cfg["connectome"].get("hemibrain", {}))
    elif src == "flywire":
        from src.connectome.flywire_loader import load_flywire_connectome
        return load_flywire_connectome(**cfg["connectome"]["flywire"])
    else:
        raise ValueError(f"未知的 connectome.source: {src}")


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("== 1. 載入連接體 ==")
    graph = load_connectome(cfg)
    print(graph.summary())

    print("\n== 2. 載入市場資料 ==")
    m = cfg["market"]
    df, source = load_ohlcv(
        symbol=m["symbol"], timeframe=m["timeframe"], limit=m["limit"],
        exchange_ids=m.get("exchange_ids"),
    )
    print(f"資料來源: {source}, 共 {len(df)} 根 K 棒")

    features = build_features(df)
    close = df["close"].values

    print("\n== 3. 切分訓練/測試集 ==")
    t = cfg["train"]
    split = int(len(df) * t["train_fraction"])
    feat_train, feat_test = features[:split], features[split:]
    close_train, close_test = close[:split], close[split:]
    print(f"訓練集 {split} 筆, 測試集 {len(df) - split} 筆")

    print("\n== 4. 建立 agent 並用 CMA-ES 訓練 ==")
    agent = ConnectomeTradingAgent(graph, n_features=features.shape[1], motor_window=t["motor_window"])
    print(f"可訓練參數量: {agent.layout.n_total} (跟連接體神經元/突觸數量無關)")

    best_theta, history = train_cma(
        agent,
        feat_train,
        close_train,
        sigma0=t["sigma0"],
        max_iter=t["max_iter"],
        popsize=t["popsize"],
        seed=t["seed"],
        fee_bps=m["fee_bps"],
        ppy=periods_per_year(m["timeframe"]),
    )

    print("\n== 5. 測試集回測 ==")
    train_actions = agent.act(feat_train, best_theta)
    train_result = run_backtest(close_train, train_actions, fee_bps=m["fee_bps"], periods_per_year=periods_per_year(m["timeframe"]))
    test_actions = agent.act(feat_test, best_theta)
    test_result = run_backtest(close_test, test_actions, fee_bps=m["fee_bps"], periods_per_year=periods_per_year(m["timeframe"]))

    def report(name, r):
        print(
            f"[{name}] sharpe={r.sharpe:+.3f}  total_return={r.total_return:+.2%}  "
            f"max_drawdown={r.max_drawdown:.2%}  avg_turnover={r.turnover:.3f}"
        )

    report("train", train_result)
    report("test ", test_result)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "best_theta.npy", best_theta)
    np.save(out_dir / "train_equity_curve.npy", train_result.equity_curve)
    np.save(out_dir / "test_equity_curve.npy", test_result.equity_curve)
    print(f"\n已將訓練結果存到 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
