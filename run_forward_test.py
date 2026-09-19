"""不訓練,只用現有 champion 在最新一段資料上跑一次回測,檢查它「現在」
是不是還在賺錢——這是持續監控用的腳本,對應「有賺錢就繼續觀察,變差
了再決定要不要重新進化」這個流程。

跟 run_evolution.py 的差別:
- run_evolution.py 會訓練新解、可能取代 champion。
- run_forward_test.py 完全不訓練,只是拿「現在存著的 champion」去跑
  最新一段資料,單純觀察、留紀錄。

建議用法:每隔一段時間(例如每天或每週)重新執行一次。它會抓最新的
市場資料,只取「最後 forward_test_bars 根 K 棒」來評分,並把結果累積
寫進 outputs/champion/forward_ledger.csv,讓你可以攤開一段時間的紀錄,
觀察 champion 的表現是穩定、變好、還是在變差(變差到一定程度,就是
「該回去跑 run_evolution.py 重新進化,或乾脆放棄這個方向」的訊號)。

用法:
    python run_forward_test.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from main import load_connectome
from src.data.market_data import load_ohlcv
from src.data.features import build_features
from src.train.evolve import ConnectomeTradingAgent
from src.train.champion import ChampionStore
from src.backtest.engine import run_backtest, run_buy_and_hold
from src.backtest.metrics import periods_per_year


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m, t, c = cfg["market"], cfg["train"], cfg["champion"]

    store = ChampionStore(c["path"])
    if not store.exists():
        print(
            "還沒有存過 champion。請先執行一次:\n"
            "    python run_evolution.py --config " + config_path
        )
        return

    theta, meta = store.load()

    graph = load_connectome(cfg)
    df, source = load_ohlcv(
        symbol=m["symbol"], timeframe=m["timeframe"], limit=m["limit"],
        exchange_ids=m.get("exchange_ids"),
    )
    features = build_features(df)
    close = df["close"].values

    fwd_bars = min(c.get("forward_test_bars", 200), len(df) - 1)
    feat_fwd = features[-fwd_bars:]
    close_fwd = close[-fwd_bars:]
    period_start, period_end = df.index[-fwd_bars], df.index[-1]

    agent = ConnectomeTradingAgent(graph, n_features=features.shape[1], motor_window=t["motor_window"])
    actions = agent.act(feat_fwd, theta)
    result = run_backtest(close_fwd, actions, fee_bps=m["fee_bps"], periods_per_year=periods_per_year(m["timeframe"]))
    baseline = run_buy_and_hold(close_fwd, fee_bps=m["fee_bps"], periods_per_year=periods_per_year(m["timeframe"]))

    print(f"champion:第 {meta['generation']} 代(存檔於 {meta.get('created_at', '未知時間')})")
    print(f"資料來源: {source}")
    print(f"檢查區間: {period_start} ~ {period_end} (最新 {fwd_bars} 根 K 棒,champion 沒訓練過這段)")
    print(
        f"\n結果:\n"
        f"  sharpe        = {result.sharpe:+.3f}   (baseline buy-and-hold = {baseline.sharpe:+.3f})\n"
        f"  total_return  = {result.total_return:+.2%}   (baseline = {baseline.total_return:+.2%})\n"
        f"  max_drawdown  = {result.max_drawdown:.2%}\n"
        f"  avg_turnover  = {result.turnover:.3f}"
    )

    if result.sharpe > 0 and result.sharpe > baseline.sharpe:
        print("\n=> 這段最新資料是賺錢的,而且贏過 buy-and-hold,champion 先繼續保留、觀察。")
    elif result.sharpe > 0:
        print("\n=> 這段有賺錢,但沒贏過 buy-and-hold baseline,代表這段時間單純長期持有可能還比較好。")
    else:
        print("\n=> 這段最新資料是虧錢的,建議多觀察幾次(單一區間雜訊很大),"
              "若持續變差再考慮重新執行 run_evolution.py 進化,或重新檢討整個方法。")

    store.append_forward_ledger(
        {
            "generation": meta["generation"],
            "period_start": str(period_start),
            "period_end": str(period_end),
            "n_bars": fwd_bars,
            "sharpe": round(result.sharpe, 4),
            "total_return": round(result.total_return, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "baseline_sharpe": round(baseline.sharpe, 4),
            "baseline_total_return": round(baseline.total_return, 4),
            "data_source": source,
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    print(f"\n已寫入監控紀錄: {store.forward_ledger_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
