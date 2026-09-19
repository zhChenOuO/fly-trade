"""持續進化腳本 —— 每次執行代表一個新「世代」的嘗試。

這是回答「有沒有賺錢、有的話要不要保留、下次再進化」的具體做法:

1. 把資料切成三段:
   - train:餵給 CMA-ES 訓練用
   - validation:訓練完之後,用來決定「這次的新解夠不夠好,值不值得
     取代現有 champion」——這段資料訓練過程完全沒看過,是真正的樣本
     外評分
   - test:從頭到尾都不參與任何決策,只在最後老實印一次「這組最終
     保留下來的參數,在完全沒用過的資料上表現如何」,避免我們自己
     用 validation 選來選去、選到最後其實也偷偷過擬合了 validation

2. 如果本機已經存有 champion(之前跑過至少一次):
   - 用「warm start」:新一代的搜尋起點設在舊 champion 參數附近(加
     一點雜訊),等於是在舊解基礎上「繼續進化」,而不是每次都從頭
     亂猜一個新解。
   - 訓練完後,新解跟舊 champion 在「同一段這次抓到的 validation 資
     料」上重新公平比較。
   - 只有新解贏過舊 champion 一定幅度(config 裡的 min_improvement,
     預設 0.05 Sharpe),才會取代成新的 champion;贏不夠多或反而更
     差,舊 champion 就原封不動留著,這次的嘗試只會被記在歷史紀錄裡。

3. 如果還沒有 champion(第一次跑),新解只要 validation Sharpe 是正
   的,就會被存成第一代 champion。

4. 不管有沒有取代 champion,這次的嘗試都會被寫進
   outputs/champion/history.csv,可以打開來看「每一代」的進化紀錄
   (有沒有進步、進步了多少、最後測試集表現如何)。

用法:
    python run_evolution.py --config configs/default.yaml

建議:每隔一段時間(例如每週)重新跑一次,讓它有機會在新資料上繼續
進化;真正想知道 champion「現在」還有沒有在賺錢,請用
run_forward_test.py,那個才是不訓練、純粹檢查最新表現的腳本。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from main import load_connectome
from src.data.market_data import load_ohlcv
from src.data.features import build_features
from src.train.evolve import ConnectomeTradingAgent, train_cma
from src.train.champion import ChampionStore
from src.backtest.engine import run_backtest, run_buy_and_hold
from src.backtest.metrics import periods_per_year


def three_way_split(n: int, train_frac: float, val_frac: float):
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))
    return n_train, n_val


def evaluate(agent, theta, features, close, fee_bps, ppy):
    actions = agent.act(features, theta)
    return run_backtest(close, actions, fee_bps=fee_bps, periods_per_year=ppy)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m, t, c = cfg["market"], cfg["train"], cfg["champion"]
    ppy = periods_per_year(m["timeframe"])

    print("== 1. 載入連接體 ==")
    graph = load_connectome(cfg)
    print(graph.summary())

    print("\n== 2. 載入市場資料 ==")
    df, source = load_ohlcv(
        symbol=m["symbol"], timeframe=m["timeframe"], limit=m["limit"],
        exchange_ids=m.get("exchange_ids"),
    )
    print(f"資料來源: {source}, 共 {len(df)} 根 K 棒 ({df.index[0]} ~ {df.index[-1]})")

    features = build_features(df)
    close = df["close"].values

    i_train, i_val = three_way_split(len(df), t["train_fraction"], t["val_fraction"])
    feat_train, feat_val, feat_test = features[:i_train], features[i_train:i_val], features[i_val:]
    close_train, close_val, close_test = close[:i_train], close[i_train:i_val], close[i_val:]
    print(f"train={len(feat_train)}  validation={len(feat_val)}  test={len(feat_test)}")

    # Buy-and-hold baseline:agent 的表現至少要能打贏「一開始就買進、
    # 全程不動」,不然代表方法(或這次的訓練結果)沒有學到有用的東西。
    bh_val = run_buy_and_hold(close_val, fee_bps=m["fee_bps"], periods_per_year=ppy)
    bh_test = run_buy_and_hold(close_test, fee_bps=m["fee_bps"], periods_per_year=ppy)
    print(
        f"buy-and-hold baseline: validation sharpe={bh_val.sharpe:+.3f} "
        f"(total_return={bh_val.total_return:+.2%})  "
        f"test sharpe={bh_test.sharpe:+.3f} (total_return={bh_test.total_return:+.2%})"
    )

    agent = ConnectomeTradingAgent(graph, n_features=features.shape[1], motor_window=t["motor_window"])
    store = ChampionStore(c["path"])

    print("\n== 3. 檢查是否已有 champion ==")
    old_theta, old_meta = None, None
    old_val_sharpe = -np.inf
    generation = 1
    if store.exists():
        old_theta, old_meta = store.load()
        generation = old_meta.get("generation", 0) + 1
        old_result = evaluate(agent, old_theta, feat_val, close_val, m["fee_bps"], ppy)
        old_val_sharpe = old_result.sharpe
        print(
            f"讀到現有 champion(第 {old_meta['generation']} 代),"
            f"用這次新抓到的 validation 資料重新評分: sharpe={old_val_sharpe:+.3f}"
        )
    else:
        print("沒有現有 champion,這是第一代。")

    x0 = None
    if old_theta is not None:
        rng = np.random.default_rng(t["seed"] + generation)
        x0 = old_theta + rng.normal(0, c.get("warm_start_noise", 0.15), size=old_theta.shape)

    print(f"\n== 4. 用 CMA-ES 訓練第 {generation} 代候選解 ==")
    print(f"可訓練參數量: {agent.layout.n_total}")
    candidate_theta, history = train_cma(
        agent, feat_train, close_train,
        sigma0=t["sigma0"], max_iter=t["max_iter"], popsize=t["popsize"],
        seed=t["seed"] + generation, x0=x0,
        fee_bps=m["fee_bps"], ppy=ppy,
    )

    candidate_train = evaluate(agent, candidate_theta, feat_train, close_train, m["fee_bps"], ppy)
    candidate_val = evaluate(agent, candidate_theta, feat_val, close_val, m["fee_bps"], ppy)

    print("\n== 5. 決定是否取代 champion ==")
    min_improve = c.get("min_improvement", 0.05)
    is_first = old_theta is None

    if is_first:
        # 第一次跑,沒有舊解可比較,但至少要求 validation 是賺錢的
        # (sharpe > 0),不然沒有意義的解不值得存成 champion。
        promoted = candidate_val.sharpe > 0
    else:
        promoted = candidate_val.sharpe > old_val_sharpe + min_improve

    if promoted:
        final_theta = candidate_theta
        final_generation = generation
        decision = "INITIAL" if is_first else "PROMOTED"
        final_val_sharpe = candidate_val.sharpe
        final_train_sharpe = candidate_train.sharpe
    elif is_first:
        # 第一代就沒訓練出賺錢的解:不存 champion,只留紀錄。
        decision = "REJECTED_NO_CHAMPION"
        final_theta = None
    else:
        final_theta = old_theta
        final_generation = old_meta["generation"]
        decision = "KEPT_OLD"
        final_val_sharpe = old_val_sharpe
        final_train_sharpe = old_meta.get("train_sharpe")

    if final_theta is None:
        store.append_history(
            {
                "generation": generation,
                "decision": decision,
                "candidate_val_sharpe": round(candidate_val.sharpe, 4),
                "old_champion_val_sharpe": "",
                "candidate_train_sharpe": round(candidate_train.sharpe, 4),
                "kept_theta_test_sharpe": "",
                "baseline_val_sharpe": round(bh_val.sharpe, 4),
                "baseline_test_sharpe": round(bh_test.sharpe, 4),
                "data_source": source,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        vs_baseline = (
            "順帶一提,連 buy-and-hold baseline 這段時間的 validation sharpe 都是 "
            f"{bh_val.sharpe:+.3f}," + ("这段行情本身就很難做。" if bh_val.sharpe <= 0 else "候選解明顯比什麼都不做還差,是方法/訓練的問題。")
        )
        print(
            f"決策: {decision}\n"
            f"這次候選解 validation sharpe = {candidate_val.sharpe:+.3f}(不是正的,不存成 champion)\n"
            f"{vs_baseline}\n"
            "建議調整 config(例如 max_iter、connectome 規模、seed)重跑,"
            "或直接接受這個方向暫時沒找到能用的解。"
        )
        return

    final_test = evaluate(agent, final_theta, feat_test, close_test, m["fee_bps"], ppy)

    meta = {
        "generation": final_generation,
        "train_sharpe": final_train_sharpe,
        "val_sharpe": final_val_sharpe,
        "test_sharpe": final_test.sharpe,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": t["seed"] + generation,
        "connectome_source": cfg["connectome"]["source"],
        "data_source": source,
        "symbol": m["symbol"],
        "timeframe": m["timeframe"],
        "n_train_bars": len(feat_train),
        "n_val_bars": len(feat_val),
        "n_test_bars": len(feat_test),
    }

    if decision != "KEPT_OLD":
        store.save(final_theta, meta)

    store.append_history(
        {
            "generation": generation,
            "decision": decision,
            "candidate_val_sharpe": round(candidate_val.sharpe, 4),
            "old_champion_val_sharpe": ("" if old_val_sharpe == -np.inf else round(old_val_sharpe, 4)),
            "candidate_train_sharpe": round(candidate_train.sharpe, 4),
            "kept_theta_test_sharpe": round(final_test.sharpe, 4),
            "baseline_val_sharpe": round(bh_val.sharpe, 4),
            "baseline_test_sharpe": round(bh_test.sharpe, 4),
            "data_source": source,
            "timestamp": meta["created_at"],
        }
    )

    print(f"決策: {decision}")
    print(f"這次候選解 validation sharpe = {candidate_val.sharpe:+.3f}  (baseline={bh_val.sharpe:+.3f})")
    if old_val_sharpe != -np.inf:
        print(f"舊 champion validation sharpe(用這次資料重新算) = {old_val_sharpe:+.3f}")
    print(
        f"\n最終保留參數(第 {final_generation} 代)在完全沒用過的 test 集上的表現:\n"
        f"  sharpe        = {final_test.sharpe:+.3f}   (baseline buy-and-hold = {bh_test.sharpe:+.3f})\n"
        f"  total_return  = {final_test.total_return:+.2%}   (baseline = {bh_test.total_return:+.2%})\n"
        f"  max_drawdown  = {final_test.max_drawdown:.2%}\n"
        f"  avg_turnover  = {final_test.turnover:.3f}"
    )
    if final_test.sharpe <= bh_test.sharpe:
        print(
            "\n注意:目前保留的參數在 test 集上並沒有贏過 buy-and-hold baseline,"
            "代表這組解暫時還不算真的學到有用的東西,建議繼續進化或重新檢討設定。"
        )
    print(f"\nchampion 存放位置: {store.path}")
    print(f"每一代的進化歷史: {store.history_path}")
    print("\n想知道 champion 現在(不重新訓練)在最新資料上是否還在賺錢,請執行 run_forward_test.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
