# 實作契約(所有 agent 必須遵守,方便並行開發後直接拼接)

方法論來源: ../research.md(必讀)。設定來源: config/experiment.yaml。
Python: `cd /Users/zh/Documents/playground/flywire && .venv/bin/python -m pytest research/tests`(用 .venv,系統 python 沒有 pandas)。
新程式碼放 `research/pipeline/`(不叫 src,避免跟 flywire/src 撞名)。重用既有: `src.connectome.*`(ConnectomeGraph、make_synthetic_connectome)、`src.network.reservoir.LeakyReservoir`(從 flywire/ 根目錄 import)。**不得修改 flywire/src、flywire/*.py 既有檔案**;不得 git commit。
只動自己負責的檔案。

## 檔案所有權
| Agent | 負責檔案 |
|---|---|
| data  | pipeline/build_dataset.py, pipeline/render_market.py, tests/test_no_future_leakage.py, data/* |
| sim   | pipeline/fly_simulator.py, pipeline/action_decoder.py, pipeline/graph_variants.py, tests/test_determinism.py, tests/test_action_mapping.py, smoke_test.py |
| stats | pipeline/statistics.py, pipeline/baselines.py, tests/test_statistics.py |
(run_experiment.py 由 Phase 2 整合,現在不要寫)

## 資料契約 (data → 其他人)
- `data/raw_ohlcv.parquet`: 欄位 timestamp(UTC, datetime64[ns]), open, high, low, close, volume;已去重、檢查 5m 連續。
- `data/samples.parquet`: 欄位 `sample_id(int, 0..N-1, 依時間遞增), timestamp(決策時間 t = 第48根K線收盤時間), image_idx(int, 對應 images.npy 第幾張), future_return, label(1=UP,0=DOWN), split(train|val|test)`。
- `data/images.npy`: uint8 (N,64,64,3),與 sample_id 同序。另存 100 張人工檢查用 PNG 到 data/audit/。
- `data/splits.json`(各 split 的 sample_id 範圍 + embargo)、`data/data_hash.txt`(raw_ohlcv.parquet 的 sha256)。
- `pipeline/render_market.py`: 純函式 `render(window: np.ndarray[48,5] (OHLCV)) -> np.ndarray uint8 (64,64,3)`。只用該 window 內資料正規化(不可用全域統計);圖片不含任何文字/時間資訊。
- 樣本 t 只能用 K 線 [t-47..t](含 t 的收盤);future_return = close[t+6]/close[t]-1。

## 模擬器契約 (sim → stats / 整合)
- `FlySimulator(graph: ConnectomeGraph, seed: int, steps=32, gain=1.0, leak=0.5, noise_std=0.0)`;`.run(images: uint8 (B,64,64,3)) -> dict(buy_score=(B,), sell_score=(B,))`。權重固定,不學習。
- 像素→sensory neuron 的映射固定、不可學習、不可隨 seed 改變(seed 只影響 noise)。BUY/SELL 為 graph.motor_idx 的兩組互斥子集;score = 該組神經元在模擬窗內的總活動。
- `action_decoder.decode(buy, sell) -> np.ndarray(0=SELL,1=BUY)`,平手→SELL。
- `graph_variants.py`: `degree_preserved_scramble(graph, seed) -> ConnectomeGraph`(保留每個節點入/出度與各邊符號分布 — 例如 double-edge-swap;sensory/motor idx 不變)。
- 一個 latency 量測欄位 latency_ms 由整合階段量測,simulator 不需處理。

## decisions 契約 (整合 → stats)
`outputs/decisions.parquet`: `sample_id, seed, group, buy_score, sell_score, action, latency_ms`;group ∈ {fly_intact, matched_random, input_shuffled, degree_scramble, constant_input}。標籤從 data/samples.parquet 以 sample_id join。

## 統計契約 (stats)
`pipeline/statistics.py` 純 numpy/pandas/scipy(sklearn 未安裝,不要加),皆為可測試的純函式:
consistency(actions_matrix), mutual_information(x_discrete, action), balanced_accuracy, mcc, precision_by_side, longest_run, buy_ratio, bootstrap_ci(stat_fn, y, pred, n, seed), empirical_p(real, null_samples) = (1+#(null>=real))/(1+n), input_shuffle_permutation_test(stat_fn, y, pred, n, seed), seed_direction_agreement(effects) 。
`pipeline/baselines.py`: matched_random(p_buy, n, seed), constant_input_profile 相關的比較函式。
`evaluate_levels(...)`: 依 research.md §6 的 Level 1–4 門檻回傳結構化 dict(每條準則 pass/fail + 數值)。
`tests/test_statistics.py`: 用合成 decisions 驗證 —— 無訊號時 p 分佈近似均勻/不會宣稱 Level 2+;植入訊號時能偵測到。
