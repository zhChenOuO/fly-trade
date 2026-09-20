果蠅 Connectome 監督式訓練實作計畫

1. 目標

建立一套可重現、無未來資料洩漏的流程，回答三個問題：

固定的果蠅 Connectome 活動是否包含市場輸入資訊？

Connectome 活動是否包含可預測未來報酬的增量資訊？

在確認預測訊號後，reward learning 是否能改善實際交易決策？

本階段先採用監督式學習。Reward learning 只有在監督式實驗通過預定門檻後才開始。

2. 核心設計

將「預測」與「交易」分開：

歷史 OHLCV / K 線圖
        ↓
Connectome 模擬
        ↓
神經活動特徵
        ↓
監督式 Decoder
        ↓
未來報酬預測
        ↓
含成本的交易規則
        ↓
BUY / HOLD / SELL

預測模型不知道目前是否持倉，只預測未來市場變化。

交易規則另外接收目前持倉、手續費、滑價與風險限制。

不直接把每根 K 線硬標成 BUY 或 SELL。

3. 實驗版本與資料治理

3.1 舊實驗處理

舊的 Test 已用於發現輸入映射問題，因此：

dataset_role = development_test_v1
result_status = INVALID_INPUT_MAPPING
highest_level = 0

舊資料仍可用於除錯，但不得再用作正式 Level 4 證據。

3.2 時間切割

必須按照時間切割，禁止隨機拆分：

Train → Validation → Future Sealed Holdout

切割邊界加入 purge／embargo，至少隔開：

input_window + prediction_horizon

例如輸入 48 根、預測未來 6 根，兩個 split 邊界至少排除相鄰 54 根資料，避免窗口重疊。

3.3 Train-only 產物

下列項目只能由 Train 計算：

平均圖、標準差與 normalization。

Connectome activity baseline。

BUY／SELL 神經群的活性校正值。

標籤 threshold。

Decoder 參數與超參數。

每個產物記錄資料期間、程式 commit、設定檔與 SHA256。

4. 資料與標籤

4.1 初始資料規格

先沿用以下起始設定，之後只能用 Train／Validation 調整：

interval: 5m
input_window_bars: 48
prediction_horizons: [1, 3, 6, 12]
primary_horizon: 6

每個樣本只能包含時間 t 當下已知的資料：

OHLC 報酬率。

成交量及成交量變化率。

過去窗口波動率。

K 線影像。

時間特徵，例如星期與時段；不能加入任何未來欄位。

4.2 主要標籤

優先預測連續未來報酬：

future_return_h = log(close[t+h] / close[t])

建議同時保留：

future_return_1
future_return_3
future_return_6
future_return_12
future_volatility_6
maximum_favorable_excursion_6
maximum_adverse_excursion_6

第一版 primary target 使用 future_return_6。

4.3 動作標籤

動作由未來報酬和成本門檻產生：

future_return >  cost_threshold → BUY
future_return < -cost_threshold → SELL
其他                            → HOLD

其中：

cost_threshold >= 雙邊手續費 + 預估滑價

若 SELL 代表建立空單，必須另外標明；若只允許現貨，SELL 應表示減倉或平倉。

5. 輸入編碼 v2

5.1 候選編碼

只在 Train／Validation 比較以下方案：

每個像素減去 Train mean image。

黑色映射為零刺激，不再映射為負刺激。

ON/OFF 雙通道：

ON  = max(pixel - train_mean, 0)
OFF = max(train_mean - pixel, 0)

編碼方案先以無標籤健康指標選擇，不以 Validation 報酬最高者直接決定。

5.2 輸出分數正規化

BUY／SELL 神經群數量不同時，禁止直接比較總活動：

buy_rate  = buy_activity  / number_of_buy_neurons
sell_rate = sell_activity / number_of_sell_neurons
margin    = buy_rate - sell_rate

若兩群仍有固定活性偏差，再使用 Train-only baseline 做標準化。

5.3 Renderer shortcut 檢查

對 margin 與下列變數計算相關性及簡單回歸解釋力：

平均亮度。

非黑像素比例。

總像素能量。

K 線垂直範圍。

上漲／下跌 K 線數量。

過去波動率。

若 margin 主要由亮度或非黑像素數量解釋，輸入編碼仍不合格。

6. 實作階段

Phase 0：凍結協議

工作：

建立 experiment_v2.yaml。

記錄資料期間與切割點。

將舊 Test 降級為開發資料。

預先寫好成功門檻。

完成條件：不看新 Holdout，也能完整描述資料、模型、指標及停止規則。

Phase 1：建立資料集

工作：

將原始 OHLCV 轉成固定時間序列樣本。

產生多 horizon 連續標籤。

產生 BUY／HOLD／SELL 輔助標籤。

在 split 邊界執行 purge／embargo。

加入未來資料洩漏測試。

完成條件：隨機抽查任一樣本，都能證明 input 的最大時間不晚於 t，target 僅來自 t 之後。

Phase 2：修正輸入編碼

工作：

只用 Train 產生 mean image。

實作三種候選編碼。

儲存編碼版本與 hash。

記錄每張圖的亮度、非黑比例及輸入電流統計。

完成條件：在至少 500 張真實 Train／Validation 圖片上沒有 action collapse、NaN、Inf 或固定 margin。

Phase 3：真實資料 Smoke Test

每個樣本保存：

sample_id
timestamp
encoding_version
seed
buy_activity
sell_activity
margin
action
mean_brightness
nonzero_pixel_ratio
latency_ms

最低門檻：

minority_action_ratio >= 0.05
margin_std > 預定數值誤差門檻
無 NaN / Inf
不同輸入的變異 > 重複模擬的變異
至少部分受控擾動會改變 margin

這些條件只代表系統沒有塌縮，不代表具有預測能力。

Phase 4：建立非 Connectome Baseline

至少建立：

Constant／majority baseline。

直接使用 OHLCV 的 Ridge regression。

直接使用 OHLCV 的 Logistic regression。

簡單 MLP；只作為較強的非 Connectome 對照。

所有 baseline 使用完全相同的 Train／Validation 樣本和標籤。

完成條件：可以產生 Validation 的連續預測、分類結果與交易模擬結果。

Phase 5：Frozen Connectome Linear Probe

流程：

固定 Connectome，不更新任何內部權重。

對所有樣本輸出神經活動向量。

只使用 Train 訓練 Ridge／Logistic decoder。

在 Validation 評估。

使用完全相同流程測試 Random network 與 Degree-preserved scramble。

此階段是整個研究的主要判斷點。

通過條件：Connectome 特徵不只具有輸入依賴性，也在預先指定指標上穩定優於 matched control；同時回報效果量與信賴區間。

Phase 6：有限度監督式訓練

只有 Phase 5 通過才執行：

優先只訓練 action decoder。

再測試只開放部分突觸或 plastic layer。

保留一組 Frozen Connectome 作控制。

每個版本使用相同 seeds 和資料。

禁止一次解凍全部權重，否則無法判斷效果來自 Connectome 拓樸還是一般神經網路容量。

Phase 7：Reward Learning

只有監督式模型在 Validation 有穩定預測訊號才執行。

Reward 應納入：

reward = 已實現損益 - 手續費 - 滑價 - 風險懲罰 - 過度換手懲罰

新增控制組：

關閉 plasticity。

Reward 時間隨機打亂。

相同預測模型搭配固定交易規則。

Random／scrambled connectome 使用相同 reward。

Reward learning 的任務是優化持倉與交易，而不是重新證明市場方向可預測。

Phase 8：Future Sealed Holdout

執行前必須凍結：

資料處理程式。

Input encoding。

模型權重。

Action threshold。

指標與報表程式。

Git commit 與設定檔 hash。

新 Holdout 只正式打開一次。若根據結果修改任何設定，該 Holdout 必須降級為開發資料。

7. 實驗矩陣

ID

輸入／模型

訓練內容

目的

B0

Constant／majority

無

最低基準

B1

OHLCV + Ridge

Decoder

線性市場基準

B2

OHLCV + MLP

全模型

非 Connectome 基準

C0

Frozen Connectome

Linear probe

檢查表示是否含預測資訊

C1

Random network

Linear probe

排除一般隨機投影效果

C2

Degree-preserved scramble

Linear probe

檢查真實拓樸價值

C3

Connectome

Decoder／部分突觸

監督式訓練效果

R1

Connectome

Reward learning

交易策略優化

8. 評估方式

8.1 預測指標

主要指標先選一個並鎖定：

連續報酬：Spearman IC 或 MAE。

三分類：Balanced accuracy 或 MCC。

輔助回報：

每類 precision、recall。

預測分布與 action 比例。

Calibration／Brier score。

跨 seeds 的平均、標準差與方向一致率。

8.2 交易指標

必須扣除手續費及滑價：

淨報酬。

Sharpe ratio。

最大回撤。

Turnover。

交易次數與市場覆蓋率。

Buy-and-hold 與固定規則基準。

8.3 統計檢定

同時回報效果量、95% CI 與 p-value。

時序資料使用 block bootstrap，不使用破壞自相關的普通 iid bootstrap。

Permutation 使用 block permutation 或時間 circular shift。

不能只因 p < 0.05 就判定通過。

預先設定最小實質效果，避免 BA 只比 0.5 高極小數值也被宣稱成功。

9. 建議專案結構

project/
├── configs/
│   └── experiment_v2.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── artifacts/
│   ├── train_mean_image.npy
│   ├── normalization.json
│   └── hashes.json
├── src/
│   ├── dataset.py
│   ├── labels.py
│   ├── image_encoder.py
│   ├── connectome_runner.py
│   ├── activity_normalizer.py
│   ├── linear_probe.py
│   ├── baselines.py
│   ├── trading_policy.py
│   └── evaluation.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── run_smoke_test.py
│   ├── extract_connectome_features.py
│   ├── train_probe.py
│   ├── evaluate_validation.py
│   └── evaluate_sealed_holdout.py
├── tests/
│   ├── test_no_future_leakage.py
│   ├── test_split_overlap.py
│   ├── test_train_only_normalization.py
│   ├── test_action_mapping.py
│   ├── test_input_sensitivity.py
│   └── test_determinism.py
└── reports/

10. 設定檔範例

experiment_id: supervised_connectome_v2

data:
  interval: 5m
  input_window_bars: 48
  prediction_horizons: [1, 3, 6, 12]
  primary_horizon: 6
  split_method: chronological
  purge_bars: 54

encoding:
  version: train_mean_subtraction_v2
  fit_split: train
  image_size: [64, 64]

model:
  connectome_frozen: true
  decoder: ridge
  seeds: 30

labels:
  primary_target: future_log_return_6
  actions: [BUY, HOLD, SELL]
  include_transaction_cost: true

evaluation:
  primary_metric: spearman_ic
  bootstrap: block
  bootstrap_samples: 2000
  permutations: 1000

governance:
  old_test_role: development_test_v1
  sealed_holdout_access: once

11. 執行順序 Checklist

標記舊 Test 與舊結果為不可正式引用。

確定 Train／Validation／Future Holdout 時間區間。

建立資料 manifest 與 purge／embargo。

產生連續未來報酬和 BUY／HOLD／SELL 標籤。

只用 Train 建立 mean image 和 normalization。

使用真實圖片完成輸入編碼 smoke test。

檢查 margin 是否被亮度、非黑像素等 shortcut 支配。

建立 OHLCV baseline。

執行 Frozen Connectome linear probe。

執行 Random 與 Degree-preserved scramble 對照。

通過預定門檻後，才進行有限度監督式訓練。

監督式預測穩定後，才加入 Reward learning。

凍結程式、模型、threshold 與報表。

收集並執行一次 Future Sealed Holdout。

12. 最終停止規則

遇到以下任一情況，不進入 Reward Learning 或正式回測：

BUY／SELL／HOLD 再次發生輸出塌縮。

不同輸入的 margin 變化不高於模擬噪聲。

Connectome linear probe 不優於 Random／scrambled control。

Connectome 不優於直接使用 OHLCV 的簡單 baseline。

結果只在少數 seeds 成立。

效果主要由亮度、非黑像素或 renderer 特徵解釋。

扣除交易成本後效果消失。

只有 p-value 顯著，但效果量沒有實質意義。

若 Frozen Connectome 沒有增量訊號，研究結論應停在「未找到 Connectome 對市場預測的額外價值」，而不是透過增加模型複雜度持續尋找正結果。