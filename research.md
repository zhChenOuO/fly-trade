果蠅 Connectome 交易決策「非隨機性」驗證研究計畫

1. 研究目標

驗證果蠅神經連線模型（Fly Connectome）輸出的 BUY / SELL 是否會穩定受到市場輸入影響，而不是隨機輸出、固定偏向 BUY，或只對模擬器雜訊作出反應。

本研究第一階段不以賺錢為主要目標。非隨機決策與具備市場預測能力是兩件不同的事：

輸入依賴性：不同市場輸入會造成可重現的不同輸出。

預測能力：輸出與未來漲跌存在樣本外關聯。

獲利能力：扣除交易成本後仍有正報酬。

本研究依序驗證 1 → 2；只有前兩項成立後，才值得研究第 3 項。

2. 研究問題與假設

RQ1：決策是否受到輸入影響？

H0-1：市場輸入與神經模型輸出無關。

H1-1：市場輸入改變時，神經模型輸出會產生穩定且可重現的改變。

RQ2：真實 Connectome 結構是否有作用？

H0-2：真實 Connectome 與相同規模的隨機連線模型沒有差異。

H1-2：真實 Connectome 在穩定性、輸入依賴性或樣本外預測上優於隨機連線模型。

RQ3：決策是否包含未來價格資訊？

H0-3：模型的樣本外方向預測不優於 matched-random baseline。

H1-3：模型的樣本外方向預測顯著優於 matched-random baseline。

3. 最小可行實驗（MVP）

3.1 固定設定

項目

第一版設定

市場

BTC/USDT 現貨

K 線週期

5 分鐘

輸入視窗

最近 48 根 K 線（4 小時）

預測區間

未來 6 根 K 線（30 分鐘）

動作

BUY、SELL，先不加入 HOLD

輸入格式

固定尺寸 K 線圖，例如 64 × 64 RGB

每組重複執行

100 次（若模型完全 deterministic，可降為 5 次確認）

隨機種子

至少 30 組

測試資料

至少 10,000 個不重疊或低重疊樣本

所有圖片尺寸、色彩、K 線數量、座標範圍和正規化方式必須固定。不可讓檔名、時間文字或未來資料進入圖片。

3.2 標籤定義

令模型在時間 t 做決策：

future_return = close[t + 6] / close[t] - 1

future_return > 0  → UP
future_return <= 0 → DOWN

主要評估：

BUY  對應 UP
SELL 對應 DOWN

可另做較嚴格版本，把交易成本 cost 納入：

future_return >  cost → UP
future_return < -cost → DOWN
其餘樣本排除或標記為 FLAT

4. 必要實驗組

組別

說明

用途

A. Fly-Intact

原始果蠅 Connectome

實驗組

B. Matched Random

BUY 機率與 A 相同的隨機代理

排除單純動作偏差

C. Input-Shuffled

Connectome 不變，但市場圖片與時間標籤隨機配對

檢驗市場輸入是否真正影響結果

D. Degree-Preserved Scramble

保留神經元數、邊數及各節點入度/出度，打亂連線

檢驗真實拓樸是否有額外價值

E. Constant Input

全黑圖、固定圖或平均圖

找出模型的基礎 BUY/SELL 偏向

若資源有限，至少完成 A、B、C；若要主張「果蠅腦結構有效」，D 不可省略。

5. 執行步驟

Step 1：準備並鎖定資料

取得 BTC/USDT 5 分鐘 OHLCV。

移除重複、缺漏或時間不連續資料。

依時間切割，禁止隨機切割：

Train：最早 60%

Validation：中間 20%

Test：最後 20%

先建立資料版本雜湊，之後不可因看到測試結果而改測試集。

MVP 不訓練時，可不使用 Train，但仍應保留 Validation / Test 分離。

交付物：

data/raw_ohlcv.parquet
data/splits.json
data/data_hash.txt

Step 2：建立統一輸入編碼

每個樣本取最近 48 根 K 線。

只使用當下及過去資料正規化價格範圍。

轉成固定的 64 × 64 RGB 圖。

同一時間點永遠產生完全相同的圖片。

隨機抽 100 張人工檢查，確認沒有未來資料與時間文字洩漏。

交付物：

sample_id,timestamp,image_path,future_return,label

Step 3：建立 Connectome → Action 映射

指定固定的視覺輸入神經元集合。

將圖片像素以固定規則映射至輸入神經元。

指定兩組互斥的輸出神經元：BUY neurons 與 SELL neurons。

在固定模擬時間窗內累積兩組神經活動：

buy_score  = BUY neurons 的總活動
sell_score = SELL neurons 的總活動

buy_score > sell_score → BUY
其他                    → SELL

平手規則必須預先固定，例如使用上一個動作或固定判為 SELL；不可臨時隨機處理。

第一階段關閉 reinforcement learning，不更新權重。

交付物：每次推論都寫入下列欄位。

sample_id,seed,group,buy_score,sell_score,action,latency_ms

Step 4：先做 Smoke Test

只用 100 個樣本確認：

全黑、全白、上漲圖、下跌圖是否能完成推論。

輸出沒有永遠相同、NaN、溢位或無活動。

同一 seed 與輸入能重現相同結果。

改變輸入後，至少部分神經活動會改變。

若 99% 以上都輸出同一動作，先修正 action mapping，不要直接進入大規模回測。

Step 5：執行「非隨機性」測試

5.1 重複輸入測試

抽取至少 200 個市場輸入，每個重跑 100 次。

consistency(x) = 該輸入最常見動作的比例

紀錄平均 consistency。若模擬器含噪聲，建議門檻為 ≥ 80%；若完全 deterministic，應接近 100%。

5.2 輸入敏感度測試

對每張圖產生受控變體：

原圖

上下翻轉價格方向

打亂 K 線時間順序

遮蔽最近 25% K 線

全黑或平均圖

計算動作改變率與 buy_score - sell_score 的變化。若所有變體的結果幾乎不變，模型雖可重現，但沒有證據顯示它在讀取市場輸入。

5.3 Input-Shuffle permutation test

使用真實輸入取得 A 組結果。

將輸入和 sample_id 隨機錯配，重做至少 1,000 次。

比較真實資料與 shuffle 分布的統計量，例如：

市場狀態與 action 的 mutual information。

buy_score - sell_score 與過去報酬、波動率的關聯。

樣本外 balanced accuracy。

使用 empirical p-value：

p = (1 + shuffle 中統計量 >= 真實統計量的次數) / (1 + shuffle 次數)

Step 6：比較對照組

對 A～E 使用完全相同的資料、輸入、模擬時間及 seed。

Matched Random 的 BUY 機率必須等於 Fly-Intact 在 validation set 的 BUY 比例。例如 Fly-Intact 有 63% BUY，基準也使用 63% BUY，而不是 50%。

主要比較：

Fly-Intact vs Matched Random
Fly-Intact vs Input-Shuffled
Fly-Intact vs Degree-Preserved Scramble

Step 7：最後才測未來方向

只在封存的 Test set 計算：

Balanced accuracy

Matthews correlation coefficient（MCC）

BUY precision / SELL precision

BUY 比例與最長連續相同動作

95% bootstrap confidence interval

對 matched-random 的 permutation p-value

Accuracy 只作輔助，因為漲跌比例不一定平衡。

6. 預先訂定的成功標準

Level 1：證明不是純粹 simulator noise

必須同時滿足：

同一輸入的平均 consistency ≥ 80%。

不同輸入能造成可測量的 score 或 action 差異。

Constant Input 無法重現真實輸入的完整行為分布。

Level 2：證明輸出與市場輸入有關

必須同時滿足：

Input-Shuffle permutation test：p < 0.05。

效果在至少 80% 的 seeds 方向一致。

結果不是只由固定 BUY/SELL 偏向造成。

Level 3：證明真實 Connectome 拓樸可能有作用

必須滿足：

Fly-Intact 在預先選定的主要指標上優於 Degree-Preserved Scramble。

95% bootstrap CI 不跨越 0，且跨 seeds 可重現。

Level 4：證明具備弱市場預測訊號

必須同時滿足：

封存 Test set 優於 Matched Random。

permutation p < 0.05，並回報效果量與 95% CI。

換另一個不重疊時期仍有相同方向結果。

未達 Level 4 時，不應宣稱「果蠅會交易」；最多只能說模型具有可重現的輸入依賴行為。

7. 第二階段：加入 Reward Learning

只有第一階段至少達到 Level 2，才加入學習。

使用 Train set 更新允許塑性的突觸權重。

BUY 後價格上升或 SELL 後價格下降時給 reward。

錯誤方向給 punishment。

Validation set 只用來選超參數，不更新權重。

Test set 只跑一次正式評估。

增加兩個控制組：

相同 Connectome，但關閉 plasticity。

reward 時間隨機打亂。

比較訓練前後 learning curve、方向預測與 action 分布。

若「真實 reward」與「亂序 reward」沒有差異，就不能認定模型真的學習。

8. 建議專案結構

fly-trading-research/
├── README.md
├── config/
│   └── experiment.yaml
├── data/
│   ├── raw_ohlcv.parquet
│   ├── splits.json
│   └── data_hash.txt
├── src/
│   ├── build_dataset.py
│   ├── render_market.py
│   ├── fly_simulator.py
│   ├── action_decoder.py
│   ├── baselines.py
│   ├── run_experiment.py
│   └── statistics.py
├── outputs/
│   ├── decisions.parquet
│   ├── metrics.json
│   └── figures/
└── tests/
    ├── test_no_future_leakage.py
    ├── test_determinism.py
    └── test_action_mapping.py

experiment.yaml 至少鎖定：

symbol: BTCUSDT
interval: 5m
window_bars: 48
horizon_bars: 6
image_size: [64, 64]
actions: [BUY, SELL]
seeds: 30
repeats_per_input: 100
primary_metric: balanced_accuracy
permutations: 1000
bootstrap_samples: 2000

9. 執行順序 Checklist

下載、清理並依時間切割市場資料

固定輸入圖片格式並檢查未來資料洩漏

固定 input neurons、output neurons 與 action 規則

使用 100 筆資料完成 smoke test

跑 Fly-Intact 並記錄完整 score，不只記 action

做重複輸入與受控變形測試

跑 Matched Random 與 Constant Input

跑 Input-Shuffled permutation test

建立 Degree-Preserved Scramble 並用相同 seeds 比較

鎖定分析程式後才打開 Test set

回報效果量、CI、p-value、所有 seeds，不只報最佳結果

Level 2 成立後才開始 reward learning

10. 最終報告應回答的問題

同一個市場輸入是否穩定產生相同決策？

改變市場輸入是否會改變神經活動或決策？

真實輸入是否顯著優於 shuffled input？

原始 Connectome 是否優於 degree-preserved scrambled graph？

是否優於具有相同 BUY/SELL 偏好的隨機代理？

效果是否能在不同 seeds、不同時期重現？

結果只證明輸入依賴，還是已證明樣本外預測能力？

建議使用以下結論格式：

本實驗顯示／未顯示 Fly-Intact 的動作會穩定受到市場輸入影響。
相較於 Matched Random、Input-Shuffled 與 Degree-Preserved Scramble，
其主要指標差異為 ___，95% CI 為 ___，permutation p-value 為 ___。
因此目前證據支持 Level ___，但尚不足以主張 ___。

11. 研究資料與參考實作

Dorkenwald et al., 2024, Neuronal wiring diagram of an adult brain, Nature. https://doi.org/10.1038/s41586-024-07558-y

Schlegel et al., 2024, Whole-brain annotation and multi-connectome cell typing of Drosophila, Nature. https://doi.org/10.1038/s41586-024-07686-5

Shiu et al., 2024, A Drosophila computational brain model reveals sensorimotor processing, Nature. https://doi.org/10.1038/s41586-024-07763-9

Stonkfly 參考實作：https://github.com/nftechie/stonkfly

Stonkfly 可用來理解資料流與工程結構，但其結果不能取代上述對照實驗，也不能直接視為果蠅 Connectome 已具備獲利能力的證據。
