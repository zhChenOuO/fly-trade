# Research v3 方向調整建議

## 一頁摘要

**建議：停止目前「Frozen Connectome 對 BTC 30 分鐘方向有增量預測力／可交易」的確認性主張；不要再以反覆改 decoder、讀出或 target 的方式搜尋 Val。** Phase 5 沒通過預先註冊的主要門檻，實際交易結果也不支持繼續把它當交易策略。若研究仍有方法學價值，只做一個有明確停點的診斷：先核對 Phase 0 正負控制的版本衝突，再以凍結的同一條 Phase 5 特徵／decoder 管線做一次「已知訊號可偵測」控制；通過後才值得規劃一個完全新、未查看的前瞻 holdout。這不是延長市場搜尋的許可。

Phase 5 的 Ridge 是預先註冊主線：real IC=0.00243，低於設定的 0.02；與 random 的 ΔIC=-0.00895，與 scramble 的 ΔIC=0.00261，兩者均低於要求的 +0.01；real IC circular-shift p=0.425；real Ridge 預測全為 HOLD、零筆交易。nuisance-only Logistic 的 BA=0.412，高於 real Logistic 的 0.357。這些來自 `research/outputs/v2/probe_val.json` 與 `research/outputs/v2/baselines_val.json`，都是已反覆查看的 Val 探索結果；不得當成確認性顯著性證據。

**唯一推薦主線：有條件地完成目前已授權的 Phase 6 探索任務，但不擴大、不改門檻、不進 Phase 7；之後封存目前 Val。** 先限定 Phase 6A（凍結特徵、只訓 decoder）；若整份已凍結的 Phase 6 任務已開始執行，依原規格完成並如實標記 `PHASE6_OVERRIDE_EXPLORATORY`，不要依中途 Val 結果停跑或調整。Phase 6B 的「上游平均活動乘以突觸權重增量」不是在原遞迴模擬器內做突觸可塑性，應另立名稱與問題；若尚未開始，不建議把它稱為 Connectome plasticity，也不建議耗費資源執行。無論 Phase 6 結果如何，都不能取代 sealed holdout。

**現在不值得投入 RTX 5070 長時間全網路重訓或收集昂貴 holdout。** 若單一、事前鎖定的正控制未能通過，停止「此表示／解碼管線含可恢復資訊」這條路；若通過，也只代表測量管線有能力恢復該合成訊號，不代表市場可預測。現有 v3 規格估計 Phase 6 為 1–2 小時（`research/REMOTE_TASK_P6.md`）；Phase 5 實際特徵萃取 metadata 合計約 17.3 分鐘（三圖 × train/val；`research/outputs/v2/features/*.json`），此估計不含後續新圖變體與除錯。

## 詳細診斷

### 1. Phase 5 結果與證據強度

| 項目 | Val 結果 | 解讀 |
|---|---:|---|
| 樣本數 | Train 31,556；Val 10,504 | `probe_val.json.metadata`；樣本依時間排列、每 6 根取樣。 |
| Real Ridge IC | 0.00243；95% CI [-0.01677, 0.02174] | 未達 yaml 中提議的 IC ≥0.02；CI 包含 0。 |
| Random Ridge IC | 0.01138 | 高於 real Ridge。 |
| Scramble Ridge IC | -0.00018 | 與 real 的微小差異 CI 跨 0。 |
| Real-Random ΔIC | -0.00895；CI [-0.02973, 0.01335] | real 沒有增量優勢。 |
| Real-Scramble ΔIC | +0.00261；CI [-0.01865, 0.02276] | 未達預註冊 +0.01，CI 跨 0。 |
| Real Ridge permutation | p=0.42458 | circular-shift 單尾檢定不支持可預測 IC。 |
| Real Logistic IC | 0.02541；CI [0.00702, 0.04303] | 探索性 decoder 指標；沒有對 Logistic 的 real-vs-random/scramble 配對推論與多重比較校正，不能替代 Ridge 主分析。 |
| Real Logistic BA／交易 | BA=0.35681；淨報酬=-75.8%；2,239 筆交易 | `probe_val.json`；不支持交易用途。 |
| Nuisance-only Logistic BA／交易 | BA=0.41175；淨報酬=-48.9%；546 筆交易 | renderer／OHLCV 摘要特徵在此分類任務優於 real Logistic，但交易仍虧損。 |
| Real Ridge 輸出 | 100% HOLD；0 筆交易 | 淨報酬 0、Sharpe 0、回撤 0 是「沒有交易」的定義性結果，不是通過交易門檻的證據。 |
| Alpha | real、random、scramble 均選 10,000（網格上限） | `baselines_v2.py` 的時序 CV 網格為 1e-3 至 1e4；只能說最佳值落在邊界，不能事後擴網格再把新結果當預註冊確認。 |

判定以預註冊的連續目標 Ridge 為準，故 Phase 5 **FAIL**（`research/REMOTE_TASK_P6.md` 治理紀錄亦列出預先判定）。pass/fail 表有 11 列、4 列顯示通過，但其中三列是 renderer R²、零淨報酬、零回撤的無效或弱判斷；逐列通過數不是研究成功率。分類 BA 相對 1/3 的差額也被標成通過，然而此項不等同交易價值，且三類比例與分類機率校準需一起看。

### 2. 最可能失敗原因排序

1. **目標與資料期本身訊號弱，無法支持「市場預測」的強結論（高信心）。** Train/Val 是同一 BTC/USDT 5m 歷史區間；Val OHLCV Ridge IC=-0.00073、MLP 平均 IC=-0.00273，Logistic IC=0.01295（`baselines_val.json`）。這顯示方向／報酬訊號稀薄，但單一時期、單一資產的 Val 不能證明所有未來市場都不可預測。可用現有資料做的僅是檢查既有 Val 的標籤分布、原始 return 自相關、按時段切出的穩定性；這些分析仍是探索性，不能恢復確認性地位。
2. **目前沒有證據顯示真實拓樸提供增量（高信心）。** Ridge real 不勝 random，對 scramble 的差異小且不確定；Phase 5 是單一 real、單一 random graph、單一 scramble graph、固定 simulator seed=0（見 `features/*.json`）。只有一個圖實例，不能估計「拓樸類別」相對不同隨機圖實例的變異。補救需多個事前指定 graph seeds、等參數／等輸入的配對比較；成本高，應先通過正控制才做。
3. **預測表徵及比較條件不對稱（高信心）。** Connectome 接受 renderer 圖像；OHLCV baseline 使用 150 維原始報酬、量、時間特徵（`baselines_v2.py`），且 renderer 每個視窗把價格範圍拉滿畫面，抹除絕對價格波動尺度（`research/SPEC_v3.md` Amendment 2）。兩者的可用資訊不同，不能把差異只歸因於 Connectome 拓樸。應同時報告「相同來源特徵／相同訓練預算」對照；另須把時間特徵視為可能的時段捷徑。
4. **讀出維度高、時序樣本有效數低且正則化未有邊界診斷（中高信心）。** DN readout 為 1,291 維、31,556 筆 train，但連續窗口高度重疊，市場 regime 自相關也不會因每 6 根取樣而消失。所有 Ridge CV 都選 alpha 上界 10,000；這可能代表需更強正則、也可能代表 CV 對弱目標只選到近常數模型。事後擴網格不能算新確認證據；新版本應在 Train 內預註冊更廣且包含「常數／縮小」基線的網格，記錄每折曲線和邊界命中率。
5. **解碼器／目標選擇與多重比較（高信心）。** 同時報告 real/random/scramble × Ridge/Logistic、連續 IC、三分類 BA、交易結果等；Real Logistic 的 IC CI 排除 0，但單一 p/CI 沒有處理 decoder、圖變體與指標的家族錯誤率。它不能事後升格為主結果。另三分類 HOLD 門檻是 ±10 bps（`experiment_v2.yaml`），HOLD 類把「小幅報酬」混在一起，模型可能主要學到幅度／波動 regime，而非方向。
6. **Phase 0 正控制證據存在，但 gate 記錄不一致，且不能直接替 Phase 5 背書（高信心）。** `phase0_report_remote.json` 中 PC-1 趨勢 BA=0.9868、斜率 R²=0.7701，PC-3 AR(1) 方向 BA=0.6223，NC-1 GBM BA=0.506（n=2,000）；但 PC-2 波動 R²=-0.0048。該檔仍把整體 S4 標為 STOP，而 `SPEC_v3.md` Amendment 2 已規定 PC-2 只報告、不設關卡；規格後文卻稱 Phase 0 GO。另一份 `phase0_report.json` 也 STOP，且 NC-1 n=10，精度不足。先查清哪份報告／程式版本是凍結的正式裁決，不能只引用 SPEC 文字。既有正控制證明特定合成趨勢和 AR(1) 可由那批活動解碼，沒有證明 Phase 5 實際輸出、目標、對照和統計推論整條鏈已具備充分檢定力。

### 3. 現有設計能否回答「超越 OHLCV 的資訊」？

**字面上的資訊量問題不能。** 市場圖像是 OHLCV 的確定性轉換；Connectome 活動是圖像與固定圖、固定動力學、固定輸入編碼的函數（另加可控模擬雜訊）。依資料處理關係，確定性轉換不會創造原始 OHLCV 中不存在的資訊；含獨立雜訊也不會增加標籤資訊。可以回答的窄問題是：「在同一可用 OHLCV、固定資料預算、指定 decoder 與成本下，這個固定 Connectome 表徵是否比預先指定的替代表徵更容易解碼，或在跨期穩健性上更好？」這是表徵／歸納偏置比較，不是 Connectome 發現市場新資訊。

目前 Phase 5 **原則上能檢驗一種狹義的樣本外表徵效用**，但尚不能給確認性答案：Val 已反覆查看；配對比較只對 Ridge 做；對 OHLCV 比較只報點估計且取「兩個 baseline 中較好 IC」與 Ridge 比，沒有配對 CI；且沒有足夠 graph instances。未來若要研究，primary estimand 應是新 holdout 上固定預測器的配對損失差（例如 real-feature Ridge 對預先指定、同容量控制的 Ridge），不是「額外資訊」。另報 OHLCV-only 與 OHLCV+Connectome 的相同 decoder 比較；若 Connectome 特徵是 OHLCV 的函數，後者只測有限樣本下的表示便利性。

## 候選方向表

| 候選 | 問題與最小實驗 | 預先註冊通過／失敗 | 成本 | 主要假陽性 | 評價 |
|---|---|---|---|---|---|
| **A. 停止市場主張（推薦底線）** | 封存 Phase 5 原結果，發表／記錄「本次設定未找到 Connectome 增量預測證據」；不再搜尋 Val、不跑 reward learning。 | 直接接受 Phase 5 FAIL；不以 Real Logistic、最佳 seed 或交易指標翻案。 | 0 GPU；文件整理半天內。 | 對某一資料期的負結果過度外推成市場普遍不可預測。 | 必須執行的結論邊界。 |
| **B. 一次性正／負控制稽核（唯一推薦的研究主線）** | 先 reconcile Phase 0 STOP/GO；再固定同一 frozen FlyWire 輸出與 Phase 5 Ridge/Logistic 管線，在 Train-synthetic 訓練、獨立 seed synthetic-test 測試：趨勢斜率、AR(1) 方向各為陽性；純 GBM 為陰性。加一個零訊號打亂標籤控制；完整流程不得碰市場 Val。 | 陽性：每一 primary 合成任務達既有 SPEC_v3 指標（PC-1 sign BA≥0.95 且 slope R²≥0.5；PC-3 BA≥0.60）；陰性：GBM BA 的 95% 區間涵蓋 0.5，區間以二項／block 方法事前鎖定。任一陽性失敗或陰性顯示洩漏即停止；不得換 readout/encoder 再試。 | Phase 0 的 remote 報告用 RTX 5070 模擬 300 Train/Val 張 gate 約有分鐘級模型運算；此新完整 train/test 數量未定，先用已量測 throughput 134.9 張/秒估算並把 renderer／訓練包含在預算中；估計總工時半日至一天（估計，需先做 dry-run）。 | 合成趨勢過於規則；不同種子雖獨立但分布太易；PC 設計後看結果調參；同一 renderer 讓 target 幾乎可直接讀出。 | 若陽性控制失敗，管線沒有資格解讀 Phase 5 負結果；若通過，只證明可恢復特定人工訊號。不能證明市場可預測。 |
| **C. 將研究改成波動／市場狀態表徵** | 另立 RQ：「表徵能否解碼過去窗口可見的市場狀態，或預測未來 realized volatility？」先修 renderer 把每窗絕對尺度正規化掉的設計缺陷；用過去 realized volatility 作可見正控制，再選一個未來波動 target。 | 只測一個事前鎖定 target；同一 OHLCV baseline、real、random、scramble；跨 seed／graph instance；新 holdout 的 paired loss CI 下界須優於 0，且效應達事前最小差異。無閾值前不開新市場 holdout。 | 改 renderer 和重萃特徵；三圖 train/val 的既有萃取約 17.3 分鐘，GPU 空間足夠（Phase 1a peak 約 2.97 GB / 12.34 GB）；端到端新版本估 1–3 天工程與驗證（估計）。 | 將已知的「過去波動」重建誤稱為未來預測；target 漂移、volatility clustering、Renderer shortcuts。 | 比方向分類更有資訊，但已是新研究問題，不可併入本次 Phase 5 結論。 |
| **D. 繼續 Phase 6A/6B** | 使用已凍結 DN features 做 MLP probe，另以 sparse upstream 表示訓練；依 `REMOTE_TASK_P6.md` 比 real/random/scramble。 | 一律 `PHASE6_OVERRIDE_EXPLORATORY`；即使 Val 達標也不得宣稱確認或打開 sealed holdout。訓練內部選超參數；交易 0 筆標 N/A；NaN/Inf、hash 不符或 mask 錯誤立即停。 | 任務估 1–2 小時；若預估 >4 小時須依任務書先停並回報。 | 使用 Val 選模、seed 多重比較、拓樸單例、SNR 挑邊、過度參數容量、事後挑 6A/6B。 | 只支持工程／表示探索。Phase 6B 不等於原 reservoir 突觸可塑性。 |

## 統計治理

1. **重新分類資料用途。** 舊 test 是 `development_test_v1`；Train 與 Val 也已在 Phase 1a、baseline、Phase 5 及事後分析中反覆查看。今後這兩者都只能作開發／探索資料，Val 不再產生新的確認性 p 值、CI 或「通過」宣告。未來 sealed holdout 尚未收集（`experiment_v2.yaml`），必須使用新資料；任何看過結果後的調整都把該 holdout 降級並要求下一個新時段。
2. **一個 primary target、一個 primary decoder、一個 primary estimand。** 若仍做市場研究，先在不碰新 holdout下鎖定 continuous `future_return_6`、一個 decoder、real 對一個主要 matched control，以及配對 block loss/IC 差。Logistic 三分類、MAE、BA、MCC、其它 horizons、交易曲線都列 secondary；以 Holm 校正預先列出的 secondary family，不可挑最漂亮結果。Real vs random、real vs scramble、real vs OHLCV 必須各自配對 bootstrap CI；不能拿兩模型獨立 CI 重疊來替代差異檢定。
3. **最小效果量先鎖、再談樣本數。** `experiment_v2.yaml` 的 IC 0.02、增量 IC 0.01、BA +0.02 都標為 `PROPOSED_NEEDS_USER_CONFIRMATION`，不是既定事實；新 holdout 前必須確認或另訂，並以 Train-only 時間區塊重抽做 power 模擬。不能只以 p<0.05 過關。交易主張另外必須在固定交易規則下，扣成本後 net return >0 且交易數足夠；目前估成本為雙邊手續費 8 bps 加滑價 2 bps，round trip 約 10 bps（`experiment_v2.yaml`），須核實滑價估計是否反映真實成交。
4. **粗略 power 只作規劃下限。** 把 IC 近似相關係數、假設獨立樣本、雙尾 α=0.05、power=80%，Fisher z 近似給 `n ≈ 3 + ((1.96+0.84)/atanh(r))²`。真實相關 r=0.02 約需 19,600 個獨立樣本；r=0.03 約需 8,700。現有 10,504 Val 若真獨立，對 r=0.02 仍不足 80% power。樣本 stride=6 表示每 30 分鐘一筆，10,000 筆約 208 天、19,600 筆約 408 天；這是日曆時間換算，不是有效樣本保證。重疊輸入視窗、自相關與 regime clustering 會降低有效 n，正式 n 必須用 Train-only block-bootstrap／時間模擬估算；ΔIC 的 power 也應直接對 paired block resamples 模擬，不能套單相關公式冒充精算。
5. **seed 不是市場樣本。** 每個樣本的結果跨 seed 先形成同一市場日期上的平均預測／效果，再以連續市場時間區塊為統計單位；seed 是模型隨機性層級。若要主張圖拓樸優勢，至少要多個事前固定的 random/scramble graph instances，對 graph instance 與市場時間都分層／配對重抽。30 seeds 共用相同樣本不能把 n 乘 30；單一 seed 更不能代表網路穩健性。
6. **封存流程。** 在新資料產生任何 label 分析前，寫下 data provenance、時間界線、資產／週期、renderer、graph hashes、decoder、alpha、cost、primary metric、實務門檻、block 長度選法、排除規則、比較家族、報表程式 hash。資料管理者只交付不含標籤的 input/features 供凍結 pipeline 運行；預測與程式 hash 鎖定後才一次解封 labels。若 holdout 期間只來自 BTC 未來資料，結論僅適用該單一未來期間；第二資產／週期應是事前指定的外部複現，而不是失敗後換標的。

## Phase 6 意見

**建議：不擴張；僅完成已明確授權、已凍結且正在跑的 Phase 6A 探索，隨後停止。不得進 Phase 7，也不得把 Phase 6 結果當成對 Phase 5 的補救。** `research_v2.md §12` 原本明定 Phase 5 不優於 random/scramble 或 OHLCV 時停止；`REMOTE_TASK_P6.md` 清楚記錄這次是使用者例外，且規定輸出 `PHASE6_OVERRIDE_EXPLORATORY: NOT CONFIRMATORY EVIDENCE`。遵守這條例外的邊界，比無限延伸更有研究誠信。

Phase 6A 可以回答「較有彈性的 decoder 能否讀出這組固定 features 上的樣本內／Val 弱關聯」，但其 Val 已污染，5 個訓練 seeds 仍共用相同市場樣本。照任務書以 Train 內部 80/20 時序切分選超參數，Val 最後只評一次，逐 seed 報告；不因 Val 成績改 epoch、loss、hidden size、feature 或門檻。OHLCV baseline 的 IC 約 -0.003、真實 Logistic 有一個未校正的 Val 訊號，兩者都不足以當 Phase 6 成功先驗。

Phase 6B 的 `z_i = Σ(W_ij+Δ_ij)x_j` 以凍結動力學下的上游平均活動作一次靜態特徵，沒有把 Δ 放回循環圖重新模擬；因此應稱「以 sparse upstream features 作監督式 readout」，不能稱 Connectome 突觸 plasticity。依 variant 各自挑 top-|w| 邊，再按全圖丟棄小邊對齊參數，會改變每個目標神經元的邊數／覆蓋，產生額外選擇差異。若 6B 還沒開始，我建議本輪不啟動；若已執行，完整照凍結規格記錄，不選擇性隱去結果，且只作探索附錄。

## Phase 5 程式修正清單（本次只列建議，不改檔）

1. **零交易不可算交易 PASS。** `train_probe_v2.py::evaluate_thresholds_table` 直接用 `net_return >= 0`、`max_drawdown <= 0.15`；Ridge 100% HOLD 時 0 淨報酬與 0 回撤被標 PASS。若 `trade_count==0`，交易相關列必須為 `N/A_NO_TRADES`，整體閘門不得視作通過。若 real/random/scramble 缺檔，`pass_fail_table=[]`；下游必須明確狀態為 `NOT_EVALUATED`，不可用空集合 `all()` 當 PASS。現有 JSON 的表非空，故本次實際的問題是零交易 PASS 語義，不是假稱當次表為空。
2. **修正 BA 的比較基線定義。** `min_practical_effect_balanced_acc` 計算 `BA - 1/3`，但 yaml 描述寫「over 0.5/majority baseline」。三分類 balanced accuracy 的 chance baseline 可為 1/3，但必須統一名稱、定義與預註冊門檻；另加逐類 recall、預測動作比例、trade_count 和成本結果的必要 gate。當預測恰為單類時不能只靠 BA 表述交易能力。
3. **Renderer shortcut R² 目前不是 R²。** `train_probe_v2.py` 把 nuisance Ridge 的 Spearman IC 平方當 `nuisance_shortcut_r2`。它既不是線性回歸的 R²，也不是 out-of-sample explained variance；本次 3.52e-8 因而沒有解釋「shortcut 很低」的證據力。改用 Train 擬合 nuisance 線性模型、在 Val 報預測 R²（允許負值）與增量／消融比較，並另報所有七個特徵的定義和共線性處置。
4. **與 OHLCV 的比較需配對且對稱。** 程式只對 real Ridge 與 random/scramble 算 paired block CI；對 OHLCV 用兩 baseline IC 的 max 算點差，沒有 CI，且 real Ridge 跟對方較佳 decoder 不同。先鎖一個可比 decoder／共同容量，使用同一 valid rows，配對 block bootstrap 所有差異；若要回答條件增量，比較 OHLCV-only 與 OHLCV+Connectome 的同一模型並固定正則化選法。
5. **補全多重比較治理及模型 seed。** Probe 同時跑 3 graph variants × Ridge/Logistic，另列多個 baseline 和 outcome；目前只對 real Ridge 做一個 circular-shift 檢定，未校正多個實際被解讀的結果。只預註冊一個 primary，其餘用 Holm。Phase 5 feature metadata 固定 `seed: 0`；新增 simulator noise seeds 及多 graph-instance 的分層報告，seed 維度不可冒充市場 n。
6. **alpha 邊界必須明示。** `baselines_v2.py::select_ridge_alpha_timeseries_cv` 網格上限 10,000，三種 graph 全選上限。報告每折 CV loss 和 edge-hit flag；本輪不事後擴網格。若要新版本，先在 Train-only 預註冊更廣網格與固定選擇規則，再從頭比較所有組別。
7. **檢查時序 CV purge／embargo。** `select_ridge_alpha_timeseries_cv` 以 expanding folds 切連續 rows，但沒有 purge。stride=6、horizon=6 時標籤區間幾乎接續，48-bar feature windows仍跨 fold 重疊。於 fold 邊界 purge 至少 input window + 最大 target horizon，再用 sample timestamp 驗證 train／CV target 不跨界；不能把「按時間排序」等同於「無依賴」。
8. **baseline 讀取範圍需符合 split 白名單。** `train_probe_v2.py` 用 parquet filter 僅載 train/val；`baselines_v2.py::run_baselines_v2` 卻未帶 filter 載入整份 `labels_v2.parquet`，之後才用 mask 留 train/val。即使下游沒計算 test 指標，讀取行為仍違反嚴格 split 隔離的意圖。新執行應在讀檔層限制 split 欄位；已產生的 baseline 數值不可因此假定曾使用 test label，但存取路徑應修正並稽核。
9. **處理無效／缺失數值與彙總狀態。** `evaluate_thresholds_table` 在缺比較時用 0 預設，在缺模型時整表為空；需將「未計算」、「不適用」、「失敗」分開，任何 NaN/Inf／缺指標均不得因比較運算偶然變成 PASS。整體決策採明確必備 gate conjunction，而非數 PASS 數量。
10. **控制 decoder 與 target 的事後挑選。** Logistic IC 雖 0.0254，不能在看到結果後替代 primary Ridge；各分類、IC、交易指標都需 primary/secondary 標記與校正。分類 head 的 HOLD 標籤同時受波動與成本閾值影響；若要研究方向，另用方向 conditional target；若研究波動，另立 target 並修 renderer，不在同一輪挑較好看的任務。

## 待辦決策與停損

- Phase 0 哪個報告是正式凍結結果？`phase0_report_remote.json` 與 `SPEC_v3.md` 的總 gate 結論不一致；需先校正記錄，禁止改寫成一致成功。
- 是否接受把研究問題由「Connectome 含超越 OHLCV 的資訊」改寫為「固定表示在指定資料預算和 decoder 下是否有預測效用」？前者資訊論上不成立。
- Phase 6B 若未開始，是否取消本輪？本計畫建議不啟動；若已開始則按原規格完整揭露為探索結果。
- IC 0.02、ΔIC 0.01、BA +0.02 和實際交易門檻仍是 yaml 的 proposed 值。新 sealed holdout 開始收集前，必須由研究負責人正式凍結 primary effect 和成本模型。
- 若一次性正控制在凍結管線下失敗、GBM 陰性控制顯示假訊號、程式存取 test split、或 Phase 6 只有少數 seeds／單一 graph 勝出，停止 Connectome 市場預測與 reward-learning 方向。若正控制通過但 fresh holdout 之 paired 增量 CI 下界未過 0 或實務最小效果未達，結論維持「未找到增量效用」，不再改 target 追結果。

### 數字來源與限制

- Phase 5／baseline：`research/outputs/v2/probe_val.json`、`research/outputs/v2/baselines_val.json`、`research/outputs/v2/probe_run.log`。
- 特徵數、sample ids、模擬 seed、輸出維度與萃取耗時：`research/outputs/v2/features/{real,random,scramble}_{train,val}.json`。
- Phase 1a Val gate：`research/outputs/v3/phase1a_remote/metrics.json`；只引用其 Val label-free 輸入健康／一致性結論，不使用 test split 的 label 或 future return。
- 合成 controls 與 renderer 尺度限制：`research/outputs/v3/phase0_report_remote.json`、`research/outputs/v3/phase0_report.json`、`research/SPEC_v3.md`。
- 門檻、資料切分及 holdout 狀態：`research/config/experiment_v2.yaml`、`research_v2.md`、`research/REMOTE_TASK_P6.md`。
- 本文件的樣本數、步數換算與粗略 power 是明列公式下的估算，不是新資料分析結果。未讀取 test split 的 label 或 `future_return`；未修改程式、設定或既有結果。

## 交叉詰問回覆（Claude 質詢後）

本節補充並修正前文的優先順序；前文保留原樣。以下對資料中尚不存在的結果只提出設計，不假稱已驗證。

### Q1. 明確行動順序與前文修正

1. **先處理目前已在遠端執行的 Phase 6。** 凍結現有任務與設定，不加模型、不改門檻、不因中途結果挑停；pytest collection error 必須先釐清並留紀錄。若錯誤涉及資料隔離、mask、feature hash 或模型正確性，停止受影響階段並報告；若不影響正在執行的固定工作，也不能把 Phase 6 結果升格為確認性證據。Phase 6 完成後不進 Phase 7、不重跑同一 Val。
2. **其後只做一個合併的探索性診斷批次：MDE 曲線是主問題；graph-instance 分布作為同批次的拓樸副分析。** MDE 回答 Phase 5 對弱訊號有沒有檢定力；多 graph instances 回答單一 random/scramble 的結果是否偶然。兩者需事前凍結，所有市場 Val 結果明確標記 exploratory。Q4 的 20+20 若只做四小時 pilot，限於估分布寬度，不能作 multiplicity-adjusted 的確認檢定。
3. **不收集目前方向交易假設的 sealed holdout。** 依目前粗略獨立樣本假設，偵測 IC=0.02 已約需 19,600 筆／408 天；若要偵測最小增量 ΔIC=0.01，粗略約需 78,400 筆／4.5 年，且時序依賴只會增加需求。這個成本對尚未在已看 Val 上顯示 real 優勢的命題不合理。
4. **停損：** MDE 若顯示目標效應量的檢出率低於 80%，就停止對此資料量下的 Phase 5 FAIL 作「不存在訊號」解讀；正式結論限於「本次設定未顯示預註冊所需效果，對更小效應無足夠排除力」。若 MDE 有力且圖分布仍未見 real 穩定勝出，停止 Connectome 市場預測與交易主張。若效果只在一個 graph、decoder 或 Val 子期出現，也停，不再換規格追結果。

**優先關係：** 已開始的 Phase 6 是使用者明確授權的有限例外，依凍結流程收尾；它先於新的研究批次，但不能改變停止規則。新研究中，MDE 是首要判斷，多 graph null 是同批次副分析，不應拿後者的單一好排名取代檢定力分析。最終主張仍依 A：承認目前市場預測未通過並停止確認性主張。前文把 B 稱「唯一推薦主線」而摘要又稱 Phase 6 唯一主線，確有矛盾；本節改為上述順序，且撤回「先跑強訊號正控制再決定」作為主要建議。Phase 0 強正控制已存在，真正缺的是**目標效應量附近的檢定力校準**。

### Q2. 弱訊號 MDE／檢定力曲線

同意。前文 B 重跑 PC-1／PC-3 的強訊號門檻，不能回答 Phase 5 對 IC 0.01–0.03 的靈敏度，應改成以下**量測檢定力**，不再把舊強訊號控制當主線：

- **資料生成：** 用固定 synthetic price generator 產生與 Phase 5 相同 window=48、stride=6、Train=31,556、Val=10,504、相同 split purge 的 OHLCV 圖。已知 latent signal (s_t) 必須以一個固定、可從過去圖窗觀察的形態編碼（例如過去 K 線中一個預註冊的弱形態／趨勢因子）；未來 6 根連續報酬由 (y_{t+6}=eta s_t+epsilon_t) 生成，再用只依 generator 的校準程序選 (eta)，使 oracle (s_t) 對目標的 Spearman IC 依次為 0、0.005、0.01、0.02、0.03、0.05。每一強度用獨立 generator/noise seed 產生多個 cohort；同一 cohort 的圖像供 real、random、scramble 共用。不能用 Val 標籤調 (eta)、編碼或 decoder。
- **完整流程：** 先以相同的 retina/render、固定動力學、feature 維度、Train-only 標準化、Ridge expanding CV、預先指定 Logistic 輔助分析及 Val 樣本數走完整 Phase 5。主結果只看連續 Ridge IC 與 real-control 配對 ΔIC；Ridge alpha CV 邊界、CI、有效樣本數、缺值、預測退化都照實回報。Logistic 的弱訊號效應容易被固定 10 bps 動作門檻壓成 HOLD，故列 secondary，不以三分類結果取代連續 primary。
- **重複數與通過定義：** 事前固定至少 500 組獨立 synthetic future continuations/強度；每組在相同過去 OHLCV 窗口上，以獨立的分塊創新項產生未來價格路徑，強度只改 signal coefficient，避免每一條路徑重萃相同的 frozen features。每個強度對三圖使用共同樣本及 common random numbers；每個強度回報：IC 偏差、配對 ΔIC 平均與 95% block CI 寬度分布（中位數、90 百分位）、CI 涵蓋率、以「primary 95% CI 下界 > 0 且點估計方向正確」定義的檢出率及其二項 Monte Carlo 95% CI。無訊號強度的 family-wise false-positive rate 應 ≤5%；real-vs-random、real-vs-scramble 的雙比較用 Holm。MDE 定義為檢出率首次達 80% 的最小注入 IC，並同時報其 Monte Carlo 區間，不插值冒充精確值。這 500 次主要量的是給定 synthetic past-window panel 和指定分塊生成器下的條件檢定力；若結果接近 80% 邊界，須再以新 past-window cohorts 作外層複現，不能把 500 條 future continuations 當 500 個獨立市場時期。
- **重要區分：** 所有圖都接收同一合成市場訊號時，此設計量出「各表示能否恢復共同弱訊號」及 ΔIC CI 的精度；它不保證拓樸差異的真實效應。若要量「已知拓樸增量 ΔIC」的 MDE，須另做統計校準：在 Train-only、事前固定的 real-feature 子空間加入已知效應，並以配對 control 分數作負對照。這個差異注入是對檢定程序的校準，因它按定義偏向 real，不能當生物學正控制或預測證據。

**如何改變 Phase 5 FAIL 解讀：** 若 MDE=0.04，Phase 5 的接近零結果仍支持「未達預註冊的 IC≥0.02／增量門檻」，但不能排除 0.02 的效應；結論需寫「估計未達門檻且設計對該大小檢定力不足」，不得寫「Connectome 沒訊號」。若 MDE≤0.02、假陽性率受控、CI 涵蓋率合理，Phase 5 近零與已報 CI 才較有力地反對達到該效應門檻；Val 反覆使用仍使其不具確認性。前文的 PC-1/PC-3 通過只說明強訊號可恢復，不能代替這條曲線。

### Q3. Holdout 時間成本與跨資產有效樣本

**(a)** 需把目標區分清楚。前文的 Fisher 近似是單一相關 IC 的規劃下限：若欲以雙尾 5%、80% power 偵測真實 IC=0.02，約需 19,600 個獨立樣本；sample_stride=6 是每 30 分鐘一筆，故約 408 天連續資料。若主要主張是 real 相對 matched graph 的**增量**，yaml 候選門檻 ΔIC=0.01（目前仍是 proposed）；套相同單相關近似約 78,400 筆、約 4.5 年，但配對 ΔIC 的真正需求取決於 real/control 預測誤差協方差，必須以 Train-only paired block simulation 另算。兩者都是未計 regime clustering 的樂觀下限；因此若問題是「增量至少 0.01」，一次合理時長的單 BTC 前瞻 holdout 不經濟。

**(b)** 是，應承認停止目前的確認性市場主張比無期限收集 holdout合理。措辭須精確：Phase 5 FAIL 是「未達已凍結的效果門檻」，不是「證明效應為零」或「現設計必然無法偵測小於 0.03」。Real Ridge 的名目 95% CI 上界是 0.0217；real-random ΔIC CI 上界是 0.01335；real-scramble ΔIC CI 上界是 0.02276（`probe_val.json`）。依該 bootstrap 方法，前兩個區間並不支持某些更大的正增量；但 block=24 是否涵蓋長 regime 依賴、Val 重複查看造成的選擇偏差，都限制這種排除解讀。故建議記錄實際點估計／區間並停止，而不把估算 power 單獨當成 FAIL 的理由。

**(c)** 多資產可以增加「資產外泛化」資訊，但不能把樣本數直接乘資產數。若 m 個資產的對齊預測誤差／訊號殘差相關係數近似共同為 ρ，等長面板的粗略有效樣本數為 (n_{eff}≈ mn/[1+(m-1)ρ])。只有 ρ=0 才接近 m 倍；正相關越高，增加越有限。此處需要估的是相同時點、同一市場 regime 下模型誤差或 target residual 的跨資產相關，不是原始報酬的相關。現有專案未提供 BTC/ETH/SOL 的對齊 residual 相關，因此我不能給實際 ρ 或實際 n_eff；須在開發期估計、再以資產為 cluster 做分層 bootstrap。即使有效樣本增加，跨資產仍是不同外部複現，不可只當同分布 i.i.d. 行數。

### Q4. Random／scramble graph 零分布

同意前文把多 graph instances 放在 B 之後，理由不足。20 random + 20 scramble 約 4 小時 GPU、遠少於 Phase 6 預估上限；而且能直接估計單一 random/scramble 圖是否代表其圖生成分布。因此它應**優先於任何新的 Phase 6／6B 探索**，並可與 Q2 MDE 放在同一個預先凍結的診斷批次；要不要執行的唯一保留理由，是該結果只會描述已看過 Val 上的拓樸敏感性，無法修復確認性證據，也不會改變停止市場主張的決定。

必須明確標記：**這會是已查看 Val 上的探索分析。** 不能將經驗零分布 p 值寫成新的確認性檢定。20 個 graph per null 的經驗 p 最小解析度約 (1/(20+1)=0.0476)；若同時檢 real-vs-random 和 real-vs-scramble，Holm 首個門檻可到 0.025，20 個樣本連解析該門檻都做不到。故 20+20 只足以作 null 寬度 pilot／排序分布報告，不能支撐校正後顯著宣稱。要讓經驗尾機率解析度達 0.01，至少需各 99 個 control graph（還未計尾部估計精度與多重比較）；以每圖約 5.8 分鐘萃取粗算約 19 小時，不是 10 小時，需重新評估是否值得。

比較單位必須是每一個獨立 graph instance 的完整 Train-CV fit + Val 預測；同一市場樣本對各圖配對，並用時間 block 與 graph-instance 兩層不確定性。random/scramble 的生成程序、degree／weight／sign 保留規則、graph seed、收斂診斷必須先凍結；20 個隨機圖不是精確置換檢定，也不能把 20×10,504 當市場樣本數。Logistic IC=0.0254 為 secondary，只可報在同一圖分布的位置，不得因排名好而變 primary。

### Q5. 原 RQ2 與「歸納偏置」的貢獻

**(a) 是，我承認原計畫把問題定義得不良。** `research_v2.md §1` 的「Connectome 活動是否含有超越 OHLCV 的增量資訊」若按資訊量解讀，與其輸入是 OHLCV 的確定性圖像轉換、活動是其函數相矛盾。原本真正能檢驗的是固定容量和樣本預算下的表示／歸納偏置效用，當初應這樣寫清楚。這是研究問題定義的錯誤，不是 Phase 5 程式失敗後才用字規避結果。

**(b)** 同意：若有正結果，最多主張「在指定 decoder、資料、成本和外部樣本下，Connectome 拓樸帶來可重現的歸納偏置優勢」，並需勝過多個同規模 random 與 degree-preserved scramble；不主張新資訊或「果蠅腦會交易」。對「真實拓樸達到實質 ΔIC≥0.01 的機率」，我的主觀先驗是 **10%**，合理但很寬的主觀範圍約 **2–25%**。這不是由現有樣本估出的後驗機率；理由是目前唯一 real graph 在 Ridge 上低於 random、對 scramble 增量只有 +0.0026 且 CI 跨 0，OHLCV baseline 弱也沒有讓 real 勝出。Phase 0 強訊號解碼僅支持計算管線可恢復某些人工特徵，不能大幅提高「真實拓樸有市場歸納偏置」的先驗。若效應定義改為任意非零小差，機率會不同；這裡明確指實質 ΔIC≥0.01、跨 controls/時段可複現。

### Q6. Phase 6 可執行判斷

**(a) 主觀先驗（不是校準過的預測）：**

- **Phase 6A 通過所有 matched-control 門檻：10%。** MLP 增加非線性讀出，可能提取 Ridge 未讀出的弱特徵；但 Phase 5 Real Ridge 未達 0.02，random 的 IC 更高，市場方向 baseline 也近零，而且 5 seeds 同用一段已查看的市場資料。對隨機初始化有利不代表真拓樸優於兩個 controls。
- **Phase 6B 通過所有 matched-control／null 門檻：5%。** 它有更多可調自由度、稀疏上游活動並非真實遞迴突觸更新，存在 fitting gains 但沒有強理由預期只在 real graph 有增量；同時還要勝 random、scramble、6B-null 及 OHLCV 等比較。可合理給 1–10% 的主觀區間，非資料估計。

**(b)** 若 6B 在 Val 上反常地大幅勝出，原樣報告全部 seed、各 graph instance、參數量、Holm 校正後比較和失敗 gate；不得改命名掩飾或把它升為證據。由於這是 Phase 5 FAIL 後的明示 override、同一 Val 已多次使用且 Phase 6 是較高自由度模型，**即使比較本身預先列在 Phase 6 任務書並通過 Holm，也仍是探索性結果**。不在同 Val 再跑第二次確認。若值得驗證，鎖定 6B 權重與程式 hash、primary estimand/threshold、交易成本與 graph generation；另收從未查看、時間上在後的 holdout，一次檢驗，並預先計畫跨 graph instances 和時間 blocks 的分層分析。前述約 4.5 年是 ΔIC=0.01 的樂觀規劃量級，故若負擔不可接受，就如實停在「探索性候選，尚未確認」，不降門檻或換已看過的歷史時段。

**(c) 名稱建議：** 把 6B 結果標題改為 **「Phase 6B — Sparse upstream-activity readout（探索性；由 top-|w| 鄰接選特徵）」**。描述採「從各圖的上游活動摘要建立稀疏監督式 readout」，不要稱「Connectome 突觸可塑性」、「plastic layer learning」或「網路內學習」；因活動在固定原圖動力學下萃取，Δ 不回饋進遞迴模擬。這只是準確標示實驗，不要求重跑或改實驗本身。

### Q7. 方向 C 的預註冊與成功機率

方向 C 必須視為**新研究問題與新版本**，不能用已看過的 Val 重新挑 renderer 或 target：

1. 只選一個 primary target，例如未來 6 根 log realized variance（明確定義回報區間與年化／對數轉換）；不在 `future_return_1/3/6/12`、MFE/MAE、volatility 閾值之間挑結果。現有分類 HOLD/非 HOLD 不是未來波動 target。
2. 只選一種尺度保留 renderer：畫面使用固定 Train-only 尺度參數；明確保留價格絕對 range／bar return scale 的數值通道。其映射、裁切分位、缺值規則、版本 hash 都在任何新驗證前鎖定；不比較多種 renderer 後以 Val 表現挑一個。舊 renderer 只作歷史對照，不回頭改判 Phase 5。
3. 主要比較固定為真實 graph 對預先產生的多個 random/scramble graph instances，以及相同 OHLCV 直接特徵的同容量 decoder。baseline 至少包含 past realized volatility 的 persistence forecast。Train-only 決定標準化與 alpha；一個 metric（建議 paired out-of-sample loss skill）作 primary，其他 IC、方向、交易都是 secondary。
4. 先用 Train-only rolling-origin folds 和 block bootstrap 做 power/MDE；新 holdout 前鎖定最低實質增量與成本／風險規則。若 MDE 顯示對該最低效果 power<80%，或合成控制不過，停止；不打開 holdout。若開封後 CI 下界未過 0 或增量低於實務門檻，C 結束，不再改 renderer／target。

既有 nuisance-only BA=0.412 只表示七個過去 OHLCV／renderer 摘要對目前三分類 action label 有分類能力；它不能證明未來 realized volatility 可預測，也不能證明 Connectome 的未來波動能力。它提醒我們 volatility/horizon 門檻會混進 HOLD 判別，故 C 應用明確連續 volatility target 與 persistence baseline。

**C 不推翻 A。** A 的結論維持為「原方向 target、renderer、Phase 5 protocol 未達預註冊標準」；C 是另立問題，不能回溯替 A 找到成功結果。若一定給「C 在新的 sealed holdout 上達到有實質增量的成功機率」，我的主觀先驗為 **20%**（粗略不確定範圍 5–40%），不是資料估計。理由是現有特徵對含未來報酬閾值的分類有非零訊號，但未來波動不是同一 target，原 renderer 又抹去尺度；修 renderer 可能保留更多可預測變異，也新增設計風險。尚無未來波動 baseline 數值、MDE 或 untouched holdout，故這個百分比只能供決策排序，不得寫成研究結果。

### 對前文的明示修改

- **修改優先順序：** 前文把強訊號正控制 B 放成唯一推薦主線、把多 graph null 留待之後；現在改為已在執行的 Phase 6 依凍結規格收尾，之後把弱訊號 MDE 作主診斷，並把 graph-instance null 分析合併到同一探索批次。20+20 僅作 null 寬度 pilot；若要求校正後顯著性，樣本數不足。
- **修改 Phase 0 描述：** `SPEC_v3.md` 不只與 `phase0_report_remote.json` 矛盾，它明載 Amendment 2 是看到 PC-2 結果後才將其降為非 gate。故不得把後改規則後的 GO 當完全先驗凍結的成功；遠端報告的原 gate 狀態是 STOP。PC-1、PC-3 個別成功仍可報告，但整體 S4 不可被重述成沒有事後改規則。
- **修改停止建議的語氣：** 停止的是目前市場／交易確認性主張與新的 holdout 成本投入；不是宣稱所有細小效果為零。Phase 5 的區間與 MDE 要分開呈現：本次效應未達門檻，以及對更小效應的辨識力各自回答不同問題。
- **更正本節的 graph 計時：** 各 99 個 random/scramble control 共 198 個 graph；以每圖約 5.8 分鐘計約 19 小時。先前寫約 10 小時是算術錯誤，以上述更正為準。

## 交叉詰問回覆 II（Phase 6 後）

### Q1. Phase 6 程式獨立審查

**總判定：樣本排序與 Train/Val 隔離的程式檢查完整；6B 的自由參數實際作用、loss、尺度與輸出記錄不足以支持「已測到突觸可塑性」或「真實拓樸帶來增量」的解讀。** `origin/remote/phase6` 的 `run_phase6.py` 只允許 `train`、`val`；輸出記錄亦標示未讀取 forbidden split。以下判斷只依程式與現存 Train/Val 摘要，不涉及 test labels。

- **特徵與標籤順序：未發現本次錯位（低嚴重度；修正不會改變結果）。** `_read_allowed_split` 兩邊都按 `sample_id` 排序並逐列比對；Phase 5 feature sidecar 另核對 sample-ID hash、feature SHA256、shape 與有限值。6A 對特徵套用 label valid mask；6B upstream 保留完整 sample rows，本次 Train/Val 都是 0 筆 NaN label，長度仍一致，任何未來非零 drop 會被 selector 長度檢查擋下。輸出為 Train 31,556、Val 10,504，兩邊 `dropped_nan_labels=0`。資料集建置器按時間順序遞增指派 `sample_id`，所以內層時間切分的排序前提有來源保證。
- **Huber 尾部：標準化無洩漏，但不是穩健尺度（中嚴重度；是否改變比較無法由摘要判定）。** target mean/std 只由內層 Train 計算，Huber `delta=1` 對標準化殘差採線性尾損失；但標準差仍會受極端值影響，沒有 clipping/winsorization。結果檔未提供 target 分位數或超過一個標準差的比例，故不能斷言尾部已造成結果，也不能確認其影響很小。
- **HOLD 處理：標籤映射正確，退化實際發生（交易判斷高嚴重度；修正訓練目標可能改變 6B，不能挽救目前失敗結果）。** action head 以固定 `BUY/HOLD/SELL` 三類及無 class weight 的 cross-entropy 訓練，沒有漏掉 HOLD 類；但 6B-real 的 Val 預測為 588 BUY、9,916 HOLD、0 SELL，6B-null 為 100% HOLD，兩者均違反 minority-action gate。連續 IC head 仍可單獨描述，但這不構成可交易的三分類結果。
- **超參數沒有用外部 Val 選擇（低嚴重度；沒有發現 Val selection leakage）。** `select_hyperparameters_train_only` 只接受 Train 陣列，內部分成按時間排序的 80/20 並 purge 60 bars，4 組候選以 Huber+CE+L2 的 combined loss 選擇；`run_phase6` 在全部模型完成後才用 Val 算結果。內層約 6.3k 筆用來調參不等於外部 Val。限制是只用一個內層切分，且選擇目標不是主要 IC，故超參數不穩定與估計目標錯配屬中高風險。
- **final refit：符合程式所述流程，epoch 數仍有抽樣不確定性（中低嚴重度；不屬洩漏）。** 每個 seed 以內層選出的 hyperparameters 與 `best_epoch` 從新初始化，在完整 Train 重算 Train-only scaler 後重訓；沒有用外部 Val 決定 epoch。此法合理但受單一內層切分影響。實際 6A 選到 2–4 epochs，6B-real 為 11、23、11、23、30；不能把這些 epoch 解讀為在外部 Val 上確認的最佳值。
- **6B weight decay 漏掉 linear heads（高嚴重度；修正可能改變 6B 結果）。** `_loss_for_batch` 對 `SparsePlasticReadout` 的 penalty 只算 active `delta`，不含 return/action linear heads；Adam 本身沒有傳 `weight_decay`。因此 YAML/grid 名稱中的 weight decay 對 6B heads 實際為零正則化，且 early-stop 的 validation combined loss 也用同一不完整 penalty。這比單純「loss 不以 IC 選擇」更直接影響 6B 的 head 大小與 action collapse。
- **6B 未標準化且未記錄必要診斷（高嚴重度；可能改變結論）。** 6B 的 upstream activity 原樣進入 `z_i=Σ(W+Δ)x`；6A 則做 Train-only feature standardization。現存 JSON 有 source 數、float16/hash 與 collapse 比率，沒有各 feature/z 的尺度、訓練後 Δ、Δ/|W|、梯度或預測對 Δ 的敏感度，因此不能查明頭部是否受尺度壓制，也不能證明 Δ 有效移動。
- **6B 不是遞迴網路可塑性（對原科學命題高嚴重度；更正命名本身不會改變 6B IC）。** 每筆市場窗口各自從零狀態開始，同一影像電流重複 32 步，最後只保留來源 neuron 的時間平均；訓練出的 Δ 不回寫 reservoir，也不參與狀態更新。測到的是凍結動力學活動上的稀疏靜態 readout，不是沿時間流動市場輸入的 reservoir memory，也不是網路內突觸學習。
- **輸入噪聲、拓樸與 seed 的外推不足（高嚴重度；會改變拓樸穩健性的結論）。** upstream 活動每圖、每 split 僅用 `noise_seed=0`；5 個 seeds 是 readout 初始化／訓練種子，不是市場樣本、模擬噪聲或 graph instances。三種圖各只有 seed 0；大圖 scramble 每邊僅嘗試 0.1 次 swap，輸出沒有成功 swap 數、邊重疊率或混合診斷，故不能確認 degree-preserved null 已充分混合。需多個 graph instances 與動力學 noise seeds 才能估這些變異。
- **統計單位與摘要欄位須分清（中高嚴重度；可翻轉模型排名）。** paired block bootstrap 對共同市場時間區塊重抽，並非把 5 seeds 當 5 倍樣本，這點正確；但主要 aggregate 指標是先平均 5 組預測後算 IC，不等於 5 個 per-seed IC 的平均。例：6B-random per-seed mean IC=0.0081，ensemble prediction IC=0.0197；6B-real 分別為 0.0173 與 0.0175，兩種摘要會翻轉 random/real 排序。正式 paired ΔIC 用的是 ensemble，全部 6B real-control CI 跨 0、Holm p=1。報告不得把 per-seed mean 當成 paired 比較所檢驗的 estimand。
- **另一比較限制（中嚴重度；可影響 real-vs-control，但不支持當前正面結論）。** 每種圖只取一個 graph seed，且按各圖 top-|w| 邊後再全域裁到共同預算；real/null 才逐 DN 匹配邊數。活動維度與固定值比例亦不同（Train 6B-real 約 23.2% 常數/零、null 約 26.8%、random 0%、scramble 約 1.1%）。這些會混入拓樸差異。`run_phase6.py` 的 split whitelist assertion 本身是恆真檢查，但實際讀檔仍由 `_read_allowed_split` 阻止其他 split，故此處是低嚴重度防護缺陷，未發現它造成 test 存取。

### Q2. 對四項診斷的判斷與數字

1. **Δ 位移小：同意量級方向，不同意「≤1」是嚴格上界。** Phase6 JSON 顯示 6B-real final refit epoch 為 11–30；Train 31,556、batch 1,024，即每 epoch 31 次 optimizer step；最大 lr=0.001。若 Adam 每步每座標的正規化更新量約為 1，名目累積位移為 0.001×31×epoch，即各 seed 約 0.34–0.93。以交叉詰問提供、但本 checkout 無 graph cache 可獨立重算的 |W| 中位數 9–21、p90 48–92、最大 600+ 計，名目相對變化分別約 1.6–10.3%、0.37–1.94%、低於 0.16%。這只是一階尺度估算：Adam 的 moment 比值沒有被程式限制在 1，Δ 沒有 clipping，故不是數學上界；|W| 也不能代表 Δx 對 z 的效果。現有檔案缺 `graph_cache.npz`、upstream `*_train.npy`、final model state/Δ，且 JSON 不存 x/z moments 或 Δ，所以精確相對位移及功能影響不可驗證。6B 即使 Δ 很小仍是在固定 z 上訓練線性 heads；稱為近似固定特徵 readout 合理，稱「實際測了遞迴突觸可塑性」不合理。
2. **6B 缺輸入標準化：同意。** 6B 明確 `scaler=None`，而 6A 以 Train 均值／標準差縮放；活動與 z 的尺度未存檔，無法估其對梯度、head 或 collapse 的實際影響。
3. **以 combined loss 選 epoch：同意，並補充 6B 正則化不完整。** Huber+CE 等權，且 validation loss 含 L2；selection 不看 Spearman IC，也不報 loss 曲線平坦程度或候選間差距。6A 短 epoch 與 6B 11–30 epoch 是不穩定選擇的警訊，但不是 Val 洩漏證據。
4. **無弱訊號訓練注入：同意。** `test_phase6_training.py` 的人工 signal 測試只比較隨機 target 與 `target + noise(scale=0.5)`，IC 約大於 0.5；它沒有訓練模型。稀疏遮罩測試另用小矩陣與 AdamW，並非實際 `_train_epochs` 的 Adam/Huber/CE 路徑。這些測試不能證明在 31,556/10,504 樣本、時序依賴及 IC 0.03/0.05 時有足夠檢定力。

### Q3. 監督式訓練已驗證到哪裡

**判定：資料對齊、split 白名單、有限值、模型 I/O、訓練函式與輸出彙總的部分工程契約有檢查；Phase 6 監督式學習對弱訊號的可恢復性尚未驗證。**

- **已驗證：** runner 使用 Train/Val 白名單；sample IDs、Phase 5 feature hashes、樣本數與 valid mask 對齊；Train/Val 均無 dropped NaN；selector API 不接受外部 Val；purge、輸出 shape、finite、mask 梯度及模型可回傳 3 logits 有小型測試。Phase 6 一致性檢查顯示前 100 筆 Val 的原始 W 加權活動與 frozen DN 讀出可對上（flattened Pearson r=0.261；每 DN r 中位數 0.849），但這只驗證部分運算對應，不驗證 target 訊號。
- **未驗證：** 在真實規模 feature 上跑完整 Train-only 選參、refit 與 Val 評估能否回收已知 IC 0.03/0.05；weak-signal power、null 假陽性率、尾部和尺度敏感度、連同 HOLD 的三類學習；6B Δ 是否真的改變預測。Phase 5 MDE 是 Ridge，不是 Phase 6 MLP/6B。`run_phase6.py` 硬讀市場 label parquet，沒有注入介面；直接呼叫相同 training functions 的 harness 能驗證 trainer，但要宣稱驗證 runner 全流程還須有獨立、合成標籤的 orchestration 測試。Phase 6 JSON 亦明列 `PHASE6_OVERRIDE_EXPLORATORY`，多項門檻標為 `PROPOSED_NEEDS_USER_CONFIRMATION`，故本次結果不是 confirmation gate。
- **最低可解讀測試：** 先做一個直接 feature-teacher 正控制，確認同一 trainer 能從預先凍結的 feature 線性組合回收 IC=0.05；再做 Batch 5 momentum 端到端測試：`s_t=log(C_{t-1}/C_{t-7})`（只取當下可見的前 6 根 bar），用已鎖定的實際 Phase 6 Train/Val features 生成 `y=βs_t+ε_t`。β 僅用 Train 校準，noise 由 Train-only 分布／事前固定 block generator 產生；Val noise 獨立，不得為了把 Val oracle IC 調到目標而另行改 β。做 IC=0、0.03、0.05 三種，合成 action 也只由合成 y 和凍結門檻生成；完整保留 4 組內層選參、5 個 model seeds、refit、同一 24-sample block bootstrap。IC 0.03/0.05 的「回收」建議定義為至少 2/3 oracle IC 且模型 Val block-CI 下界大於 0；100 組獨立合成 noise cohort 中，0.03 的回收率 Wilson 95% 下界須 ≥0.80，IC=0 的家族錯誤率上界須 ≤0.05。直接 teacher 失敗表示 trainer/尺度有問題；teacher 通過、momentum 失敗表示輸入表徵未保留此訊號。Val 標籤在此是獨立合成標籤，不是市場未來標籤；這仍只校準固定 feature panel 下的條件 power，不代表新市場樣本 power。
- **成本：** Phase 6 全流程記錄 7,813 秒（2.17 小時）RTX 5070 elapsed，其中 upstream 萃取約 19 分鐘，GPU active time 沒有獨立記錄。若快取 features，一組含 real 6A/6B 與 IC 0/0.03/0.05 六個模型條件的合成 cohort 約 1.6–1.9 GPU-equivalent 小時；100 組約 160–187 小時，按既有 elapsed 線性縮放，另加一組 direct-feature teacher 約 0.5–0.6 小時及 harness 工時。單一 cohort 可用約 2 小時作工程 sanity check，但不能宣稱已估得 80% power。精確檢定力驗證成本不符合本計畫建議上限，故不建議為重跑 Phase 6B 花這筆預算。

### Q4. 可行方向、成功率、成本與唯一推薦

**仍有繼續研究的可能，但限於改問「真實連接體拓樸是否對已知時間序列計算任務提供歸納偏置」；原市場交易確認性路線停止。** 下列機率均是以明確成功事件為條件的主觀決策先驗，不是資料估計；三條路線的成功事件不同。

- **A. 標準合成時間序列 reservoir 任務（推薦）。** 主測 NARMA10，Mackey–Glass 與 Lorenz 只作事前 secondary；主要成功事件定為真實圖相對至少 20 個 random 與 20 個充分混合 degree-preserved scramble graph 的 median，NARMA10 holdout NMSE 至少降低 5%，且 graph-instance/時間 block 配對區間下界大於 0。主觀機率約 **15%（範圍 5–30%）**。這比市場資料更直接測非線性時間記憶，但不能回答可交易性。輸入須採可說明的感覺路徑：優先用已有 annotation/retinotopic 座標把標量時間序列映成固定全視野亮度，再由 photoreceptor 輸入；不得隨機投影或看結果換 sensory group。simulation 必須跨連續 t 保留 state 並事前鎖定 washout；Phase 6 現在每窗口歸零、重播 32 步再平均的流程不能直接拿來測時間記憶。1291 個既選 DN 可保留為同一固定活動向量，丟掉 BUY/SELL 語義，改用單一 Train-only ridge scalar readout；若改成全神經元 readout，視為另一研究，不得看 benchmark 結果後挑。估計 3–4 個工作天、至多 8 GPU-h，前提是首日確認可追蹤 state 的模擬器與固定 sensory mapping；此為工作量上限而非已量測工時。
- **B. 修正 6B 再跑市場。** 即使改成 Δ=W⊙ρ、Train-only 標準化、以 IC 早停，也是在看過 Val 後的新模型；同 Val 只能 exploratory，不能確認。要回到市場確認需再做上述弱訊號驗證並另找未碰過的未來 holdout；偵測 ΔIC≈0.01 的先前樂觀規劃量級約 4.5 年，10,504 筆則約 208 天，但樣本相依會增加成本。主觀機率：取得「新資料上真實拓樸市場 ΔIC≥0.01 且可交易」約 **2%（範圍 0.5–5%）**；單跑同 Val 得正點估計不計成功。修正與單輪重跑約 1–2 工作天、2–4 GPU-h；弱訊號 power 若按 Q3 達標需約 160–190 GPU-h，另加新 holdout 等待時間。因此不推薦。
- **C. 停止並封存市場問題。** 新增正面科學發現機率為 0；依既有 frozen gates 封存「本設定未達所需效果」的決策確定性為 100%。成本 0 GPU、小於半個工作天。此路線正確保留負結果與結論界線，但放棄測試拓樸一般計算能力。

**唯一推薦 A，總預算上限 8 GPU-h、4 工作天；不得用超支或換 benchmark 補救失敗。** 先做 stateful simulation 與固定 sensory/readout mapping 的小型正負控制；超過 1 個工作天仍無法證明 sample/state 對齊，或 0.05 planted signal 未被既定 ridge readout 回收，就停。通過後才做凍結的 NARMA10 主測；real 未達 5% NMSE 增益、區間包含 0、或與 controls 的差異只出現在單一 seed/graph，就結案，不再改 mapping、readout 或 primary task。

### Q5. 先驗校準與前一輪建議修正

- **6A 全 matched-control pass：舊先驗 10%。** 這次實際結果已知為 FAIL；對一個未來、獨立複現仍照原市場 protocol 通過的主觀機率下修至 **2%**。五 seed mean IC 雖為 0.0121，並未顯示穩定 real-control 增量。
- **6B 全 matched-control/null pass：舊先驗 5%。** 實際結果 FAIL，且 real 有 0 SELL、與 100% HOLD 的 null 在 IC 上相近。若只是修 code 後對舊 Val 重跑，通過的確認性機率仍是 **0%**；若指新資料的未來市場成功，估 **1–2%**。
- **市場真實拓樸 ΔIC≥0.01：舊先驗 10%。** 下修到 **約 2%**（主觀寬範圍 0.5–8%）；6B-real 對 random 的正式 ensemble ΔIC=-0.0023、95% CI [-0.0239, 0.0202]；對 scramble Δ=0.0124、CI [-0.0127, 0.0376]；對 null Δ=0.0025、CI [-0.0147, 0.0187]，Holm p 均 1。點估計不能抵銷不確定性、action collapse 及單一 graph instance。
- **波動方向 C：舊先驗 20%。** Phase 6 target 是 future return/action，不是 future volatility，故不以本次結果更新；維持 **20%** 僅為尚未校準的主觀先驗。

先前若把 6B「五個 seed 的穩定度」說成拓樸訊號較可信，判斷過度：五個 seeds 共用相同市場樣本與 `noise_seed=0` 活動，且 real-null 都穩定、random 的 ensemble 排序反而翻轉。先前對 Phase 6B 可解讀性的正面期待應撤回；現有 evidence 只支持精確報告探索性輸出與停止原市場確認性主張。

### Q6. 6B 穩定 IC 的不回看 Val 判別法

**目前不能在「真實上游活動含弱預測訊號」與「固定特徵加低變動線性 head」之間作因果判定。** 6B-real per-seed mean IC 0.0173、SD 0.00068；null 為 0.0149、SD 0.00101。兩者都使用 real connectome 的相同 upstream activity bank，只改抽取的 incoming edge set；這種相近且穩定的結果不支持真實局部拓樸優勢。per-seed 預測分布不是獨立市場樣本，亦未涵蓋 simulator-noise 變異。

可先用 Train-only 作診斷：凍結目前 real Train activities，在相同 purged expanding-time folds 重訓三個版本——`Δ=0` 的固定 W linear head、現有 sparse-Δ head、matched 6B-null head；每一 outer fold 的 hparam selection 只能用該 fold 內資料，並輸出 out-of-fold IC/loss、有效 Δ/|W| 分布、z 尺度與零訊號 permutation 結果。若 Δ-head 對 W-only 的 OOF 增益未達 0.01 IC 且區間含 0，結果與固定表示/readout 足以解釋相容，不能宣稱 Δ 貢獻或拓樸效果；若增益跨 folds、seed 與新 simulator-noise seed 重複，才是「Train 內可學到」的證據，仍不是外部市場泛化證據。此為看過研究結果後的 Train-only 診斷，不具確認性；若要判定它能否延續到新市場，仍需新資料。
