# G1 v2 Stage A/B 獨立審查

日期：2026-09-22。範圍：Q1–Q6、目前程式、規格、既有 Stage B JSON 與測試原始碼；未執行模型、測試或實驗。本報告為唯一新增檔案，未修改程式、規格、結果摘要或 README，未 commit。

## 總判斷

**Q6 選（b），高信心：能力 gate 受到確定的因果可達性設計缺陷污染，必須先修正規格與驗收，再決定是否另版驗證。** 這不是已證明 Ridge 算錯，也不是一次有意義的非線性能力陰性發現。數值上的 d2 FAIL 很可能正確反映目前設定，但這個設定讓 DN 在評分時根本尚未取得 target 所需的當下獨立輸入。

關鍵責任在規格：`G1_SPEC_v2_draft.md:68` 規定同步更新，`:104` 又要求以 DN 的 x[t+1] 預測含 v[t] 的乘積；這兩項配上互不重疊的 R1-6 與 DN，使 d2 的母體最優 R² 不大於零。**這是本審查者先前設計規格時漏掉的限制，不能歸咎於實作者，也不能以預先註冊掩蓋錯誤。** raw-u 修正了奇偶對稱問題，沒有修正訊號傳播延遲。

Stage C/D 應繼續保持關閉，但理由應是「目前能力診斷設計不具鑑別力，尚未取得有效能力證據」，而非已證實 FlyWire 的非線性能力不足。既有 JSON 的觀測值及歷史 status 應保留；不能靜默將它改成 PASS，也不應將「依規格正確執行」混同「規格能回答科學問題」。

### 證據適用範圍

- 依 `g1_manifest.py:98` 的來源 hash 規則唯讀重算，目前 source snapshot 與 `research/outputs/v3/g1_stage_b_real_cuda.json:21` 完全相同：`24c88f77e4868937c36e9f3d36909a87acafdc19c3689f11c345d3b23a4bbdb1`。目前 spec SHA 也與該 JSON `:20` 相同。這排除了「本次只看到執行後另一份程式」這項疑慮。
- 唯讀檢視既有 `research/data/flywire/graph_cache.npz` 的 sensory_idx、buy_idx、sell_idx：全部 10,582 個 photoreceptors 與 1,291 個 DN 的交集為 **0**。R1-6 是前者子集，因此同樣與 DN 不重疊。未執行 simulator 或重建圖；此查核本身不是實驗。
- JSON 僅提供彙總分數、rank、condition number 等欄位；沒有完整 X、check predictions、每 α 的 OOF losses 或每 sequence 負控制誤差。因此不能聲稱已由數據定位負控制微小正值的唯一根因。
- 嚴重度：Blocker 表示足以否決本次科學推論；P1 表示重要驗收／推論缺陷；P2 表示需記錄的數值或規格一致性問題，但未證明造成目前分數。

## Q1. d2 target 與輸入／state 對齊

**判斷：target 算式及陣列切片正確；因果時間對齊有 Blocker 級設計缺陷。影響 RESULTS_G1_v2.md 的核心科學解讀。**

### 陣列與 lag 並未寫反

- `research/pipeline/g1_v2.py:181` 將同一 u 轉為 float64，`:186` 計算 v=u−0.25；`:199` 的迴圈在 t≥9 使用 v[t−9]，`:206`、`:207` 分別設定 d1 和 d2，`:209` 才切除 washout。
- `research/pipeline/g1_stage_b.py:544`／`:555` 將各 sequence 的 `s["u"]` 直接交給 reservoir；`:550`／`:561` 將輸出切為 out_states[washout:]。`:614`–`:617` 的 target 也是同一批 `s["u"]` 與同一 washout。
- `research/pipeline/g1_reservoir.py:467` 的 input_centering 預設 False，`:506`–`:522` 先更新，再將 x[t+1] 寫到輸出第 t 列。production 沒有再減 0.25；target 的 v 不會回寫 u。未發現這段把 lag 9 寫成 lag −9 或切片錯一列。
- Stage B 未傳入 u_negative，但有效點從 t=500 開始，d2 的最早 lag 為 t=491，故不影響此次兩個 diagnostic targets。另有規格一致性缺口：`g1_manifest.py:286`–`:297` 的 production generator 使用零負歷史，未依 spec `:45` 先生成獨立 u[−9…−1]；它不是這次 d2 失敗的原因。

### 真正限制：DN 的當下輸出沒有當下輸入

`g1_reservoir.py:506` 先以舊 X 做 W@X，`:510` 只向 sensory indices 加 u[t]，`:511` 同步更新所有神經元，`:522` 隨即讀取 DN。對不在 sensory 集合的 DN j：

`x_j[t+1] = (1−leak)x_j[t] + leak·tanh((W x[t])_j)`。

右式不含 u[t]。即使有 R1-6→DN 直接邊，該邊本步讀到的也是 R1-6 的舊 state；u[t] 最早要再一個更新才可能抵達 DN。多跳路徑還需要更多時間。

令 F_(t−1) 為截至 t−1 的輸入歷史。零初態、固定圖下的 DN x[t+1] 是此歷史的函數，而 v[t] 獨立且均值為零，因此：

`E[v[t]v[t−9] | F_(t−1)] = v[t−9]E[v[t]] = 0`。

對任何只看當列 DN state 的 predictor f（不只線性 Ridge），母體平方誤差為 `E[d2²]+E[f²]`，最佳 predictor 為零、最佳母體 R² 為零。有限樣本 R² 可以小幅正負波動，並不反駁此界。R²≥0.10 的能力 gate 因此要求了一個在此資訊集合下不可達的效果。

`RESULTS_G1_v2.md:12` 的接近零分數符合這項理論限制；它無法區分「DN 會不會計算歷史乘積」。`:29` 將此作為一般計算能力路線的陰性結案依據，應撤回該科學歸因。停止目前流程本身仍正確。

## Q2. DN 變異、有效秩與條件數

**判斷：現有欄位顯示大量維度被活動門檻移除及高共線性；不能據此證明其餘維度沒有非線性資訊。P2；不推翻 Q1 的確定因果限制。**

- `g1_bench.py:296`–`:306` 以 fit std≥10⁻⁸ 保留欄位。Stage B JSON `:78`–`:81`、`:93`–`:96` 顯示 active_p=277、effective_rank=277、discarded=0。由 1,291−277 可知有 **1,014 欄在 final fit 的 std gate 被排除**；不是 SVD 又丟掉了 1,014 個奇異方向。低活動不等於生物上不存在訊號，也不等於全部 DN 完全常數。
- retained_condition_number=70,909,443 是保留之 X 的條件數，數值上偏高；rank=277 是指定容差下的數值 rank，不是 277 個可靠、獨立的任務訊息通道。可全 rank 又高度共線。
- `g1_reservoir.py:489`、`:524` 輸出 float32；`g1_bench.py:298`、`:299` 先在輸入 dtype 算 mean/std、`:306` 標準化，直到 `:331` 才轉 float64 做 QR/SVD。float64 solver 無法恢復上游 float32 已損失的精度；近常數欄位的尺度估計及標準化也可能放大 rounding。這是應檢查的數值風險，**沒有現成證據足以認定它造成 277 rank 或 d2 FAIL**。
- 對 d1 的 α=0，病態方向較值得關注；d2 的 α=100 對小奇異值有抑制，不能直接拿未正則化 X 的條件數判定 d2 solver 已失效。兩個 head 報相同條件數是使用同一 X 的結果。
- 飽和率零只表示沒有計到 |x|>0.9，不能判斷 states 是否幾乎零、是否處於近線性區間，或是否包含所需交互作用。現有 JSON 未存 DN mean/std 分布、preactivation 分布、完整奇異值或 check feature statistics，故上述機制未能定量區分。

對摘要的影響：可以保留 d1 分數作觀測，但不能將 rank 或健康 PASS 當成 readout 充分性的證明。即使改善數值精度，Q1 所缺的 u[t] 仍不會出現。

## Q3. Ridge/SVD、CV 與 α=100

**判斷：主路徑的 Ridge 公式與 sequence CV 未見能解釋 d2 假陰性的代數錯誤；α=100 沒有理論上保證足夠的依據。另有 P2 規格差異，尚無證據改變本次選擇。**

- Stage B 用的是 `G1RidgeReadout`（`g1_stage_b.py:622`、`:627`），不是 `g1_v2.py:669` 的獨立 direct solver。不能因後者正確就宣稱 production 已驗證。
- `g1_bench.py:334`–`:363` 的 QR＋SVD 與 filter `s/(s²+nα)` 符合含 1/n 的 Ridge objective；α=0 用預註冊相對奇異值 cutoff。`:365` 的 intercept 亦符合中心化解。
- `g1_bench.py:392`–`:428` 每折留兩條完整 sequence、fit-only 標準化，最後以 pooled OOF NMSE 選 α；`:441`–`:454` 用全部 fit refit。未見把 Diagnostic-check target 用來選 α。
- **P2：tie-break 不完全依規格。** spec `G1_SPEC_v2_draft.md:84` 要求「數值完全同分」才選較大 α；程式 `g1_bench.py:436` 將相對差≤10⁻⁹ 也當 tie，可能在極小差異時偏向較大 α、甚至碰上界。封存 JSON 沒有每個 α 的 OOF losses，因此未證實本次受到該分支影響，不能斷言 α=100 是這個 bug 造成。
- spec `:82` 規定 y mean/std 標準化；production `g1_bench.py:409`–`:411` 只有中心化。對目前單一 target、固定 α、線性 Ridge 與正確反尺度而言，y 乘常數只會使權重同比縮放，原單位預測在精確算術下等價；此差異本身不能證明 d2 受到系統性懲罰。

α 的合理大小取決於標準化 X 的 `XᵀX/n` 特徵值和 target 投影。標準化後各欄變異為一，不代表其聯合特徵值最大只有一；p=277 時 trace 約為277，最大特徵值可以遠大於100。另一方面，完全沒有可泛化訊號時，最佳收縮可以趨近 α→∞。所以100既不是已證明的完整搜尋上限，撞上界也不等於「證明沒有訊號」。

本次 d2 在既定網格下選強正則化、check 近零，與 Q1 的不可預測性相容；不需要擴網格便能判斷其能力推論無效。本報告不建議事後擴網格。原 alpha-boundary no-go 是工程規則，不是資訊不存在的定理。

## Q4. 負控制的微小正值

**判斷：未發現常數基準／模型使用不同 target 或不同抽樣分母的 bug；微小正值的實際根因未定。負控制配對存在 P1 推論設計問題，摘要的既定歸因不足。**

- `g1_v2.py:482`–`:485` 確實各 split 內循環移位一條；`:495`、`:496` 重新 CV/refit，`:503` 的常數是錯配 fit target 的 pooled mean。沒有使用 check mean 訓練 baseline。
- `g1_v2.py:509`–`:522` 對同一 y_b、p_b 比較模型與常數 MSE，再除同一常數 MSE；未見所問的分母不一致。`:510` 抽樣的是 **10 個完整 check sequence pairs**，不是把 20,000 點當獨立 bootstrap n。因此 `RESULTS_G1_v2.md:17` 將原因直接定為「20,000 點下檢定力過高」沒有充分依據。
- 很小的 G 仍可能有窄 CI：差分 MSE 會消去大量共同 target 變異；強收縮模型若接近常數，其 G 本來就落在很小的尺度。窄 CI 本身不證明洩漏，也不證明數值錯誤。一次隨機錯配亦可能出現偶然拒絕。
- **P1：循環配對後的 pairs 並非已證明獨立。** pair s 使用 X_s 與 y_(s+1)，pair s+1 又使用 X_(s+1) 與 y_(s+2)；y_(s+1) 與 X_(s+1) 來自同一原始 stream。將這些 pairs 當 iid units bootstrap，未保留這種共享來源的依賴。fit 的 sequence CV 也只按 pair 索引切分，不能保證原始 stream 不跨 train/validation 角色。這不是跨 Diagnostic-fit/check 的直接 label 洩漏，但會削弱「iid null 下的 CI／CV」解釋。現有證據不能確定偏差方向或量值。
- 上述循環錯配其實遵循 spec `:110`，因此也是規格設計問題，不能單方面稱實作違規。應撤回「已知根因、無須深究」的說法；也不能因效果小就事後改為 PASS。

兩能力 head 與負控制共用 `G1RidgeReadout`，但正常能力 gate 沒有循環配對，其 R² bootstrap 與負控制的 G bootstrap 也不是同一函式（`g1_v2.py:355` 對 `:452`）。故此問題不直接證明 d2 的正常分數算錯；d2 的主要無效原因仍是 Q1。僅憑此 JSON 無法在數值誤差、偶然拒絕、配對依賴等解釋間作唯一判定。

## Q5. 健康 PASS 與工作點疑慮

**判斷：健康 PASS 不足以證明任務所需動力學正常；P1 驗收覆蓋缺陷。這次結果既沒有印證，也沒有推翻「換 rho/leak 能改善可達任務」的猜測。**

- `g1_stage_b.py:569`–`:585` 的健康判定僅涵蓋飽和、初態遺忘與有限值。它們沒有檢查當下訊號是否已到 DN、tanh 曲率、非線性記憶或 target 可達性。快速遺忘也可能伴隨過弱的有效記憶／驅動。
- **oracle 是自我比較。** `g1_stage_b.py:183`–`:186` 將 check target 與自己計算 R²，與 reservoir states、Ridge、獨立 target 重算完全無關。只要 target 有有效變異，R²=1 就是算式恆等式；不能驗證 lag 或 simulator-to-target 對齊。`research/tests/test_g1_v2.py:352`–`:355` 同樣自比，未補足這項覆蓋。
- **go fixture 改變了關鍵資訊條件。** `g1_stage_b.py:240`–`:242` 明確讓 sensory 和 DN 重疊，當下 u 因此直接進入 readout neurons。`research/tests/test_g1_stage_b.py:48` 的 end-to-end go 測試使用此 fixture，不能驗證真圖 sensory/DN 分離時的 d2 可達性。
- 300-node ESN fixture 在 `research/tests/test_g1_v2.py:306`、`:314` 使用 full states，包含 sensory nodes，並不是固定 disjoint DN readout。它可以驗證 raw-u 打破奇偶對稱，但不能為 real DN 的同時刻乘積任務提供陽性對照。

飽和為零可能與近線性運算相容，但沒有 preactivation／曲率數據，不能定為根因。更重要的是：只改 rho/leak 不改輸入、讀出與同步時序，仍然無法讓當列 DN 取得 u[t]；不能用工作點搜尋解決 Q1 的限制。先前「共同 rho 選擇」疑慮在本次沒有得到有效實驗回答。

## Q6. 最終結論、摘要影響與最小後續驗證

**（b），高信心；已發現須修正的設計及驗收缺陷，另有具體 tie-break 規格不一致。未發現 production target 算式寫錯或 SVD 代數錯誤。**

對結果的逐項判斷：

| 現有敘述 | 審查判斷 |
|---|---|
| d1 R²=0.5775、d2 R²≈0、α=100 | 保留為封存執行的觀測；未重算，沒有證據需篡改數字 |
| 此次門檻按程式回傳 FAIL | 判定算式符合主要 gate／上界規則；不能因此推得任務設計有效 |
| d2 FAIL 證明此 DN reservoir 缺少非線性乘積能力 | 不成立；所選 target 含當下尚未可達的獨立輸入 |
| 健康與 oracle PASS 排除管線問題 | 不成立；oracle 自比及重疊 fixture 掩蓋關鍵時序條件 |
| d1 負控制 fail 已知只是檢定力過高 | 證據不足，且循環配對的推論單位有依賴問題 |
| 現在進 Stage C/D 或解鎖市場方向 | 不支持；應保持暫停，不把無效診斷當通過 |

因此建議結果另行標註 **「診斷設計無效／不能作能力推論」**，而不是自動把依規格執行的歷史 run 改成「程式算錯」。`RESULTS_G1_v2.md:3`、`:17`、`:29` 的解讀需要更正；本次依唯讀邊界未改該檔。市場既有負面證據不受本審查推翻。

最小後續驗證僅描述，未執行：

1. **先做因果 fixture，不重跑真圖 grid。** 建立 sensory/DN 不重疊的單邊及多跳小圖；兩條輸入只改 u[t]，確認同列 DN 不變，而改變最早在圖路徑允許的後續列出現。獨立手算 d1/d2，取代 oracle 自比。這可直接驗證 Q1，不依賴擇優性能。
2. **若要研究乘積能力，先明定計算延遲。** 在至少能取得兩項輸入的時刻讀出過去的乘積；最短路徑只給出必要延遲，不保證足夠能力。延遲規則應由結構／計算時序事先決定，跨圖使用同一規則，不掃 lag 挑最好 R²；延遲重建任務也不能冒稱原本即時預測。此項須新規格與新的未觸碰 fit/check streams，不能在已看過的 check 上救援舊 gate。
3. **負控制先查可稽核的誤差分解。** 若已保存預測，列各原始 stream 的 SSE、baseline SSE、prediction mean/std 及 fit mean，並核對 pair 依賴；若未保存，需另行授權後才能重現。新的 null 可使用與 states 來源完全獨立的 target streams，使每個 pair 的原始來源互不重疊，或校準保留共享來源的完整隨機化程序；不可直接沿用目前 iid pair CI 作根因證明。
4. **數值檢查保持同一設定。** 在固定、具近常數／病態特徵的小型 fixture 對照 float64 scaling＋獨立 Ridge reference，查每 α 的 OOF loss 與精確 tie-break。無須也不授權擴 α 網格。此驗證不能替代因果修正。

允許修正的理由是可由同步方程與索引集合獨立證明的設計缺陷，不是結果「接近門檻」或希望提高分數；這也不授權無限新版本、追加預算或直接開始新確認性實驗。新研究是否值得投入須另作決策。
