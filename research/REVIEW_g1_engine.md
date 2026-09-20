# G1 引擎審查結果

## 總判定

**有會改變結論的缺陷；目前不放行確認性 G1。** 關鍵風險是 spectral-radius 失敗 fallback 可能把圖縮放到錯誤動態區，及 α=0 的 Gram 矩陣絕對 cutoff 不等同規格宣稱的 Moore–Penrose／lstsq 解。另有全腦 gate 的 GPU 記憶體超限與 sensory map 不一致，故現有 8 小時估算和 small-ESN 結果不能證明 FlyWire G1 已可執行。

本文件是靜態程式審查；本次未重跑測試或 CUDA。154+ 測試與 300 節點 ESN 的 NMSE 是任務書提供的結果，未在本次獨立驗證。現有 G1 測試主要使用小型合成矩陣；CUDA 不可用時，CPU/Torch 對照測試會選 CPU Torch，不能據此宣稱已驗證 RTX 執行。

## 符合規格的部分

- NARMA10 reference recurrence 實作正確；t=0…8 的 u[t−9] 明確設為 0，第一個非零延遲乘積在 t=9，reservoir 更新後的 x[t+1] 對齊 y[t+1]。ring/history 實作逐項依相同順序累加 y 歷史。
- Scipy 與 Torch 每個 time step 都執行一次 W_eff @ X、加入 u−0.25，再更新並記錄 x[t+1]。將 W 視為 row=post、column=pre 時，稀疏矩陣乘 state 的方向正確，且與 ConnectomeGraph schema 一致。
- 批次以神經元×stream 矩陣乘法運算；每欄各自演化，輸入也是逐 stream 加到同一 sensory index，不會因矩陣乘法讓不同 stream 互相混合。跨呼叫延續由呼叫端把前次 final_state 明確傳為 initial_state，並非 simulator 內部自動保存；Scipy 測試涵蓋 chunk continuation。
- Ridge 截距透過中心化後回算，不受 α 懲罰；每一 fold 的 feature mean/std 僅由該 fold Train 求得；預設 5-fold 每 fold 留出兩條完整 sequence，未把同一條序列的時間點拆到 Train/validation；同分選較大的 α。這些部分符合草案。
- 輸入注入公式及矩陣方向本身正確，但前提是 caller 傳入的 sensory_idx 已符合 G1 的 R1-6 限定；目前 timed pilot 未符合此前提。

## 缺陷與風險

| 嚴重度 | 位置與判斷 | 是否可能改變結論 | 放行條件 |
|---|---|---|---|
| Blocker | g1_reservoir.py:38-56：eigs 任何例外都退回 50 次 power iteration，最後以 abs(vᵀWv) 當 spectral radius。這對一般 signed、非對稱圖不成立；例如二維旋轉矩陣的 spectral radius 為 1，該 Rayleigh quotient 卻可為 0。呼叫端接著以該值縮放，且沒有縮放後獨立核對。 | **是。** 若 FlyWire 或 control graph 走到 fallback，rho 工作點、動態、MC 與 NMSE 對比全部可能不再是規格所稱的條件。是否在實圖觸發尚無證據。 | 不收斂就停止；使用收斂且殘差可核對的 eigensolver，記錄 graph hash、方法、殘差，並核對縮放後 rho。不得把 Rayleigh fallback 當成等價替代。 |
| Blocker | g1_bench.py:248-269：α=0 以 G=XᵀX 的特徵值大於絕對值 1e-12 決定逆矩陣方向。這個尺度相關 cutoff 不等同 np.linalg.lstsq 的相對奇異值 cutoff；先形成 Gram matrix 也會平方設計矩陣條件數。docstring 的「matches np.linalg.lstsq」只可能在一般良態例子近似成立。 | **是。** α=0 已在任務書提供的 300 節點例子出現；DN 共線情況下，此差異能改變係數、CV 選擇及 Test NMSE。該例數值未在本次驗證。 | 使用 fold-local centered design matrix 的 float64 薄 SVD，以固定 machine-epsilon 相對秩 cutoff 求 Moore–Penrose 解；回報有效秩、discarded directions、保留子空間條件數。此規則不增加可調參數。 |
| 高 | g1_reservoir.py:117-178、235-282；g1_pilot.py:168-175；g1_bench.py:380-410：全腦 saturation gate 需要所有 neuron 的 post-washout state。現有 pilot 以 batch 10 要求 full states；float32 輸出約 30.5 GB（28.4 GiB），尚未含其他張量。Scipy 預設路徑超過 16 GB Mac 記憶體；Torch 路徑超過 RTX 5070 12GB VRAM。timed pilot 只保留 1,291 維 readout，未量到 gate 完整路徑。 | **會改變可執行性與預算判斷。** 照現有 pilot 實作會 OOM；略過 gate 則不能宣稱通過。 | 在 simulator 內線上累計飽和、非有限值與分母，只輸出摘要；以完整 gate 路徑重新計時。 |
| 高 | flywire_graph.py:9、142-145；g1_reservoir.py:204-208、232-233；g1_pilot.py:157-166、177-183：graph.sensory_idx 定義為 R1-6、R7、R8 全部 photoreceptors，pilot 把該陣列原樣傳入 G1 engine。故 R7/R8 也收到 u−0.25，且被排除於 saturation gate 的 non-sensory 分母。G1_SPEC_draft.md 只允許有限 retina 座標的 R1-6。engine 只依傳入 index 注入，沒有驗證類別。 | **會改變輸入條件與 gate；若沿用於主分析，會改變可支持的結論。** | caller 對齊 graph.meta 的 photoreceptor_type、u、v 和 sensory_idx，只傳有限座標的 R1-6；對選中索引與 graph hash 留存摘要並作一致性斷言。 |
| 高 | g1_bench.py:473-545；g1_pilot.py:200-208：pilot 雖抽出 DN readout states，但逐 sequence 在同一批 observations 上 fit 和 score MC；compute_memory_capacity_fast 以未 rank-revealing 的 QR 投影所有欄位，常數／共線欄位可能加入虛假方向。樣本內 R² 對高維特徵偏樂觀，且會作為 rho 選點依據。 | **可能。** MC 排序能改變共同 rho，進而改變主要 Test 對比。 | MC 用固定 DN 維度與 washout 後資料，按完整 sequence 做 5-fold Train-only CV；fit/score 分離，採 rank-aware 解法，每 lag 僅在 held-out sequences 算 R²。 |
| 高 | g1_bench.py:380-410；g1_pilot.py:173-183：saturation helper 未驗證有限值；NaN 比較 abs(x)>threshold 為 false，會被算成未飽和；空輸入也回傳比例 0 而通過。pilot 有明確切除 washout，這部分符合規格，但 helper 本身不保證呼叫者會切除。 | **可能。** 非有限或空狀態可錯誤通過 gate；若後續 readout 沒讀到該 state，可能不會由 NMSE 自動攔下。 | 空輸入、NaN、Inf 一律 fail；保留 caller 的 post-washout 切片並以線上計數實作。 |
| 中 | g1_bench.py:159-178、573-613：generate_g1_splits 一次建立並回傳 Test 的 u、y_next；oracle 函式會讀三個 split。g1_pilot.py 預設直接產生 Train，因此沒有證據顯示目前 Pilot 已讀 Test；但通用 helper 沒有 freeze 邊界，不能作為 sealed Test 保證。 | **流程上可能。** 若 Train-only pilot 使用全 split helper，Test 會在設定凍結前暴露。 | Test 建立／oracle 評估限於所有設定、source 與 hash 鎖定後；任何較早存取均使該次 confirmatory run 失效。 |
| 高 | g1_pilot.py:111-127、238-272：Train-only guard 只拒絕 VAL_SEEDS、TEST_SEEDS 或明確 val/test 標記，沒有要求 seed 必須屬於 TRAIN_SEEDS；run_g1_pilot 也接受任意外部 train_sequences。 | **可能。** 未登錄序列可改變 rho 的 MC 排序與 gate 結果，讓「Train-only」超出預註冊資料。 | 嚴格要求 seed 集合等於凍結 TRAIN_SEEDS，且 split metadata 明確為 train；拒絕額外、缺漏或重複 sequence。 |
| 中 | g1_bench.py:363-366、test_g1_bench.py:156-194：alpha_at_boundary 對 0 與 100 都回傳 true，測試也把 α=0 定義為 boundary。這和修訂後只將上端點列為 grid no-go 相衝。 | **是，會造成假 no-go。** | 報表拆成 upper-bound flag 與 alpha=0 diagnostic；不要沿用舊 bool 作 no-go。 |
| 中 | g1_pilot.py:240-242、365-366、451-478：candidate_rhos 可由 API 及 CLI 覆寫，但輸出的 spec_rules_hash 是依固定預設網格預先計算，沒有納入實際傳入網格。雖然輸出另列 candidate_rhos，hash 本身不能證明規則未漂移。 | **治理上可能改變結論。** 不同 rho 搜尋可沿用相同規格 hash，破壞 hash 作為預註冊版本鎖的用途。 | 確認性 run 固定拒絕不同網格，或以有效參數產生不同版本與 hash，並在產生任何 performance 結果前凍結。 |
| 中 | g1_reservoir.py:289-323：run_timed_pilot 收到 graph_path，但呼叫 load_flywire_graph(use_cache=True)，未使用該路徑；同時直接用所有 sensory_idx。 | **不直接改變 NMSE，但會使 pilot 無法證明指定圖和輸入映射的實際成本。** | pilot 必須接受並回報實際 graph cache 路徑與 hash，且採用 G1 的 R1-6 索引。 |
| 中 | g1_reservoir.py:202、235-258；research/tests/test_g1_reservoir.py:228-276：Torch 固定 float32；Scipy dtype 可設 float64。對 G1 主設定 float32 可比，但 CPU/Torch 對照僅對實際選到的 device 有效。Torch class 在 CUDA 不可用時把非 CPU 請求靜默轉成 CPU；測試也會在無 CUDA 時選 CPU。 | **對程式正確性不必然改變；對「已完成 GPU 驗證」的主張會改變。** | 記錄並斷言實際 device、dtype；無 CUDA 時明確標為 CPU-only，不能算 GPU gate 通過。 |

## G1 放行判斷

公式、單步 state 更新、序列 group CV、fold-local 標準化與 unpenalized intercept 的靜態檢視結果為符合。確認性 G1 仍需先解決 spectral radius fail-closed、α=0 固定 SVD、R1-6 index、MC held-out、有限值／washout gate、streaming 全腦 saturation 與 Test access boundary；現有 timed pilot 不足以支撐 FlyWire 完整工作量估算。

α 邊界裁定：只把 α=10² 的上端飽和比例列為網格 no-go；α=0 是有效候選，不因 endpoint 身分失敗。它必須使用固定的數值秩 SVD 偽逆並回報有效秩與條件數；目前程式尚未符合，修訂規格不代表程式已修復。
