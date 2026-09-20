# G1 v2 規格草案：先驗證直接能力，再做固定拓樸對照

狀態：使用者已確認研究方向；本文件是待實作驗收與封存的規格，不代表程式已符合或任何 gate 已通過。v2 取代 `G1_SPEC_draft.md`；歷史結果及規格保留。不得讀取市場 Test、label 或 future_return；所有新任務均為合成序列。不得把本研究解讀為交易能力證據。

## 1. 修訂紀錄與數值來源

2026-09-21：

- 撤回 `q=1.5u[t]u[t−9]`、`r`／`r+βq` 與 `A_q` 讀出 gate。q 含常數及線性項，原基準又可能因平均值不同產生改善，不能辨識純乘積能力；不實作該舊 gate。
- 輸入改為原始 `u[t]`，不減 0.25。置中輸入、零初態、無 bias 的 tanh states 對整條輸入反號呈奇對稱；仿射讀出無法恢復對稱輸入分布下的純偶函數乘積。原始輸入打破此限制，但不保證 FlyWire 成功。
- 固定 rho=0.95、leak=0.5；撤回 rho 搜尋、MC 選點及 nested rho CV。保留僅用 fit/Train 的 α CV，不搜尋任何 dynamics。
- 撤回三 family 選擇與 61 圖；唯一主要對照為 20 張 `scramble_mixed`，加 1 張 real，共 21 圖。取消 weight shuffle、random endpoints、Mackey–Glass、Lorenz；結論因此只針對指定重接 null。
- 能力 gate 改為直接預測 `v[t−9]` 與 `v[t]v[t−9]`，其中 `v=u−0.25` 僅用來定義 target，不能重新置中 reservoir input。
- primary CI 改為 graph × 完整獨立 sequence 的交叉重抽；撤回序列內 500-step block 重抽。完整序列保留時間相依，避免同一分析重複計入序列與 block 變異。
- power/FPR 改為明確零差異及 5% 母體效果的成對 residual 校準，匹配 1 對 20；Monte Carlo 區間而非點估計決定放行。能力 gate 與統計推論校準分開。

數值標記：**[文]** 為文獻定義／依據；**[測]** 為本專案已有結果或使用者回報；**[設]** 為已確認或本草案凍結的主觀／工程選擇，並非實測成功率。公式推得的數值另標 **[推]**。本文件所有樣本數、seed 規則、容差、分位數、預算與門檻，若未另標，一律為 [設]；日期、版本及檔案行號為識別資訊。

使用者於本次任務回報 [測，尚未於本輪重跑／取得完整 artifact]：置中輸入、零初態下 `max|x[−v]+x[v]|=0.0`；300 節點 ESN、20 條序列（10 fit／10 check，每條 2,000 有效步＋500 washout），held-out R² 如下。這些是方向修正的探索性證據，不作 v2 階段 A 或 B 的自動通過憑證；原 seed、graph、程式 hash、預測與 target 應保留以供重現。

| 輸入 | 延遲 v[t−9] | 乘積 v[t]v[t−9] |
|---|---:|---:|
| 置中 v | 0.49 | −0.019 |
| 原始 u | 0.478 | 0.333 |

對稱論點及 input shift 的一般依據見 [Herteux & Räth, Breaking Symmetries of the Reservoir Equations in Echo State Networks](https://arxiv.org/abs/2010.07103) [文]；本文不以該文替本計劃 rho、R² 或 power 門檻背書。

## 2. 問題與結論邊界

在指定 FlyWire v783 specimen、固定 R1-6 輸入、固定 DN 線性讀出及固定 dynamics 下，real 在 NARMA10 的 NMSE 是否比 degree-preserved `scramble_mixed` 分布中位數低至少 5% [設]，且差異的雙側 95% CI 下界 >0 [設]？

- 能力 gate 失敗：結論為「此固定輸入、動力學與 DN 線性讀出未達預定能力下限；本版不進入拓樸確認性實驗」。不稱拓樸無效，也不籠統宣稱對比在數學上不可識別。
- power／預算 no-go：結論為「在固定資源與設計下無足夠證據進行預定效果的確認」，不是陰性生物學結論。
- Test 未達門檻：只支持「未證實此固定系統相對指定重接 null 至少 5% 的改善」。未拒絕零效果不等於證明等價。
- Test 通過：只支持指定 specimen、任務、輸入／讀出與 null 下的優勢；不代表其他個體、真正視覺運算、純拓樸因果效果或市場能力。

## 3. 資料、索引與封存

### 3.1 NARMA10 及 state/target 對齊

`u[t] ~ iid Uniform[0,0.5]`；

`y[t+1] = 0.3y[t] + 0.05y[t]Σ(k=0…9)y[t−k] + 1.5u[t]u[t−9] + 0.1`。

係數、10 階及輸入區間採既有 NARMA10 benchmark [文，見 [Liang et al.](https://www.nature.com/articles/s41467-022-29260-1)]。本版初始化 [設]：獨立 RNG 先產生 `u[−9…−1]`，令 `y[−9…0]=0`、`x[0]=0`；自 t=0 起每步接收 u[t]、更新一次得到 x[t+1]，預測 y[t+1]。warmup 為 t=0…499，有效點自 t=500 起。負索引須明確映射，不能用陣列負索引意外讀到尾端；不得跨 sequence 接 lag。

### 3.2 分割與數量 [設]

| 資料集 | 獨立序列數 | 每條有效步／washout | 用途與可讀時間 |
|---|---:|---:|---|
| Diagnostic-fit | 10 | 2,000／500 | B 的兩個能力 head：5-fold sequence CV 選 α，再完整 fit |
| Diagnostic-check | 10 | 2,000／500 | B 一次性能力 gate；不得回饋 α 或其他設定 |
| Main-Train | 10 | 5,000／500 | C：各圖 NARMA head 的 5-fold sequence CV 及 refit |
| Main-Val | 3 | 2,000／500 | C：只檢查有限值、shape、state integrity；不計算預測表現作選擇 |
| Calibration | 10 | 5,000／500 | C：Main-Train 模型對新序列的 residual tensor；只供預定 power/FPR 校準 |
| Main-Test | 10 | 5,000／500 | D：所有 gates／hash 凍結後才建立 u、y 與評分 |

Diagnostic-fit/check 各 20,000 有效點、Main-Train/Test/Calibration 各 50,000、Main-Val 6,000 [推]。序列是獨立單位，時間點不是額外獨立 n。所有圖使用相同 split 的相同完整序列，以保留配對。

A 使用獨立 `fixture` namespace 的小圖與 synthetic streams，禁止使用任何上述正式 seeds。manifest 在 A 凍結各 namespace 的 seed 清單、RNG 算法與版本、sequence generator hash、scramble seeds、bootstrap/MC seeds；各清單互斥且不沿用已看過表現的舊 streams。seed 可預先登錄，但 Main-Test 的 u/y 不得提前建立、載入或傳遞；提供按 split 建立的 API，禁止一次生成所有 split。未知 seed／split fail closed。Calibration 不能用來挑模型或換門檻。

## 4. 固定輸入、動力學與讀出

### 4.1 Input 與 recurrence

- R1-6 由同一份 metadata 的 sensory index 與 photoreceptor type 逐項對齊，限定 `photoreceptor_type == R1-6` 且 retina u/v 有限。非 R1-6（含 R7、R8）外部電流為零。
- 對每個合格 R1-6 注入 `I_i[t]=u[t]`，input gain=1、noise_std=0、額外 recurrent bias=0 [設]。retina u/v 是位置欄位，與時間訊號 u[t] 不同。這是 scalar photoreceptor-group injection，不聲稱生物視覺刺激模型。
- `x[t+1]=0.5x[t]+0.5tanh(W_eff x[t]+I[t])`。leak=0.5 為歷史 Phase 0 使用設定 [測背景]、於 v2 固定採用 [設]；rho=0.95、recurrence gain=1 固定 [設]，沒有候選 grid 或 MC selector。
- 每張圖各自以自身 spectral radius 縮放至 0.95；保留 unscaled/scaled matrix hashes。需收斂 eigensolver、特徵對殘差與縮放後獨立驗證；禁止不收斂時使用 Rayleigh quotient fallback。相對特徵對殘差 ≤10⁻⁶，兩次獨立初始化的最大模特徵值估計相對差 ≤10⁻³，縮放後 rho 相對誤差 ≤10⁻³ [設]。小圖另對照 dense eigenspectrum；大圖驗證屬數值證據，不宣稱殘差可證明找到所有特徵值。
- 每條 sequence 零初態、跨時間保留 state；chunk 邊界不可重設。只保存 DN trajectories，非 sensory 的健康計數逐步累計，禁止配置完整時間×全腦張量。

### 4.2 Dynamics gates [設]

所有使用的 state、輸入、目標與預測必須有限；空資料、NaN、Inf 立即阻擋。只計 washout 後 non-sensory `|x|>0.9` 的比例，須 <5%。non-sensory 集合為固定 R1-6 集合之補集。

forgetting 使用零初態對 iid Uniform[−0.1,0.1] 初態，每 sequence 4 個配對；10 條 fit/Train streams 共 40 對，至少 38 對在 t=500 的 `RMS(x_a−x_b)/max(RMS(x_a),RMS(x_b),10⁻⁶)<10⁻³`。B 在 real Diagnostic-fit 評估；C 在全部 21 圖 Main-Train 評估；任一圖不合格即 no-go，不刪除不合格 scramble。Main-Val、Calibration、Main-Test 仍檢查有限值與飽和。有效執行的動力學失敗是 gate fail；證明實作違反方程才是 run invalid。

### 4.3 DN 與 Ridge

固定 real metadata 的 `buy_idx∪sell_idx` 所有 1,291 DN [測：既有集合，執行前須核對 hash／數量]；所有圖保持相同 node identity 和順序。不採用 Buy/Sell 語義，不按活動或表現篩選，不增加 x²、全腦投影、input 或 target 作 feature。數量／metadata 不符屬 provenance 阻擋，不能補選節點。

每個 scalar target 各自 fit Ridge，含不受懲罰 intercept。X 與 y 的 mean/std 僅由該 fold 的 fit 資料估計；std<10⁻⁸ 的 X 欄固定移除 [設]，以 fit mean 處理 check/Test；y std 非正或非有限阻擋。反標準化後評分。

objective 為 `(1/n)||Y−XB||²+α||B||²`；固定 α 網格 `{0,10⁻⁸,10⁻⁷,…,10²}` [設]。10 條 fit/Train 依 manifest 順序分 5 folds，每 fold 留 2 條完整序列；每個圖／target 選 pooled OOF NMSE 最低 α，數值完全同分選較大 α。不按 check/Val/Calibration/Test 選 α；選完用全部 fit/Train refit。

α=0 合法；採 float64 薄 SVD 與 `s_i > ε64·max(n,p)·s_max` 的固定 Moore–Penrose cutoff [設，標準數值秩慣例]，不得用 Gram 絕對 cutoff 取代。報告 active p、有效秩、丟棄方向及保留子空間條件數。非零 α 同樣使用穩定 solver，須符合含 1/n 的 objective。

B 的兩個能力 head 任一選 α=10² 即 no-go；C 的 21 個 NARMA heads 若 >10% 選 α=10²，即至少 3 個，no-go [設／推]。上界碰撞不授權擴網格。不得把 α=0 當碰撞失敗。

## 5. 圖、對照與主要評分

只有 real FlyWire v783 一張與 20 個預先凍結 seed 的 `scramble_mixed`，共 21 圖 [設]。在 spectral scaling 前驗證節點、逐節點 in/out degree、source out-strength、權重多重集不變，無自環／重複邊，edge overlap≤5% [設，混合診斷而非均勻抽樣的證明]；索引與 metadata/hash 完整。縮放後 out-strength 絕對值允許依各圖尺度改变，須記錄尺度，不混淆兩個階段的不變量。swap 計數與停止規則須在 A manifest 凍結；mixing 失敗不換 seed。

對固定 graph g，全部有效 sequence/time 點等權 pooled：

`NMSE_g = mean((yhat_g−y)²) / mean((y−mean(y))²)`，variance 使用 ddof=0 [設]。分母在各圖共用同一批 y；零／非有限分母阻擋。`C=median(NMSE_scramble_1,…,NMSE_scramble_20)`，偶數 median 取中間兩值算術平均 [設]；`Δ=(C−NMSE_real)/C`。

Main-Test 主要成功須 `Δ≥0.05` 且雙側 95% CI 下界 >0，並所有 gates 通過 [設]。只有一個主要 contrast，不從其他 family 或 task 救援。

CI 固定 2,000 次 percentile bootstrap [設]：每次對 10 條完整 sequence 有放回抽 10 條，該抽樣索引同時用於 real 與所有 controls；獨立對 20 個 control graph identities 有放回抽 20 個。每次重算 pooled variance、各圖 NMSE、control median 及 Δ，取 2.5%／97.5% 分位數（線性插值）[設]。real 始終為同一 specimen，不重抽成假想 real 個體。所有圖共用 sequence 抽樣，不在每張圖內各抽不同 sequences；序列內不再抽 block。區間僅描述固定模型在新同分布序列及指定 control 分布的變異。

## 6. 直接能力診斷與負控制

兩個 target：`d1[t]=v[t−9]`、`d2[t]=v[t]v[t−9]`，`v[t]=u[t]−0.25`；lag 9 對應 NARMA10 input term [文／推]。predictor 始終是接收原始 u 後的 DN `x[t+1]`。使用 §3 Diagnostic-fit/check；不把 d1/d2 或歷史 u 添入 DN predictor。

check 上 `R²=1−SSE/SST`，SST 為同一 target 全部 check 點相對其 pooled mean 的平方和；不截斷負值。兩個 target 各自都須 R²≥0.10 且完整獨立 sequence bootstrap（2,000 次、10 抽 10）的雙側 95% CI 下界 >0 [設]。check mean 只作分母，不用來訓練 intercept。兩項是必須同時通過的能力條件，不挑其中一項、不把二者當兩次擇優發現。

oracle positive controls：用明確 lag access 直接回傳 d1/d2，R²≥1−10⁻⁶；NARMA oracle 用真 recurrence RHS，NMSE<10⁻⁶ [設]。這些只驗證管線，不能充當 DN 能力。

負控制：fit 和 check 各自在固定 seed 順序循環移位一條，把第 s 條 DN states 配給第 s+1 條完整 target；末條回接首條，兩個 split 不互相配對。負控制重新執行相同 fit-only α CV／refit，只在原始單位比較同一錯配 target；常數基準使用錯配 fit target 的平均。禁止只打亂 time points，禁止挑多個 permutation 中最弱者。兩項各計算相對常數的 `G=(MSE_constant−MSE_model)/MSE_constant` 與 sequence-bootstrap 雙側 95% CI；任一 G 的下界 >0 則負控制 fail/no-go [設，保守工程判準，仍可能有隨機誤拒]。負控制不要求每個點估計必為負。小圖 fixture 的 loss-of-signal 測試另須能拒絕常數 states 的能力通過。

## 7. Power/FPR：明確零差異與 Monte Carlo 放行

### 7.1 Calibration 來源及範圍

C 使用全部 21 圖、Main-Train 選 α 並 refit 的固定 NARMA heads，在獨立 Calibration 的 10 條完整 5,000-step 序列取得 `e[g,s,t]=yhat−y`；不使用 in-sample residual，不使用 Main-Test，不再選 α。儲存相同 sequence 順序的 target／residual tensor 或其完整 sequence SSE、sum(y)、sum(y²)、count；只存充分統計時仍保留可稽核的來源 hash。

本節校準的是凍結模型的統計偵測程序，不是 DN 表示能力；不得把 q 直接加進預測當能力證據。此 empirical residual model 的 Monte Carlo 區間是條件於有限 Calibration templates，不能聲稱涵蓋所有未知市場、connectome 或 error distributions。

### 7.2 零效果模型 [設，明確定義的經驗校準母體]

令 `m_g=mean_s,t(e[g,s,t]²)`，`M=median(m_1,…,m_20)`。要求 m_real 與 M 嚴格正且所有值有限。固定：

`e_null[real,s,t] = e[real,s,t]·sqrt(M/m_real)`；controls 保留原 e。

將 10 個完整 sequence templates 等機率視為校準母體，20 個 control graph templates 等機率視為 control 母體。在這個母體上，pseudo-real 的期望 MSE 恰為 M，control 母體的 graph-level MSE 中位數恰為 M，故母體 Δ=0 [推]。**零差異在母體，不強迫每一個抽樣 cohort 的觀測 Δ 等於零。** 原 real/control 未縮放 residual 本來可能有差異，不能直接稱 null；單純將 c 設零或反轉效果差符號也不成立。

每個外層 replicate 有放回抽 10 個完整 sequence templates，所有圖共用同一索引；再有放回抽 20 個 control templates，保留單一固定 pseudo-real template identity。擷取每条完整 residual／target sequence，不逐圖獨立生成 Gaussian residual，不打亂時間。固定的正比例縮放保留序列內殘差形狀及 real/control 共享難度的相關结构；control 間平均誤差差異仍保留。每個 cohort 恰為 1 pseudo-real 對 20 controls，禁止用 10 對 10 或 1 對 10 代替。

每個 cohort 使用 §5 完全相同的 estimator、2,000 次交叉 bootstrap 及 CI。內層 bootstrap 只重抽該 cohort，不重新估計 null 縮放係數；縮放值由 Calibration 一次鎖定。

### 7.3 5% alternative 與定義

在相同母體及相同配對抽樣下，僅將 `e_alt[real]=sqrt(0.95)·e_null[real]` [推，自 Δ=0.05]；controls 不變。母體 real MSE 為 0.95M，故母體 Δ=0.05。每個 replicate 不依其觀察 median 再縮放，避免移除抽樣不確定性。

FPR event 定義為 null cohort 的主要雙側 95% CI 下界 >0；power event 為 5% alternative cohort 的同一 CI 下界 >0。另報完整成功 event「下界 >0 且點估計 Δ≥0.05」的通過率與區間，**不能將其稱為 80% power**；真值恰在效果門檻時，兩者通過率不同。能力診斷不取代本節。

### 7.4 Monte Carlo 明確放行規則

固定 N=1,000 個獨立外層 replicates [設]，每個有獨立 RNG stream；同一 replicate 的 null/alternative 可共用抽樣索引以減少比較噪音。N 不得看到結果後增加。令 k0 為 FPR events 數、k1 為 power events 數。

使用單側 exact Clopper–Pearson bounds，尾機率 a=0.025 [設]：

- `U_FPR = BetaQuantile(1−a; k0+1,N−k0)`；k0=N 時 U=1。
- `L_power = BetaQuantile(a; k1,N−k1+1)`；k1=0 時 L=0。

此為各自 97.5% 單側 bound，對兩個 Monte Carlo 聲明用 union bound 提供至少 95% 同時覆蓋 [推]。**只有 `L_power≥0.80` 且 `U_FPR≤0.05` 才 go** [設]；點估計達標但 bound 不達仍 no-go/inconclusive，不補 replicates、不放寬容差。CI endpoint、Beta quantile 與事件判斷均需獨立 fixture 驗證。

Bootstrap 次數不當作新增獨立序列或生物個體。10 條 Calibration templates 帶來的分布估計誤差不包含在上述二項區間；報告必須明示此限制，不能宣稱真實世界 power 已精確證明。校準模型、seed、bounds 與實際 cohort 大小不符合規格時屬 run invalid。

## 8. A–D 階段、預算與 go/no-go

全部時限 [設]：4 工作天／8 GPU 小時為整個 v2 A–D 總上限，不是各阶段可重置的額度；工作天按 8 小時人工投入計，共 32 工時。GPU 小時計單機 GPU job 實際占用 wall time，包含測試、warmup、失敗與合法重跑，不是只計 kernel time；CPU/Ridge/hash/bootstrap 成本另記 wall time 與峰值 RAM。下列分配是硬上限，不因其他階段省時就自行挪用或追加。

| 階段 | 工作與必要交付 | 人工／GPU 上限 | 放行／失敗 |
|---|---|---|---|
| A：小圖與管線驗收 | §10 全項；CPU reference、直接診斷、null/MC fixtures、封存 manifest、runbook 對齊 | 0.5 工作天／0 GPU h | 清單全過才進 B；未完成不啟動真圖，按 invalid 或 feasibility no-go 記錄 |
| B：real 能力與計時 | 開頭以小圖核對 Windows CUDA/CPU；單張真圖 Diagnostic-fit/check、負控制、健康 gates；完整模擬＋Ridge 計時 | 1 工作天／1 GPU h | 兩能力條件、負控制、α、健康、CPU/GPU 一致性全過；任一有效 fail 即結案 |
| C：21 圖與推論校準 | 全部圖 integrity／rho／Train 健康；Main-Train α CV/refit；Main-Val integrity；Calibration residual；固定 MC power/FPR；封存正式模型與輸出 schema | 1 工作天／4 GPU h | 圖與 α gates 全過、MC bounds 合格、D 完整外推不超其剩餘上限；否則 no-go，不建立 Test |
| D：一次確認性 Test | 同機完成 21 圖 × 10 Test streams；oracle/integrity、主要 NMSE/Δ/CI、失敗亦完整封存報告 | 1.5 工作天／3 GPU h | 主要門檻與所有 gates 全過才 PASS；其餘 FAIL/inconclusive 或有證據的 invalid |

合計 4 工作天／8 GPU h [推]。在 B、C 的入口及出口，以已量測 throughput、Ridge、I/O、graph generation、bootstrap 工作量估算剩餘成本；不可只由 static replay 或純 SpMV 外推。預估已超任一階段上限就停止，不先消耗到超額。1,000×2,000 的統計校準優先對 sequence 充分統計向量化，不重跑 reservoir；仍須計入 CPU 工時。

Mac 僅跑 A 的小圖／CPU fixtures；B–D 在同一台 Windows RTX 5070 12GB [測：使用者設備] 完成，記錄 OS、driver、CUDA、PyTorch 及精度，不混合不同環境的主結果。A 的 CPU 對照不是跨機正式分數。GPU/CPU 小圖 state 最大絕對誤差≤10⁻⁴ [設]；B 首次上 GPU 才做此實際硬體驗收，不聲稱 A 的 CPU Torch 等於 CUDA 驗收。

## 9. Valid fail 與 run invalid

沿用 `REVIEW_stop_rule.md` Q5 的分類。有效執行的能力／健康／α／power／FPR fail、成本上限、near-miss 或 control 勝出，均結束本版；不得換 input、readout、rho、gain、alpha grid、seed、null、效果門檻或增加資料救援。

可修復例外僅限下列 run-invalidating 事件；保留原 artifacts、spec/code/graph/config/seed hashes、環境、首次發現時間、獨立證據、哪些結果已被看過、變更檔案／新版 hash 和剩餘預算：

1. 結果盲化且由獨立 fixture/reference 證明的程式、索引、solver 或資料路徑缺陷。已看過性能的舊 Test 只留探索性，新確認性 run 不重用它。
2. 尚未讀取結果的外部 OS/GPU job／電源／環境中斷；同設定可重跑，換機或數值環境則另版完整重跑，不拼接。OOM 僅在證明為外部暫時資源占用時屬中斷；可重現 OOM 是容量／實作限制，只有證明是 bug 才循第 1 項處理。
3. graph/data 版本、hash、split boundary 或 Test 封存失效；立即 invalid，不能以重新產生 cache 靜默覆寫。Test 已看過時使用全新未觸碰 Test seeds。
4. 任何性能指標讀取前發現規格矛盾，可先澄清另版凍結；已看過 Train MC/gate/性能則不能追溯套用，另版需新 Train 及 Test seeds。

每次例外需明列不是依效果大小調整的證據；重跑仍計入本版總資源，不自動授予新預算。A 中預先規劃的 fixture 修正是實作驗收，不能被宣稱為科學 gate 通過。v2 是使用者明確核准的設計修訂，未將舊 run 改標通過；不構成日後任意換版本的通行證。

## 10. 階段 A 驗收清單（24 項，供 agy 逐項實作／測試）

每項須記錄 fixture、可執行命令、預期／實際結果、source hash 與 PASS/FAIL；本清單是驗收要求，未宣稱已執行。A 禁止市場資料及正式 Test。數值容差均為 [設]。

- [ ] A01：v2 設定入口固定 raw u、rho=0.95、leak=0.5、noise=0、DN-only、唯一 scramble_mixed；舊 q／rho grid／family selection 覆寫須明確拒絕。
- [ ] A02：獨立 NARMA reference 逐步核對 generator（float64 最大絕對差≤10⁻¹²），驗證負歷史、washout 與 y[t+1] 索引；oracle NMSE<10⁻⁶。
- [ ] A03：pulse／手算小圖核對每時刻恰更新一次、x[t+1] 對 target、sequence 間 reset、chunk 間保留 state。
- [ ] A04：小圖確認 production sensory current 恰為 u[t]；R1-6/type/retina index 對齊，R7/R8/其他電流為零；空／錯位 metadata 必須失敗。
- [ ] A05：僅 fixture 的置中 ±v 配對驗證奇對稱（float64 最大殘差≤10⁻¹²）；production raw-u 路徑不套用此置中。不可用上次 0.0 數字替代測試。
- [ ] A06：direct lag／product teacher 的兩 target R²≥1−10⁻⁶；驗證 target 不會出現在 DN predictor；pure product 與 raw q 必須區別。
- [ ] A07：兩個 diagnostic split 數量／長度／lag/washout 邊界符合 §3；禁止跨序列 lag，check loader 不接受 fit seeds。
- [ ] A08：Ridge 的 X/y scaling、constant-column 處理、intercept、反標準化僅使用 fold fit；用改動 check 分布的 fixture 證明不影響 fit 係數。
- [ ] A09：Ridge objective 1/n 與獨立小矩陣解吻合（float64 預測誤差≤10⁻⁸）；rank-deficient α=0、SVD cutoff／rank metadata、上端點規則均符合 §4。
- [ ] A10：5-fold sequence CV 不混時間點，選 α 後完整 refit；診斷兩 target 可各有 α，但不能靠 check 換 α；Main-Val/Calibration/Test 均不能參與選擇。
- [ ] A11：pooled R²/NMSE（ddof=0）、偶數 median、Δ 以手算 fixture 核對；負 R² 保留，零分母／NaN／空資料 fail closed。
- [ ] A12：診斷 CI 只重抽完整 check sequences；常數 states 不可通過能力 gate；兩 target 的 AND 判定與邊界 ≥0.10／CI >0 可由固定數值 fixture 驗證。
- [ ] A13：負控制 split 內循環移位一條、重新 fit、同 target 常數基準與 G/CI 判定符合 §6；不偷用 check mean fit intercept、不打亂單點。
- [ ] A14：小圖 unscaled scramble 的 degree、source strength、weights、loop/duplicate/hash invariants；固定 seed 與 mixing 停止規則寫入 manifest，無效圖不替補。
- [ ] A15：小圖 spectral radius 對 dense eigenspectrum，殘差／target tolerance／不收斂 fail closed；確認各圖獨立縮放、保留前後 hashes。
- [ ] A16：飽和只計 washout 後 non-sensory、串流與小圖 full-state reference 相同；NaN/Inf/空資料不被當低飽和；forgetting 的 40 對／38 對規則可核對。
- [ ] A17：DN-only 與小圖 full-state slice 相符，chunk 輸出一致；記憶體不配置完整 T×N 歷史。CPU reference 與 Torch CPU 小圖誤差≤10⁻⁴；CUDA 驗收明列留待 B、不能先標通過。
- [ ] A18：替換舊 power 路徑；對原 `g1_power.py:589` 未定義 `gamma_star` 問題做端到端返回值測試，不只改名稱。v2 回傳 null/alt scale、seed、events、bounds，不把外加 q prediction 當 gate。
- [ ] A19：手造不等 MSE residual tensor，證明 §7 null 母體 real MSE=M、alt=0.95M（相對誤差≤10⁻¹²）；未縮放資料不能誤標 null；保留 control 平均誤差差異。
- [ ] A20：每個 MC cohort 是 1 fixed pseudo-real＋20 抽樣 controls＋10 共用完整 sequences；加入 sequence-ID／共同難度 fixture 證明配對未被破壞，且無獨立 Gaussian／任意 sign-flip null。
- [ ] A21：主要 graph×sequence 交叉 bootstrap 與診斷 sequence bootstrap 的實作分開；重抽時重新計算 median/variance/Δ、real identity 不變，CI 分位數與固定 seed 可重現。
- [ ] A22：用固定 event counts 驗證 exact binomial bounds（含 k=0/N）、N=1,000、a=0.025、power 下界≥0.80／FPR 上界≤0.05；完整成功率和 rejection power 分開，禁止 adaptive MC 追加。
- [ ] A23：manifest 完整列 seed namespaces／graph/simulator/readout/scorer hashes／版本；A–C 建立 Test 必須被拒絕；故意 hash mismatch、未知 seed、缺檔須 fail closed。
- [ ] A24：更新執行端的 v2 runbook／report schema，使其允許本規格 fit-only target 診斷並移除舊 gate；以小型端到端 fixture 驗證 A→B→C→D 狀態機、valid fail/invalid、預算累計與 Test freeze 邊界。本項不要求 A 執行真圖或正式 1,000-cohort 校準。

## 11. 封存及未決證據

實作交付應含本 spec SHA、source snapshot SHA（工作樹有未提交變更時不得只記 HEAD）、環境、全部資料／圖／模型／split hashes、每項 gate、資源 ledger、失敗亦保留的輸出與禁止 Test 提前建立的稽核。文件凍結不要求本次 git commit。

待取得的證據：使用者 ESN 對稱／能力實驗 artifacts；真 FlyWire raw-u 的飽和、forgetting、直接能力與成本；實際 graph/dynamics metadata 是否符合 manifest；21 圖 inference calibration 的 bounds。未取得前不填入成功旗標、不拿主觀成功機率替代 gate。階段 A 的每項驗收與 B 的實際 CUDA 核對通過前，不能宣稱 v2 runner 已就緒。
