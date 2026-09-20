# G1 預註冊規格草案：stateful FlyWire Reservoir 的合成時序基準

**狀態：** 草案；任何 G1 主分析前須凍結本文件、程式 commit、圖檔 hash、序列 seed 清單與完整輸出目錄。若改任何 gate，另開版本並視為新研究，不沿用本規格的 confirmatory 名稱。

## 1. 研究問題與可支持的結論

主要問題：在單一固定 FlyWire v783 connectome、固定 input map、固定 DN readout 與預註冊 dynamics 下，真實連接圖在 NARMA10 上是否比事先指定的隨機／拓樸／權重 controls 有至少 5% 的 NMSE 改善？

陽性結果最多支持「這個指定 FlyWire 圖作為固定 stateful reservoir，在此合成 benchmark 下勝過指定 controls」。不支持「果蠅在生物上實際執行 NARMA10」、其他 FlyWire 個體具相同表現，或任何市場預測／交易結論。G1 不讀取、不需要任何市場 split 或市場 label。

## 2. 任務及序列

### 主任務：NARMA10

採常見離散 NARMA10 版本：獨立輸入 u[t] ~ Uniform[0, 0.5]；令初始歷史 y[-9…0]=0，依序產生

`y[t+1] = 0.3 y[t] + 0.05 y[t] Σ(k=0…9)y[t−k] + 1.5 u[t]u[t−9] + 0.1`

reservoir 在 t 接收 u[t]，讀出預測 y[t+1]。序列 generator、索引對齊與 recurrence 以獨立參考實作逐值核對；第一個有效 target index 及初始 10 步規則寫入 metadata。公式與 U[0,0.5] 來源為 reservoir computing 的常用 NARMA10 benchmark；公式版本必須固定，不得在看結果後換 indexing 或係數。[benchmark 原文示例](https://www.nature.com/articles/s41467-022-29260-1)

### Split、樣本量與 washout

每條序列連續產生；同一條序列的每一時間步只更新 reservoir 一次。Train、Val、Test 使用彼此不重疊且預先凍結的獨立 RNG seeds：

| Split | 序列數 | 每條有效時間步 | 每條前置 washout | 用途 |
|---|---:|---:|---:|---|
| Train | 10 | 5,000 | 500 | 5-fold sequence-group CV 選 Ridge α；Train-only 選共同工作點 rho。 |
| Val | 3 | 2,000 | 500 | 僅作有限值、輸入/輸出 shape 與 simulator integrity 檢查；不看 performance 選模型、alpha 或 gate。 |
| Test | 10 | 5,000 | 500 | 唯一主要分數；完成所有 graph、rho、alpha、readout 和 code freeze 後才解封合成 Test。 |

Train 共 50,000 個有效輸出點、Val 6,000、Test 50,000；washout 的 500 步不計入 NMSE。每個 split 的各序列獨立且連續，split 之間以獨立 seed 和新 state 起始；每條序列內不得逐樣本清零。500 步 washout、序列數及長度為本專案規劃值（主觀），不是由 FlyWire 文獻推得。若 5% MDE power gate 不足，不得縮短 washout 或選擇性增加 Test；應報告 G1 inconclusive／no-go。

### 次要任務：Mackey–Glass、Lorenz

僅在 NARMA10 的主分析和 integrity gates 全部完成後執行；不改變 G1 通過判定。Mackey–Glass 固定使用 `dx/dt = 0.2x(t−17)/(1+x(t−17)^10) − 0.1x(t)`、dt=0.1；Lorenz 固定使用 σ=10、ρ=28、β=8/3、dt=0.01 的標準方程。初值、積分器、transient、forecast horizon、train/val/test seeds 與輸出 hash 必須在執行前登錄。若四工作天／8 GPU 小時上限已用完，次要任務取消且不影響主結果。

## 3. 固定 FlyWire 輸入與讀出映射

### Input

使用 `research/pipeline/flywire_graph.py` 已依 annotations 選出的兩眼 R1-6 photoreceptor group，限 `photoreceptor_type == "R1-6"` 且既有 `u,v` retina coordinates 有限的神經元。每一步對每個 R1-6 注入同一個全視野均勻 luminance current：

`I_i[t] = u[t] − 0.25`（R1-6）；R7、R8 與其他 neuron 的外部電流為 0。

0.25 是 U[0,0.5] 的固定數學均值；input current 不再乘可調 input gain，沒有隨機投影、圖像 renderer 或市場 train mean。retinotopic 座標只用來鎖定／稽核既有 R1-6 群；因 NARMA 是 scalar temporal task，不對 retinal 座標加未有生物依據的空間 pattern。這是「photoreceptor-group injection」，不是視覺 stimulus 的生物模型。

noise_std=0，作為 deterministic 主任務；不在主分析加噪聲。任何 noise robustness 必須另版預註冊且使用對每個 graph family 相同的絕對/相對尺度與 RNG policy。此規則移除先前自訂 5% sensory noise 的自由參數，不把它誤稱為 connectome 性質。

### Reservoir recurrence 與 stateful 定義

固定方程為 `x[t+1] = (1−leak)x[t] + leak·tanh(W_eff x[t] + I[t])`，其中 `W_eff` 是按本文件共同 rho 工作點規一後的圖；`leak=0.5`（本專案 Phase 0 measured setting），recurrence gain 固定 1.0，避免同時設定重複的 gain 與 rho。

每一完整序列由初始 state 開始，逐時刻做**一次** sparse matrix × state update，並將 `x[t+1]` 保留到下一時間步；經 500 步 washout 後，逐步記錄 DN readout state。不得把同一 NARMA input 重播 32 次、每個 time window 重設 state、或將 32 步平均成一個 sample。現有 `TorchReservoir.simulate_batch` 在一次呼叫內跨 step 保留 state，但每個 batch 開頭設為零；G1 每條連續序列須在單一 stateful call 內完成。它與市場 Phase 5/6 的每張圖像獨立初始化、同一 sensory current 靜態重播 32 步之差異，必須在輸出 schema 中明確標記。

### Readout

唯一 readout 為現有 real graph metadata 的所有 1,291 個 left/right descending neurons（DN；使用 `buy_idx∪sell_idx` 這個集合，但不使用 Buy/Sell 標籤語義），每一時間點的 reservoir state `x[t+1]` 為一筆 1,291 維特徵。不得改用全腦、按表現挑選神經元、只取活躍神經元或時間平均。

對每個 graph 以 Train folds 計算 feature mean/std，標準化只用該 fold 的 Train；std<10⁻⁸ 的維度記為 constant 並固定移除。固定 Ridge 目標標準化、含 unpenalized intercept，訓練目標為 y[t+1]。最小化的明確 objective 為 `(1/n)||Y−XB||² + α||B||²`。α 網格固定 `{0, 10⁻⁸, 10⁻⁷, …, 10²}`，同一 grid 用於所有 graph/rho；選 10 個 Train streams 為 group 的 5-fold CV mean NMSE 最低 α，同分選較大 α。Val 不選 α。

若任何主分析 graph/rho 有超過 10% 的 instance 最佳 α 落在網格端點 0 或 10²，判定 alpha grid 未包住解，主分析 no-go；不得事後擴網格重跑後仍稱原預註冊通過。標準化、objective、alpha grid 與 10% 邊界比例均為本規格主觀／工程設定；舊 Phase 5 alpha 數字因特徵尺度和 objective 不同，不直接移植。

## 4. Controls、graph instances 與工作點

| Family | Instances | 固定／改變 |
|---|---:|---|
| Real FlyWire v783 | 1 | 完整圖、固定 input/readout index。 |
| `scramble_mixed` | 20 個獨立 seed | in/out degree 序列、每條權重與 source out-strength 固定；端點交換至 edge overlap ≤5%。 |
| Source-wise sign-stratified weight shuffle | 20 個獨立 seed | 拓樸、degree、每 source 正/負 out-strength、edge sign 與權重多重集固定；同 source 同符號內置換幅度。 |
| Uniform random endpoints | 20 個獨立 seed | 節點、邊數及全圖有符號權重多重集固定；端點均勻無放回抽樣、自環排除；度數與節點 strength 自然改變。排序後取前 E 的現有算法不得用於主分析。 |

控制圖的 input/readout indices 與 real 完全相同。各家族 20 instances 的 seed 清單在 Train 前凍結；control seeds 是圖層級變異，不當作市場樣本或時間序列樣本。真圖只有 1 instance，所有推論限定於這張 specimen。

### Spectral radius 工作點

共同候選網格 `rho ∈ {0.90, 0.95, 0.99}`；每一 graph family 都依自身 unscaled ρ 將 `W` 規一到同一 target。rho 網格和 selection rule 在解封任何 Test sequence 前凍結。1.05 不納入主網格；不能因 0.99 表現最好才事後增加。global rho matched 是明確的比較條件，不宣稱所有生物網路只由 rho 決定。

在 1 real + 每個 null family 5 graphs 的 16-graph Pilot 上，三個候選 rho 都只使用 Train sequences 做：(a) 飽和 gate：washout 後 non-sensory |x|>0.9 的比例 <5%；(b) initial-state forgetting gate：每條 Train 序列以 x₀=0 及各自獨立的 iid Uniform[−0.1,0.1] 初態，使用同一段前 500 個 input steps；在 t=500 計算 `d_rel = RMS(x_a−x_b) / max(RMS(x_a), RMS(x_b), 10⁻⁶)`，每張圖至少 4 個初態 pair × 10 條 Train 序列共 40 組中有 38 組（95%）達 `d_rel <10⁻³`；(c) MC 記憶容量：`Σ(k=1…50)max(R²(u[t−k]),0)`。只保留所有四個 family、每張 Pilot 圖均通過 (a)(b) 的共同 rho；依四 family 等權的 instance-level mean MC 最大者選單一 rho，同分選較低值。Pilot 的 rho selection 不讀 Val/Test，也不按某個 graph 各自選工作點。全量 61 張圖在選定 rho 的 Train 檢查也必須通過 (a)(b)，否則 no-go。若沒有共同合格 rho，停止 G1。初態分布與 40 組設計是本規格主觀診斷值。

0.99 是文獻候選值，來源為 Costi & Izzo 的特定果蠅 reservoir/CR3BP 實驗；0.90/0.95/0.99 網格、MC lag 50、500 步 forgetting 與 gate 數字是本計劃選擇，不能標成文獻定律。[Costi & Izzo 2025](https://www.mdpi.com/2313-7673/10/5/341)

## 5. 主要 endpoint、統計與比較

唯一主要 endpoint 是 Test NARMA10 NMSE：`NMSE = mean((ŷ−y)²) / Var(y)`。測試以每個獨立 sequence seed 為基本重抽單位；長序列再切為 500-step paired blocks，避免將高度相依的單點當獨立 n。對 graph controls 的抽樣以 graph instance 為另一層級；真實 FlyWire 只有一張圖，區間只描述此固定圖與控制圖分布在新 NARMA streams 上的對比，不描述真實 connectome 個體母群。

Primary control family 在解封 Test 前，用 Train 5-fold CV 的 mean NMSE 從三個 control family 中選表現最佳（NMSE 最低）者並鎖定。主要改善量為 `Δ = (median(NMSE_control) − NMSE_real) / median(NMSE_control)`。通過須同時符合：

1. Test 上 `Δ ≥ 0.05`（至少相對 NMSE 改善 5%；主觀的最小技術效果門檻，與市場交易成本無關）。
2. 以 graph instances × independent sequence seeds × 500-step blocks 做 2,000 次階層 paired bootstrap，Δ 的雙側 95% CI 下界 >0（2,000 與 95% 為規劃設定）。
3. 上述 simulator、alignment、state、alpha-boundary、negative/positive control gates 全部通過。

三個 control family 的其他 contrasts、Mackey–Glass、Lorenz 是次要結果，使用 Holm 校正，不能取代未過的 primary。Val 是診斷 split，不作成功證據。控制 family 以 Train 選定、Test 僅作一次評分；Test 結果解封後不得變更 input map、rho、readout、alpha、baseline、NMSE 定義或主要對照。

## 6. Positive/negative controls、假陽性率與 MDE

### Integrity positive control

以已知 NARMA recurrence 建立 oracle feature teacher，特徵包含所需歷史 y 與 recurrence RHS 的 input term；它只檢查 generator、索引位移與 scorer，不能算作 reservoir 能力。各 split oracle NMSE 必須 <10⁻⁶。若失敗，stateful graph 結果不解讀。

### Null／假陽性校準

在每個 null family 內，以 graph instance 為單位把 20 張圖劃成兩個 pseudo-group（10/10），共享相同 sequence seeds、套用 primary paired-block pipeline 並做 1,000 次 group-label permutation；5% nominal 門檻下假陽性比例必須 ≤5%。若評估 primary family-selection 對 FPR 的影響，每個 permutation 都重跑完整 Train-only family-selection 規則，再用獨立 calibration streams 評分。另以同一組 test sequences 配對「真 NARMA u」與來自另一獨立 seed 的錯配 u 作無訊號負控制；負控制相對 constant-mean baseline 不得呈現正向顯著改善。1,000 permutations 與 ≤5% 是本計劃工程判準。

### 5% MDE 校準

在不屬於主 Test seeds 的獨立 synthetic calibration streams 上，加入一個固定 delay-line feature `q[t]=1.5u[t]u[t−9]` 作為 planted memory feature；只用 calibration Train 選其縮放係數，使預期 NMSE 優勢恰為 5%，係數鎖定後在 held-out calibration streams 重複；若無法維持此 5% effect，校準失敗並停止。以與主分析相同的 graph/sequence/block 層級推論做 1,000 個模擬 replicates。要求 5% planted effect 的檢出率 ≥80%，且 null 假陽性 ≤5%。若 5% effect 的 power <80%，G1 對 5% MDE 沒有足夠檢定力，停止並報告 inconclusive；不解封主 Test 後再加資料。80%、1,000 replicates 為事前規劃門檻，屬主觀設計，不是已完成的 power 結果。

## 7. G0/G1 gate 與停損

| 階段 | Go 條件 | No-go／停損 |
|---|---|---|
| G0a：圖和 controls integrity | base graph / 每個 variant hash 與 metadata 完整；in/out degree、source positive/negative strength、weight multiset、loops、edge uniqueness、target rho 均符合對應 family；random endpoints 在 source index / neuron class 上無抽樣偏差。 | 任一統計不符、快取來源不明或未達 overlap≤5%，不跑 readout。 |
| G0b：stateful engine / alignment | 同一小圖、同一 input 下，GPU Torch 與 CPU reference states 在 float32 tolerance 內一致（absolute state error ≤10⁻⁴）；跨時間狀態更新一次且只更新一次；oracle NMSE<10⁻⁶；錯配 input negative control 不通過主要改善 gate。 | 首工作日結束仍無法證明 state/time alignment，或 oracle/negative control 失敗，即終止 G1。不得用靜態 32-step replay 冒充 stateful。 |
| G1a：Pilot | 1 real + 5 instances per null family 對同一 rho grid 完成 Train-only MC/穩定性 gate；找到一個四 family 共用合格 rho；5% MDE power≥80%、null FPR≤5%；全程預估 GPU≤8 h。 | 沒有共用 rho、5% MDE 無 80% power、假陽性>5%，或成本預估超 8 GPU h，no-go；不臨時改 input/noise/alpha/grid。 |
| G1b：鎖定主分析 | 凍結 source、graphs、seeds、weights hash、readout、rho、alpha 規則、primary family 選擇規則與 report schema 後，一次執行 61 graphs 的 Test。 | 任一 Test 讀取前後規格漂移、輸出缺檔或非有限值，整批結果標為 invalid；不得補跑後挑最好版本。 |
| 最終判定 | primary Δ≥5% 且 paired 95% CI lower bound>0，所有 integrity gates 通過。 | 任一條件不滿足：G1 FAIL／inconclusive，終止 connectome market prediction 路徑；不得改做新 market split 以挽救 G1。 |

主要門檻來源：5% NMSE 改善、500-step block、500-step washout、20 graph instances/family、Train/Val/Test 各自 10/3/10 sequence seeds、95% CI、2,000 bootstrap、80% power、5% FPR、8 GPU 小時與 4 工作天均為本計劃／使用者指定的研究規劃值或主觀門檻；FlyWire Phase 0 non-sensory saturation <5% 與 leak=0.5 為本專案實測規則。文獻來源只支援 NARMA10 benchmark 形式與 rho=0.99 候選，不支援本文件的效應量門檻。

## 8. 資料、設備與工時

G1 不需要市場 K 線、market Test、future_return、交易 labels、網路下載或新 sealed market holdout。FlyWire v783 graph 若無既有 cache，需有本機以下原始資料（欄位以目前 loader 需求為準）：

| 檔案 | 必要欄位／用途 |
|---|---|
| `Completeness_783.csv` | 第一欄 neuron root IDs，建立 0…N−1 component index。 |
| `Supplemental_file1_neuron_annotations.tsv` | `root_id`、`cell_type`、`super_class`、`side`、`pos_x`、`pos_y`、`pos_z`；選 R1-6、DN readout，計算 retina u/v。 |
| `Connectivity_783.parquet` | `Presynaptic_Index`、`Postsynaptic_Index`、`Excitatory x Connectivity`；建立 W[post, pre] CSR。 |
| 現有／生成 cache | `graph_cache.npz` 含 CSR、sensory/motor indices、photoreceptor type、eye、u/v、spectral radius；每個變體另有 hash 與 manifest。 |

上列檔名及欄位來自目前 `research/pipeline/flywire_graph.py`；執行前核對下載版本、資料授權、檔案 SHA 與 loader 欄位。若 annotations 未含上述位置／類型欄，retinotopic group mapping 不成立，先停而非隨機選 sensory neurons。

**硬體順序：** Mac CPU 只做 generator/oracle、小圖 state-alignment、圖統計及 ridge smoke tests；不得用 Mac 跑 61× full FlyWire 的慢速 CPU 主分析。只有 G0b 通過且 1,000-step CUDA timed pilot 的全流程外推可在 8 GPU 小時內完成時，才使用 Windows Ryzen 7 5800X3D + RTX 5070 12GB 跑完整 13.8 萬神經元 stateful grid/主分析。GPU 只負責 sparse reservoir state updates；graph generation、hash、資料檢查、Ridge 等由 CPU 執行。圖生成器每張 well-mixed 約 2.35–2.4 CPU 分鐘／真圖實測，20 張約 47–48 CPU 分鐘。G1 GPU 成本以 Phase 6 的 train/val 32-step static replay 實測速率外推，stateful 小 batch 尚未實測；timed pilot 超預算就停，不在主結果後換 OS／硬體接續同一批結果。

工時上限 4 個工作天：第 1 天 engine/target alignment + controls hash；第 2 天 GPU Pilot、power/FPR 校準與 freeze；第 3 天完整主分析；第 4 天品質檢查、階層區間與封存報告。這是排程上限（主觀）。跨作業系統浮點不保證逐位元一致；單一確認性 run 全程在同一台 Windows GPU 機器完成，記錄 OS、driver、CUDA、PyTorch、GPU 與 commit hash。
