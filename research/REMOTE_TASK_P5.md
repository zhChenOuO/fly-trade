# REMOTE_TASK_P5 — 在 GPU 機器上跑 Phase 5（Frozen Connectome 監督式 Linear Probe）

你是 remote codex。先與 origin 同步 main 分支（用系統 git，這台沒有 rtk），再建立新分支 `remote/phase5`（不要推 main、不要 force push）。

## 1. 必讀文件與設計原則
- 必讀：`research_v2.md`（§6 Phase 5 監督式 Linear Probe）、`research/SPEC_v3.md`、`research/pipeline/{extract_features_v2,train_probe_v2,graph_variants,fly_simulator,torch_sim}.py`、`research/REMOTE_TASK_P1.md`。
- **嚴格禁令**：
  1. **絕對不讀取/評估 dev_test_v1 與 holdout**；只處理 train 與 val（白名單鎖定，違規直接報錯）。
  2. **不得為了結果好看而改動 alpha 選擇方式、門檻或特徵**（Ridge alpha 僅由 Train 內時序 CV 決定，不得看 Val）。
  3. **特徵 .npy 檔案不要 commit**（`research/outputs/v2/features/*.npy` 已被 gitignore，體積龐大），只 commit JSON、指標與 log。

---

## 2. 執行流程與精確指令

請在專案根目錄依序執行下列步驟（GPU 環境已具備 torch 與 CUDA）：

### Step -1: 確認資料前置（images.npy 與 FlyWire 檔案）
`research/data/images.npy` 與 FlyWire parquet/npz 被 gitignore，不在 repo 內。若 `research/data/images.npy` 不存在，先重建（會與 `research/data/audit/*.png` 逐位元比對，不一致就停止並回報；不得重新抓交易所資料）：
```bash
test -f research/data/images.npy || python -m research.pipeline.rebuild_images
```
FlyWire 檔案（`research/data/flywire/` 下）依 `research/data/flywire/DOWNLOAD.md`，前幾個 Phase 已在這台機器上準備好；若缺失請停止並回報。
另請用 `sha256sum research/data/raw_ohlcv.parquet` 對照 `research/data/data_hash.txt`，不一致就停止。

### Step 0: 前置一致性檢查（Smoke Consistency Check）
開跑全量前，先以 100 個 Val 樣本驗證 readout 萃取與 Phase 1a intact 模型的數值一致性：
```bash
python -m research.pipeline.extract_features_v2 --variant real --split val --limit 100 --backend torch
```
檢查輸出中的 consistency check 是否通過（max diff 應小於 1e-5）。若失敗立即停止並回報。

### Step 1: 特徵萃取（三種 Graph 變體 × train / val）
對真實 Connectome（real）、拓撲對照（random）、度保持打亂對照（scramble）分別萃取 readout 活動向量：

```bash
# --- 變體 1: real (真實 FlyWire Connectome) ---
python -m research.pipeline.extract_features_v2 --variant real --split train --backend torch
python -m research.pipeline.extract_features_v2 --variant real --split val --backend torch

# --- 變體 2: random (隨機拓撲對照網路) ---
python -m research.pipeline.extract_features_v2 --variant random --split train --backend torch
python -m research.pipeline.extract_features_v2 --variant random --split val --backend torch

# --- 變體 3: scramble (度分佈保持打亂網路) ---
python -m research.pipeline.extract_features_v2 --variant scramble --split train --backend torch
python -m research.pipeline.extract_features_v2 --variant scramble --split val --backend torch
```

*特徵儲存位置*：`research/outputs/v2/features/{variant}_{split}.npy` 與對應的 `{variant}_{split}.json` 元資料。

### Step 2: Probe 訓練、評估與統計檢定
執行統一的訓練與評估腳本（自動進行 Train-only 標準化、時序 CV alpha 選擇、Ridge/Logistic 解碼器評估、Nuisance-only 控制組評估、與 OHLCV baseline 比較、Block Bootstrap 95% CI、以及 1000 次 Circular Shift 區塊置換檢定）：

```bash
python -m research.pipeline.train_probe_v2 --bootstrap-samples 1000 --permutations 1000 2>&1 | tee research/outputs/v2/probe_run.log
```

*評估結果輸出*：`research/outputs/v2/probe_val.json`。

---

## 3. 實測建構數據與耗時預估（估計值）

### (A) 圖變體建構實測（真實 FlyWire 1500 萬邊，本機 CPU 實測）
- `random_matched_graph`：
  - 邊生成耗時：5.33 秒
  - 譜半徑 eigs 計算：3.35 秒（原始 $\rho \approx 94.03$，目標歸一化至 2164.2939）
  - 記憶體消耗：約 1.2 GB
- `degree_preserved_scramble`（100 萬次邊交換）：
  - 邊交換耗時：3.30 秒（採用打包 int64 邊集合優化）
  - 譜半徑 eigs 計算：0.73 秒（原始 $\rho \approx 2016.48$，目標歸一化至 2164.2939）
  - 記憶體消耗：約 1.4 GB

### (B) 運行耗時預估（估計值，依 Phase 1a 約 240 張/秒推算）
- **Train split**（31,556 樣本）：約 131 秒（~2.2 分鐘）/ 變體
- **Val split**（10,504 樣本）：約 44 秒（~0.7 分鐘）/ 變體
- **三個變體特徵萃取總耗時**（6 次 run）：估計約 17.5 分鐘
- **Probe 訓練與檢定**（含 1000 次 bootstrap 與 1000 次 circular shift）：估計約 1.5 ~ 2.5 分鐘
- **全流程總預估耗時**：**約 20 分鐘**

---

## 4. 停止規則（Stop Rules，立即停止並回報）
遇下列任何情況時，必須立即終止執行並回報錯誤日誌，不得自行調參或掩蓋：
1. **數值異常**：特徵向量或預測值中出現任何 `NaN` 或 `Inf`。
2. **活動塌縮（Activity Collapse）**：在 Train split 中，超過 90% 的特徵維度全為 0 或方差為 0。
3. **一致性檢查失敗**：在 100 個 Val 樣本測試中，由 readout 重構之分數與 Phase 1a `buy_score / sell_score` 差異超過 $10^{-5}$。
4. **顯存/記憶體溢出（OOM）**：若 GPU VRAM 超標，檢查 batch chunk size（預設 256/512），不得改動圖結構。

---

## 5. 交付與回報格式

### Git 操作
在 `remote/phase5` 分支上 commit 並 push 以下檔案：
- `research/outputs/v2/features/*.json`（元資料檔案，記錄 SHA256 與 sample 順序）
- `research/outputs/v2/probe_val.json`（完整指標、比較、Bootstrap CI 與 pass/fail 表）
- `research/outputs/v2/probe_run.log`（執行日誌）
- Commit 訊息結尾請加上：`Co-Authored-By: Codex <noreply@openai.com>`

### 完成回報內容
請在回報中附上：
1. 各組 Validation Spearman IC（Real、Random、Scramble、Nuisance-only、OHLCV Baseline）。
2. 增量效果量與 95% CI：$\Delta\text{IC}(\text{real} - \text{random})$、$\Delta\text{IC}(\text{real} - \text{scramble})$、$\Delta\text{IC}(\text{real} - \text{OHLCV})$。
3. Circular Shift 置換檢定之 p-value。
4. `pass_fail_evaluation` 逐項門檻判斷結果表。
5. 實際各步驟耗時與 GPU 吞吐量（張/秒）。
