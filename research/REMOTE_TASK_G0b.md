# REMOTE_TASK_G0b — 在遠端 GPU 機器執行 G0 修正對照網路特徵萃取與 Ridge Probe 評估（EXPLORATORY）

你是 remote codex。先與 origin 同步 `main` 分支（用系統 git），再建立新分支 `remote/g0_controls`（不要推 main、不要 force push）。

> **執行環境與資源**：
> - 特徵萃取（Step 1）需要 GPU（RTX 4090，以 PyTorch FlySimulator 執行 13.8 萬神經元 600 步模擬）。
> - Probe 評估（Step 2）為 CPU 運算（Ridge CV、Logistic 與 Block Bootstrap）。

---

## 1. 必讀文件與設計原則

- **必讀文件**：
  - `research/RESULTS_market_v1.md`（已知缺陷段落）
  - `research/PREREQUISITES_v3.md`（§1 P6、§3 G0）
  - `research/outputs/v2/probe_val.json`
  - `research/pipeline/graph_variants.py`
  - `research/pipeline/extract_features_v2.py`
  - `research/pipeline/train_probe_v2.py`
- **核心審計與治理原則**：
  1. **ORIGINAL_VERDICT: FAIL（已定案，維持不變；見 probe_val.json）**。
  2. **本任務屬於「事後探索性分析（EXPLORATORY）」**：檢驗文獻（PMC12109256）指出「權重分布貢獻大於拓樸」之現象，**絕對不得引述為改變 Phase 5 原判定之證據**。
  3. **絕對不讀取/評估 dev_test_v1 與 holdout**；只處理 train 與 val（白名單鎖定，違規直接報錯）。
  4. **不得為了結果好看而改動任何超參數、門檻、特徵或 alpha 選擇方式**（Ridge alpha 嚴格由 Train 內時序 CV 決定，含 `purge_samples=10` embargo，不得偷看 Val）。
  5. **不得覆寫、改動或取代既有 probe_val.json 或 outputs/v2 既有檔案**；本次 Probe 評估一律輸出至新檔案 `research/outputs/v2/probe_controls_val.json`。
  6. **特徵 .npy 檔案不要 commit**（`research/outputs/v2/features/*.npy` 已被 gitignore）。
  7. **嚴禁修改 README**。
  8. **日誌與回報格式硬性規定**：實驗日誌與完成回報只記結果（數值、判定、檔案路徑），絕對不寫過程敘述與贅詞。

---

## 2. 執行流程與精確指令

請在專案根目錄依序執行下列步驟：

### Step 0: 確認既有特徵與環境
確認既有 3 個變體特徵（real, random, scramble）已存在於 `research/outputs/v2/features/`：
```bash
ls -lh research/outputs/v2/features/{real,random,scramble}_{train,val}.npy
```
若缺少，請先依據既有 worktree 進行符號連結（symlink）或複製。

### Step 1: 萃取 G0 新對照特徵（scramble_mixed 與 weight_shuffle）

本步驟共需產出 4 個特徵檔案：
1. `scramble_mixed`（充分混合 degree-preserved scramble，邊重疊率 $\le 0.05$）
2. `weight_shuffle`（保留完全相同的有向邊拓樸，分別對興奮性與抑制性權重多重集隨機置換，保留 Dale's principle）

執行指令：
```bash
mkdir -p research/outputs/v2/features

# 1. 萃取 scramble_mixed
python -m research.pipeline.extract_features_v2 --variant scramble_mixed --split train 2>&1 | tee research/outputs/v2/extract_scramble_mixed_train.log
python -m research.pipeline.extract_features_v2 --variant scramble_mixed --split val 2>&1 | tee research/outputs/v2/extract_scramble_mixed_val.log

# 2. 萃取 weight_shuffle
python -m research.pipeline.extract_features_v2 --variant weight_shuffle --split train 2>&1 | tee research/outputs/v2/extract_weight_shuffle_train.log
python -m research.pipeline.extract_features_v2 --variant weight_shuffle --split val 2>&1 | tee research/outputs/v2/extract_weight_shuffle_val.log
```

萃取完成後，計算並記錄 4 個新特徵檔案之 SHA256：
```bash
sha256sum research/outputs/v2/features/scramble_mixed_{train,val}.npy research/outputs/v2/features/weight_shuffle_{train,val}.npy
```

### Step 2: 執行 G0 擴充 Ridge Probe 評估（不覆寫既有 probe_val.json）

使用更新後的 `train_probe_v2` 介面，將 5 個變體（`real`, `random`, `scramble`, `scramble_mixed`, `weight_shuffle`）並列評估：
```bash
python -m research.pipeline.train_probe_v2 \
    --variants real random scramble scramble_mixed weight_shuffle \
    --output-path research/outputs/v2/probe_controls_val.json \
    --bootstrap-samples 1000 \
    --permutations 1000 \
    2>&1 | tee research/outputs/v2/probe_controls_run.log
```

產出結果路徑：
- `research/outputs/v2/probe_controls_val.json`
- `research/outputs/v2/probe_controls_run.log`

---

## 3. 耗時預估與計算成本分析

- **特徵圖生成與快取**：
  - `scramble_mixed`：真實 FlyWire 圖（15,091,983 邊）進行 24,901,770 次 double-edge swap，本機實測約 **141.3 秒**（2.35 分鐘，成功 swap 24,544,329 次，重疊率降至 4.38%）。產生之圖快取保存於 `research/data/flywire/graph_scramble_mixed_seed42.npz`。
  - `weight_shuffle`：權重陣列置換，實測約 **1.13 秒**。圖快取保存於 `research/data/flywire/graph_weight_shuffle_seed42.npz`。
- **特徵萃取模擬（GPU）**：
  - 依 Phase 5 基準（RTX 4090），每圖每 split 約 **5.8 分鐘**。
  - 共 4 次萃取 $\times$ 5.8 分鐘 $\approx$ **23.2 分鐘**。
- **Ridge Probe 評估（CPU）**：
  - 5 個變體之時序 CV（8-alpha）+ 1,000 次 Paired Bootstrap + 1,000 次 Permutations，約 **2 ~ 3 分鐘**。
- **總估計耗時**：**約 27 ~ 30 分鐘**。

---

## 4. 停止規則（Stop Rules，立即終止並回報）

遇下列情況之一時，必須立即終止執行並回報錯誤日誌：
1. **重疊率超標**：`scramble_mixed` 的 `scramble_diagnostics.overlap_ratio` $> 0.05$ 或 `target_reached` 為 `false`。
2. **度數未保持**：`scramble_mixed` 的 `in_degree_preserved` 或 `out_degree_preserved` 為 `false`。
3. **拓樸或權重破壞**：`weight_shuffle` 的 `topology_preserved` 或 `weights_identical_multiset` 為 `false`。
4. **數值異常**：任何特徵矩陣、預測值、IC、係數出現 `NaN` 或 `Inf`。
5. **既有變體數值漂移**：`real`、`random`、`scramble` 在本次評估中的 Validation Spearman IC 與原始 `probe_val.json` 差異 $> 10^{-4}$。
6. **違規資料讀取**：若有任何程式碼嘗試讀取 `dev_test_v1` 或 `holdout`。

---

## 5. 交付與回報格式

### Git 操作
在 `remote/g0_controls` 分支上 commit 並 push 以下檔案：
- `research/outputs/v2/probe_controls_val.json`
- `research/outputs/v2/probe_controls_run.log`
- `research/outputs/v2/features/scramble_mixed_train.meta.json`
- `research/outputs/v2/features/scramble_mixed_val.meta.json`
- `research/outputs/v2/features/weight_shuffle_train.meta.json`
- `research/outputs/v2/features/weight_shuffle_val.meta.json`
- **注意**：`.npy` 特徵檔案嚴禁 commit！
- Commit 訊息結尾請加上：`Co-Authored-By: Codex <noreply@openai.com>`

### 完成回報內容（只寫結果）
依照 `CLAUDE.md` 規則，回報只寫結果（數值、判定、檔案路徑），不寫過程敘述與贅詞。請附上下列數據：
1. **新特徵檔案 SHA256 與混合診斷摘要**：
   - `scramble_mixed`: 成功 swap 次數、最終重疊率（需 $\le 5\%$）、入/出度保持確認。
   - `weight_shuffle`: 拓樸一致性確認、正負權重多重集保持確認。
2. **5 個變體 Validation 表現並列對照表**：
   | 變體名稱 | 變體類型描述 | Val Spearman IC | 95% CI | $\Delta\text{IC}$ vs Real (95% CI) | Ridge Alpha |
   | :--- | :--- | :---: | :---: | :---: | :---: |
   | `real` | 完整真實果蠅連接組 | -0.00752 | [-0.0381, 0.0229] | 0.0000 (基準) | 1000000.0 |
   | `random` | 權重洗牌 + 隨機端點 | 0.00335 | [-0.0270, 0.0335] | -0.0109 [-0.0528, 0.0312] | 100000.0 |
   | `scramble` | 舊版 swap（混合不足，重疊 ~80%） | 0.00223 | [-0.0277, 0.0322] | -0.0097 [-0.0506, 0.0315] | 100000.0 |
   | `scramble_mixed` | 充分混合 swap（重疊 $\le 5\%$） | ? | ? | ? | ? |
   | `weight_shuffle` | 保留真實拓樸，洗牌突觸權重 | ? | ? | ? | ? |
3. **科學判定結論**：
   - 充分混合後的 `scramble_mixed` 是否仍與 `real` 無顯著差異？
   - `weight_shuffle`（保留拓樸但洗牌權重）之 IC 是否與 `real` 接近？文獻 PMC12109256 之假說（權重分布主導能力）在此特徵空間是否獲得支持？
