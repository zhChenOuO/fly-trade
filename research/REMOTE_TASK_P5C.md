# REMOTE_TASK_P5C — 在遠端機器執行 Phase 5 缺陷修正重測與 MDE 標籤注入實驗（EXPLORATORY）

你是 remote codex。先與 origin 同步 `main` 分支（用系統 git，這台沒有 rtk），再建立新分支 `remote/phase5c`（不要推 main、不要 force push）。

> **執行排程註記**：遠端 GPU 目前正在執行 Phase 6。**本任務（P5C）完全不需要 GPU，主要負載為 CPU 上的 Ridge CV 與 Block Bootstrap，可安排在 Phase 6 之後執行，或在 CPU 空閒核心上以 nice 優先權執行。**

---

## 1. 必讀文件與設計原則

- **必讀文件**：
  - `research/outputs/v2/probe_val.json`
  - `research/outputs/v2/baselines_val.json`
  - `research/pipeline/mde_injection_v2.py`
  - `research/pipeline/baselines_v2.py`
  - `research/pipeline/train_probe_v2.py`
  - `research/PLAN_v3_direction.md`
- **核心審計與治理原則**：
  1. **ORIGINAL_VERDICT: FAIL（已定案，維持不變；見 probe_val.json）**。
  2. **本任務屬於「事後探索（EXPLORATORY_POST_HOC）」，絕對不得引述為 PASS 之證據**。
  3. **絕對不讀取/評估 dev_test_v1 與 holdout**；只處理 train 與 val（白名單鎖定，違規直接報錯）。
  4. **不得為了結果好看而改動 alpha 選擇方式、門檻或特徵**（Ridge alpha 僅由 Train 內時序 CV 決定，含 `purge_samples=10` embargo，不得偷看 Val）。
  5. **不得覆寫、改動或取代既有 probe_val.json 或 baselines_val.json**；結果必須原樣如實回報（包含不利於假設之結果）。
  6. **特徵 .npy 檔案不要 commit**（`research/outputs/v2/features/*.npy` 已被 gitignore）。
  7. **嚴禁修改 README**（不因實驗或排查問題修改 README，只在專案用法或結構變動時才改）。
  8. **日誌與回報格式硬性規定**：實驗日誌與完成回報只記結果（數值、判定、檔案路徑），絕對不寫過程敘述與贅詞。

---

## 2. 執行流程與精確指令

請在專案根目錄依序執行下列步驟：

### Step 0: 連結既有特徵檔案（不重新萃取）
特徵 `.npy` 檔案先前已在遠端 Phase 5 產生，路徑位於 `.worktrees/phase5/research/outputs/v2/features/`。
新 worktree 請建立目錄並以符號連結（symlink）或複製方式接入同名路徑：

```bash
mkdir -p research/outputs/v2/features

# 嘗試自既有 worktree 建立軟連結
if [ -d "../phase5/research/outputs/v2/features" ]; then
    ln -sf ../phase5/research/outputs/v2/features/*.npy research/outputs/v2/features/
elif [ -d ".worktrees/phase5/research/outputs/v2/features" ]; then
    ln -sf $(pwd)/.worktrees/phase5/research/outputs/v2/features/*.npy research/outputs/v2/features/
else
    echo "請確認 phase5 特徵 .npy 檔案目錄位置並連結至 research/outputs/v2/features/"
fi

# 確認 6 個特徵檔案齊全
ls -lh research/outputs/v2/features/*.npy
```

### Step 1: 檔案 SHA256 驗證與 P5B 事後分析（若尚未執行）
在執行 MDE 實驗前，先執行快速預檢：
1. 驗證所有 6 個 `.npy` 特徵檔的 SHA256 與 `probe_val.json` 之 `metadata.feature_hashes` 逐一相符。
2. 重建 Ridge 與 Logistic 模型，比對 Spearman IC 是否與 `probe_val.json` 數值完全一致（容許誤差 $\le 10^{-4}$）。

```bash
python -m research.pipeline.probe_posthoc_v2 --check-only
```

*若出現任何 hash 不符或 IC 不一致，立即終止並回報，不要往下執行！*

> 註：若遠端環境尚未產生 `research/outputs/v2/probe_posthoc_val.json`，可先補跑 P5B 事後分析：
> ```bash
> python -m research.pipeline.probe_posthoc_v2 --bootstrap-samples 2000 2>&1 | tee research/outputs/v2/probe_posthoc_run.log
> ```

### Step 2: Phase 5 修正版 Baselines 重跑對照（Section A4 驗證）
驗證 `baselines_v2.py` 在加上 parquet split filters 後，輸出指標與原始 `baselines_val.json` 逐位一致：

```bash
python -c "
import json
from pathlib import Path
from research.pipeline.baselines_v2 import run_baselines_v2

# 輸出至暫存檔，絕對不得覆寫 outputs/v2/baselines_val.json
tmp_out = Path('scratch_baselines_val.json')
res = run_baselines_v2(output_path=tmp_out)
saved = json.loads(Path('research/outputs/v2/baselines_val.json').read_text(encoding='utf-8'))

print('=== Baselines Verification Check ===')
all_matched = True
for m in ['B0_constant_majority', 'B1a_ridge', 'B1b_logistic']:
    ic_new = res['models'][m]['continuous']['spearman_ic']
    ic_old = saved['models'][m]['continuous']['spearman_ic']
    diff = abs(ic_new - ic_old)
    print(f'Model {m:<25}: new_ic={ic_new:<22} old_ic={ic_old:<22} diff={diff:.2e}')
    if diff > 1e-7:
        all_matched = False

tmp_out.unlink(missing_ok=True)
assert all_matched, 'Baseline metrics do not match baselines_val.json!'
print('Baselines verification PASSED: all deterministic metrics match bitwise.')
"
```

### Step 3: 執行 MDE 標籤注入實驗（mde_injection_v2.py）

本實驗測量管線對三種弱訊號形態（`momentum`, `mean_reversion`, `image_projection`）在目標 IC $\{0.0, 0.005, 0.01, 0.02, 0.03, 0.05\}$ 下的檢定力。

#### 預設完整執行（$R=100$, bootstrap=500）
```bash
python -m research.pipeline.mde_injection_v2 \
    --repetitions 100 \
    --bootstrap-samples 500 \
    2>&1 | tee research/outputs/v2/mde_injection.log
```

#### 縮減快速執行選項（若總耗時限制在 2 小時內）
若評估 CPU 資源受限或需在 2 小時內完成，可將重複次數調整為 $R=20$，bootstrap 次數調為 300：
```bash
python -m research.pipeline.mde_injection_v2 \
    --repetitions 20 \
    --bootstrap-samples 300 \
    2>&1 | tee research/outputs/v2/mde_injection.log
```

*產出位置*：`research/outputs/v2/mde_injection.json` 與 `research/outputs/v2/mde_injection.log`。

---

## 3. 耗時預估與計算成本分析

- **特徵準備（Step 0）**：0 秒（直接 symlink）。
- **重現預檢（Step 1）**：約 15 ~ 20 秒（若補跑完整 P5B 約 3 ~ 5 分鐘）。
- **Baselines 驗證（Step 2）**：約 20 秒。
- **MDE 標籤注入實驗（Step 3）**：
  - **計算成本結構**：
    $\text{總擬合次數} = 3 \text{ 種形態} \times 6 \text{ 個目標 IC} \times R \text{ 次重複} \times 3 \text{ 個網路變體} = 54 \times R \text{ 次}$。
    每次擬合需進行：
    - 5 折時序擴展窗 CV（含 `purge_samples=10` 隔離期）× 8 個 candidate alpha。
    - 1 次全 Train 集 Ridge 擬合（1291 維度）。
    - 配對移動區塊重抽（Paired Block Bootstrap，長度 10,504，block=24）計算 95% CI 與 $\Delta\text{IC}$ CI。
  - **單次擬合耗時**：在單一 CPU 核心上約 7.5 秒。
  - **各配置耗時對照**：
    | 配置 | $R$ | Bootstrap 次數 | 總擬合次數 | 預估總耗時 | 適用場景 |
    | :--- | :---: | :---: | :---: | :---: | :--- |
    | **預設標準（Default）** | 100 | 500 | 5,400 次 | **約 11 ~ 12 小時** | 夜間完整精確測試、Monte Carlo 誤差最小 |
    | **縮減選項（Reduced）** | 20 | 300 | 1,080 次 | **約 2.0 ~ 2.5 小時** | 兩小時內快速驗證檢定力階梯趨勢 |
    | **快速測試（Fast）** | 10 | 200 | 540 次 | **約 1.1 小時** | 極速驗收 pipeline 運作 |

---

## 4. 停止規則（Stop Rules，立即停止並回報）

遇下列任何情況時，必須立即終止執行並回報錯誤日誌，不得自行調參或掩蓋：
1. **SHA256 不符**：任一特徵檔 SHA256 與 `probe_val.json` 記錄之 hash 不一致。
2. **重現檢查失敗**：任一模型重新訓練之 Validation Spearman IC 與 `probe_val.json` 差異 $> 10^{-4}$。
3. **Baselines 數字不一致**：Step 2 中重跑的 Baselines 與 `baselines_val.json` 差異 $> 10^{-7}$。
4. **校準收斂失敗**：二分法振幅校準在 40 輪內無法使 oracle Spearman IC 達到目標值（容差 0.001 內）。
5. **數值異常**：任何回歸係數、預測、bootstrap 信賴區間出現 `NaN` 或 `Inf`。
6. **違規資料讀取**：若有任何程式碼嘗試讀取 `dev_test_v1` 或 `holdout`。

---

## 5. 交付與回報格式

### Git 操作
在 `remote/phase5c` 分支上 commit 並 push 以下檔案：
- `research/outputs/v2/mde_injection.json`（MDE 檢定力實驗結果）
- `research/outputs/v2/mde_injection.log`（執行日誌）
- （若在本次補跑 P5B，附帶 `probe_posthoc_val.json` 與 `probe_posthoc_run.log`）
- **注意**：`.npy` 特徵檔案嚴禁 commit！
- Commit 訊息結尾請加上：`Co-Authored-By: Codex <noreply@openai.com>`

### 完成回報內容
依照 `CLAUDE.md` 規則，回報只寫結果（數值、判定、檔案路徑），不寫過程敘述與贅詞。請附上下列數據：
1. **Step 2 Baselines 驗證差值**（確認與原始 `baselines_val.json` 逐位一致）。
2. **MDE 總結表**（各形態與變體之最小可偵測 IC，若未達 80% 報 NOT_REACHED）：
   | 訊號形態 | Real 網路 MDE | Random 網路 MDE | Scramble 網路 MDE |
   | :--- | :---: | :---: | :---: |
   | `momentum` | ? | ? | ? |
   | `mean_reversion` | ? | ? | ? |
   | `image_projection` | ? | ? | ? |
3. **各目標 IC 下的偵測率（Detection Rate）及 95% Wilson CI**：
   - 目標 IC = 0.0（假陽性率 FPR）：是否 $\le 10\%$？
   - 目標 IC = 0.005, 0.01, 0.02, 0.03, 0.05：偵測率是否隨 IC 上升？
4. **配對比較 $\Delta\text{IC}$ 檢定力**：Real vs Random 與 Real vs Scramble 的顯著偵測率與 CI 寬度。
5. **Ridge Alpha 上界撞擊比例**（`alpha_hits_upper_bound_ratio`）：是否隨注入訊號強度增強而由上界回落？
6. **誠實限制**：確認 JSON 輸出頂端包含 `EXPLORATORY_POST_HOC` 與 `DOES_NOT_CHANGE_PHASE5_VERDICT`，並記錄限制聲明。
