# REMOTE_TASK_P5B — 在 GPU 機器上跑 Phase 5 事後探索分析（EXPLORATORY_POST_HOC）

你是 remote codex。先與 origin 同步 main 分支（用系統 git，這台沒有 rtk），再建立新分支 `remote/phase5b`（不要推 main、不要 force push）。

## 1. 必讀文件與設計原則
- 必讀：`research/outputs/v2/probe_val.json`、`research/pipeline/probe_posthoc_v2.py`、`research_v2.md`（§8、§12）、`research/REMOTE_TASK_P5.md`。
- **核心審計與治理原則**：
  1. **ORIGINAL_VERDICT: FAIL（已定案，維持不變；見 probe_val.json）**。
  2. **本任務屬於「事後探索（EXPLORATORY_POST_HOC）」，絕對不得引述為 PASS 之證據**。
  3. **絕對不讀取/評估 dev_test_v1 與 holdout**；只處理 train 與 val（白名單鎖定，違規直接報錯）。
  4. **不得為了結果好看而改動 alpha 選擇方式、門檻或特徵**（Ridge alpha 僅由 Train 內時序 CV 決定，不得看 Val）。
  5. **不得覆寫、改動或取代既有 probe_val.json**；結果必須原樣如實回報（包含不利於假設之結果）。
  6. **特徵 .npy 檔案不要 commit**（`research/outputs/v2/features/*.npy` 已被 gitignore）。

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

### Step 1: 檔案 SHA256 驗證與重現檢查（Pre-flight Reproduction Check）
在執行 bootstrap 前，先執行快速預檢：
1. 驗證所有 6 個 `.npy` 特徵檔的 SHA256 與 `probe_val.json` 之 `metadata.feature_hashes` 逐一相符。
2. 重建 Ridge 與 Logistic 模型，比對 Spearman IC 是否與 `probe_val.json` 數值完全一致（容許誤差 $\le 10^{-4}$）。

```bash
python -m research.pipeline.probe_posthoc_v2 --check-only
```

*若出現任何 hash 不符或 IC 不一致，立即終止並回報，不要往下執行！*

### Step 2: 執行完整事後探索分析（2000 次 Paired Block Bootstrap）
執行統計分析，包含：
- 2000 次配對移動區塊重抽（block=24 樣本，同一組重抽索引套用於所有比較組別）。
- Logistic 與 Ridge 各組之 $\Delta\text{IC}$、95% CI、雙尾 $p$ 值，以及 Holm-Bonferroni 多重比較校正 $p$ 值。
- 訊號冗餘性與殘差 IC（線性回歸剔除 OHLCV 與 Nuisance 分數後之增量 IC）。
- 4 個時序等分區間（Quarters）之子期間穩定性與符號一致性。
- 修正版審計表（標註 0 交易空 PASS、並排競爭性控制組 BA、非零 Logistic 之 Nuisance $R^2$）。

```bash
python -m research.pipeline.probe_posthoc_v2 --bootstrap-samples 2000 2>&1 | tee research/outputs/v2/probe_posthoc_run.log
```

*產出位置*：`research/outputs/v2/probe_posthoc_val.json`。

---

## 3. 耗時預估（估計值）

- **特徵準備**：0 秒（直接 symlink 既有檔案）。
- **Step 1 重現預檢**：約 15 ~ 20 秒。
- **Step 2 統計分析**：
  - Nuisance 特徵萃取：約 10 ~ 15 秒。
  - OHLCV 特徵提取與模型配適：約 10 秒。
  - 2000 次多模型配對區塊重抽：約 1.5 ~ 3 分鐘。
  - 殘差 IC 與子區間分析：約 30 秒。
- **全流程總預估耗時**：**約 3 ~ 5 分鐘**。

---

## 4. 停止規則（Stop Rules，立即停止並回報）
遇下列任何情況時，必須立即終止執行並回報錯誤日誌，不得自行調參或掩蓋：
1. **SHA256 不符**：任一特徵檔 SHA256 與 `probe_val.json` 記錄之 hash 不一致。
2. **重現檢查失敗**：任一模型重新訓練之 Validation Spearman IC 與 `probe_val.json` 差異 $> 10^{-4}$。
3. **數值異常**：任何回歸係數、殘差、預測或統計量出現 `NaN` 或 `Inf`。
4. **記憶體異常**：若遇 OOM，檢查陣列維度，不得改動分析邏輯。

---

## 5. 交付與回報格式

### Git 操作
在 `remote/phase5b` 分支上 commit 並 push 以下檔案：
- `research/outputs/v2/probe_posthoc_val.json`（事後探索完整分析結果）
- `research/outputs/v2/probe_posthoc_run.log`（執行日誌）
- **注意**：`.npy` 特徵檔案已在 `.gitignore`，**嚴禁 commit**。
- Commit 訊息結尾請加上：`Co-Authored-By: Codex <noreply@openai.com>`

### 完成回報內容
請在回報中附上下列數據：
1. **重現預檢結果**：各模型 IC 重現差值。
2. **Logistic 配對比較表**：
   - Real vs Random：$\Delta\text{IC}$、95% CI、raw $p$、Holm-adjusted $p$
   - Real vs Scramble：$\Delta\text{IC}$、95% CI、raw $p$、Holm-adjusted $p$
   - Real vs OHLCV B1b：$\Delta\text{IC}$、95% CI、raw $p$、Holm-adjusted $p$
   - Real vs Nuisance-only：$\Delta\text{IC}$、95% CI、raw $p$、Holm-adjusted $p$
3. **Ridge 配對比較表**（對照組同上）。
4. **訊號冗餘性與殘差 IC**：
   - Real 與 OHLCV、Nuisance 之相關係數。
   - 剔除控制組後的殘差 Spearman IC 及其 95% CI。
5. **時序穩定性**：4 個 Quarter 之 IC 與正值比例。
6. **修正版審計表摘要**：0 交易列旗標、BA 並排比較數值、Logistic 之 Nuisance $R^2$。
