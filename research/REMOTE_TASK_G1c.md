# REMOTE_TASK_G1c — 在遠端 GPU 機器執行 G1 v2 階段 C 主測（21 圖 NARMA10 + Δ + Power/FPR 校準）

> **前置依賴（Prerequisites）**：
> 本任務書在 `REMOTE_TASK_G1b` 完成且判定 Stage B 診斷設計缺陷（STAGE_B_INVALID）後啟動；依據 `research/RESULTS_G1_v2.md` 與 `research/DISCUSSION_next_target.md` 最新決策，Stage C 維持原時序對齊 $x[t+1] \to y[t+1]$ 執行比較式主測，不需延遲修正，不再以 $d_1/d_2$ 能力 gate 作為前置條件。

你是 remote codex。在遠端 GPU 環境切換至新分支 `remote/g1_stage_c`（基於包含 Stage C 執行器的分支），執行確認性階段 C 21 圖主測與推論校準。

- **執行環境 Python 直譯器**：
  `/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python`
- **硬體需求**：NVIDIA CUDA GPU（如 RTX 4090 或 RTX 5070，顯存 $\ge 12$ GB）。

---

## 1. 核心治理與科學審計原則（SPEC v2 §3.2, §4, §5, §7, §8, §9）

1. **21 圖完整拓樸對照組**：
   - 1 張真實 FlyWire v783 全腦圖（138,639 神經元，1,509 萬突觸）。
   - 20 張 `scramble_mixed` 重接圖（固定種子 35001..35020，以 `g1_manifest.SEED_NAMESPACES["scramble"]` 為準，此處數字僅供參考）。
   - 嚴格驗證每張 scramble 圖的無縮放不變量（edge overlap $\le 5\%$、in/out degree、source out-strength、權重多重集、無自環/重複邊）。
   - 全 21 張圖各自獨立縮放譜半徑至 $\rho=0.95$（收斂殘差 $\le 10^{-6}$，獨立初始化相對差 $\le 10^{-3}$）。
2. **固定輸入、動力學與時序契約**：
   - 原始輸入 $u[t] \sim \text{Uniform}[0, 0.5]$ 注入 R1-6（7,932 個，排除 R7/R8），不置中、無額外 recurrent bias。
   - 時序對齊：狀態 $x[t+1]$ 預測 NARMA10 目標 $y[t+1]$（`x[t+1] -> y[t+1]`）。
   - 讀出固定 1,291 個 DN（`buy_idx U sell_idx`），不增加非線性 head，不逐圖搜尋超參數。
3. **資料分割使用與嚴格密封邊界**：
   - **Main-Train**（10 條序列，有效 5,000 步＋washout 500 步）：每張圖執行 5-fold sequence CV 選最優 $\alpha \in \{0, 10^{-8}, \dots, 10^2\}$，以 10 條全量 fit。
   - **Main-Val**（3 條序列，有效 2,000 步＋washout 500 步）：僅檢查有限值、shape、state integrity；**絕對不計算預測分數、不參與模型選擇**。
   - **Calibration**（10 條序列，有效 5,000 步＋washout 500 步）：產出 21 圖殘差張量 $e[g, s, t]$，計算 Calibration pooled NMSE 與 $\Delta$。
   - **Main-Test 嚴格密封**：階段 C 嚴禁讀取、生成或評估 Main-Test 序列（manifest sealed，違規直接 raise PermissionError）。
4. **主要指標與統計推論校準**：
   - 主要指標：$\Delta = (C - \text{NMSE}_{\text{real}}) / C$，其中 $C = \text{median}(\text{NMSE}_{\text{ctrl}_1}, \dots, \text{NMSE}_{\text{ctrl}_{20}})$（偶數 median 取中間兩值平均）。
   - 主要 CI：2,000 次 graph $\times$ sequence 交叉 bootstrap（10 條完整序列 $\times$ 20 控制圖，real 恆定），取 95% 雙側百分位數區間。
   - 推論校準：依 §7.2 / §7.3 殘差縮放（$e_{\text{null}}$ 與 $e_{\text{alt}}$），執行 1,000 cohort Monte Carlo 校準，要求 $L_{\text{power}} \ge 0.80$ 且 $U_{\text{FPR}} \le 0.05$（Exact Clopper–Pearson 97.5% 單側界限）。
5. **Stage C Go / No-Go 放行規則（SPEC v2 §8）**：
   - 動力學健康 gate：全 21 張圖 non-sensory 飽和率 $< 5\%$，初態遺忘 $\ge 38/40$ 配對合格。
   - $\alpha$ 上界碰撞檢查：21 圖中選取 $\alpha = 10^2$ 之比例不得 $> 10\%$（即至多 2 張圖）。
   - 推論校準 gate：$L_{\text{power}} \ge 0.80$ 且 $U_{\text{FPR}} \le 0.05$。
   - 預算 gate：Stage C 實際 GPU 耗時 $\le 4.0$ 小時，全案累計 $\le 8.0$ GPU 小時。
   - **全部通過** $\implies$ `STAGE_C_GO`（允許凍結並排定 Stage D）。
   - **任一不過** $\implies$ `STAGE_C_NO_GO`（有效結案，不得調整超參數重跑）。
6. **停止規則（Stop Rules，立即中斷停機並回報）**：
   - 任何狀態或預測出現 `NaN` 或 `Inf`。
   - 顯存溢出（OOM）。
   - 實際或預估 GPU 耗時超過 4.0 小時。
   - 任何嘗試存取或解封 Main-Test 的操作。
7. **嚴禁修改 README**，回報只寫結果。

---

## 2. 執行指令

在專案根目錄建立輸出目錄並執行 Stage C（使用真實全腦圖 + 20 個 scramble_mixed、CUDA GPU 與 Torch 引擎）：

```bash
mkdir -p research/outputs/v3

/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python -m research.pipeline.g1_stage_c \
    --graph-source real \
    --device cuda \
    --require-cuda \
    --prior-gpu-hours 0.026 \
    2>&1 | tee research/outputs/v3/g1_stage_c.log
```

---

## 3. 交付與回報格式

### Git 操作
在 `remote/g1_stage_c` 分支上 commit 並 push 以下檔案：
- `research/outputs/v3/g1_stage_c_report.json`
- `research/outputs/v3/g1_stage_c.log`
- Commit 訊息結尾請加上：`Co-Authored-By: Codex <noreply@openai.com>`

### 完成回報內容（只寫結果）
依照 `CLAUDE.md` 規則，回報只寫結果（數值、判定、檔案路徑），不寫過程敘述與贅詞。請附上下列數據：

1. **圖完整性與縮放（Graph Provenance）**：
   - 21 張圖載入與 invariants 驗證（PASS / FAIL）
   - 譜半徑縮放驗證（全 21 圖 verified rho，最大相對誤差 $\le 10^{-3}$：PASS / FAIL）
2. **動力學健康檢查（Dynamics Health Gates）**：
   - 飽和 gate（全 21 圖飽和率 $< 0.05$：PASS / FAIL，列出最大飽和率）
   - 遺忘 gate（全 21 圖通過配對 $\ge 38/40$：PASS / FAIL，列出最低通過數）
3. **Ridge 模型擬合與 $\alpha$ 分布**：
   - Real 圖最佳 $\alpha$、Main-Train NMSE、Calibration NMSE
   - 20 張 Control 圖 Calibration NMSE 中位數 $C$
   - $\alpha = 10^2$ 上界碰撞數與比例（$\le 10\%$：PASS / FAIL）
4. **主要對比與推論校準（Primary Contrast & Power/FPR）**：
   - Calibration $\Delta = (C - \text{NMSE}_{\text{real}}) / C$ 點估計
   - 2,000 次交叉 Bootstrap 95% CI：`[ci_lower, ci_upper]`
   - 1,000-cohort Power 下界 $L_{\text{power}}$（$\ge 0.80$：PASS / FAIL）
   - 1,000-cohort FPR 上界 $U_{\text{FPR}}$（$\le 0.05$：PASS / FAIL）
   - MDE 校準綜合判定（PASS / FAIL）
5. **資源消耗與總結判定**：
   - 實際 GPU 耗時（小時 / 分鐘）
   - Stage C 預算判定（$\le 4.0$ GPU h：PASS / FAIL）
   - Stage C 總體結論（`STAGE_C_GO` / `STAGE_C_NO_GO` / `RUN_INVALID`）
   - 若 `STAGE_C_GO`，提示：**可排定 Stage D（一次確認性 21 圖 Main-Test）**
