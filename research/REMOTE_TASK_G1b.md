# REMOTE_TASK_G1b — 在遠端 GPU 機器執行 G1 v2 階段 B 真圖能力診斷與負控制

> **前置依賴（Prerequisites）**：
> 本任務書必須在 `REMOTE_TASK_G1a` 成功通過（CPU vs Torch 狀態對齊 PASS，且 1,000 步 CUDA Timed Pilot 驗收合格，外推階段 B $\le 1.0$ GPU 小時、全案 $\le 8.0$ GPU 小時）後方可執行。若 G1a 未通過，嚴禁啟動本任務。

你是 remote codex。在遠端 GPU 環境切換至分支 `remote/g1_stage_b`（基於已包含 G1a 成果的分支），執行確認性階段 B 能力診斷與負控制。

- **執行環境 Python 直譯器**：
  `/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python`
- **硬體需求**：NVIDIA CUDA GPU（如 RTX 4090 或 RTX 5070，顯存 $\ge 12$ GB）。

---

## 1. 核心治理與科學審計原則（SPEC v2 §4, §6, §8, §9）

1. **唯一單圖真實連接體評估**：
   - 階段 B 僅評估 1 張真實 FlyWire v783 全腦圖（138,639 神經元，1,509 萬突觸）。
   - 固定動力學：原始 $u[t] \sim \text{Uniform}[0, 0.5]$（不置中）、固定譜半徑 $\rho=0.95$、固定 leak=0.5、recurrent bias=0、DN-only 讀出（1,291 節點）。
   - **完全撤回舊版 $\rho$ 網格搜尋、多家族選擇與舊 $q=1.5u[t]u[t-9]$ 讀出 gate**。
2. **直接能力診斷（SPEC v2 §6）**：
   - 兩個明確能力目標：
     - $d_1[t] = v[t-9]$
     - $d_2[t] = v[t] \cdot v[t-9]$，其中 $v[t] = u[t] - 0.25$（僅用於定義 target，不得將 $v$ 作為 reservoir 輸入）。
   - **Diagnostic-fit split**（10 條序列，每條有效 2,000 步＋washout 500 步）：
     - 執行 5-fold sequence CV，在固定網格 $\alpha \in \{0, 10^{-8}, 10^{-7}, \dots, 10^2\}$ 選出 pooled OOF NMSE 最低之 $\alpha$，再以全部 10 條 fit 序列 refit。
     - **上端點碰撞檢查**：任一 head 選取 $\alpha = 10^2$ 視為 no-go（不得擴張網格）。
   - **Diagnostic-check split**（10 條序列，每條有效 2,000 步＋washout 500 步）：
     - 一次性在 check 序列評估 $R^2 = 1 - \text{SSE} / \text{SST}$。
     - **能力通過條件（AND 判定）**：$d_1$ 與 $d_2$ **兩者皆必須**達 $R^2 \ge 0.10$ 且 2,000 次完整 sequence bootstrap 之 95% CI 下界 $> 0$。
3. **負控制（SPEC v2 §6）**：
   - 在 Diagnostic-fit 與 check split 內分別執行循環移位一條（第 $s$ 條 DN states 搭配第 $s+1$ 條 target，末條回接首條）。
   - 以錯配 fit target 重新執行 5-fold sequence CV 選 $\alpha$ 並 refit，在 check 上計算相對常數基準之改善量 $G = (\text{MSE}_{\text{constant}} - \text{MSE}_{\text{model}}) / \text{MSE}_{\text{constant}}$。
   - **負控制通過條件**：兩目標的 sequence bootstrap 95% CI 下界均不得 $> 0$（若任一 $G$ 下界 $> 0$ 則宣告負控制失敗／no-go）。
4. **動力學健康檢查（SPEC v2 §4.2）**：
   - 飽和檢查：washout 後非 sensory 節點 $|x| > 0.9$ 的比例必須 $< 5\%$。
   - 初態遺忘檢查：40 對配對（零初態 vs $\text{Uniform}[-0.1, 0.1]$），至少 38 對在 $t=500$ 步時正規化 RMS 差異 $< 10^{-3}$。
5. **預算硬上限與終止規則（SPEC v2 §8）**：
   - 階段 B 硬上限：$\le 1.0$ 工作天（8 人工工時）／$\le 1.0$ GPU 小時（全案總上限 4 工作天／8 GPU h）。
   - 若階段 B GPU 耗時預計或實際超過 1.0 GPU 小時，立即停止（`BUDGET_EXCEEDED`）。
   - 有效執行的能力／負控制／健康失敗均為 `VALID_GATE_FAIL`，**直接結案，不得調整超參數重跑**。
6. **嚴格保護後續 Splits**：
   - 嚴禁讀取、生成或評估 Main-Train、Main-Val、Calibration 或 Main-Test 序列。
7. **嚴禁修改 README**，回報只寫結果。

---

## 2. 執行流程與指令

在專案根目錄建立輸出目錄並執行階段 B（使用 TorchG1Reservoir 引擎）：

```bash
mkdir -p research/outputs/v3

/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python -c "
import json
import time
from pathlib import Path
from research.pipeline.flywire_graph import load_flywire_graph
from research.pipeline.graph_variants import compute_graph_sha256
from research.pipeline.g1_manifest import create_g1_manifest, generate_split_sequences
from research.pipeline.g1_reservoir import TorchG1Reservoir
from research.pipeline.g1_runner import BudgetLedger

t0 = time.time()
print('Initializing Stage B runner and manifest...')
manifest = create_g1_manifest('.')
manifest.advance_stage('B')

print('Loading base FlyWire v783 connectome...')
graph = load_flywire_graph(use_cache=True)
graph.meta['sha256'] = compute_graph_sha256(graph)

print('Generating Diagnostic-fit sequences (10 streams x 2,500 steps)...')
fit_seqs = generate_split_sequences('diagnostic-fit', manifest)

print('Generating Diagnostic-check sequences (10 streams x 2,500 steps)...')
check_seqs = generate_split_sequences('diagnostic-check', manifest)

print('Running Stage B execution on CUDA GPU...')
# GPU Reservoir simulation and Ridge evaluation
# (Outputs saved to research/outputs/v3/g1_stage_b_report.json)
" 2>&1 | tee research/outputs/v3/g1_stage_b.log
```

---

## 3. 完成回報內容（只寫結果）

依照 `CLAUDE.md` 規則，回報只寫結果（數值、判定、檔案路徑），不寫過程敘述與贅詞。請附上下列數據：

1. **動力學健康檢查（Health Gates）**：
   - 飽和比例（Saturation Ratio）：數值與判定（$< 0.05$：PASS / FAIL）
   - 初態遺忘通過數（Forgetting Passed Pairs）：數值（$\ge 38 / 40$：PASS / FAIL）
2. **能力診斷（Capability Diagnostics）**：
   - 延遲目標 $d_1$: 最優 $\alpha$、Check $R^2$ 點估計、95% CI 下界、判定（$R^2 \ge 0.10$ 且 CI 下界 $> 0$：PASS / FAIL）
   - 乘積目標 $d_2$: 最優 $\alpha$、Check $R^2$ 點估計、95% CI 下界、判定（$R^2 \ge 0.10$ 且 CI 下界 $> 0$：PASS / FAIL）
   - 能力門檻綜合判定（AND 判定：PASS / FAIL）
3. **負控制（Negative Controls）**：
   - 錯配 $d_1$: Check $G$ 點估計、95% CI 下界、判定（CI 下界 $\le 0$：PASS / FAIL）
   - 錯配 $d_2$: Check $G$ 點估計、95% CI 下界、判定（CI 下界 $\le 0$：PASS / FAIL）
   - 負控制綜合判定（PASS / FAIL）
4. **階段 B 資源消耗與總結**：
   - 實際 GPU 耗時（小時 / 分鐘）
   - 階段 B 預算判定（$\le 1.0$ GPU h：PASS / FAIL）
   - 階段 B 總體結論（STAGE_COMPLETED / VALID_GATE_FAIL / RUN_INVALID）
