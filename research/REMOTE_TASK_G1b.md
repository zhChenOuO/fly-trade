# REMOTE_TASK_G1b — 在遠端 GPU 機器執行 G1 Pilot 工作點選擇（16 張圖 × 3 rho，Torch 引擎）

> **前置依賴（Prerequisites）**：
> 本任務書必須在 `REMOTE_TASK_G1a` 成功通過（CPU vs Torch 狀態對齊 PASS，且 1,000 步 CUDA Timed Pilot 外推耗時 $\le 8.0$ GPU 小時）後方可執行。若 G1a 未通過，嚴禁啟動本任務。

你是 remote codex。在遠端 GPU 環境切換至分支 `remote/g1_pilot`（基於已包含 G1a 成果的分支），執行確認性 Pilot 工作點掃描。

- **執行環境 Python 直譯器**：
  `/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python`
- **硬體需求**：NVIDIA CUDA GPU（如 RTX 4090 或 RTX 5070，顯存 $\ge 12$ GB）。

---

## 1. 核心治理與科學審計原則（SPEC §4, §7）

1. **嚴格 Train-Only 評估**：
   - 僅使用預凍結之 10 條 Train 序列（`TRAIN_SEEDS = (1001, ..., 1010)`，有效長度 5,000 步，前置 washout 500 步）。
   - **嚴禁讀取、生成或解封任何 Val 或 Test 序列**（程式若偵測到非 Train seed 將直接拋出 `ValueError`）。
2. **嚴格 Train-Only delayed-product 讀出正控制（SPEC §9, REVIEW_stop_rule Q2）**：
   - Pilot 僅在 5-fold nested sequence CV 的 Train 序列上擬合已知 delayed-product $q[t]=1.5u[t]u[t-9]$ 正控制探針（讀出目標 $r=y[t+1]-q[t]$ 與 $z=r+\beta^* q$），於 outer held-out Train sequences 評估讀出充分性指標 $A_q$。
   - **絕對不得計算或輸出 NARMA 在 Val 或 Test split 上的預測性能，絕對不解封 Test split**。
   - 評估指標包含：(a) 飽和 gate、(b) 初態遺忘 gate、(c) 線性記憶容量 MC（描述性次要診斷，不作硬 gate）、(d) Task-specific $q$ readout positive control gate（$A_q \ge 0.05$、95% CI 下界 $>0$、真 5% oracle probe 檢出率 $\ge 80\%$、錯配 $q$ null FPR $\le 5\%$）。
3. **嵌套 CV 與共用工作點選擇規則（CONFIRMATORY RULE, SPEC §9）**：
   - 候選譜半徑網格：$\rho \in \{0.90, 0.95, 0.99\}$。
   - 5-fold nested Train CV：每折留 2 條 Train 序列作外層評估，內層 8 條序列在 16 張 Pilot 圖上檢驗動力學 gates 並以四家族等權 MC 選出 $\rho_f^*$；外層僅評估該 $\rho_f^*$ 之 $q$ 正控制讀出充分性 gate。
   - 若任一折無合格 $\rho$ 或外層合併 $q$ gate 失敗，宣告 `NO-GO`（`GATE_FAILURE`）並停止。
   - 全部 5 折完成後，依凍結規則以完整 10 條 Train 序列獨立選出最終共用工作點 $\rho^*$（$q$ 結果不反饋工作點選擇）。
   - 若無任何合格 $\rho$，宣告 G1 Pilot NO-GO 並終止主分析。
4. **嚴禁修改 README**。
5. **日誌與回報格式**：只寫結果（數值、判定、檔案路徑），不寫過程敘述與贅詞。

---

## 2. Pilot 圖集規格（共 16 張圖）

Pilot 階段使用 16 張真實規模圖（138,639 神經元，1,509 萬突觸）：
1. **Real FlyWire v783**：1 張（instance 0，固定輸入/讀出索引）。
2. **`scramble_mixed`**：5 張（instances 0..4，種子 1001..1005，overlap $\le 5\%$）。
3. **`weight_shuffle_source`**：5 張（instances 0..4，種子 2001..2005，保持 source 符號分層強度）。
4. **`random_endpoint`**：5 張（instances 0..4，種子 3001..3005，無偏均勻隨機端點）。

---

## 3. 執行指令

在專案根目錄建立輸出目錄並執行 Pilot（使用 TorchG1Reservoir 引擎）：

```bash
mkdir -p research/outputs/v3

/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python -c "
import json
from research.pipeline.flywire_graph import load_flywire_graph
from research.pipeline.graph_variants import (
    well_mixed_scramble,
    source_wise_weight_shuffle,
    random_endpoint_graph,
    compute_graph_sha256,
)
from research.pipeline.g1_reservoir import TorchG1Reservoir
from research.pipeline.g1_pilot import run_g1_pilot

print('Loading base FlyWire v783 connectome...')
base_graph = load_flywire_graph(use_cache=True)
base_graph.meta['sha256'] = compute_graph_sha256(base_graph)

# Provider for 16 real-scale graphs
def remote_pilot_provider(family: str, instance_idx: int):
    if family == 'real':
        return base_graph
    seed = 1000 * (1 if family == 'scramble_mixed' else 2 if family == 'weight_shuffle_source' else 3) + instance_idx + 1
    if family == 'scramble_mixed':
        g = well_mixed_scramble(base_graph, seed=seed, target_overlap=0.05)
    elif family == 'weight_shuffle_source':
        g = source_wise_weight_shuffle(base_graph, seed=seed)
    elif family == 'random_endpoint':
        g = random_endpoint_graph(base_graph, seed=seed)
    else:
        raise ValueError(f'Unknown family: {family}')
    g.meta['sha256'] = compute_graph_sha256(g)
    return g

def gpu_engine_factory(weights, sensory_idx, readout_idx, target_rho, **kwargs):
    return TorchG1Reservoir(
        weights=weights,
        sensory_idx=sensory_idx,
        readout_idx=readout_idx,
        target_rho=target_rho,
        device='cuda',
        **kwargs,
    )

print('Executing G1 Pilot across 16 graphs x 3 rhos on GPU...')
results = run_g1_pilot(
    graph_provider=remote_pilot_provider,
    candidate_rhos=(0.90, 0.95, 0.99),
    engine_factory=gpu_engine_factory,
)

out_path = 'research/outputs/v3/g1_pilot_gpu.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f'G1 Pilot successfully completed. Saved to {out_path}.')
" 2>&1 | tee research/outputs/v3/g1_pilot_gpu.log
```

---

## 4. 耗時預估與停止規則（Stop Rules）

### 預估耗時與硬上限（REVIEW_stop_rule Q4）
- 每張圖每 $\rho$ 工作點之 Train 序列：10 條 $\times 5,500$ 步 $= 55,000$ 狀態更新。
- 16 張 Pilot 圖 $\times 3$ 個 $\rho = 48$ 次模擬。
- 總狀態更新數 $= 48 \times 55,000 = 2,640,000$ 次 state updates。
- 以 Timed Pilot 基準速率（約 $0.15 \sim 0.30$ 毫秒 / update）：
  $$\text{Extrapolated GPU Reservoir Time} \approx \frac{2,640,000 \times 0.00025\text{ s}}{3600} \approx 0.18 \sim 0.35\text{ GPU 小時（約 11–21 分鐘）}$$
- CPU 圖生成與快取開銷：約 30–50 分鐘。
- **預計總耗時：約 45–70 分鐘**。
- **可行性 Pilot 硬上限（Hard Cap）：預估或實際執行超過 3.0 GPU 小時即停並回報**（Codex 交叉詰問 IV 建議僅做有硬上限的可行性 pilot，不投入完整 8 GPU 小時）。

### 停止條件與分類（Stop Rules, SPEC §7, §9, REVIEW_stop_rule Q5）

1. **有效 Gate 失敗（Valid Gate Failure $\implies$ 結案並回報，不重跑、不調參）**：
   - 任一 nested CV outer fold 無合格 $\rho$、任一候選 $\rho$ 無法在 16 張圖通過飽和或遺忘 gate。
   - Task-specific $q$ positive-control gate 失敗（pooled $A_q < 0.05$、95% CI 下界 $\le 0$、oracle power $< 80\%$、或錯配 $q$ null FPR $> 5\%$）。
   - 以上均屬確認性執行之有效科學結果，宣告 `NO-GO`（`GATE_FAILURE`）並正式結案，**不得更換 readout、不得調整 input/rho/alpha/seeds，不得以新版本挽救同一命題**。
2. **無效執行例外（Run-Invalidating Exceptions, 引用 `research/REVIEW_stop_rule.md` Q5）**：
   - 僅限以下三類可客觀證明之缺陷，方可標記為 `INVALID_RUN` 並另版修復重跑：
     (a) **結果盲化的程式/數值錯誤**：以獨立 fixture 或參考實作證明之索引錯誤、數值 solver bug，且尚未查看主要性能結果；
     (b) **外部執行中斷**：OS/GPU 驅動崩潰、硬體電源中斷，且 Test 未解封；可重現之 deterministic OOM 屬容量限制，非中斷例外；
     (c) **預註冊或資料邊界失效**：圖檔 SHA256 不符、Train/Test 邊界洩漏。
   - 任何重跑均須保留原 run 之完整 artifacts、標記 `INVALID_RUN`、記錄發現時間與修復版本 hash，不可覆蓋或混淆舊記錄。
3. **耗時超支**：執行超過 3.0 GPU 小時仍未完成即終止並回報。
4. **違規操作**：任何嘗試評估 Val/Test NARMA 預測表現或接觸 Test split 之行為。

---

## 5. 交付與回報格式

### Git 操作
在 `remote/g1_pilot` 分支 commit 並 push：
- `research/outputs/v3/g1_pilot_gpu.json`
- `research/outputs/v3/g1_pilot_gpu.log`
- Commit 訊息結尾請加上：`Co-Authored-By: Codex <noreply@openai.com>`

### 完成回報內容（只寫結果）
1. **Pilot 總體結果**：
   - 候選 $\rho$ 合格狀態（0.90 / 0.95 / 0.99 各自為 PASS / FAIL）
   - 各 $\rho$ 之四家族等權平均 MC
   - 選定之共用 $\rho$（或 NO-GO 判定及失敗原因）
2. **各家族表現摘要**：
   - real: MC, 飽和比例, 遺忘通過數
   - scramble_mixed: 平均 MC, 飽和比例, 遺忘通過數
   - weight_shuffle_source: 平均 MC, 飽和比例, 遺忘通過數
   - random_endpoint: 平均 MC, 飽和比例, 遺忘通過數
3. **實測耗時與資源**：
   - 總執行耗時（秒 / 分鐘）
   - 峰值 GPU 顯存（MB）
   - 產出檔案路徑
