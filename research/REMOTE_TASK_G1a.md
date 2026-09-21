# REMOTE_TASK_G1a — 在遠端 GPU 機器執行 G1 v2 階段 B 引擎與成本驗證（CPU vs Torch 狀態對齊與 1,000 步 CUDA Timed Pilot）

你是 remote codex。先與 origin 同步 `main` 分支（用系統 git），再建立新分支 `remote/g1_engine`（不要推 main、不要 force push）。

> **執行環境與直譯器**：
> Python 直譯器：`/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python`
> 硬體需求：NVIDIA CUDA GPU（RTX 4090 或 RTX 5070 12GB）與 PyTorch。

---

## 1. 必讀文件與設計原則

- **必讀文件**：
  - `research/G1_SPEC_v2_draft.md`（§4 固定輸入/動力學、§8 預算、§9 valid fail/invalid）
  - `research/pipeline/g1_reservoir.py`
  - `research/pipeline/g1_manifest.py`
  - `research/pipeline/g1_runner.py`
- **核心審計與治理原則（SPEC v2）**：
  1. **輸入與動力學凍結**：注入原始 $u[t] \sim \text{Uniform}[0, 0.5]$（不減 0.25）、固定譜半徑 $\rho=0.95$、leak=0.5、recurrent bias=0、DN-only 讀出（1,291 節點）。撤回 $\rho$ 搜尋與候選網格。
  2. **對照組唯一化**：全量 G1 v2 僅 21 張圖（1 real + 20 `scramble_mixed`），撤回 weight shuffle、random endpoint 及多家族選擇。
  3. **絕對不評估任何 NARMA 預測表現**：本任務只驗證動力學引擎的數值精度與硬體耗時，不得執行 Ridge probe 或計算 NMSE。
  4. **嚴格保護 Test split**：階段 A–C 嚴禁生成、讀取或評估 Main-Test 序列（manifest 處於 sealed 狀態，違規存取直接 raise PermissionError）。
  5. **嚴禁修改 README**。
  6. **回報格式硬性規定**：只寫結果（數值、判定、檔案路徑），不寫過程敘述與贅詞。

---

## 2. 執行流程與精確指令

請在專案根目錄依序執行下列步驟：

### Step 0: 環境檢查
確認 GPU 驅動、CUDA 與 PyTorch 正常運作：
```bash
nvidia-smi
/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Step 1: CPU vs Torch 逐步狀態一致性驗證（小型 2,000 節點合成圖）
驗證在相同稀疏權重矩陣與原始 $u[t]$ 輸入激勵下，PyTorch GPU CSR sparse mm 與 Scipy CPU 參考引擎之逐步狀態差異在浮點誤差容許範圍內（$\le 10^{-4}$）：
```bash
/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python -m pytest research/tests/test_g1_reservoir.py -k test_cpu_vs_torch_state_alignment -v -s
```
*若出現任何錯誤或最大絕對誤差 $> 10^{-4}$，立即終止並回報！*

### Step 2: 執行 1,000 步真實 FlyWire CUDA Timed Pilot
使用真實果蠅全腦圖（138,639 神經元，1,509 萬突觸）在固定目標譜半徑 $\rho=0.95$、批次維度 $B=10$ 條序列下，實測 1,000 步更新耗時與峰值顯存，並外推 G1 v2 全量成本。
（引擎內部強制使用 `select_r16_indices` 僅對有限視網膜座標之 R1-6 神經元注入電流、排除 R7/R8，並使用線上串流飽和追蹤 `track_saturation=True` 避免配置全腦狀態；加上 `--require-cuda` 確保嚴格在 GPU 上執行）：

```bash
mkdir -p research/outputs/v3

/var/home/zh/Documents/github/fly-trade/.worktrees/phase0-gpu/.venv/bin/python -m research.pipeline.g1_reservoir \
    --timed-pilot \
    --steps 1000 \
    --batch-size 10 \
    --rho 0.95 \
    --device cuda \
    --require-cuda \
    2>&1 | tee research/outputs/v3/g1_timed_pilot.log
```

---

## 3. G1 v2 成本外推基準與停止規則（SPEC v2 §8）

### 預算總額與階段硬上限
- **v2 總上限**：4 工作天（32 人工工時）／8.0 GPU 小時（全階段累計硬上限，不重置、不挪用）。
- **階段預算分配**：
  - **階段 A**：0.5 工作天／0.0 GPU h（已於本地 CPU 驗收完成）。
  - **階段 B**：$\le 1.0$ 工作天／$\le 1.0$ GPU h（單圖 real 能力診斷、計時與健康檢查）。
  - **階段 C**：$\le 1.0$ 工作天／$\le 4.0$ GPU h（21 圖 Main-Train、Main-Val、Calibration 與 1,000-cohort MC 校準）。
  - **階段 D**：$\le 1.5$ 工作天／$\le 3.0$ GPU h（一次確認性 21 圖 Main-Test）。
- **GPU 小時計算法**：包含測試、warmup、失敗與合法重跑之實際單機 wall time，不是只計 kernel time。

### v2 總狀態更新數外推基準
- **圖數量**：固定 21 張（1 real + 20 scramble_mixed），固定單一工作點 $\rho=0.95$。
- **序列更新數**：
  - 階段 B：1 real $\times 20$ 條 Diagnostic (fit+check) $\times 2,500$ 步 $= 50,000$ updates。
  - 階段 C：21 圖 $\times$ (Main-Train 55,000 + Main-Val 7,500 + Calibration 55,000) $= 21 \times 117,500 = 2,467,500$ updates。
  - 階段 D：21 圖 $\times$ Main-Test 55,000 $= 1,155,000$ updates。
  - **v2 總狀態更新數** $\approx 3,672,500$ 次 state updates。
- **總 GPU 耗時外推**：
  $$\text{Extrapolated GPU Hours} = \frac{3,672,500 \times t_{\text{update}}}{3600}$$

### 停止規則（Stop Rules，立即終止並回報）
1. **外推時間超標**：階段 B 外推總 GPU 耗時 $> 1.0$ 小時，或全案外推總 GPU 耗時 $> 8.0$ 小時。
2. **狀態不一致**：Step 1 之 CPU vs Torch 最大絕對誤差 $> 10^{-4}$。
3. **數值異常**：任何狀態更新出現 `NaN` 或 `Inf`。
4. **顯存溢出（OOM）**：顯存使用超過 GPU 實體上限。
5. **違規操作**：任何嘗試評估 NARMA 預測分數或碰觸 Test split。

---

## 4. 交付與回報格式

### Git 操作
在 `remote/g1_engine` 分支上 commit 並 push 以下檔案：
- `research/outputs/v3/g1_timed_pilot.log`
- Commit 訊息結尾請加上：`Co-Authored-By: Codex <noreply@openai.com>`

### 完成回報內容（只寫結果）
依照 `CLAUDE.md` 規則，回報只寫結果（數值、判定、檔案路徑），不寫過程敘述與贅詞。請附上下列數據：
1. **Step 1 狀態對齊結果**：
   - 最大 Readout 狀態誤差（Max Readout Error）
   - 最大全狀態誤差（Max Final State Error）
   - 對齊判定（PASS / FAIL）
2. **Step 2 Timed Pilot 實測與外推數據**：
   - 1,000 步 $\times 10$ 序列實測耗時（秒）
   - 單步耗時（毫秒 / step）
   - 單狀態更新耗時（毫秒 / update）
   - 峰值顯存使用（Peak VRAM MB）
   - G1 v2 總更新數（3,672,500）之總外推 GPU 耗時（小時）
   - 預算判定（階段 B $\le 1.0$ 小時且全量 $\le 8.0$ 小時：PASS / FAIL）
