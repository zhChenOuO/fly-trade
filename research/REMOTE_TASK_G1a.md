# REMOTE_TASK_G1a — 在遠端 GPU 機器執行 G1 第 1 天工程驗證（CPU vs Torch 狀態對齊與 1,000 步 CUDA Timed Pilot）

你是 remote codex。先與 origin 同步 `main` 分支（用系統 git），再建立新分支 `remote/g1_engine`（不要推 main、不要 force push）。

> **硬體需求**：本任務需要在具備 NVIDIA CUDA GPU（如 RTX 4090 或 RTX 5070）與 PyTorch 的環境執行。

---

## 1. 必讀文件與設計原則

- **必讀文件**：
  - `research/G1_SPEC_draft.md`（§3 輸入/stateful/readout、§7 G0b 條件、§8 資料設備工時）
  - `research/REVIEW_v3_codex.md`（Q2、Q3）
  - `research/pipeline/g1_reservoir.py`
  - `research/tests/test_g1_reservoir.py`
- **核心審計與治理原則**：
  1. **本任務屬於工程狀態與成本驗證（G0b Integrity Gate）**。
  2. **絕對不評估任何 NARMA 預測表現**：本任務只驗證動力學引擎的數值精度與硬體耗時，不得執行 Ridge probe 或計算 NMSE。
  3. **絕對不解封 Test split**（嚴禁讀取、生成或評估 Test sequences）。
  4. **嚴禁修改 README**。
  5. **日誌與回報格式硬性規定**：實驗日誌與完成回報只記結果（數值、判定、檔案路徑），絕對不寫過程敘述與贅詞。

---

## 2. 執行流程與精確指令

請在專案根目錄依序執行下列步驟：

### Step 0: 環境檢查
確認 GPU 驅動、CUDA 與 PyTorch 正常運作：
```bash
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Step 1: CPU vs Torch 逐步狀態一致性驗證（小型 2,000 節點合成圖）
驗證在相同稀疏權重矩陣與輸入激勵下，PyTorch GPU CSR sparse mm 與 Scipy CPU 參考引擎之逐步狀態差異在浮點誤差容許範圍內（$\le 10^{-4}$）：
```bash
python -m pytest research/tests/test_g1_reservoir.py -k test_cpu_vs_torch_state_alignment -v -s
```

*若出現任何錯誤或最大絕對誤差 $> 10^{-4}$，立即終止並回報！*

### Step 2: 執行 1,000 步真實 FlyWire CUDA Timed Pilot
使用真實果蠅全腦圖（138,639 神經元，1,509 萬突觸）在目標譜半徑 $\rho=0.95$、批次維度 $B=10$ 條序列下，實測 1,000 步更新耗時與峰值顯存，並外推全量 G1 成本：

```bash
mkdir -p research/outputs/v3

python -m research.pipeline.g1_reservoir \
    --timed-pilot \
    --steps 1000 \
    --batch-size 10 \
    --rho 0.95 \
    --device cuda \
    2>&1 | tee research/outputs/v3/g1_timed_pilot.log
```

---

## 3. 成本外推基準與停止規則

### 總狀態更新數外推基準（SPEC §8, REVIEW_v3_codex Q2）
- 每圖每 $\rho$ 工作點之序列更新數：
  - Train: 10 條 $\times 5,500$ 步 $= 55,000$ updates
  - Val: 3 條 $\times 2,500$ 步 $= 7,500$ updates
  - Test: 10 條 $\times 5,500$ 步 $= 55,000$ updates
  - 小計：**117,500 state-sample updates / 圖 / $\rho$**
- G1 全量規模：**61 張圖**（1 real + 20 scramble_mixed + 20 source-wise shuffle + 20 random endpoints）$\times$ **3 個候選 $\rho$ 網格**（$\{0.90, 0.95, 0.99\}$）
- **G1 總狀態更新數** $= 61 \times 3 \times 117,500 = 21,502,500$ 次 state updates。
- **總 GPU 耗時外推**：
  $$\text{Extrapolated GPU Hours} = \frac{21,502,500 \times t_{\text{update}}}{3600}$$

### 停止規則（Stop Rules，立即終止並回報）
1. **外推時間超標**：外推總 GPU 耗時 $> 8.0$ 小時（預算上限）。
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
   - G1 總更新數（21,502,500）之總外推 GPU 耗時（小時）
   - 預算判定（$\le 8.0$ 小時：PASS / FAIL）
