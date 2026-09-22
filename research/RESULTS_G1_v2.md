# G1 v2 結果（connectome-as-reservoir，合成基準）

判定：**VALID_GATE_FAIL（能力 gate）**。依 `G1_SPEC_v2_draft.md` §2 預先寫定的結論：此固定輸入、動力學與 DN 線性讀出未達預定能力下限；本版不進入拓樸確認性實驗（21 圖 NARMA10 對照）。不得解讀為「拓樸無效」或「連接體無資訊」。

## 設定
真實 FlyWire v783（138,639 神經元 / 15,091,983 突觸）；輸入原始 `u[t]~U[0,0.5]` 注入 R1-6（7,932 個，排除 R7/R8）；rho=0.95、leak=0.5；讀出固定 DN（buy_idx∪sell_idx=1,291 維）；Diagnostic-fit/check 各 10 條序列（2,000 步+500 washout）；Ridge、5-fold sequence CV、α∈{0,1e-8,…,1e2}。

## 結果（Stage B，真圖）
| 目標 | 最佳 α | Check R² | 95% CI | 判定 |
|---|---|---|---|---|
| `d1 = v[t-9]`（線性延遲） | 0（無正則化） | 0.5775 | [0.5692, 0.5848] | PASS |
| `d2 = v[t]·v[t-9]`（非線性乘積） | 100（撞上界） | -0.0000037 | [-0.00039, 0.00006] | **FAIL** |
| 綜合（AND） | | | | **FAIL** |

健康 gate：飽和 0%、遺忘 40/40、無非有限值，全 PASS。Oracle 正控制 R²=1.0，PASS。

負控制：`d1` 錯配 G=0.0000213，CI [0.0000011, 0.0000427]（下界 >0，判定 FAIL，但效應量比 `d1` 訊號小 5 個量級，20,000 樣本點下屬統計檢定力過高導致的邊界情況，未深究機制，不影響整體結論——`d2` 已獨立判定失敗）。`d2` 錯配 PASS。

GPU 成本：Stage B 實測 0.0138 GPU 小時；累計（含 G1a 三次嘗試）0.026 GPU 小時，預算 PASS，未觸及上限。

證據：`research/outputs/v3/g1_stage_b_real_cuda.json`、`.log`；`research/outputs/v3/g1_timed_pilot.log`（Stage B 硬體驗證）。

## Stage A/B 過程中修正的 3 個 run-invalidating bug（與本結果無關，已修復並記錄）
1. `compute_spectral_radius` 對近簡併特徵值叢集用 k=1 ARPACK 不穩定 → 改 k=6 取最大幅值（`da4d797`）。
2. `test_g1_reservoir.py` 缺少 `import torch`，Mac 因 skip 而未發現，GPU 機器才會炸（`ea59c65`）。
3. CLI timed-pilot 外推公式殘留 pre-v2 的 61 圖×3 rho 舊數字（`6b8c264`，僅列印用，未影響任何預算判定）。

## 對研究路線的影響
- G1（連接體儲庫的一般計算能力）：結案，不進 Stage C/D。
- `PREREQUISITES_v3.md` 的 G2（BTC 已實現波動度）、G3（微結構）：維持不解鎖，因為它們的先決條件是 G1 通過。
- 市場預測路線（`RESULTS_market_v1.md`）：獨立結論不受影響，本來就已封存。
