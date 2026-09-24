# G1 v2 結果（connectome-as-reservoir，合成基準）

## 最終判定（2026-09-24）：STAGE_C_NO_GO，G1 結案

主測（Stage C，21 圖：1 real + 20 scramble_mixed，NARMA10）已完整執行，結果為 **NO-GO**。依預先議定的停損規則（`REVIEW_stop_rule.md`），有效 gate 失敗即結案，不換 readout／rho／alpha／seed／效果門檻重跑。**G1（connectome-as-reservoir 一般計算能力）研究到此為止。**

### 結果
| 指標 | 數值 |
|---|---|
| real 圖 Calibration NMSE | 0.5120 |
| 20 個 scramble_mixed 圖 NMSE 範圍 | 0.4025 – 0.4224（中位數 0.4086） |
| Δ = (中位數−real)/中位數 | **−0.2530**（real 比全部 20 個對照都差，不是「沒有優勢」，是「輸給每一個」） |
| 95% CI（2,000 次交叉 bootstrap） | [−0.2747, −0.2341]（完全不含 0，更不含 +5%） |
| Health / α 撞上界 / Power / Budget gate | 全 PASS（0/21 圖撞 α 上界；health 全過） |
| FPR gate | **FAIL**（點估計 0.102，97.5% 上界 0.1224，門檻 ≤0.05；空校準本身超出容許誤判率，獨立於 Δ 為負這件事） |
| GPU 成本 | Stage C 1.747 小時；累計 1.773 / 8 小時 |

Claude 已獨立核對：20 個控制圖的 α 全部是 0（無正則化）、NMSE 緊密分布在 0.40–0.42，real 的 0.512 明顯高於全部 20 個控制圖，不是邊界情況。

### 判讀
- **不是「real 沒有優勢」，是「real 在此任務下比隨機重接還差」。** 這本身是有資訊量的負結果。
- **FPR gate 另外失敗**（獨立於 Δ）：空校準（null calibration）的假陽性率 0.102 超出 0.05 門檻，代表這套統計程序本身在「真的沒有差異」時容易誤判有差異。這不會反過來讓 Δ=−0.253 這個强烈負值變得可疑——CI 完全落在負值區，遠離 0，不太可能是 FPR 過鬆造成的假訊號（那只會讓「real 更好」的誤判機率增加，不會系統性地讓 real 看起來更差）。但這代表若未來要重用同一套統計框架，需要另外查 FPR 偏高的機制。
- 依規格 §2：結論僅限「在此固定 specimen、固定輸入/讀出/動力學、此組 20 個 scramble_mixed 對照下，real 未證明優於對照，且觀測值方向為劣」，不代表其他個體、其他 benchmark、或市場預測能力。

### 過程記錄（本次執行前，連續攔下 4 輪程式/文件缺陷，未浪費 GPU 時間）
1. MC 校準內層 bootstrap 次數被限制在 500，與規格要求的 2,000 不符（`dbd8e85`）。
2. 任務文件的 scramble 種子過時、Main-Val/Calibration 缺飽和 gate、GPU 帳本沒有累計既有用量（`41cba30`）。
3. `TorchG1Reservoir` 收到不支援的 `dtype` 參數，Mac 本機測試從未真正跑過 Torch 路徑而漏掉此問題（`4389d3e`）。
4. Git 分支分歧（先前失敗紀錄的 commit 造成 `--ff-only` 失敗），改開新分支 `remote/g1_stage_c_v2` 解決，未 force push、未遺失任何資料。

證據：`research/outputs/v3/g1_stage_c_report.json`、`.log`。

## 更正紀錄（2026-09-22，Codex 獨立審查後，Claude 已親自驗證）
原判定「VALID_GATE_FAIL，DN 線性讀出無非線性能力」**已撤回**。Codex 唯讀審查發現 Blocker 級設計缺陷：`d2 = v[t]·v[t−9]` 這個 target 在數學上不可能被 Stage B 使用的讀出狀態預測到，與拓樸、讀出、演算法能力全部無關。Claude 已獨立讀碼與實測驗證此缺陷成立（見下）。**判定改為 STAGE_B_INVALID（診斷設計缺陷），結論未定，非「拓樸/連接體/監督式學習無能力」。**

## 設定
真實 FlyWire v783（138,639 神經元 / 15,091,983 突觸）；輸入原始 `u[t]~U[0,0.5]` 注入 R1-6（7,932 個，排除 R7/R8）；rho=0.95、leak=0.5；讀出固定 DN（buy_idx∪sell_idx=1,291 維）；Diagnostic-fit/check 各 10 條序列（2,000 步+500 washout）；Ridge、5-fold sequence CV、α∈{0,1e-8,…,1e2}。

## 缺陷詳情（Blocker，已獨立驗證）
`g1_reservoir.py` 的模擬迴圈：`S = W @ x[t]`（用舊狀態）→ 只對 sensory 列做 `S[sensory] += u[t]` → `x[t+1] = decay·x[t] + leak·tanh(S)`，並把這個 `x[t+1]` 存成 `out_states[:, t, :]`。對任何非 sensory 節點 j（含全部 1,291 個 DN）：`x_{t+1,j}` 完全不含 `u[t]`，只是 `x[t]`（即 `u[0..t-1]` 的函數）的函數。

`g1_v2.py::build_diagnostic_targets` 把 `d2[t] = v[t]·v[t-9]` 對齊到同一個索引 `t` 的 `out_states[:, t, :]`。但 `v[t]` 是均值為 0、獨立於歷史的雜訊項，而讀出狀態根本還沒看過 `u[t]`。因此對任何 predictor `f`（不限線性）：

`E[d2[t] | F_(t-1)] = v[t-9]·E[v[t]] = 0`

母體最佳 R² 恆為 0，跟連接體、拓樸、讀出方式、Ridge 或任何模型的能力完全無關——這個 target 在此設計下**不可能被任何方法預測**。

**Claude 獨立驗證**（不只讀程式，另跑實測）：
- R1-6（7,932 個）與 DN（1,291 個）索引完全不重疊。
- 用真實圖的稀疏權重矩陣直接查詢：**R1-6 到 DN 之間沒有任何一條直接突觸**（nnz=0）。也就是說 `u[t]` 連經過 1 跳都到不了 DN，`d2` 需要的「當下」資訊在讀出當下必然不存在，不只是慢一步的問題。

`d1 = v[t-9]`（9 步前的舊資訊）不受此缺陷影響，因為舊資訊有充分時間傳播到 DN；其 R²=0.5775 應視為真實觀測（DN 讀出保有強線性記憶），但尚不構成「拓樸優於隨機」的證據（還沒做真實圖對隨機圖的對照）。

## Stage B 原始觀測（保留為封存執行紀錄，不重新賦予科學意義）
| 目標 | 最佳 α | Check R² | 95% CI | 原判定 |
|---|---|---|---|---|
| `d1 = v[t-9]`（線性延遲） | 0（無正則化） | 0.5775 | [0.5692, 0.5848] | PASS（可信觀測） |
| `d2 = v[t]·v[t-9]`（非線性乘積，**設計上不可達**） | 100（撞上界） | -0.0000037 | [-0.00039, 0.00006] | 無效，非科學發現 |

健康 gate：飽和 0%、遺忘 40/40、無非有限值，PASS——但 oracle 正控制是「target 對自己算 R²」的恆等式，未驗證 lag/對齊，不構成獨立驗證。

負控制：`d1` 錯配 G=0.0000213，CI 下界 0.0000011（技術上判 FAIL）。Codex 審查指出循環移位配對可能不是真正獨立單位（pair s 用 X_s 配 y_(s+1)，y_(s+1) 與 X_(s+1) 同源），根因未定，不可再用「檢定力過高」帶過。

GPU 成本：Stage B 實測 0.0138 GPU 小時；累計 0.026 GPU 小時，預算 PASS，未觸及上限（此結論不受本次更正影響）。

證據：`research/outputs/v3/g1_stage_b_real_cuda.json`、`.log`；獨立審查見 `research/REVIEW_G1_v2_final_audit.md`。

## Stage A/B 過程中修正的 3 個 run-invalidating bug（與本次缺陷不同，已修復並記錄）
1. `compute_spectral_radius` 對近簡併特徵值叢集用 k=1 ARPACK 不穩定 → 改 k=6 取最大幅值（`da4d797`）。
2. `test_g1_reservoir.py` 缺少 `import torch`，Mac 因 skip 而未發現，GPU 機器才會炸（`ea59c65`）。
3. CLI timed-pilot 外推公式殘留 pre-v2 的 61 圖×3 rho 舊數字（`6b8c264`，僅列印用，未影響任何預算判定）。

## 二次更正（同日，Codex 追問後撤回 L=9 建議）
Codex 原建議「改成 L=9 延遲讀出才能跑 Stage C」已被自己撤回。經 Claude 追問並獨立驗算：Stage C 的判定是**比較式**指標 `Δ=(median(NMSE_scramble)−NMSE_real)/median(NMSE_scramble)`，不是絕對 R² 門檻。用正交分解可證：`y[t+1]=m_t+ε_t`（`m_t` 為可預測歷史項，`ε_t=1.5(u[t]−0.25)u[t−9]` 為不可約 innovation，`E[ε_t|F_(t−1)]=0`），因此任一圖的母體風險 `R_g=1/256+A_g`（`A_g` 為對可預測部分 `m_t` 的誤差）。1/256 這個共同不可約項在 Δ 的**分子**完全抵消（只留在分母，壓縮相對改善幅度，不會讓比較失效）。Claude 已獨立驗算 1/256 這個數字。

**結論：Stage C 不需要 L=9 或任何延遲修正，維持原本 `x[t+1]→y[t+1]` 對齊即可執行比較式主測。** d2 的「全歸零」情況（純外生雜訊乘積，無可預測分量）不能類推到完整 NARMA10 target（含歷史項 `m_t`，是可預測的）。

固定延遲 L=9 本身也有未解決的公平性風險（scramble 與 real 到 DN 的傳播速度未驗證是否相同，固定 L 可能反映傳播速度差異而非非線性運算品質），這條路徑已撤回，不再考慮。

完整討論見 `research/DISCUSSION_next_target.md`。

## Stage C 前的最小工程驗收清單（不含延遲修正，範圍已縮小）
1. **獨立 oracle**：現有 oracle 正控制是 target 對自己算 R²（恆等式），需換成獨立重算的 target 與 runner 輸出逐項比對。
2. **因果 fixture**（保留作一般完整性驗證，非 Stage C 前置條件）：sensory/DN 不重疊的小圖，獨立手算驗證同列 DN 不含 `u[t]`；用於證明程式忠實對齊規格的時序定義，不是用來決定要不要延遲。
3. **Ridge tie-break 與 spec 一致**：目前 `g1_bench.py` 用相對差 ≤1e-9 當同分，規格要求「數值完全相同」，需對齊（P2，Codex 審查發現）。
4. **`d1` 負控制的微小異常**：循環移位配對可能不是真正獨立單位，根因未定；與 Stage C 的 power/FPR 統計機制不同（Stage C 用的是已獨立驗證過的 null/alt residual 縮放，不受此問題影響），可平行處理不阻擋 Stage C。
5. **Diagnostic-fit/check 的既有結果**：`d1`/`d2` 數字保留為封存觀測，不重跑、不追加新 seeds（Stage B 的能力 gate 已撤除，不再是 Stage C 的前置條件）。

## 對研究路線的影響
- G1：**Stage B 的能力 gate 已撤除，不再是 Stage C 前置條件**。Stage C（21 圖 Main-Train/Val/Calibration + power/FPR 校準）可在完成上述工程驗收清單後執行，原時序對齊（`x[t+1]→y[t+1]`）不需修改。
- `PREREQUISITES_v3.md` 的 G2、G3：維持不解鎖，因為 Stage C/D 尚未執行，G1 仍未有正式結論。
- 市場預測路線（`RESULTS_market_v1.md`）：不受影響，本來就已獨立封存。
