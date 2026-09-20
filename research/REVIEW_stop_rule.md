# G1 Pilot 停損規則交叉詰問 IV

## 總判斷

**有條件同意。** 有效執行後的預註冊 gate 失敗，應結束本版 G1 confirmatory 路徑；不啟動替代 readout、不調 input／rho／alpha、不以新 seed 或新版本挽救同一命題。若 run 因可證明的程式／資料／環境缺陷而無效，則屬另一類事件，可按下列盲化與版本規則修復；不得把無效 run 改標成 gate 通過。

## Q1. 硬停損與替代 readout

建議維持本版硬停損，不預先加入 v2 fallback。固定 1,291-DN readout 是目前 G1 estimand 的一部分；切換至全狀態或隨機投影後，問題會變成「候選 readout 集合中哪一個較好」，不再是固定 DN readout 下的拓樸對比。即使替代 readout 只用 Train、跨 graph family 共用一條選擇規則且 Test 只評一次，也必須預註冊讀出集合、選擇器與新的 estimand，並承擔 Pilot winner's curse；它不能替固定 DN gate 失敗作解釋。

有限替代路徑在程序上可以是未來的獨立研究，但現在加入會回應已知 DN MDE 衰減結果，容易把本版的失敗轉成選擇面更廣的確認性問題。可在新研究中比較最多一個 Train-only 選定的 readout，但須另有獨立 Test，不能當本版 no-go 的自動延伸。故本版不應保留「失敗後換一個」的救援選項。

## Q2. 讀出充分性與低低 MC

**先前 `MC_DN / median(MC_random_projection) ≥ 1` 單獨作為硬 gate 不成立。** 若兩者都接近零，比例仍可能通過；而 20 個隨機投影的中位數也不是任務訊號的充分性上界。它只能描述 DN 相對於 generic projection 的線性記憶，不足以放行 NARMA10。

建議用一個任務特定、固定 readout 的絕對 positive control：在 Train-only outer sequence folds 內，以 `q[t]=1.5u[t]u[t−9]` 作為 NARMA10 已知的 delayed-product term，令 `r[t]=y[t+1]−q[t]`。inner Train 只用真實圖校準一個共同 `β`，使已知 oracle correction `r̂+βq` 對 `r` readout 的 NMSE 改善恰為 5%；同一個 `β` 套用所有 graph families。外層分別以原本固定 1,291 DN states 讀出 `r` 與 `z=r+βq`，不把 q 加進 predictor，並比較 `A_q=(MSE(z,r̂)−MSE(z,ẑ))/MSE(z,r̂)`。讀出只在 `A_q` pooled point estimate ≥0.05、單側 95% block-bootstrap 下界 >0、真 5% oracle probe 的檢出率 ≥80%、錯配 q 的 null FPR ≤5% 時通過。這個 5% 是沿用 G1 最小相對 NMSE 改善的工程門檻，並非文獻定律或與拓樸 Δ 完全等價。

錯配 q 固定採 frozen `TRAIN_SEEDS` 順序循環移位一位，不能挑較容易的 null；`r` head 選出的 alpha 同時用於 `z` head。MC 比值可作次要描述，不作單獨 pass/fail。通過上述 gate 僅支持固定 DN readout 能恢復這個校準的 NARMA delayed-product 訊號，不代表其他 nonlinear terms 或拓樸效應已有足夠 power。

## Q3. rho gate

要求 0.90、0.95、0.99 三個未選 rho 全部通過 readout gate 過嚴；未使用候選點較差，不應單獨否決最後工作點。建議以固定 5-fold nested Train CV 避免在同一批 MC 上選 rho、再用同一批資料替選出的 rho 背書：每折留 2 條 Train sequences 作外層，其餘 8 條及全部 16 個 Pilot graphs 依原本 dynamics gates 與四 family 等權 MC 規則選一個共同 rho；外層只評估該 rho 的 q readout positive control。若任一折無共同合格 rho，或外層合併的 readout gate 失敗，Pilot no-go。五折完成後，再依原本凍結 selector 使用全部 10 條 Train 選最終 rho；q probe 結果不得回饋 rho、alpha 或 family selection。

這將 selector 的工作點選擇與 readout gate 評估分離；所有 graph families 使用同一 nested rule，Test 仍完全封存。候選 rho 仍須通過原本所有家族的 saturation／forgetting 條件才有資格被選，只有 readout adequacy 不再要求未選 rho 全數通過。現有 `g1_pilot.py` 尚未實作 nested selector 或 q gate，故規格先行不代表程式已符合。

## Q4. 是否值得 Pilot 與成功機率

以下為**主觀決策先驗，不是從實驗頻率估計**。封存的方向建議先前把 G1 的主要成功機率估為 15%（5–30%）；納入 Phase 5/6 無市場增量預測證據、同 DN＋Ridge 對 IC=0.05 訊號僅還原約 16–41%（image projection 約 16%）、MDE 全部未達與 α 54/54 撞上界後，建議下修為 **5%（1–15%）**。市場 IC 和 NARMA NMSE 的 estimand 不同，所以不是歸零；但共享 DN readout／Ridge 衰減提高了 G1 讀出不足的風險。[市場結果摘要](RESULTS_market_v1.md)；先前 15% 估計見 [PLAN_v3_direction.md](archive/market_v1/PLAN_v3_direction.md)。

在新 gate 與實作全部對齊後，G1a 全部 Pilot go 的主觀機率約 **15%（5–35%）**。目前可執行狀態則沒有有效 go：`g1_pilot.py` 未實作本次增加的 q gate／nested rho selector；`g1_power.py:523-531` 把 q 直接加在 prediction 上，繞過 DN 是否能讀出 q；`g1_power.py:589` 回傳未定義的 `gamma_star`，呼叫會在完成模擬後失敗。測試只涵蓋 planted-scale tuning 與獨立的合成 power helper，未涵蓋此端到端函式。`REMOTE_TASK_G1b.md` 仍寫明 Pilot 不得擬合 NARMA target，也與新增的 Train-only q positive control 不一致。v3 目前沒有可供估計 G1 Pilot pass rate 的真實 FlyWire G1 pilot 結果。

Costi 等人的 reservoir 結果只能支持「connectome reservoir 有合理研究價值」，不能直接支持本規格的 5% NMSE 成功率：該文的主要任務為多軌跡三體問題預測，設定含譜半徑 0.99、input scaling 0.7、更新參數 0.9、每個 reservoir 神經元承接三維座標之一，並以 reservoir 節點讀出；與本研究的 R1-6 scalar injection、leak 0.5、固定 1,291 DN readout、NARMA10 與 controls 不同。[Costi et al., *The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/)。

**不建議投入原定完整 4 工作天／8 GPU 小時確認性執行。** 只值得在規格與 runner 對齊、`g1_power` 修正並經小型 synthetic fixture 驗證後，執行有硬上限的 G0/G1a feasibility pilot；若 readout gate／完整計時仍需超出既定成本，停止。現有 8 GPU 小時估算缺少 nested CV、q probe 與完整 Ridge/power 工作量，不能當作已驗證預算。

## Q5. 合法重跑的例外

可另版修復並重跑的情況限於會使 run 無效、而非科學門檻未過：

1. **結果盲化的程式錯誤：** 索引錯誤、非預期資料路徑、數值 solver bug 或與規格不符的實作；以獨立 reference／固定 fixture 證明，保存首次發現時間和測試證據。若已看過性能結果，舊 Test 只留作探索性，不得同 Test 重跑確認性分析。
2. **外部執行中斷：** OS／GPU job 中斷、電源或環境損壞，且尚未讀取輸出；記錄硬體、driver、commit、graph／seed／config hashes 及 checkpoint。原規格仍適用；任何換機或數值設定變更另開版本並完整重跑，不拼接不同環境輸出。可重現的 deterministic OOM 是容量／實作問題，不能當瞬時中斷忽略。
3. **資料或預註冊封存失效：** graph hash／版本錯、Train/Test boundary 錯或 Test 提前建立／存取；立即標記整次 run invalid，保留原 artifacts。修正需先新增修訂紀錄與新 spec hash；若 Test 已被看過，必須換全新且未碰過的 Test seeds。
4. **規格內部矛盾：** 只可在任何性能指標讀取前澄清並凍結新版本。若已看過 Train MC、gate 或性能分數，修正不能追溯套用到同一版；新版本需新 Train seeds 與全新 Test seeds。

不屬於例外：valid Pilot gate fail、rho 不合格、MC／q recovery 太低、power 不足、成本超支、結果接近門檻或 control 勝出。這些均是本版研究結果，應照 no-go 結案，不得改參數後沿用同一確認性名稱。

## 規格同步結果

已修訂 [G1_SPEC_draft.md](G1_SPEC_draft.md)：以 task-specific q positive control 取代 MC ratio 單獨放行；使用 nested Train CV 只驗證選出的 rho；新增無效執行與有效 gate no-go 的區別及不可重用已解封 Test 的條件。未修改程式、runbook、README 或其他文件；程式／`REMOTE_TASK_G1b.md` 與新規格尚未對齊，G1 Pilot 不得啟動。
