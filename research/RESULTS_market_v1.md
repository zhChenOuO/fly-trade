# 市場預測研究 v1 結果封存（tag: market-v1-negative-result）

結論：預先註冊判定 FAIL。未找到 Frozen Connectome 對 BTC/USDT 5m、horizon=6（30 分）報酬的增量預測價值。此結果不代表任何更小效應為零（MDE 尚未量測）。

## 設定
- 資料：BTC/USDT 5m，窗口 48 根，Train 31,556 / Val 10,504（stride=6）；dev_test_v1 僅供除錯；sealed holdout 未收集。
- 管線：K 線圖 → 視網膜編碼 → 凍結 FlyWire v783（138,639 神經元 / 15,091,983 邊）32 步 → DN 讀出 1291 維 → decoder（僅訓練 decoder；6B 開放 sparse Δ）。
- 資料溯源：`research/data/manifest_v2.json`、`research/outputs/v2/probe_val.json`、`phase6_val.json`。

## 結果（Val）
| 階段 | 結果 | 檔案 |
|---|---|---|
| Phase 1a Level 1-2 | 通過（consistency 0.99905；between/within 20072） | outputs/v3/phase1a_remote/metrics.json |
| OHLCV baseline IC | Ridge -0.0007 / Logistic 0.0130 / MLP -0.0027 | outputs/v2/baselines_val.json |
| Phase 5 IC（Ridge，主要指標） | real 0.0024 / random 0.0114 / scramble -0.0002；ΔIC vs random -0.0090、vs scramble +0.0026（CI 皆跨 0）；permutation p=0.425；判定 FAIL（7/11 項） | outputs/v2/probe_val.json |
| Phase 5 Logistic IC（探索） | real 0.0254 [0.007, 0.043] / random 0.0119 / scramble 0.0129 | 同上 |
| Phase 6（override，探索；5 seeds 平均 IC） | 6A real 0.0121 / random 0.0116 / scramble 0.0062；6B real 0.0173 / random 0.0081 / scramble 0.0051 / null 0.0149；全部配對 ΔIC 的 CI 跨 0，Holm p=1；7 模型全 FAIL | outputs/v2/phase6_val.json |
| 交易（含成本） | 有交易的模型淨報酬約 -75%（buy&hold -37.8%） | 同上 |

## 已知缺陷（影響解讀）
- scramble 對照對 1500 萬邊圖只用 0.1×E 次 swap（≤約 20% 邊被改動），非充分打亂；Phase 5/6 的 real-vs-scramble 比較無效。
- 6B 的 Δ 相對 |W|（中位數 9–21）位移小，實為固定特徵上的線性 head；weight decay 未作用於 head；無輸入標準化；不是遞迴突觸可塑性。
- 早停以 loss 而非 IC 選擇；未用已知訊號注入驗證 Phase 6 訓練管線。
- `nuisance_shortcut_r2`（Phase 5 原輸出）定義為 Spearman IC 平方，非 R²；零交易列曾被判 PASS（已在 strict_gates 修正）。
- yaml BA 增量描述為 "0.5/majority"，程式用 1/3。
- 單一 graph 實例、單一 noise seed；Val 已被反覆使用，後續分析皆為探索性。

## P5B 事後分析（探索性，remote/phase5c；不改變判定）
- 重現檢查：10 個 IC 與 probe_val.json 完全一致（最大絕對差 0）；baseline B0/B1a/B1b 差值 0。
- Logistic ΔIC（real−對照）：vs random +0.0136 [-0.0075, 0.0343]、vs scramble +0.0125 [-0.0075, 0.0336]、vs OHLCV B1b +0.0125 [-0.0092, 0.0342]、vs nuisance +0.0208 [-0.0023, 0.0453]；Holm 校正後 p 0.31–0.58（全域 0.62–1.0）。Ridge 全部 |ΔIC|<0.01，CI 跨 0。
- 殘差 IC（real Logistic 分數對 OHLCV+nuisance 分數回歸後）：0.0232 [0.0042, 0.0424]（未校正）。
- 分季 IC（real Logistic）：0.0198 / 0.0192 / 0.0394 / 0.0200，四季皆正；random 3/4、scramble 3/4 為正；real Ridge 2/4。
- 檔案：outputs/v2/probe_posthoc_val.json。

## MDE 標籤注入（探索性；remote/phase5c，R=100、bootstrap=500，耗時 685 秒）
- 設計：固定母體係數（Train 校準，K=50 雜訊抽樣）；訊號形態 momentum / mean_reversion / image_projection；目標 IC 0–0.05；偵測＝Val IC 的 95% CI 下界 >0。
- MDE（偵測率首次 ≥80%）：三形態 × 三網路全部 NOT_REACHED（最高目標 IC 0.05 仍未達）。
- real 圖偵測率（目標 IC 0.02 / 0.03 / 0.05）：momentum 7% / 9% / 31%；mean_reversion 9% / 11% / 53%；image_projection 6% / 8% / 14%。random 與 scramble 同量級或更低。
- 目標 IC 0.05 時，real 圖 Ridge 實際還原的 IC（目標＋平均偏差）：momentum 0.013、mean_reversion 0.021、image_projection 0.008（約 16–41% 的植入訊號）。
- Val IC 的 95% CI 中位寬度 0.037–0.038；IC=0 假陽性率 2% / 2% / 4%（real / random / scramble）。
- Ridge α 撞上界（1e4）比例：54/54 配置皆 100%。
- 判讀：Phase 5/6 的 DN 讀出管線對植入訊號（含影像的線性函數）還原率低，且 Val 樣本量只能分辨 IC ≳0.04；對 IC ≤0.05 的效應檢定力不足。Phase 5 FAIL 對 ≤0.05 的效應無資訊，不代表連接體表徵沒有資訊。
- 檔案：outputs/v2/mde_injection.json、mde_injection.log。

## 未完成
- 無（市場命題封存）。

## 封存文件（research/archive/market_v1/）
research.md、research_v2.md、PLAN_v2.md、SPEC.md、SPEC_v3.md、REMOTE_TASK.md、REMOTE_TASK_P1.md、REMOTE_TASK_P5.md、REMOTE_TASK_P6.md、PLAN_v3_direction.md（Codex 審查與兩輪交叉詰問）。程式註解中對這些文件的引用指向此目錄。
