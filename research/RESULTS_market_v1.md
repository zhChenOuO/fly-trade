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

## 未完成
- MDE 標籤注入（`research/REMOTE_TASK_P5C.md`，預估約 54 分鐘）；P5B 事後分析（`research/REMOTE_TASK_P5B.md`）。

## 封存文件（research/archive/market_v1/）
research.md、research_v2.md、PLAN_v2.md、SPEC.md、SPEC_v3.md、REMOTE_TASK.md、REMOTE_TASK_P1.md、REMOTE_TASK_P5.md、REMOTE_TASK_P6.md、PLAN_v3_direction.md（Codex 審查與兩輪交叉詰問）。程式註解中對這些文件的引用指向此目錄。
