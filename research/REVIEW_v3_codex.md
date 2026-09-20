# 交叉詰問 III：G0 對照網路與 G1 前置設計審查

## Q1｜程式缺陷與結論影響

| 嚴重度 | 發現與證據 | 判斷與影響 |
|---|---|---|
| **重大** | Phase 5 的大圖 `scramble` 使用 `n_swap_multiplier=0.1`（`extract_features_v2.py` 的變體分支）；以 1,509 萬條邊計，約 151 萬次嘗試，預期仍約 82% 邊未改變。新 `well_mixed_scramble` 的真實圖輸出則為 4.38% overlap、逐節點 in/out degree 相等、權重仍附著於 presynaptic source（交叉詰問所列實測）。 | 舊 scramble 不能支持「真實拓樸與充分隨機拓樸相近」的推論。這改變的是 Phase 5 的拓樸歸因：舊 real-vs-scramble 不可作拓樸證據。它**不**改變既定 Phase 5 FAIL／市場路徑停止的判定，也不構成真圖有預測技能的證據。新 mixed 結果解決了邊重疊問題，但不能單憑 overlap 證明 swap chain 已達均勻穩態。 |
| **中** | `degree_preserved_scramble` 將 CSR 轉 COO，保留每條邊的數值和 source，只交換 destination；因此 source 的正、負加權 out-strength 各自精確保持，target in-strength 會改變。CSR 對同一 neuron-pair 的重複連結先聚合成一個加權邊，故保持的是「不同鄰居數」degree，而不是原始突觸筆數。交換時會拒絕新增自環與重複端點；但 loader 未先明確移除或拒絕輸入自環，既有自環可能保留。 | 對目前由 CSR 表示的真實圖，沒有重複 neuron-pair 邊時，交換後 CSR 聚合不會再改變度數；程式診斷也會逐節點比較度數。仍應在 G1 輸入 gate 報告對角非零數、CSR canonical 狀態與聚合前後 nnz。repo 測試只用無自環的小型合成圖，未證明真實圖自環為零。若自環存在，需把此 null 說成「保留原自環的度數隨機化」，不可說成 loop-free 隨機圖。 |
| **中** | `weight_shuffled_graph(separate_signs=True)` 完全保留拓樸、每條邊的符號及全圖正／負權重多重集；但權重幅度在全圖同符號邊間置換，故改變各神經元的加權 in/out-strength 及其與 degree 的關聯。現有測試只驗證微型合成圖的拓樸、權重多重集與可重現性。 | 作為「拓樸固定、權重幅度與節點 strength 一併打散」的強 null 合理；作為唯一的純權重配置 null 則過強且混淆因素。主要 null 應在每個 presynaptic neuron 的出邊內、按正負符號分層置換幅度，保留每個來源神經元的正／負 out-strength；全圖置換僅作 secondary stress control。局部出邊置換仍不保留 target in-strength，需清楚限定問題。 |
| **中** | `random_matched_graph` 先抽候選端點，再以 `np.unique` 排序後取 `unique_packed[:E]`；排序鍵包含 source index，因此不是對所有候選邊做均勻隨機抽樣，末端 source index 的入選率較低。它會保留 N、E、全圖權重值及正負比例並排除自環，但不保留節點 degree、節點 strength 或 presynaptic 的 Dale sign。測試未涵蓋此函式。 | 不能將現有實作直接稱為無偏的 ER graph。需先以不排序截斷的均勻無放回端點抽樣替代，並以 source index 分層檢查 out-degree、edge inclusion 與 neuron class。偏差是否改變既有 Phase 5 random 結果大小尚未量測；該結果應保留為原樣本的探索性輸出，不回頭宣稱它代表完整 ER 零假設。 |
| **中** | `load_or_create_variant_graph` 的快取鍵只有 variant/seed；讀取快取時不核對 base graph hash、生成器版本、交換門檻或 target rho。metadata 也不足以重建完整生成參數。 | 若快取來自舊程式或不同 FlyWire 版本，可能靜默地用錯圖。G1 每張圖須記錄 base graph SHA-256、變體 SHA-256、生成參數、seed、degree/strength/sign/loop 統計及實測 spectral radius；不符即停止。這是可預防的 provenance 風險，不足以單獨推翻已完成的 Phase 5 指標。 |

測試覆蓋判斷：`test_graph_controls_v2.py` 對 synthetic 100-node graph 驗證 mixed overlap、degree、weight shuffle 不變量與 deterministic seed；沒有驗證真 FlyWire 圖的 loop/strength、random_matched 無偏性、≥1,000 萬邊的大圖分支、stale cache，亦沒有驗證 NARMA stateful 對齊。故目前可稱「局部圖不變量有測試」，不可稱「G0 controls 已完整通過科學驗證」。

## Q2｜最小而足夠的對照家族

| 網路／條件 | 建議 instances | 固定與改變的因素 | 能回答的問題 |
|---|---:|---|---|
| 真實 FlyWire v783 | 1 | 固定一個已鎖定的完整圖 | 本次指定 specimen 在 NARMA10 的條件式表現；一張 connectome 不能估計果蠅個體間變異。 |
| 充分混合的 degree-preserving scramble | 20 個獨立 seed | node 數、in/out degree、邊數、每條邊權重及 source out-strength 固定；端點改變，overlap ≤5% | 在 degree 序列及 source-strength 分布固定下，真實 wiring 相對於充分打亂 wiring 是否重要。 |
| source-wise、sign-stratified weight shuffle | 20 個獨立 seed | 真實拓樸、degree、每個 source 的正／負 out-strength、符號及全圖權重多重集固定；同 source 同符號的幅度重新配到出邊 | 特定 synapse 的幅度配置是否有用，而非節點 strength 異質性本身是否有用。 |
| 無偏 random-endpoint graph | 20 個獨立 seed | node 數、邊數、全圖有符號權重多重集固定；端點隨機化，degree/strength 改變 | 一般 degree distribution 與整體非生物 wiring 的基準；須修正排序截斷偏差後才能作正式 null。 |
| 全圖範圍 weight shuffle | 不列入主清單；若要重現論文 hybrid，另做 20 個 seed | 拓樸、邊符號與全圖幅度分布固定；各節點加權 strength 改變 | 節點 strength/weight-degree 關聯是否重要；單獨無法把「權重配置」與「strength 異質性」分開。 |
| 共用譜半徑工作點 | 不增加 graph instances；對選定 Pilot 全部 graph families 使用相同網格 | 同一張圖以統一方法逐一縮放 | 譜尺度對表現／動力學的影響。它是各 graph family 的共同因子，不應再製作一種「譜半徑 null」。 |

最終主分析為 1 張 real + 20 個 scramble + 20 個 source-wise weight shuffle + 20 個無偏 random-endpoint，合計 61 張圖。全圖 weight shuffle 是回答節點 strength 問題的附加 null，對 G1 主判定非必要。20 個控制 instances 是規劃值（主觀）；它們不等於 20 個獨立真實 connectome。

`random_matched_graph` 現有實作的確以權重陣列置換保留全圖權重值與正負比例；不保留各 source 的符號組成／Dale identity、degree 或 node strength。`load_or_create_variant_graph` 會把生成圖規一至 base graph 的 rho，但應在每一個預註冊 rho 上重新計算／核驗；「生成器保留權重多重集」與「每種網路的實際尺度相同」是兩件事。

**成本估計。** Phase 6 日誌中，32 步靜態重播的 train+val 萃取時間為 real 423 秒、random 359 秒、舊 scramble 361 秒；各圖合計 42,060 個影像樣本。G1 規格的每圖每 rho 約 117,500 個 state-sample updates，粗略線性外推約 31–37 秒／圖／rho；stateful 小 batch 的 GPU 效率未量測，預算採 1–3 分鐘／圖／rho（規劃估計，不是實測）。61 張主分析圖在單一選定 rho 約 1.0–3.1 GPU 小時；另以 1 real + 每個 null family 5 張 Pilot 掃描三個 rho，約 0.4–1.2 GPU 小時。合計約 1.4–4.3 GPU 小時，尚有 8 小時上限內的驗證／重跑緩衝。若首個 timed pilot 推算總量 >8 GPU 小時，停止並回報超支；不在看見 NMSE 後縮短序列或減少 seeds。well-mixed scramble 生成約 2.35–2.4 分鐘／張，20 張約 47–48 CPU 分鐘（已提供的真圖實測外推）；source-wise shuffle 與 random graph 生成成本未量測，先以 CPU pilot 記錄，不占用 GPU 預算。

## Q3｜工作點建議

**確認性網格：全域 spectral radius {0.90, 0.95, 0.99}；不把 1.05 納入主要網格。** 這三值對 real、scramble、source-wise shuffle、random graph 使用完全相同的規一方法與候選網格；規一後 recurrence gain 固定為 1.0、leak 固定為 0.5。既有 0.95/ρ 與 Phase 0 leak=0.5 是本專案設定，不是 NARMA 的最佳值。rho=1.05 只會把有效工作點推到標準 ESP 警戒範圍以外；若研究者仍想看它，必須另列探索性 stress run，不得拿來選主要結果或通過門檻。

在 16 張 Pilot 圖上只使用 Train sequences 選**一個所有圖共用的 rho**：先排除任何不滿足以下條件的 rho：非感覺神經元中 |x|>0.9 的比例 ≥5%；或相同輸入下、從兩個不同有界初始 state 開始，經 500 步後 normalized RMS state difference 未在至少 95% 測試序列降至 <10⁻³。這些數字是沿用 Phase 0 的 5% 飽和門檻與本計劃主觀設定的 state-convergence 門檻。對其餘 rho，以 Train-only 線性記憶容量 MC=Σ(k=1…50) max(R²(u[t−k]),0) 作選擇；先在每個 graph instance 計算，再讓 real、scramble、source-wise shuffle、random 四 family 等權，選 family-balanced 平均 MC 最高者；同分選較低 rho。若沒有共同合格 rho，G1 no-go。

此規則不對每張圖各選最佳 rho，因此保留 real/control 的公平比較；Train-only MC 不看 NARMA Val/Test，降低以主任務挑工作點造成的自由度。global rho matching 對照的是「相同線性尺度下的圖結構」，會改變各圖的絕對權重尺度，必須保存並報告縮放因子。非 hub 子圖 rho 暫不作主設定：hub 定義與子圖邊界會另增自由度，也可能讓不同 graph family 的尺度不可比。

Costi & Izzo 的果蠅儲庫研究在其特定抽樣 reservoir、CR3BP 任務、ESN readout 與 matched-size/ρ 設計下報告 0.99 勝過 0.25/0.5/0.75，並報告拓樸與權重皆有作用；這是選網格的動機，不能推論 0.99 必然適合本專案 138,639-neuron leaky-rate system，更不能單憑它說權重普遍大於拓樸。[原文](https://www.mdpi.com/2313-7673/10/5/341)

## Q4｜G0b 市場舊特徵重跑

**判斷：不跑。** Phase 5 已封存為 FAIL，Val 曾重複用於選擇，且 G0b 的新變體比較不會產生新的確認性證據；跑出有利結果也不能改變封存判定。它只會回答舊 scramble 缺陷對描述性差異的影響，這不是 G1 前置條件，且容易重新打開已結案市場命題。

封存標註應精確寫成：「Phase 5 的 `scramble` 使用 0.1E 次 swap 嘗試，約 82% 原邊預期未移動；故 real-vs-scramble 不能視為充分拓樸隨機化比較，拓樸歸因無效。Phase 5 原始結果保留為歷史探索結果，不重算、不改 FAIL 判定；新 `scramble_mixed`／weight controls 若另行重跑，也只列事後探索性附錄，不得解封或升級市場確認證據。」交叉詰問已指出 market-v1 以 tag 封存；任何額外執行都不在本階段。

## Q6｜`PREREQUISITES_v3.md` 逐表審查

| 區塊 | 具體證據問題／過度延伸 | 建議修正 |
|---|---|---|
| 證據等級定義 | 「A＝已讀原文/官方摘要」把全文、摘要與官方二手摘要混成同一等級；來源是否讀過不等於設計能否外推。 | 拆成「來源型態／是否讀原文／證據能支持的命題／外推限制」。所有摘要型來源最多只能支持摘要中的結果。 |
| P1 微結構 | [arXiv:2602.00776](https://arxiv.org/abs/2602.00776) 是 preprint，且本文的秒級 Binance Futures 資料與 3 秒 horizon 不等於 5m BTC/USDT spot 圖或 30m 方向交易；International Journal of Forecasting 2024 在表中僅有摘要。 | 將「OHLCV 不含 LOB 狀態」保留為資料限制；把「應轉向微結構」降為待測假說，先補原文、交易成本與跨交易所/時段複驗。不得將統計可預測性等同可交易獲利。 |
| P2 頻率與 horizon | [ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S221484502600089X) 只有搜尋摘要（B）；「頻率越高定價效率越低」是廣泛因果陳述，未由表中來源直接建立，也混淆資料採樣頻率與預測 horizon。 | 改為「不同 horizon／流動性 regime 的可預測性需分開驗證」；標示目前需查原文。 |
| P3 效應量 | Gu–Kelly–Xiu 的 0.26% 月 R² 來源在表內屬二手摘要，市場／目標／頻率也與本案不同；IC 0.02–0.05 是二手整理而非通用交易門檻。val IC ±0.02 及「19,600 獨立樣本」未處理重疊 label、序列相依、交易成本與策略曝險。 | 將每一數值標成待查原文或專案內部估計；使用有效獨立 block 數而非 row 數做 power；市場主張另報淨交易成本後的經濟效果。 |
| P4 多重檢定 | Harvey–Liu–Zhu 的 t>3 是資產定價因子挖掘的建議，不是所有預測器的普遍門檻；MinBTL 約 45 個配置來自特定假設且目前只見二手整理。Val 已反覆用過，試過的模型／圖／參數數量亦須計入。 | 將兩者當研究設計參考，不列為硬性普世門檻；寫出完整試驗數與一次性 primary endpoint，採時間序列/圖 instance 層級推論與事前 MDE。MinBTL 的例值需查原文。 |
| P5 目標 | Corsi HAR-RV 是波動度基準的來源，但表內未連原文；Bitcoin volatility review 不是本案資料的功效證據。HAR 可預測波動度不推出 FlyWire 儲庫優於 HAR。 | 補 Corsi 原始引用與本案 realized-volatility horizon 定義；G2 必須和 HAR、歷史均值及簡單波動持續性基準並列，holdout 不供調參。 |
| P6 reservoir 工作點 | [Costi & Izzo 2025](https://www.mdpi.com/2313-7673/10/5/341) 的直接結果支持其指定 CR3BP/ESN 條件下以 0.99 作候選點，且 topology 與 weights 均有作用；不能從該文抽成「權重貢獻大於拓樸」的普遍結論。其低/高 β 的結論不可直接映射成本案 Ridge α：feature scaling、目標、readout、損失除數與 alpha 單位均可能不同。Phase 5 舊 scramble 混合不足，確實削弱舊 topology control；新 weight shuffle 也尚未跑 G1。 | 把文獻結論寫成任務限定的候選假說；不得用不同論文的 β 和本案 α 數值直接比較。G1 同尺度比較、固定 control、stateful 動力學驗證應先於 G2/G3。 |
| P7 衍生品 | 表內明確承認沒有嚴謹來源；「OI 描述部位、不預測方向」依據是實務文章，無法支持方向否定結論。 | 保持弱證據標記；目前從先決條件移除，不以它排除任何假說。 |
| 第2節 資料表 | “新 sealed holdout（未來資料）”與 “LOB 深度快照／逐筆成交”只是需求，尚無供應、欄位、時間戳對齊、缺失、費率及保留政策。 | 不把資料源存在當成已取得；取得後先建 train-only feature contract、交易成本、點時資料可用性及獨立時間窗。G1 不需任何市場資料。 |
| 第3節 G0→G3 順序 | 先 graph null/positive control，再 synthetic stateful NARMA，後 volatility/HAR、microstructure 方向，是低成本且能逐層否證的合理順序。問題是 G0b 舊 market Val rerun 被放在 G0 容易看似解封市場；G1 尚須明確 state carry、輸入/讀出、operating-point 與 MDE。 | G0 僅含 null network 驗證、state-alignment 與正負對照，不做市場重評；G1 合格後才決定 G2。G3 不應由「某微結構論文有訊號」自動解鎖。 |
| 第4節 市場結論 | 專案本身 Phase 5 FAIL、Phase 0 STOP 支持「目前市場 pipeline 沒有足夠證據」；不支持「市場方向報酬在文獻和本案都不具預測性」這種泛化結論。第2節的加密貨幣回報研究也顯示跨變數、horizon 與樣本期結果不同。 | 2% 是明確主觀先驗，可保留為主觀值而非證據估計；改寫為「本 pipeline／5m OHLCV-to-image 不值得再作確認性測試」。對收益方向避免普遍市場效率論斷。 |
| 第5節 待查證 | 願意明列 B 級項目是正確的；但 P1、P2 的新文獻結果也同樣需要來源型態與原文檢查。 | 列入 arXiv:2602.00776、ScienceDirect 2026、International Journal of Forecasting 2024、MinBTL、Gu–Kelly–Xiu 數值、Grinold IC、Corsi/HAR-RV、Suárez 2021。未查原文者保留「需查原文」。 |

表內應補的對照文獻：金融時序的 Reservoir Computing 並非空白領域；例如 [Ballarin、Capra、Dellaportas (2025), Multi-Horizon Echo State Network Prediction of Intraday Stock Returns](https://arxiv.org/abs/2504.19623) 是直接 ESN/日內報酬研究的 preprint，應列作金融 task baseline 文獻，但其結果不能替 FlyWire connectome 背書。加密貨幣微結構也須寫出效應界線：[Bozzetto、Sifat、Nahidi (2026)](https://link.springer.com/chapter/10.1007/978-3-032-18109-1_13) 用 Binance BTC/USDT 高頻資料發現 order-flow imbalance 與 mid-price change 的關係依費率 regime 改變；同文報告其單步預測力可忽略。這支持「資料來源與 horizon 重要」，不等於可扣除成本後獲利。另一組樣本外研究發現，多數常見加密貨幣預測變數未顯著，少數變數才有 out-of-sample 解釋力；[Physica A (2022)](https://www.sciencedirect.com/science/article/pii/S0378437122002928)。這反對把「加密貨幣市場有時可預測」概括成任一 OHLCV 模型可獲利。

**總判斷：** G1 synthetic NARMA10 可作為 reservoir 動態是否有可測計算價值的最小下一步；它只驗證指定單一 FlyWire 圖與固定輸入/讀出下的 benchmark 能力，不證明生物功能或市場預測。G0b 不跑；市場命題仍封存。
