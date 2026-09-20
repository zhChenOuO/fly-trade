# 預測市場的先決條件與新研究順序（v3 草案，經 Codex 審查修訂）

判定：本專案的 pipeline（5m OHLCV → K 線圖 → 連接體 → 30 分方向）不值得再做確認性測試。市場命題封存；新路徑先在合成基準（G1）驗證連接體儲庫本身，再依序解鎖。

## 證據標記
- 讀原文＝已讀全文；摘要＝僅見摘要或搜尋摘要；二手＝他人整理。摘要與二手來源只能支持其摘要內的結果，引用前需查原文。
- 所有來源的資產、頻率、horizon 與目標都與本案不同，只作「待測假說」的動機，不作本案的證據。

## 1. 先決條件（待測假說）
| # | 假說 | 來源（型態） | 對本專案的意義與限制 |
|---|---|---|---|
| P1 | 高頻可預測性主要來自委託簿/訂單流，而非 OHLCV | 秒級 Binance Futures 永續 tick、3 秒 horizon，order imbalance/spread/VWAP-mid 偏離主導（[arXiv 2602.00776](https://arxiv.org/abs/2602.00776)，preprint，摘要型抓取，績效數字需查原文）。mid-price 高頻可預測性（[Int. J. Forecasting 2024](https://www.sciencedirect.com/science/article/pii/S0169207024000062)，摘要）。OFI 與價格變動關係隨費率 regime 改變、單步預測力可忽略（[Bozzetto et al. 2026](https://link.springer.com/chapter/10.1007/978-3-032-18109-1_13)，Binance BTC/USDT 高頻，摘要） | 「OHLCV 不含 LOB 狀態」是資料限制；轉向微結構是待測假說。統計可預測性不等於扣成本後可獲利；資料源、欄位、時間戳對齊、費率尚未取得 |
| P2 | 不同 horizon 與流動性 regime 的可預測性需分開驗證 | Bitcoin 效率研究（[ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S221484502600089X)，搜尋摘要）；多數常見加密貨幣預測變數樣本外不顯著、少數有解釋力（[Physica A 2022](https://www.sciencedirect.com/science/article/pii/S0378437122002928)，摘要） | 不作「頻率越高效率越低」的因果陳述；不作「市場方向不可預測」的普遍結論 |
| P3 | 效應量小，IC 需以有效獨立樣本估計 | Gu-Kelly-Xiu（[RFS 2020](https://academic.oup.com/rfs/article/33/5/2223/5758276)；0.26% 月 R² 為二手數值，需查原文；股票月頻）；IC 0.02–0.05 為二手整理 | 本專案 Val IC 雜訊底線約 ±0.02（本專案實測）；「19,600 獨立樣本」是單相關 Fisher 近似，未計重疊標籤、序列相依、成本，實際需以有效獨立 block 數估計 |
| P4 | 多重檢定與試驗次數需計入 | Harvey-Liu-Zhu（[RFS 2016](https://academic.oup.com/rfs/article/29/1/5/1843824)，t>3 為資產定價因子建議）；Bailey et al. MinBTL（[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)，「5 年約 45 個配置」為二手例值，需查原文） | 設計參考，非普世門檻。本專案 Val 已反覆使用，試驗數（模型/圖/參數）須計入；只保留一個 primary endpoint |
| P5 | 波動度有已知可預測結構，可作有訊號目標 | HAR-RV（Corsi 2009，未連原文）；Bitcoin 應用綜述（[Reading](https://centaur.reading.ac.uk/87932/1/R&R_Forecasting%20the%20Volatility%20of%20Bitcoin_complete.pdf)，摘要） | HAR 可預測不推出連接體優於 HAR。G2 須與 HAR、歷史均值、波動持續性並列 |
| P6 | 儲庫工作點與正則化影響連接體儲庫表現 | 果蠅連接體儲庫（[Costi & Izzo 2025](https://www.mdpi.com/2313-7673/10/5/341)，CR3BP 任務、特定抽樣 reservoir 與 ESN readout；讀原文摘要/PMC 版）：譜半徑 0.99 優於 0.25/0.5/0.75，拓樸與權重皆有作用。人類連接體儲庫（[Suárez et al. 2021](https://www.nature.com/articles/s42256-021-00376-1)，摘要）：近臨界動力學表現最佳 | 只支持把 0.99 列為候選工作點。不作「權重貢獻大於拓樸」的普遍結論；論文的 β 不可與本案 Ridge α 直接比較；舊 scramble 混合不足、隨機圖抽樣有偏（見 REVIEW_v3_codex.md），拓樸歸因需重做 |
| 已移除 | 衍生品特徵（funding/OI/liquidation） | 僅實務文章，無嚴謹來源 | 不作為先決條件，不用於排除任何假說 |

補充文獻（金融儲庫計算並非空白）：[Ballarin et al. 2025](https://arxiv.org/abs/2504.19623)（ESN 預測日內股票報酬，preprint），僅作金融任務基準文獻，不為 FlyWire 背書。

## 2. 需要的資料（需求，尚未取得）
| 資料 | 用途 | 狀態 |
|---|---|---|
| 委託簿深度與逐筆成交 | 微結構方向預測（G3） | 無來源、欄位、時間戳、缺失與費率規格；取得後先建 train-only feature contract |
| 多尺度過去已實現波動度 | 波動度目標與 HAR 基準（G2） | 可由現有 raw_ohlcv 計算 |
| 交易成本模型 | 淨績效判定 | yaml 為 PROPOSED，需確認 |
| NARMA10（主）、Mackey-Glass、Lorenz（次） | 連接體儲庫的正/負控制（G1） | 需新增；不需市場資料 |
| 新的 sealed holdout | 市場確認性證據 | 未收集 |

## 3. 新研究順序（逐關通過，任一關不過即結案）
- G0（工具）：充分混合 scramble、權重洗牌、無偏隨機圖、圖 provenance；MDE 標籤注入（母體係數設計）。不重評市場舊 Val；G0b（舊市場特徵重跑新對照）不做。
- G1（合成基準）：規格見 `research/G1_SPEC_draft.md`。上限 8 GPU 小時、4 個工作天。
- G2（僅 G1 通過）：BTC 已實現波動度，對照 HAR，新 holdout。
- G3（僅 G2 通過）：微結構資料；不由「某論文有訊號」自動解鎖。

## 4. 判定
- 本 pipeline（5m OHLCV→圖像）不值得再作確認性測試。這不推出「市場方向不可預測」的普遍結論。
- 「先驗約 2%」為 Codex 主觀估計，非資料估計。
- 波動度與微結構路徑取決於 G1 是否顯示連接體儲庫有可測計算價值。
