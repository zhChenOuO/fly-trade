# 預測市場的先決條件與新研究順序（v3 草案）

判定：以「5m OHLCV → K 線圖 → 連接體 → 30 分方向」不具備預測市場的先決條件。可行路徑需先通過連接體本身的基準驗證，再依序解鎖資料與目標。

證據等級：A＝已讀原文/官方摘要；B＝僅見搜尋摘要或二手整理，引用前需查原文；本專案實測＝本 repo 輸出。

## 1. 先決條件與證據
| # | 先決條件 | 證據 | 等級 | 對本專案的意義 |
|---|---|---|---|---|
| P1 | 資訊來源含市場微結構（委託簿/訂單流），而非只有 OHLCV | 1 秒 Binance Futures 永續 tick（2022-01～2025-10）、3 秒 horizon；SHAP 顯示 order imbalance、spread、VWAP-to-mid 偏離主導，且僅使用 LOB/trade 特徵（[arXiv 2602.00776](https://arxiv.org/html/2602.00776v1)）。高頻 mid-price 報酬可預測性普遍，表現依賴委託簿表示方式（[Int. J. Forecasting 2024](https://www.sciencedirect.com/science/article/pii/S0169207024000062)，僅見摘要） | A/B | K 線圖不含這些資訊；連接體只是其確定性函數，無法補回 |
| P2 | 時間尺度與資訊來源匹配 | 日級支持弱式效率、日內或條件於流動性可能仍有可預測性；頻率越高定價效率越低（[ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S221484502600089X)，搜尋摘要） | B | 30 分鐘 horizon 對微結構訊號太長、對慢訊號太短 |
| P3 | 期望值校準：效應量小 | 900+ 預測變數的線性模型樣本外 R²<0；加懲罰/降維後約 0.26% 月 R²（[Gu-Kelly-Xiu, RFS 2020](https://academic.oup.com/rfs/article/33/5/2223/5758276)，數值見二手摘要）。IC 0.02–0.05 屬實務可預測性量級（Grinold 基本法則的二手整理） | B | 本專案 Val IC 雜訊底線 ±0.02（本專案實測），偵測 IC=0.02 需約 19,600 獨立樣本（≈408 天） |
| P4 | 多重檢定與回測長度控制 | 新因子 t 值需 >3.0（[Harvey-Liu-Zhu, RFS 2016](https://academic.oup.com/rfs/article/29/1/5/1843824)）；5 年資料下獨立配置不宜超過約 45 個（Bailey et al. MinBTL，二手整理）（[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)） | A/B | 本專案已在同一 Val 上試了數十種配置；Val 只能算探索資料 |
| P5 | 目標選擇：有已知可預測結構的目標 | 已實現波動度可由 HAR-RV 預測（Corsi 2009），已應用於 Bitcoin（[綜述搜尋](https://centaur.reading.ac.uk/87932/1/R&R_Forecasting%20the%20Volatility%20of%20Bitcoin_complete.pdf)） | B | 方向報酬近隨機；波動度可作為有訊號的目標，但必須勝過 HAR 基準 |
| P6 | 連接體儲庫的工作點與正則化 | 果蠅連接體儲庫（CR3BP 混沌軌跡預測，無金融資料）：優勢是低正則化下抗過擬合，高正則化（β=1e-3）優勢消失；譜半徑 0.99 明顯優於 0.25–0.75；權重分布的貢獻大於拓樸（[PMC12109256](https://pmc.ncbi.nlm.nih.gov/articles/PMC12109256/)）。人類連接體儲庫：接近臨界動力學時表現最佳，由拓樸驅動（[Suárez et al., Nat. Mach. Intell. 2021](https://www.nature.com/articles/s42256-021-00376-1)，僅見摘要） | A/B | 本專案 Ridge alpha 選到上限 1e4（強正則化，優勢消失區）；動力學遠非臨界（SPEC_v3 已知限制）；scramble 對照無效；缺「保留拓樸、洗牌權重」對照 |
| P7 | 衍生品特徵（funding、OI、liquidation） | 未找到嚴謹研究；實務文章稱 OI 描述部位、不預測方向 | B（弱） | 只能當預先註冊的候選特徵，不當先決條件 |

## 2. 需要注入的背景資料
| 資料 | 用途 | 依據 | 狀態 |
|---|---|---|---|
| 委託簿深度快照與逐筆成交（OFI、spread、VWAP-mid） | 微結構方向預測 | P1 | 未取得；需確認來源、歷史長度、成本 |
| 過去已實現波動度（多尺度：日/週/月） | 波動度目標與 HAR 基準 | P5 | 可由現有 raw_ohlcv 計算 |
| 交易成本模型（maker/taker、滑價） | 淨績效判定 | P3 | yaml 為 PROPOSED，需確認 |
| 標準時間序列基準（NARMA10、Mackey-Glass、Lorenz、CR3BP） | 連接體儲庫的正/負控制 | P6 | 需新增，無需市場資料 |
| 新的 sealed holdout（未來資料） | 唯一確認性證據 | P4 | 未收集 |
| 衍生品特徵 | 探索性候選 | P7 | 未取得 |

## 3. 新研究順序（每關獨立通過/停止）
- G0 工具修正（無 GPU 成本）：scramble 改為充分混合（swap ≥ 數倍邊數，並輸出邊重疊率）；新增「保留拓樸、洗牌權重」對照；以標籤注入驗證訓練管線能回收已知 IC（MDE，約 54 分鐘）。
- G1 基準任務（上限 8 GPU 小時、4 個工作天）：stateful 模擬（跨時間保留狀態、固定 washout），工作點掃描含譜半徑 0.99 與低正則化，比較真實圖 / ≥20 random / ≥20 充分打亂 scramble / 權重洗牌；主要任務 NARMA10，Mackey-Glass 與 Lorenz 為次要。通過標準：真實圖 NMSE 相對 controls 中位數至少低 5%，且區間下界 >0。未通過即結案，不再改設定。
- G2（僅 G1 通過）：BTC 已實現波動度，對照 HAR 基準，新 holdout，一個 primary 指標。
- G3（僅 G2 通過）：微結構資料（P1）的方向預測。
- 停止條件：任一關未過；超出預算；controls 於單一 seed/graph 外無法重現。

## 4. 對「能否預測市場」的判定
- OHLCV 方向報酬（30 分）：本專案實測與文獻皆不支持；先驗機率約 2%（Codex 主觀估計，非資料估計）。
- 波動度預測：有文獻支持可預測性；連接體是否優於 HAR 與 random 儲庫，取決於 G1。
- 微結構方向預測：有文獻支持資料具訊號；連接體的角色未證明，且需新資料與基礎設施。

## 5. 待查證（B 級證據）
Grinold IC 量級、Gu-Kelly-Xiu 的 0.26% 數值、MinBTL 的 45 個配置例、ScienceDirect 2026 與 Suárez 2021 的全文結論；arXiv 2602.00776 的績效數字（年化 1.25–7.00%、IR 0.32–8.97）來自抓取摘要，需讀原文確認後才可引用。
