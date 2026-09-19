# 果蠅 Connectome 交易決策「非隨機性」研究：Phase 2 研究計劃 (PLAN_v2)

**文件版本**：v2.0-FROZEN  
**狀態**：待使用者核准預先註冊 (Pre-registration Pending)  
**標的專案**：Drosophila Connectome Market Dynamics Research  
**檔案路徑**：`research/PLAN_v2.md`

---

## 0. 結論摘要 (Executive Summary)

### 0.1 核心診斷：當前管線並未讀取市場，而是一個「高敏隨機雜湊函數」
在第一輪實驗 (v1) 因背景直流偏差導致動作崩潰 (BUY 比例 0.0004) 作廢後，v2 健康檢查的三種編碼方案（`mean_sub`、`zero_black`、`on_off`）在 Val 集合上**全數未通過預先註冊 Gate**：
1. **Between/Within 變異比不足**：分別為 2.99、1.41、1.82（未達門檻 3.0）。
2. **隨機雜湊指紋確證**：價格翻轉 (`flip_price`)、時間打亂 (`shuffle_time`)、遮蔽 (`mask_recent25`)、全黑與平均圖的擾動變化量全部落在 $\Delta \text{margin}/\text{SD} \approx 0.96 \sim 1.25$。唯讀重算證實：**翻轉價格後的輸出與原圖輸出的相關係數為 $-0.0032$，與隨機抽取一張獨立圖片的相關係數 ($-0.0723$) 及變化量 ($1.179\sigma$) 完全一致**（理論上兩個獨立標準常態變量之差的期望值即為 $2/\sqrt{\pi} \approx 1.128\sigma$）。
3. **動力學硬飽和崩潰**：既有合成連接體之譜半徑 $\rho(W) \approx 6.23$（未歸一化），80% 為興奮性突觸，導致 32 步模擬後超過 **58.0% 的神經元處於 $|x| > 0.95$、42.2% 處於 $|x| > 0.99$ 的極限飽和區**；同時任意自訂的 `noise_std = 0.1` 佔據了輸入電流標準差的 $35\% \sim 60\%$。系統本質上處於非線性混沌或極限截斷狀態，將任何微小擾動放大為獨立隨機跳變。

### 0.2 是否值得繼續？
**答案：有條件值得，但必須立即大幅改寫研究框架，且嚴格設立停損線。**
- **不值得繼續的做法**：若繼續沿用「K 線像素圖 $\to$ 隨機高斯投影 $\to$ 未調控譜半徑的 Reservoir $\to$ 直接預測真實市場」的架構，無論換到真實 FlyWire 還是投入 Windows GPU，都是在對一個 14 萬節點的「大型隨機雜湊器」浪費算力，成功機率為 0。
- **值得繼續的做法**：果蠅連接體在生物學上具備模組化、稀疏投射與微迴路抑制等天然特性。若將研究目標從「幻想未訓練大腦能自發獲利 (Level 4)」修正為「**驗證生物連接體能否作為具備記憶與局部幾何保真度的動態特徵萃取器 (Level 1–2)**」，並以「**合成可預測訊號的陽性對照**」作為先決通過門檻，本研究具有嚴肅的計算神經科學與仿生計算價值。

### 0.3 建議下一步行動 (Next Actions)
1. **立即凍結與標記**：將已開箱之 `test` split 永久封存為除錯用途之 `development_test_v1`；啟動更早歷史 (2020–2023) 建立不可逆雜湊的 `sealed_holdout_v2`。
2. **架構微創重構 (Phase 0: 陽性對照與動力學解飽和)**：
   - 譜半徑強制縮放至 $\rho(W) = 0.95$（重獲 Echo State 邊緣記憶性質），將神經元飽和率降至 5% 以下。
   - 廢除像素隨機投影，改採 **直接數值特徵注入 (Direct Feature Injection)** 或 **視網膜拓撲 (Retinotopic) 柱狀映射**。
   - 先以已知趨勢/波動的**合成幾何訊號 (Positive Control)** 進行驗證。若連 100% 規律的合成訊號都無法單調響應，立即終止研究 (Hard Stop)。
3. **分階段推進**：通過 Phase 0 陽性對照後，才進入真實市場 Train/Val 健康檢查 (Phase 1)；通過後才在 Windows 機器部署 FlyWire GPU 版 (Phase 2)；最後才開啟新封存集 (Phase 3)。

---

## 1. 核心決策問題回答 (Answers to A–G)

### A. 假設評估、取捨與排序 (Hypotheses Assessment & Priority)

| 排序 | 假設代號 | 假設內容 | 資訊價值 | 執行成本 | 決策處置與理由 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0 (新增)** | **RQ0 (陽性對照)** | **管線訊號保真度**：在已知強規律的合成價格訊號下，輸出是否能單調區分趨勢/波動？ | **極高 (否決性)** | **極低 (本機 CPU 數分鐘)** | **立即新增**。若模型連純幾何階躍或純線性趨勢都無法單調反映，後續所有真實市場測試皆無科學意義。 |
| **P1** | **RQ1 (改寫)** | **結構平滑輸入依賴**：模型輸出是否對市場結構具備非雜湊式、局部平滑 (Lipschitz 有界) 的可重現響應？ | **極高 (基礎門檻)** | **低 (本機 CPU，僅 Train/Val)** | **保留並改寫**。原 H1-1「輸入改變造成輸出改變」是偽命題 (雜湊函數也符合)；必須改寫為「幾何鄰近性保真度與信噪比檢驗」。 |
| **P2** | **RQ2** | **真實拓撲作用**：真實 FlyWire 拓撲在動態容量或特徵分離度上是否顯著優於同度數隨機重組 (Scramble)？ | **中高 (神經科學核心)** | **高 (下載 6GB 資料、Windows GPU 移植)** | **暫時擱置，設為條件式啟動**。只有在 RQ0/RQ1 通過、證實架構非雜湊後，引進 14 萬節點全腦才有意義。 |
| **P3** | **RQ3** | **零樣本市場預測**：未經塑性學習的果蠅大腦輸出是否顯著優於 Matched Random？ | **極低 (先驗機率 $\approx 0$)** | **中 (需耗用昂貴的封存 Holdout)** | **降級為終端探索假說**。前置 CMA-ES 實驗已證實連續 Reservoir 即使優化後 OOS Sharpe (+0.75) 仍低於隨機期望最大值 (+0.97)；弱型效率市場下零樣本自發獲利近乎不可能，預期接受 H0-3。 |

---

### B. 「輸入依賴」的操作型定義與防雜湊診斷設計 (Operational Definition & Anti-Hash Diagnostics)

#### 1. 當前設定的本質缺陷
隨機高斯投影矩陣 $W_{in} \in \mathbb{R}^{12288 \times 48}$ 作用於 1 像素寬的高稀疏 K 線圖，微小位移（如價格上下平移 2 像素）會導致非零像素在隨機投影維度上完全錯位；疊加 $\rho(W) = 6.23$ 的發散遞迴動力學，使得輸出 margin 的幾何流形發生劇烈摺疊。系統表現為典型的**局部不敏感雜湊 (Locality-Insensitive Hash)**。Claude 原先設定的 $G4 \ge 0.2\sigma$ 存在嚴重邏輯倒置：因為獨立隨機抽樣的期望變異就是 $1.128\sigma$，$1.209\sigma$ 正好是隨機雜湊的典型特徵。

#### 2. 正確的操作型定義
一個合格的「結構輸入依賴」必須同時滿足以下三大數學準則：
1. **局部幾何平滑性 (Local Lipschitz Regularity)**：輸入空間的微小擾動 $\epsilon$ 只能引起輸出空間成比例的微小變化，局部相關性接近 1，而非跳變至不相關狀態。
2. **語意對稱性/單調響應 (Semantic Symmetry / Directional Coupling)**：宏觀結構顛倒（如價格上下翻轉 `flip_price`）必須引起輸出向相反方向偏移（顯著負相關），或在形狀識別模式下保持高度結構共變，絕不允許出現 $Corr \approx 0$ 的統計獨立。
3. **高信噪比 (Signal Dominance over Internal Noise)**：跨市場輸入的變異 (Between) 必須實質壓倒內部模擬器隨機噪聲的變異 (Within)。

#### 3. 預先註冊的防雜湊診斷指標與門檻

```
[市場影像/特徵 x] ───> [輸入編碼] ───> [譜半徑受控 Reservoir] ───> margin(x)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      微擾動 x + ε (ε=1%)          結構翻轉 flip(x)
              │                           │
              ▼                           ▼
      margin(x + ε)                 margin(flip(x))
              │                           │
              ▼                           ▼
  [檢驗 1: 平滑度]            [檢驗 2: 結構反轉響應]
  Corr ≥ 0.90                 Corr ≤ -0.30 (方向型)
  Δ/SD ≤ 0.25 (拒絕雜湊)      或 |Corr| ≥ 0.40 (形態型)
```

1. **微擾 Lipschitz 檢驗 ($\epsilon$-Perturbation Test)**：
   - 方法：對 Val 樣本加入 1% 強度之高斯像素噪聲或特徵微擾 $\epsilon \sim \mathcal{N}(0, (0.01 \cdot \sigma_x)^2)$。
   - 指標：$R_{\epsilon} = \frac{\mathbb{E}[|\text{margin}(x+\epsilon) - \text{margin}(x)|]}{\sigma_{\text{train}}(\text{margin})}$ 與 $\rho_{\epsilon} = \text{Corr}(\text{margin}(x), \text{margin}(x+\epsilon))$。
   - **預先註冊門檻**：$R_{\epsilon} \le 0.25$ 且 $\rho_{\epsilon} \ge 0.90$。（隨機雜湊在此測試下 $R_{\epsilon} \approx 1.13, \rho_{\epsilon} \approx 0$，直接被一票否決）。
2. **結構翻轉檢驗 (Structural Inversion Test)**：
   - 指標：$\rho_{\text{flip}} = \text{Corr}(\text{margin}(x), \text{margin}(\text{flip}(x)))$。
   - **預先註冊門檻**：若模型宣稱捕捉方向，則必須 $\rho_{\text{flip}} \le -0.30$ 且 $p < 0.001$；若模型宣稱捕捉波動/形態，則必須 $|\rho_{\text{flip}}| \ge 0.40$。當前之 $-0.0032$ 直接判定為無結構反應。
3. **相對於隨機 ESN / 隨機雜湊對照組的超額互資訊 (Excess MI over Random ESN)**：
   - 建立同等規模、具備標準 Echo State 規範（$\rho=0.95$）的隨機 Erdős-Rényi 網絡。
   - **預先註冊門檻**：真實/合成 Connectome 與離散市場狀態之互資訊 $I(\text{Connectome}) - I(\text{Random ESN}) > 0$ 且 bootstrap 95% CI 下界 $> 0$。

---

### C. 輸入編碼、神經元映射與無偏參數決定規則 (Input Encoding & Unbiased Tuning)

#### 1. 編碼方案取捨與架構收斂

| 編碼方案 | 實作機制 | 優點 | 致命缺點 | 推薦度 |
| :--- | :--- | :--- | :--- | :--- |
| **方案 A：直接數值特徵注入 (Direct Feature Injection)** | 將 48 根 K 線的價格變化率、振幅、成交量等連續特徵正規化後，直接 1-to-1 注入感覺神經元 | 無渲染偽影、無 DC 背景偏差、具完美局部平滑性、計算速度提升 100 倍 | 形式上背離原始 64x64 圖片規格 (但符合神經生物學嗅覺/機械感覺通道真實機制) | **最優先推薦 (Primary)** |
| **方案 B：視網膜拓撲映射 (Retinotopic Columnar Projection)** | 橫軸 48 欄直接 1-to-1 對應 48 顆感覺神經元；每顆神經元僅接收該 K 線柱之垂直位置與振幅 | 保留時序因果性與 2D 幾何鄰近性，徹底消除隨機投影雜湊 | 仍受制於像素化帶來的離散跳變 | **次選 (Secondary, 備用)** |
| **方案 C：全域隨機高斯投影 (現行 v1/v2 方案)** | 12,288 像素 $\times$ 高斯隨機矩陣 $\to$ 48 感覺神經元 | 實作簡單 | 空間資訊全毀、維度災難、幾何非連續、局部雜湊化 | **徹底廢除 (Deprecate)** |

#### 2. 真實連接體 (FlyWire) 神經元選取規則 (客觀凍結，杜絕事後 Cherry-picking)
- **Sensory 神經元**：
  - 嚴格限定讀取官方註釋表 `classification.csv`。
  - 若採方案 A (特徵注入)：篩選 `super_class == "sensory"` 中之 ORN (嗅覺受體神經元) 或 mechanosensory 神經元，依 `root_id` 數字由小到大排序，固定取前 $N_{\text{sensory}}$ 顆（如 48 顆）。
  - 若採方案 B (視覺拓撲)：篩選視覺投射神經元（Lobula/Lobula Plate 下行通道），依其解剖重心 $X$ 軸坐標排序分箱後映射。
- **Motor 神經元 (BUY / SELL)**：
  - 嚴格限定篩選 `super_class == "descending"` (下行神經元 DN，果蠅大腦運動指令唯一中樞)。
  - **利用雙側對稱性天然劃分**：依神經元胞體所在半腦（Left DN vs Right DN）劃分 BUY 與 SELL。生物學上左/右轉彎具有天然的結構對稱性與動態平衡，徹底從結構根源杜絕 BUY/SELL 基礎偏向 (DC Bias)。

#### 3. 自由參數 (gain, leak, noise_std) 的無監督決定準則 (禁止以市場標籤/回測指標調參)
1. **譜半徑與增益 ($gain$)**：
   - 透過計算連接體稀疏權重矩陣之最大特徵值模長 $\rho(W)$，設定 $gain = 0.95 / \rho(W)$。
   - 原理：嚴格保證系統工作在「混沌邊緣 (Edge of Chaos)」，擁有最長之衰退記憶 (Fading Memory) 同時避免非線性發散。
2. **神經元飽和度約束**：
   - 在 Train 集合（純輸入，不含標籤）上模擬，驗證神經元平均激活狀態。
   - 要求：神經元激活絕對值 $|x_i| > 0.90$ 的比例必須 $< 5\%$。若高於此值，自動等比例調降 $gain$ 直至達標。
3. **固有神經噪聲 ($noise\_std$)**：
   - 廢除人為拍腦袋決定的 `0.1`。
   - 規則：在確定性驗證階段設定 $noise\_std = 0.0$；在生物噪聲魯棒性驗證階段，設定為 Train 感覺輸入電流標準差的精確 5%（即 $\text{SNR} = 20\text{dB}$），並在開啟 Val/Test 前凍結為常數。

---

### D. Sealed Holdout 與資料計劃 (Data Plan & Power Analysis)

#### 1. 資料集重劃與 Sealed Holdout 隔離架構

```
時間軸 (2020-09-19 至 2026-09-19 共 6 年 BTC/USDT 5m)
├───────────────────────┬───────────────────────┬───────────────────────┐
│ 歷史資料 (新增)        │ 原 Train / Val        │ 污染標記區 (不可作證據) │
│ 2020-09-19 ~ 2023-09-19│ 2023-09-19 ~ 2026-02-12│ 2026-02-12 ~ 2026-09-19│
├───────────┬───────────┼───────────┬───────────┼───────────────────────┤
│ Train_v2  │ Val_v2    │ Train_v1  │ Val_v1    │ development_test_v1   │
│ (40%)     │ (20%)     │ (既有)     │ (既有)     │ (已開箱，僅供除錯)     │
└───────────┴───────────┴───────────┴───────────┴───────────────────────┘
            ▲
            └─ ★ 封存其中 10,500 個不重疊樣本為 sealed_holdout_v2 (GPG 加密)
```

- **舊 Test 處置**：原 2026-02-12 至 2026-09-19 之 10,504 筆樣本已在 v1 中被開箱計算過 Balanced Accuracy，永久降級為 `development_test_v1`，僅作為 Phase 0/1 內部除錯，嚴禁在正式報告中作為 Level 4 證據引用。
- **新 Sealed Holdout 建立**：
  - 抓取 Binance BTC/USDT 5m 歷史資料：`2020-09-19 00:00:00` 至 `2023-09-19 00:00:00` (共 3 年，包含 2020 減半牛市、2021 雙頂及 2022 深度熊市)。
  - 在該 3 年區間內，依 stride=6 切分，提取前 40% 為 `train_v2`，中間 20% 為 `val_v2`，**最後 40% (約 21,000 筆樣本) 中抽取連續 10,500 筆不重疊樣本，凍結為 `sealed_holdout_v2`**。
  - **物理封存機制**：以 SHA-256 雜湊記錄原始行情與樣本索引，並使用 AES-256/GPG 加密保存。在所有 Phase 0/1/2 健康檢查與陽性對照通過前，不得有任何解密動作。

#### 2. 統計檢定力分析 (Statistical Power Analysis)
令虛無假設 $H_0: \text{BA} = 0.500$。假設標籤平衡（$N_{up} \approx N_{down} \approx N/2$），Balanced Accuracy 之標準誤為：
$$\sigma_{\text{BA}} \approx \frac{1}{2\sqrt{N}}$$
- 當 $N = 10,500$ 時，$\sigma_{\text{BA}} \approx \frac{1}{2\sqrt{10500}} \approx 0.00488 \ (0.488\%)$。
- 在雙尾檢定 $\alpha = 0.05$ ($z_{\alpha/2} = 1.960$) 與統計檢定力 $1 - \beta = 0.80$ ($z_{\beta} = 0.842$) 下，最小可偵測效果量 (Minimum Detectable Effect, MDE) 為：
  $$\Delta \text{BA}_{\text{min}} = (1.960 + 0.842) \times \sigma_{\text{BA}} \approx 2.802 \times 0.00488 \approx 0.0137 \ (1.37\%)$$
- **結論**：10,500 個樣本僅足以可靠偵測到 $\text{BA} \ge 0.514$ (即勝過隨機 $1.4\%$) 的訊號。
  - 若預先註冊的最小效果量設為 $\Delta \text{BA} = 0.005$ (即 $\text{BA} = 0.505$)，需要樣本數 $N \ge \left(\frac{2.802}{2 \times 0.005}\right)^2 \approx 78,500$ 筆樣本（在 stride=6 下需 4.5 年不間斷資料）。
  - 因此，**宣稱 $\text{BA} = 0.505$ 具有統計顯著性在 10,500 樣本下是不成立的偽發現**。

#### 3. Seeds 偽重複 (Pseudo-replication) 的嚴格統計修正
- 30 個 seeds 共用同一批市場樣本，種子之間具有強烈的樣本依賴性（同一個市場極端行情對 30 個 seeds 的衝擊高度相關）。
- **處理規範**：
  1. **禁止 Pooled Sample T-test**：絕對禁止將 $30 \times 10,500 = 315,000$ 筆決策混為一談計算標準誤。
  2. **Ensemble 決策為主要單元**：以 30 個 seeds 在每個樣本點的平均輸出（Consensus Voting 或 Mean Margin）聚合為單一預測序列，只在樣本時間軸 $N = 10,500$ 上進行單一檢定。
  3. **Cluster-Block Bootstrap**：對單獨 seed 的檢驗，Bootstrap 重抽樣必須以 `sample_id` 為單元進行整塊抽取（即 30 個 seeds 的對應行同時被抽入），保留種子間的跨列相關結構。

#### 4. 具備交易經濟意義的最小效果量 (Economic Significance)
- 交易成本假設：Binance 現貨 VIP0 手續費單邊 0.04%，加上滑價與點差 0.02%，單邊成本 0.06%，來回完整換手成本 $C = 0.12\%$。
- 在 5m K 線、6 根 bar (30 分鐘) 持有期內，BTC 之平均絕對價格波動 $\mathbb{E}[|\Delta P/P|] \approx 0.45\%$。
- 每次決策皆交易的情境下，打平手續費所需之盈虧平衡 Balanced Accuracy 為：
  $$\text{BA}_{\text{break-even}} \approx 0.50 + \frac{C}{2 \cdot \mathbb{E}[|\Delta P/P|]} \approx 0.50 + \frac{0.0012}{2 \times 0.0045} = 0.50 + 0.133 = 0.633 \ (63.3\%)$$
- 即便引入高信心過濾（例如只在 $|\text{margin}|$ 最高的 10% 樣本交易，該分位數平均波動提升至 1.2%），盈虧平衡 BA 亦高達：
  $$\text{BA}_{\text{filter, break-even}} \approx 0.50 + \frac{0.0012}{2 \times 0.012} = 0.550 \ (55.0\%)$$
- **預先註冊門檻**：
  - **純學術統計顯著門檻**：$\text{BA} \ge 0.514$ 且 Permutation $p < 0.01$。
  - **交易實用門檻**：$\text{BA} \ge 0.550$（若低於此值，即便統計顯著，扣除手續費後期望值依然為負）。

---

### E. 分階段路線圖與 Go/No-Go 決策樹 (Roadmap & Decision Tree)

```mermaid
flowchart TD
    Start([啟動 PLAN_v2]) --> P0[Phase 0: 陽性對照與動力學解飽和]
    P0 --> C0{幾何趨勢檢驗通過?<br/>Lipschitz R_eps <= 0.25?<br/>神經元飽和率 < 5%?}
    C0 -- No --> Stop0[Hard Stop: 終止研究<br/>Connectome Reservoir 喪失幾何保真度]
    C0 -- Yes --> P1[Phase 1: 改進編碼與真實 Train/Val 健康檢查]
    
    P1 --> C1{Val 雙側對稱 Between/Within >= 3?<br/>翻轉相關 |Corr| >= 0.40?<br/>Nuisance R2 < 0.05?}
    C1 -- No --> Stop1[Hard Stop: 終止研究<br/>市場特徵無法在無訓練下形成穩定表徵]
    C1 -- Yes --> P2[Phase 2: Windows GPU 部署與 FlyWire 拓撲驗證]
    
    P2 --> C2{Fly-Intact 在特徵分離度/記憶容量<br/>顯著優於 Degree-Scramble?}
    C2 -- No --> Fork2[接受 H0-2: 果蠅真實拓撲無特殊優勢<br/>降級為普通隨機網絡]
    C2 -- Yes --> P3[Phase 3: Sealed Holdout 終端盲測]
    Fork2 --> P3
    
    P3 --> C3{Sealed Holdout BA >= 0.514<br/>且 Permutation p < 0.01?}
    C3 -- No --> AcceptH03[接受 H0-3: 零樣本無預測能力<br/>轉向第二階段 Reward Plasticity]
    C3 -- Yes --> ClaimL4[達成 Level 4: 證實具備自發市場預測偏向]
```

#### 階段具體細節表

| 階段 | 目標與交付物 | 依賴資料與工具 | 算力平台 | 預估工時 | 通過標準 (Go) | 失敗處理 (No-Go / 停損) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 0** (陽性對照) | 1. 修正譜半徑 $\rho=0.95$<br/>2. 實作合成幾何價格產生器<br/>3. 微擾平滑度檢驗腳本 | 合成純趨勢/階躍數據 | 本機 Mac (CPU) | 3 工作天 | 1. 純趨勢響應 $|\rho_{\text{rank}}| \ge 0.50$<br/>2. $R_{\epsilon} \le 0.25$<br/>3. 飽和率 $< 5\%$ | **Hard Stop**：終止研究。證明此類型的 Reservoir 根本無法處理時序訊號。 |
| **Phase 1** (真實編碼檢查) | 1. 實作特徵直接注入與雙側對稱 Motor 映射<br/>2. 執行 Val 健康檢查 | 既有 `raw_ohlcv.parquet` 之 Train/Val | 本機 Mac (CPU) | 4 工作天 | 1. Between/Within $\ge 3.0$<br/>2. $|\rho_{\text{flip}}| \ge 0.40$<br/>3. Nuisance $R^2 < 0.05$ | **Hard Stop**：終止研究。真實市場輸入無法產生高於噪聲的平滑表徵。 |
| **Phase 2** (真實 FlyWire) | 1. 手動下載 FlyWire 數據<br/>2. 實作 `torch.sparse` GPU 模擬器<br/>3. 預先快取 Scramble 網絡 | FlyWire Codex 下載檔 (見下方清單) | Windows (RTX 5070 12GB VRAM) | 6 工作天 | Fly-Intact 在狀態分離度或記憶回溯上顯著勝過 Degree-Scramble (p < 0.01) | 宣告 RQ2 失敗 (H0-2 成立)。不終止研究，但論文/報告明確宣稱果蠅真實拓撲無特殊優勢。 |
| **Phase 3** (Sealed 盲測) | 1. 解密 `sealed_holdout_v2`<br/>2. 執行單次終端評估輸出 | `sealed_holdout_v2` (10,500 樣本) | Windows GPU 或 Mac CPU (擇一) | 1 工作天 | Ensemble $\text{BA} \ge 0.514$<br/>Permutation $p < 0.01$ | 宣告 RQ3 失敗 (H0-3 成立)。**結束未訓練實驗**，轉入帶 Plasticity 的強化學習。 |

#### FlyWire 資料下載與欄位需求清單 (供使用者手動下載)
- **下載網站**：FlyWire Codex (`https://codex.flywire.ai/download`)
- **必備檔案一：突觸連接表 (`connections.csv.gz` 或 `synapses.csv.gz`)**
  - 所需欄位：
    1. `pre_root_id` (int64)：突觸前神經元全腦唯一 ID。
    2. `post_root_id` (int64)：突觸後神經元全腦唯一 ID。
    3. `syn_count` (int)：突觸連接數量（作為權重初始值）。
    4. `nt_type` (str)：神經傳導物質類型（用於 Dale's law：GABA/Glutamate 判為抑制性，其餘判為興奮性）。
- **必備檔案二：細胞分類與註釋表 (`classification.csv` 或 `cell_types.csv.gz`)**
  - 所需欄位：
    1. `root_id` (int64)：神經元 ID。
    2. `super_class` (str)：大腦分區與功能類別（必須包含 `sensory`、`descending`、`visual_projection` 等關鍵字）。
    3. `cell_type` (str)：精細細胞類型名稱。
    4. `side` (str)：細胞體所在側（`left` 或 `right`，用於無偏左右半腦 BUY/SELL 分組）。

---

### F. 對前期方法論與 Gate 設計缺陷的獨立批判及隱藏偏差 (Critical Review & Hidden Biases)

#### 1. Claude 前期四大方法論錯誤診斷
1. **Gate 4 指標概念致命倒置**：
   - Claude 設計的 $G4$ 門檻為：`max(flip, shuffle, mask) >= 0.2 SD`。
   - **實質錯誤**：在標準常態分佈下，若兩個變量完全獨立不相關，其差值的絕對值均值為 $\mathbb{E}[|X - Y|] = \sqrt{2} \cdot \sqrt{2/\pi}\sigma \approx 1.128\sigma$。當測試跑出 $1.209\sigma$ 時，這**絕非模型對輸入有反應，而是模型輸出發生了徹底的統計獨立跳變（白雜訊化）**。把「最大化隨機跳變」當作「通過敏感度測試」，是導致 v2 盲目推進的根本盲點。
2. **視覺隨機高斯投影徹底摧毀物理時空拓撲**：
   - 採用 $12288 \to 48$ 的隨機投影矩陣 $W_{in}$，忽視了金融時間序列沿橫軸演進的因果性與縱軸價格的尺度關係。隨機投影使模型淪為 Locality-Insensitive Hash，完全喪失特徵空間平滑度。
3. **未校準動力學引發 58% 神經元硬飽和與混沌**：
   - 稀疏矩陣之譜半徑達 6.23，興奮性比例達 80%，在沒有譜半徑歸一化下運算 32 步，網絡早已經進入非線性硬截斷區（$|x| > 0.95$ 達 58%）。神經元狀態被鎖死在 $\pm 1$，失去了作為 Reservoir 該有的「狀態線性分離與局部記憶」能力。
4. **Smoke Test 設計形式化**：
   - v1 的 Smoke test 僅以合成全白/全黑圖片跑通、確認無 NaN 與進程未崩潰，卻未檢查真實圖片上的輸出分佈，導致重大的直流偏置問題在浪費 190 萬次模擬後才被發現。

#### 2. 深入排查挖掘出的隱藏偏差 (Hidden Biases)
1. **Renderer 的離散跳變偽影 (Renderer Discretization Artifacts)**：
   - 在 `render_market.py` 中，蠟燭圖顏色依據收盤價與開盤價的高低決定為綠色 `(46, 204, 113)` 或紅色 `(231, 76, 60)`。
   - 當一根 K 線的漲跌幅僅有 $+0.0001\%$ 與 $-0.0001\%$ 時，物理價格無實質變化，但渲染顏色在紅綠通道上卻發生數百點的離散跳變。隨機投影會把這種微小價格噪聲放大為劇烈的感覺電流衝擊。
2. **Seed 偽重複導致的顯著性虛胖 (Seed Pseudo-Replication Bias)**：
   - 雖然模擬器有 30 個 seeds，但所有 seeds 面對的市場樣本時間點完全相同。若在 Bootstrap 或置信區間計算時未對 `sample_id` 整塊抽樣，有效樣本量會被虛增 30 倍，產生極端虛假的 $p$ 值。
3. **Nuisance 線性模型的非線性盲區 (Nonlinear Nuisance Blindspot)**：
   - 目前健康檢查使用線性回歸 $R^2 < 0.01$ 宣稱已排除亮度、非黑像素等干擾變量。但 Reservoir 是高度非線性系統，即使線性相關為 0，其高階交互作用或非線性互資訊（Mutual Information）仍可能完全被圖片非黑像素比例所主導。

---

### G. 陽性對照 (Positive Control) 與陰性對照設計 (Positive & Negative Controls)

在信噪比極低的真實金融市場中，若模型跑出 $\text{BA} = 0.500$，存在兩種截然不同的可能：
- **解釋 A**：金融市場完全弱型有效，任何未經學習的模型本就無法預測。
- **解釋 B**：整個計算管線（編碼、動力學、解碼）存在根本性斷裂，即使面對 100% 規律的訊號也無法輸出。

若無陽性對照，研究者永遠無法排除解釋 B。因此，必須在進入真實市場前建立「合成基準測試鏈」：

```
                    ┌─────────────────────────┐
                    │    合成基準測試鏈       │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
┌──────────────────┐                           ┌──────────────────┐
│ 陽性對照組 (PC)  │                           │ 陰性對照組 (NC)  │
└────────┬─────────┘                           └────────┬─────────┘
         │                                               │
  PC-1: 純幾何趨勢 (正/負斜率直線)                 NC-1: 幾何布朗運動 (純隨機漫步)
        驗證: margin 單調遞增/遞減                       驗證: BA 嚴格符合二項分佈
  PC-2: 方波週期跳變 (已知高低態)                   NC-2: 等規模隨機 ESN 網絡
        驗證: 狀態快速切換能力                           驗證: 排除隨機拓撲固有偏置
  PC-3: 可預測 AR(1) 序列 (自相關=0.8)             NC-3: 輸入隨機雜湊 (Random Hash)
        驗證: Fading Memory 滯後相關                     驗證: 排除高敏跳變偽依賴
```

#### 1. 陽性對照組 (Positive Controls)
1. **PC-1 (純幾何斜率趨勢測試)**：
   - 生成無噪聲的標準線性趨勢數據 $P(t) = P_0 \cdot (1 + k \cdot t)$，斜率 $k \in [-0.01, +0.01]$ 等間距取 100 組。
   - 預期：解碼出的 $\text{margin}$ 必須與斜率 $k$ 呈現強單調相關（Spearman $|\rho| \ge 0.80$）。
2. **PC-2 (週期方波/狀態跳變測試)**：
   - 模擬市場暴拉後橫盤與暴跌後橫盤兩種極端巨觀狀態。
   - 預期：模型在兩類狀態下的 $\text{margin}$ 分佈必須在統計上完全分離（Cohen's $d \ge 1.5$）。
3. **PC-3 (強自相關時序 AR(1) 訊號)**：
   - 生成 $x_t = 0.8 x_{t-1} + \epsilon_t$ 之連續信號，其未來 6 步方向具有確定性數學期望。
   - 預期：Connectome 必須能藉由其遞迴動態提取時序依賴，達到 $\text{BA} \ge 0.65$。

#### 2. 陰性對照組 (Negative Controls)
1. **NC-1 (純隨機漫步幾何布朗運動 GBM)**：
   - 生成漂移項為 0、波動率符合真實 BTC 的純隨機序列。
   - 預期：$\text{BA}$ 嚴格落在 $[0.490, 0.510]$ 之二項抽樣區間內，Permutation 檢定 $p$ 值服從 $[0, 1]$ 均勻分佈。
2. **NC-2 (隨機 Echo State Network)**：
   - 保持神經元數與稀疏度相同，但權重完全隨機且滿足 Dale's law。
   - 目的：評估果蠅真實拓撲 (Fly-Intact) 是否具有超越隨機網絡的統計顯著優勢。

---

## 2. 預先註冊清單 (Pre-registration Checklist)

以下所有參數、統計檢驗門檻與執行規則在開啟任何未見過資料前**全面凍結**，未經正式修訂程序不得改動：

### 2.1 數據與分割凍結
- **標的與週期**：Binance BTC/USDT 現貨，5 分鐘 K 線。
- **決策窗與預測窗**：輸入最近 48 根 K 線 (4 小時)，預測未來第 6 根 K 線 (30 分鐘) 之收盤價漲跌。
- **取樣步長 (Stride)**：`sample_stride = 6`（樣本標籤期間嚴格無重疊）。
- **Embargo 機制**：Split 之間保留 $48 + 6 = 54$ 根 K 線的隔離帶，禁止邊界時序滲漏。
- **資料雜湊**：`raw_ohlcv.parquet` (2020-2026) 產生後立即計算 SHA-256 並寫入 `data_hash.txt`。

### 2.2 模擬器與動力學參數凍結
- **有效譜半徑**：$\rho_{\text{eff}}(W) = 0.95$（透過全域縮放固定）。
- **神經元洩漏率**：`leak = 0.5`。
- **模擬步數**：`steps = 32`。
- **飽和率門檻**：Train 激活絕對值 $|x| > 0.90$ 比例 $< 5.0\%$。
- **噪聲標準差**：微擾與平滑性測試 $noise\_std = 0.0$；環境噪聲魯棒性測試固定為輸入信號標準差的 $5.0\%$。

### 2.3 健康檢查與 Gate 通過門檻凍結
- **G1 (動作非崩潰)**：Val 上少數動作比例 $\ge 15.0\%$（原 5% 過於寬鬆）。
- **G2 (數值有限性)**：$\text{margin}$ 必須全數有限且 $\text{std} > 0$。
- **G3 (Between/Within 變異比)**：$\ge 3.0$（且在雙側對稱映射下重算）。
- **G4 (微擾平滑度 Lipschitz)**：$\epsilon = 1\%$ 噪聲下，相對變化率 $R_{\epsilon} \le 0.25$，自相關 $\rho_{\epsilon} \ge 0.90$。
- **G5 (結構翻轉反應)**：$|\text{Corr}(\text{margin}(x), \text{margin}(\text{flip}(x)))| \ge 0.40$。
- **G6 (干擾變量不相關)**：Nuisance 線性 $R^2 < 0.03$，且互資訊 $I(\text{Nuisance}, \text{Action}) < 0.01\text{ nats}$。

### 2.4 Level 評估標準凍結
- **主要評估指標**：Ensemble Balanced Accuracy ($\text{BA}$)。
- **Level 1 (平滑輸入依賴)**：通過 G1–G6，且陽性對照 PC-1 之 $|\rho_{\text{rank}}| \ge 0.50$。
- **Level 2 (超越隨機與非雜湊)**：Val 上 Input-Shuffle Permutation $p < 0.01$，且顯著優於 NC-2 隨機 ESN。
- **Level 3 (真實拓撲優勢)**：Fly-Intact 在特徵分離度或記憶容量上顯著超越 Degree-Preserved Scramble，Bootstrap 95% CI 下界 $> 0$。
- **Level 4 (真實預測能力)**：在未見過的 `sealed_holdout_v2` 上，Ensemble $\text{BA} \ge 0.514$，單尾 Permutation $p < 0.01$，且跨 2 個不重疊子時期方向一致。

---

## 3. 風險與未決事項 (Risks & Uncertainties)

1. **果蠅視覺系統與抽象 K 線圖的根本演化斷層**：
   - 果蠅視覺演化主要用於偵測物體運動邊緣、天敵陰影與光流導航，從未處理過人工定義的抽象幾何價格圖表。即便採用視網膜拓撲映射，其內在感受野能否表徵財務波動仍具有高度理論不確定性。
2. **無學習 Reservoir 的記憶容量極限**：
   - 48 根 K 線的輸入資訊量遠大於靜態氣味濃度。未經突觸塑性（Plasticity）調節的固定權重矩陣，其高維狀態空間可能迅速被過去的隨機波動淹沒，導致時序遺忘過快。
3. **Windows GPU 浮點精度與跨平臺可重現性**：
   - Mac (Apple Silicon ARM NEON / Accelerate) 與 Windows (x86_64 NVIDIA CUDA cuSPARSE) 在非線性函數 ($\tanh$) 與浮點稀疏矩陣相乘時，無法保證逐 bit 嚴格一致。
   - **處置規範**：所有正式註冊實驗（Phase 2 與 Phase 3）的完整資料、中間特徵與最終指標，必須鎖定在單一 Windows GPU 環境下一氣呵成跑完並留存 checkpoint，不可混合平臺結果。
4. **FlyWire 突觸權重的不確定性**：
   - `connections.csv` 提供的是突觸接觸數量 (`syn_count`)，並非生理學上的真實突觸後電位強度 (EPSP/IPSP)。以突觸數量作為權重存在簡化失真。

---

## 4. 需要使用者決定的問題清單 (Decisions Required from User)

在進入代碼實作與資料下載前，請使用者針對以下關鍵決策進行選擇或確認：

1. **輸入形式的路線選擇 (Architecture Choice)**：
   - **選項 A (推薦)**：放棄 64x64 圖像渲染，改採 **直接數值特徵注入 (Direct Feature Injection)**。將 48 根 K 線的價格變化率、振幅、成交量等直接注入神經元。徹底消除黑色背景偏差與隨機投影雜湊，速度快 100 倍。
   - **選項 B**：堅持使用 64x64 圖像，但必須重構為 **視網膜拓撲 (Retinotopic) 柱狀映射**（每欄對應特定視覺感受神經元），廢除隨機高斯投影。
2. **歷史 Sealed Holdout 資料取得方式**：
   - **選項 A (推薦)**：抓取 Binance BTC/USDT 5m 2020-09-19 至 2023-09-19 更早歷史（完全未被當前模型窺探過），封存為 `sealed_holdout_v2`。
   - **選項 B**：保持現有時間區間，改抓取同週期的 ETH/USDT 5m 作為跨資產封存檢驗集。
3. **FlyWire 數據下載授權**：
   - 是否願意至 `codex.flywire.ai` 手動下載 `connections.csv.gz` 與 `classification.csv`（約 2~6 GB），並傳輸至 Windows 機器？（若暫不下載，將在 Phase 0 陽性對照完成後再行決定）。
4. **運算資源調度時機**：
   - 確認 Phase 0 與 Phase 1 僅在 Mac 本機以 CPU 快速執行原型驗證；待通過 Phase 1 健康檢查後，才正式開機 Windows RTX 5070 GPU 進行全腦規模模擬。
