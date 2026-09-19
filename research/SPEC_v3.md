# SPEC_v3 — 真實 FlyWire v783 + 視網膜(複眼)映射。凍結於任何 v3 實驗執行之前。
延續 SPEC.md 的規則(只動自己負責的檔案、不 commit、不碰 test split 的 label、用 .venv/bin/python)。
資料: research/data/flywire/{Connectivity_783.parquet, Completeness_783.csv, Supplemental_file1_neuron_annotations.tsv}(SHA256SUMS.txt)。
事實(已驗證): Presynaptic_Index/Postsynaptic_Index 對齊 Completeness 的列序(0..138638);annotations 有 super_class/cell_type/side/pos_x,y,z。
光感受器 cell_type: R1-6(左4423/右4029)、R7(672/670)、R8(670/654);下行神經元 super_class=descending: left 646 / right 649 / center 8。

## 設計(預先註冊,不得因看到結果而改)
- 圖: W[post,pre] = Excitatory(±1) x Connectivity(突觸數),float32 CSR,神經元順序=Completeness 列序;neuron_ids=str(root_id);neuron_types=cell_type。
- sensory_idx = 全部 R1-6/R7/R8(依 root_id 遞增);motor_idx = descending 且 side∈{left,right}(center 排除)。
- BUY = 左半腦 DN,SELL = 右半腦 DN。因兩組數量不同(646 vs 649),graph.meta 必須提供 buy_idx / sell_idx,simulator 優先使用它們(沒有時退回既有的對半切分,舊 tests 不得壞)。score 用「每神經元平均活動」再以 Train 基準 z-score(既有 baseline 機制)。
- 視網膜映射: 每隻眼各自對 R1-6 的 3D 座標(pos_x,y,z)做 PCA,取前兩主軸標準化到 [0,1]^2 得 (u,v)(軸符號用固定規則:第一軸與 pos_x 正相關、第二軸與 pos_y 正相關;左右眼各自獨立做);兩眼都看見完整 64x64 圖(u→影像 x, v→影像 y)。R7、R8 使用同眼 R1-6 的 PCA 平面座標(以最近鄰 R1-6 的 (u,v) 代入)。每個光感受器取 (u,v) 處雙線性取樣的值: R1-6 ← 亮度 (mean RGB), R7 ← R 通道, R8 ← G 通道;取樣前皆已做 encoder mean_sub 編碼 (x - train_mean)/128。currents = 取樣值 * 1.0(input_scale 固定為 1)。
  輸出為 (B, n_sensory) 電流,FlySimulator 必須有「直接電流」模式(不經隨機投影)。
- 動力學: leak=0.5, steps=32。gain = min(0.95/ρ(W), g_sat),ρ(W)=最大特徵值模長(scipy.sparse.linalg.eigs);g_sat = 使 Train 真實圖片上非 sensory 神經元最後一步 |x|>0.9 的比例 <5% 的最大 gain(二分搜尋,只用 Train 輸入,不用 label)。gain 一旦決定即凍結並存入 outputs/v3/dynamics.json。
- noise_std: 確定性檢驗用 0;魯棒性檢驗用「Train 感覺電流標準差的 5%」。

## Phase 0 關卡(全在 Train/Val 真實圖片與合成價格上,label-free;全部通過才進 Phase 1)
 S1 飽和: 非 sensory 神經元 |x|>0.9 比例 < 5%(Train 300 張,最後一步)。
 S2 不塌縮: Val 500 張上 margin 全有限且 std>0,少數動作比例 >= 0.05。
 S3 語意平滑: Val 上相鄰時間窗(往後平移 1 根 K 線,47/48 根重疊)的 margin 相關 >= 0.5;同時「隨機獨立兩張圖」的相關作為基準必須 < 0.2。(雜湊型系統相鄰窗也會不相關。)另報 1% 像素噪聲下的 Δ/SD 與相關(僅報告,不設關卡,因任何連續系統都會通過)。
 S4 陽性對照(資訊是否存在於群體活動):合成價格(見下)渲染成同樣的圖,取全部 DN 的最後一步活動(1295 維)當特徵,用 ridge(alpha 以 Train-synthetic 內的 5-fold CV 選)在合成 train 集訓練、合成 test 集(不同隨機種子)評估:
    - PC-1 趨勢斜率 k∈[-0.01,0.01]/根: 斜率符號 balanced accuracy >= 0.95;斜率 R² >= 0.5。
    - PC-2 波動水準(3 檔): R² >= 0.5。
    - PC-3 AR(1) φ=0.8 序列,預測未來 6 根方向: BA >= 0.60(僅對可預測的合成訊號)。
   陰性對照 NC-1: GBM 純隨機漫步,同樣流程的 BA 必須落在 [0.49,0.51](95% 二項區間內)。任何陰性對照通過偏離 => 管線有洩漏,停止。
   注意: 零樣本固定 margin(左DN-右DN)不要求單調響應斜率,只報告 Spearman(僅報告)。S4 檢驗的是「資訊是否存在」,不是「任意固定讀出是否恰好有用」。
 S5 信噪: 200 個 Val 輸入 x 10 次不同 noise(5%)重複,between/within 變異比 >= 3。
決策: S1-S5 全過 => Go Phase 1(真實市場 label-free 健康檢查,之後才輪到 Windows GPU 與 sealed holdout);S4 失敗 => Stop(管線本身無法攜帶資訊);其他失敗 => 依失敗項修設計並記為新版本,不得靜默調參。
## 合成價格(pipeline/synthetic_prices.py):
 輸出 OHLCV 窗 (48,5),經 render() 渲染;trend: close=P0*(1+k*t)+小雜訊(相對波動 0.02%);vol: GBM,per-bar 波動 σ∈{0.05%,0.15%,0.4%};ar1: 對數報酬 AR(1) φ=0.8;gbm: 零漂移隨機。high/low/open 以 close 為中心加小幅合理擴張(需通過 render() 的 OHLC 檢查)、volume 取對數常態。固定種子,train/test 用不同種子。

## 修訂 1(2026-09-19 21:00,於「任何真實 FlyWire 的 Phase 0 執行之前」,起因:合成圖開發試跑中 NC-1 BA=0.484 落在 [0.49,0.51] 之外)
NC-1 原區間 [0.49,0.51] 是規格錯誤:n_test=1000 時純隨機 BA 的二項標準誤約 0.016,該區間會使正確的管線也常失敗。
改為: NC-1 的 BA 必須落在 0.5 ± 1.96*0.5/sqrt(n_test) 的雙側 95% 二項區間內(n_test=1000 時約 [0.469,0.531]);同時盡量把 NC-1 的 n_test 提高到 >= 2000。
此修訂只放寬「統計上不合理的過窄區間」,不改變任何其他關卡或參數;合成圖上的 NC-1 結果須以新區間重新判定。

## 修訂 2(2026-09-19 22:06;誠實註記:是在看到 Mac 部分結果 PC-2 R²=-0.019 之後才發現,但理由由「渲染器性質」獨立成立,並有下方回歸測試佐證)
PC-2(波動水準 R²>=0.5)在本規格下不可能通過,與管線好壞無關:render() 會把每個窗的價格正規化到整個畫面高度,絕對波動被抹除。
實測(50 條相同隨機衝擊,σ 放大 3x/8x): 平均每像素差 0.12/0.40 (0-255),而不同隨機路徑差 20.47 => 波動水準在圖上實質上看不見。
處置: PC-2 改為「僅報告(不設關卡)」,Phase 0 的 S4 判定只看 PC-1、PC-3、NC-1。其餘關卡與門檻不變。
研究含意(須寫入最終報告): 此渲染方式丟棄絕對尺度(波動水準、單根幅度),任何以這類圖片為輸入的零樣本管線都無法使用波動 regime 資訊。
S3 說明: Mac 部分結果 corr_adjacent=0.4894(n=200 對,標準誤約 0.05)在 0.5 邊緣,屬統計上不確定;門檻不變,改用更大的 n(>=500 對)重測,以更精確的估計為準。
