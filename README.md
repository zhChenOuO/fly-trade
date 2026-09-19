# flybrain-trader

用果蠅(*Drosophila*)全腦連接體(connectome)的真實拓撲結構,當作一個
固定的大型遞迴神經網路(reservoir),搭配一個小型可訓練的輸入/輸出層,
把加密貨幣價格特徵轉換成買/賣/持有訊號。

## 核心設計:為什麼不是「訓練整個連接體」

苍蝇全腦連接體(FlyWire)大約有 **13-14 萬顆神經元、3000 萬個突觸連
結**。如果想對這 3000 萬個權重做端對端反向傳播訓練,即使在桌機等級
的 GPU 上都很吃力,在 M1 筆電上並不實際。

這個專案採用的是 **reservoir computing(儲備池運算)** 的做法,這在
運算神經科學裡有明確的先例(拿固定的、有生物結構的遞迴網路當作
「隨機特徵產生器」,只訓練讀出層):

- 連接體的突觸權重 **完全不訓練**,只當作固定的遞迴矩陣。它的角色是
  提供一個有真實生物結構(模組化、局部群聚、少數長程投射)的非線性
  動態系統,把輸入的價格特徵轉換成高維、帶時間記憶的神經活動模式。
- 真正被訓練的參數只有:
  1. **輸入編碼層**:市場特徵 → 感覺神經元電流(數量 = 感覺神經元數
     × 特徵數,通常幾百個參數)
  2. **讀出層**:運動神經元活動 → 買/賣/持有 3 個動作的分數(數量 =
     運動神經元數 × 3)
  3. 兩個全域純量:訊號增益(gain)、神經元漏電係數(leak)
- 用 **CMA-ES**(一種不需要梯度的演化策略)訓練這幾百個參數,目標是
  最大化回測 Sharpe ratio。CMA-ES 不需要對 reservoir 做反向傳播,每
  一代只需要「跑一次前向模擬 → 算一個分數」,計算瓶頸只在一次稀疏矩
  陣乘法(複雜度正比於突觸數量,而不是神經元數量的平方)。

這個設計的直接好處:**訓練所需的運算量幾乎跟連接體大小無關**,即使
之後把連接體從幾千顆神經元的合成資料換成 13 萬顆神經元的 FlyWire 全
腦資料,訓練這一步的成本增加的其實有限(主要成本轉移到「模擬一次前
向動態」,而不是訓練參數量)。

## 在 M1 Mac 上跑得動嗎?

跑得動,而且不需要 GPU:

- 前向模擬只是「稀疏矩陣 × 向量」重複幾千次,用 numpy + scipy.sparse
  就能跑,M1 的 CPU(尤其搭配 Apple 的 Accelerate framework,numpy 在
  macOS 上預設會用到)處理這種規模綽綽有餘。
- 合成資料(幾千顆神經元)或 hemibrain(2.5 萬顆神經元、2000 萬突觸)
  規模,在 8GB 機型上就能跑;完整 FlyWire 全腦(13-14 萬顆神經元、
  3000 萬突觸)建議至少 16GB 統一記憶體,單純「載入 + 前向模擬」沒問
  題,但一次性下載/組矩陣那步會花比較久時間、吃比較多記憶體,建議只
  做一次然後存快取(見下方「使用真實連接體」)。
- 如果之後想升級成真正的 spiking neural network(逐一模擬神經元的放
  電時序,而不是這裡簡化的 leaky-rate 模型),可以考慮 `mlx-snn`
  (原生跑在 Apple MLX 上,吃到 M 系列晶片的統一記憶體架構)或
  `snnTorch` / `Norse`(建在 PyTorch 上,PyTorch 的 MPS 後端可以用到
  M1 的 GPU),但這兩者目前都沒有現成的「不需要反向傳播」訓練方式,
  要嫁接演化策略需要自己寫。目前這個專案先用最務實、最容易驗證的
  leaky-rate reservoir,把訓練方式的複雜度降到最低。

## 市場資料要 API key 嗎?

不用。`main.py` / `run_evolution.py` / `run_forward_test.py` 抓的都是
交易所的「公開行情」端點(K 線/OHLCV),任何人都能查,不需要註冊帳
號,更不需要 API key/secret(那是下單、查自己帳戶餘額才需要的東西)。

程式預設會依序嘗試 `binance → okx → kraken → coinbase → bybit`(可在
`configs/default.yaml` 的 `market.exchange_ids` 調整順序或增減),第
一個連得上就用它,全部都連不到才會退回合成資料。這次我在雲端環境裡
測試時,這五個交易所的公開 API 全部被沙盒的網路白名單擋掉了(不是
因為缺 API key),所以你看到的示範結果都是合成資料;但在你自己的
Mac 上,一般家用/公司網路通常至少有一個連得上,連上後就會自動改用
真實的 BTC/USDT 歷史 K 線。

## 「有賺錢就留著,下次再進化」怎麼做:champion 機制

除了 `main.py`(單次示範用,每次都從頭訓練、不會記住結果)之外,專案
另外提供兩個腳本,實作你想要的「保留 + 持續進化」流程:

### `run_evolution.py` —— 訓練 + 決定要不要取代目前最好的解

```bash
python3 run_evolution.py --config configs/default.yaml
```

每次執行是一個新的「世代」:

1. 資料切成 train(訓練)/ validation(決定要不要保留這次的新解)/
   test(完全不參與決策,只在最後老實報一次成績)三段。
2. 如果本機已經存有 champion(之前跑過至少一次),新一代會從舊
   champion 的參數附近開始搜尋(warm start),等於在舊解基礎上繼續
   進化,而不是每次從頭亂猜。
3. 新解只有在 validation Sharpe 顯著贏過舊 champion(預設要多贏 0.05
   以上,可在 config 的 `champion.min_improvement` 調整,避免雜訊型
   的假進步洗掉原本能用的解)才會取代它;贏不夠多或更差,舊 champion
   原封不動保留,這次嘗試只會被記錄下來,不會覆蓋掉任何東西。
4. 第一次跑(還沒有任何 champion)時,新解必須 validation Sharpe 是
   正的才會被存下來當作第一代 champion;訓練不出賺錢的解就不存,你
   可以調整 config(神經元規模、訓練代數、種子)重跑。

每次執行的結果(不管有沒有取代 champion)都會累積寫進
`outputs/champion/history.csv`,可以打開來看每一代的進步軌跡。

### `run_forward_test.py` —— 不訓練,只檢查 champion「現在」還有沒有在賺

```bash
python3 run_forward_test.py --config configs/default.yaml
```

這個腳本完全不訓練,只是把現有 champion 套用在**最新一段**市場資料
上(config 的 `champion.forward_test_bars`,預設抓最後 200 根 K 棒),
算一次沒有偷看未來、也沒有重新訓練的「樣外」表現,結果會累積寫進
`outputs/champion/forward_ledger.csv`。

建議用法:**定期(例如每天或每週)重新執行這個腳本**,把好幾次的紀錄
攤開來看,觀察 champion 的表現是穩定、變好、還是在變差——這比只看單
一次回測結果可靠很多,單一區間的 Sharpe 雜訊非常大。如果持續變差,
再考慮回頭執行 `run_evolution.py` 讓它繼續進化,或乾脆承認這個方向
在目前的市場條件下不管用。

champion 相關的檔案都在 `outputs/champion/`:
- `theta.npy` / `meta.json`:目前保留的最佳參數與其資訊(第幾代、
  train/val/test 表現、訓練時間等)
- `history.csv`:每一次執行 `run_evolution.py` 的嘗試紀錄
- `forward_ledger.csv`:每一次執行 `run_forward_test.py` 的監控紀錄

## 快速開始

```bash
cd flybrain-trader
python3 -m venv .venv && source .venv/bin/activate   # 建議用虛擬環境
pip install -r requirements.txt

# 跑一次煙霧測試,確認環境裝好、程式邏輯沒問題(用很小的合成資料,幾秒內跑完)
python3 tests/test_smoke.py

# 跑完整 pipeline(預設用合成連接體 + 會先嘗試抓 BTC/USDT 真實資料,
# 抓不到就自動退回合成價格序列)
python3 main.py --config configs/default.yaml
```

跑完會在 `outputs/` 產生:
- `best_theta.npy`:訓練好的參數(輸入編碼層 + 讀出層 + gain/leak)
- `train_equity_curve.npy` / `test_equity_curve.npy`:訓練集/測試集
  的權益曲線,可以自己用 matplotlib 畫圖看

## 專案結構

```
src/
  connectome/
    schema.py            # 共用的 ConnectomeGraph 資料結構
    synthetic.py          # 合成小世界連接體產生器(不需要任何帳號就能跑)
    hemibrain_loader.py   # 從 Janelia neuprint 抓 hemibrain 真實連接體
    flywire_loader.py      # 讀取 FlyWire codex 下載的本機 CSV(全腦連接體)
    build_graph.py         # 連接體的存檔/讀取快取
  network/
    reservoir.py           # leaky-rate 遞迴動態模擬(核心運算)
    encoding.py            # 市場特徵 -> 感覺神經元電流
    readout.py             # 運動神經元活動 -> 買/賣/持有動作
  data/
    market_data.py         # ccxt 抓 Binance OHLCV,含合成資料備援
    features.py             # 特徵工程(報酬率、波動度、RSI...)
  train/
    evolve.py               # CMA-ES 訓練迴圈
  backtest/
    engine.py               # 向量化回測(避免 look-ahead bias)
    metrics.py               # Sharpe / 最大回撤 / 總報酬
  train/
    evolve.py               # CMA-ES 訓練迴圈(支援 warm start)
    champion.py              # champion(目前最好解)的存讀 + 歷史紀錄
main.py                      # 單次示範:端到端跑一次,不記住結果
run_evolution.py             # 持續進化:訓練 + 決定要不要取代 champion
run_forward_test.py          # 不訓練,檢查 champion 在最新資料上是否還在賺
configs/default.yaml         # 所有可調參數
tests/test_smoke.py          # 小規模端到端煙霧測試
outputs/champion/            # run_evolution.py / run_forward_test.py 的存檔與紀錄
```

## 換成真實的果蠅連接體

### 選項 A:hemibrain(局部連接體,2.5 萬顆神經元,較快上手)

1. 到 https://neuprint.janelia.org 註冊帳號並登入,在 Account 頁面複
   製你的 API token
2. `export NEUPRINT_APPLICATION_CREDENTIALS="你的token"`
3. 把 `configs/default.yaml` 的 `connectome.source` 改成 `hemibrain`
4. 執行前建議先用 `hemibrain_loader.load_hemibrain_connectome(max_neurons=5000)`
   跑一個子集合測試,確認流程沒問題再抓全量

### 選項 B:FlyWire(全腦連接體,13-14 萬顆神經元,規模最大最完整)

1. 到 https://codex.flywire.ai 註冊帳號,登入後在帳號頁面複製 API
   token
2. FlyWire 官方不提供「重複呼叫」的批次查詢 API,而是希望使用者從
   Codex 網頁的 Download 頁面手動下載靜態檔案,至少需要:
   - `connections.csv`(神經元之間的突觸連結與強度)
   - `classification.csv`(每個神經元的細胞類型標註)
3. 把兩個檔案路徑填進 `flywire_loader.load_flywire_connectome(...)`
4. 建好 `ConnectomeGraph` 之後,務必用 `build_graph.save_cache(...)`
   存成本機快取,之後訓練直接讀快取,不要每次重新解析 CSV(全量資料
   解析可能要幾分鐘到十幾分鐘)

無論哪個資料源,實際的 sensory/motor 神經元判斷都是用細胞型別名稱
關鍵字(`ORN`、`PN`、`DN` 之類)做的粗略猜測,連接體的官方標註方式
偶爾會改版,拿到資料後建議先印出 `neuron_types` 檢查一下,必要時自己
調整 `SENSORY_TYPE_PREFIXES` / `MOTOR_KEYWORDS`。

## 老實講的限制(這不是免責聲明客套話,是真的要注意)

跑過一次預設設定(合成連接體 + 合成價格資料)之後你會看到類似這樣的
結果:

```
[train] sharpe=+7.53  total_return=+120%   ...
[test ] sharpe=-3.25  total_return=-21.5%  ...
```

**訓練集表現很好、測試集直接翻負,這是過擬合的典型樣子,不是程式
有 bug。** 這件事值得特別強調:

1. **連接體的拓撲跟金融市場完全沒有因果關係。** 果蠅演化出這套神經
   迴路是為了處理氣味、視覺、飛行控制,不是為了預測 BTC 價格。把它
   當 reservoir 用,本質上等同於用一個「有特殊統計結構的隨機特徵產
   生器」,跟拿一個隨機初始化的 Echo State Network 做交易訊號,在方
   法論上沒有本質差異——連接體只是提供一種比較有趣、有生物意義的隨
   機性,不代表訊號會更準。
2. **CMA-ES 的參數量雖然不多(幾百個),但金融時間序列雜訊很高、樣
   本量相對有限,一樣很容易過擬合訓練區間的雜訊。** 這裡示範的
   train/test 切分是最基本的防線,實務上應該再加上 walk-forward 驗
   證、多組隨機種子測試穩定性、以及跟簡單 baseline(例如買入持有、
   移動平均交叉)比較,才能判斷這個方法是不是真的學到什麼,還是純
   粹在擬合雜訊。
3. **回測不考慮滑價、真實成交深度、資金費率(若做永續合約)等因
   素**,`fee_bps` 只是一個粗略的手續費估計,實際交易成本通常更高。

簡單說:這是一個很適合拿來玩、學習 reservoir computing / 演化策略 /
連接體資料的專案,但目前的結果**不構成任何交易建議**,不要直接拿訓
練出來的參數去接真實資金的自動交易。
