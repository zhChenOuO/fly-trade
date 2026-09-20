# REMOTE_TASK — 在這台機器(GPU)上跑 FlyWire Phase 0
你是 remote codex。repo=fly-trade(本目錄)。先 `git pull origin main`。設計與關卡已凍結,**不得更動任何 SPEC / 關卡門檻 / 參數**。

## 背景(必讀)
research.md(方法)、research/SPEC.md、research/SPEC_v3.md(含「修訂 1」)、research/PLAN_v2.md、research/pipeline/{fly_simulator,flywire_graph,retina,phase0,synthetic_prices,render_market}.py。
Mac 上的 Phase 0 在單核 CPU 跑真實圖(約 1 秒/張)太慢,所以改到這台機器的 GPU 跑。**不要使用 test split 的 label / future_return**(phase0 也不需要)。

## 步驟(依序;每步先驗證再往下)
0. 回報環境:`nvidia-smi` 或 `rocm-smi`、GPU 型號與 VRAM、CPU、RAM、磁碟、python 版本。建 `.venv` 並 `pip install -r requirements.txt`;另裝符合此 GPU 的 torch(RTX 50 系列需 CUDA 12.8+ 的 wheel;AMD 則用 ROCm 版,若該 GPU 無法支援,改用 CPU 多行程並說明)。
1. 資料:依 research/data/flywire/DOWNLOAD.md 下載三個檔,`shasum -a 256 -c SHA256SUMS.txt` 必須全部 OK。
2. 重建 `research/data/images.npy`(已被 gitignore):寫 `research/pipeline/rebuild_images.py`,由 research/data/raw_ohlcv.parquet + research/data/samples.parquet,用 `render()` 與 `research/run_experiment.py` 的 `get_windows` 相同的視窗規則(決策 bar = timestamp-5min,取其前 48 根含自身)逐樣本渲染成 uint8 (N,64,64,3),順序依 image_idx。**必須**與 research/data/audit/*.png 逐位元一致(那 100 張檔名含 sample_id),否則停止並回報。不要重新抓交易所資料(會讓資料集漂移)。
3. 寫 `research/pipeline/torch_sim.py`:與 `src/network/reservoir.py::LeakyReservoir.simulate_batch` 相同動力學(x <- (1-leak)x + leak*tanh(gain*(W@x) + I),I 只注入 sensory),float32、`torch.sparse_csr_tensor` 於 GPU、分批使 VRAM 用量 <= 80%;並在 `FlySimulator` 加 `backend="scipy"|"torch"`(**預設 scipy,舊行為與既有 85 個 tests 不得壞**)。分數定義與 z-score、buy_idx/sell_idx、noise 行為都要與 scipy 版相同。
4. 驗證等價:(a) 合成圖(config 的 synthetic 設定)上 torch vs scipy 的 margin Pearson >= 0.9999 且 action 一致率 >= 99.9%;(b) 真實圖、20 張真實 Train 圖(direct_currents + retina_encoder)margin Pearson >= 0.999。寫成 tests/test_torch_sim.py(無 GPU 時 skip)。全部 tests:`.venv/bin/python -m pytest research/tests -q` 必須全過。
5. 回報吞吐(張/秒,多個 batch size)與 VRAM 峰值。
6. 執行 Phase 0(真實圖):`.venv/bin/python research/pipeline/phase0.py --graph flywire ...`;phase0.py 只允許做最小修改以支援 `--backend torch`(以及輸出檔名加 `_remote` 後綴:outputs/v3/phase0_report_remote.json、dynamics_remote.json),不得改任何關卡邏輯或門檻。樣本數若 GPU 夠快請用較大值以提高檢定力:S1=300、S2=500(規格固定值),S3 相鄰窗 >=500 對、S4 合成集 >=1000、NC-1 n_test>=2000、S5=200 輸入x10 次重複。
7. 交付:在分支 `remote/phase0-gpu`(**不要推 main,不要 force push**)commit 並 push:程式、tests、rebuild_images.py、outputs/v3/*_remote.json。不要 commit 任何大資料檔(images.npy、FlyWire parquet/npz)。commit 結尾加 `Co-Authored-By: Codex <noreply@openai.com>`。
8. 最後回報:硬體、吞吐、等價驗證數值、Phase 0 各關卡數值與 GO/STOP、任何偏離本文件之處(不得自行偏離,只回報)。若任何一步失敗,停下並如實回報失敗輸出,不要為了過關而改關卡或門檻。
