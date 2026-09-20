# REMOTE_TASK_P6 — Phase 6 有限度監督式訓練（OVERRIDE，EXPLORATORY）

你是 remote codex（GPU 機器）。先 `git fetch origin`，以 `origin/main` 建立新 worktree `.worktrees/phase6` 與分支 `remote/phase6`（不要推 main、不要 force push；用系統 git，這台沒有 rtk）。

## 0. 治理紀錄（先讀，這是本任務的前提）
- Phase 5 的預先註冊判定為 **FAIL**（real Ridge IC=0.0024 < 0.02；ΔIC vs random=-0.009、vs scramble=+0.003、vs OHLCV=-0.011；permutation p=0.42；見 `research/outputs/v2/probe_val.json`）。依 `research_v2.md` §12，本應停止、不進 Phase 6/7。
- **使用者於 2026-09-20 明確決定不跳過、繼續 Phase 6 實驗。** 這是對預先註冊停止規則的人為例外。因此：
  1. 本任務所有輸出一律標記 `PHASE6_OVERRIDE_EXPLORATORY`，**即使全部門檻通過，也不得視為確認性證據**（Phase 5 已失敗、Val 已被看過；只有 Future Sealed Holdout 才能提供確認性證據，而目前尚未收集）。
  2. 本規格在執行前凍結並 commit；執行後不得因結果修改任何設定。
  3. 不進 Phase 7（reward learning）。本任務只做 Phase 6。
- 絕對不得讀取/評估 `dev_test_v1` 與 holdout；只用 train 與 val。所有超參數只能用 **Train 內部時序切分** 選擇，**禁止用 Val 選任何東西**（含 epoch 數、lr、weight decay、k、hidden size）。
- 不得為了結果好看改動 alpha/超參數選擇方式、門檻、特徵、seeds。發現 bug 才可修正，並在報告註明「修正發生在看到哪個結果之前/之後」。

## 1. 必讀
`research_v2.md` §6 Phase 6、§8、§12；`research/config/experiment_v2.yaml`（thresholds，PROPOSED 者照用）；`research/pipeline/{extract_features_v2,train_probe_v2,baselines_v2,graph_variants,fly_simulator,torch_sim}.py`；`research/outputs/v2/probe_val.json`；`research/REMOTE_TASK_P5.md`。

## 2. 資料前置
- Phase 5 的特徵 `.npy` 在 `.worktrees/phase5/research/outputs/v2/features/`。新 worktree 用 symlink 放到同名路徑，並驗證 SHA256 與 `probe_val.json.metadata.feature_hashes` 一致，不一致就停止。
- `images.npy`、FlyWire 檔案、graph cache 沿用 phase5 worktree 的（symlink 或複製）；random/scramble 圖用與 Phase 5 相同的 seed 重建，並確認 spectral radius 歸一化與 Phase 5 相同（記錄 graph SHA256 與 phase5 的 metadata 比對）。

## 3. 凍結設計

原則（§6 Phase 6）：**不得一次解凍全部權重**；先只訓練 decoder，再只開放少量突觸；每個版本使用相同 seeds 與資料；保留 Frozen Connectome 作控制；**random 與 degree-preserved scramble 必須接受完全相同的訓練流程與可訓練參數預算**，才能區分「拓撲的貢獻」與「一般網路容量」。

共同設定：
- 網路：real / random / scramble 三種（與 Phase 5 相同圖）。輸出目標：primary = `future_return_6`（連續，Huber loss，delta=1 個標準化單位）；輔助 = 三類 action（交叉熵）。連續預測用於主要指標 Spearman IC。
- 特徵/標籤標準化只由 Train 計算。NaN 標籤剔除並記錄。
- 訓練 seeds：0..4（5 個；決定初始化與 batch 順序）。報告每個 seed 的 IC、跨 seed 平均 ± SD、方向一致率（IC>0 的 seed 比例）。
- 超參數只在 **Train 內部時序切分** 選擇：Train 前 80% 訓練、後 20% 驗證，兩者之間留 60 bar 的 purge（以樣本間距換算，並用測試證明不重疊）。網格固定如下，不得擴充：`lr ∈ {1e-3, 3e-4}`、`weight_decay ∈ {1e-4, 1e-2}`；early stopping patience=5、最多 50 epoch。選定後用整個 Train 重訓（epoch 數取 early stopping 的最佳 epoch）。Val 只在最後評估一次。

### Phase 6A — 只訓練 decoder（凍結 Connectome）
- 輸入：Phase 5 已萃取的 DN readout（1291 維，Train-only 標準化）。
- 模型：一個隱藏層 MLP，hidden=64，ReLU，dropout=0.1；同時輸出連續預測頭與三類 action 頭。
- 對照：real / random / scramble 各一份；OHLCV B2 MLP 已於 `baselines_val.json`（IC≈-0.003），直接引用，不重跑。

### Phase 6B — 只開放少量突觸（sparse plastic readout layer）
定義：對每個 readout 神經元 i（BUY∪SELL 的 DN，共 1291 個），在**該變體自己的圖**中取其入邊 |w| 最大的 k=16 條（不足 16 條則全取）作為可訓練突觸集合 E_i。可訓練參數為這些邊的權重增量 Δ_ij（初始 0，L2 正則）與一個線性 head。
- 上游來源神經元 j（E_i 的突觸前端）的活動特徵 x_j = 該圖在凍結動力學下、與 Phase 5 相同設定（noise seed 0、32 步）模擬窗口內的**時間平均活動**。需新增萃取程式（例如 `research/pipeline/extract_upstream_v2.py`），對 train/val 全部樣本輸出來源神經元活動，存 float16 `.npy`（**不 commit**，加入 .gitignore）；沿用 Phase 5 的一致性檢查思路（例如同一批 100 個 Val 樣本：用 W_ij 加權的 x_j 近似值與 DN readout 的相關係數，回報；不要求逐位一致，因為動力學是非線性）。
- 模型：`z_i = Σ_{j∈E_i} (W_ij + Δ_ij) · x_j`，`y = head(z)`（連續頭 + 三類頭）。W_ij 為該圖原始權重（凍結），只訓練 Δ 與 head。Δ 只存在於 E_i 的邊上（sparse mask，禁止任何其他邊被更新——寫測試證明 mask 外參數梯度為 0 且未被改動）。
- **可訓練參數預算對齊**：三個變體的可訓練突觸總數取三者最小值；較多者以「丟棄 |w| 最小的邊」的確定性規則裁到相同數量。在報告中列出三者裁切前後的邊數與 head 參數量。
- 對照：real/random/scramble 相同流程；另加 6B-null：用 real 圖但 E_i 改為**隨機選取**同樣數量的入邊（同 seed 規則），檢驗「挑選拓撲上最強邊」是否有貢獻。

### 評估（Val，一次）
每個模型：Spearman IC（+ block bootstrap 95% CI，block=24 樣本，2000 次）、MAE、balanced accuracy、MCC、每類 precision/recall、預測分布、含成本交易模擬（沿用 baselines_v2，成本取自 yaml；`trade_count==0` 時交易指標標 `N/A_NO_TRADES`，不得算 PASS）。
比較（皆用**配對** block bootstrap，同一組重抽區塊；報 ΔIC、95% CI、雙尾 p，並對整組比較做 Holm 校正）：real−random、real−scramble、real−(6B-null)、real−B1b(Logistic OHLCV)、real−B2(MLP OHLCV)、6B−6A（同一網路，是否開放突觸有增量）。
門檻：沿用 `experiment_v2.yaml` thresholds（PROPOSED 者照用並標示 status）；另外必須同時滿足「real 相對 random、scramble 的 ΔIC 95% CI 下界 > 0」才可稱為「優於 matched control」。**不得只因 p<0.05 宣稱通過**；效果量不足即使顯著也判 FAIL。輸出逐項 PASS/FAIL，並固定加上一行 `PHASE6_OVERRIDE_EXPLORATORY: NOT CONFIRMATORY EVIDENCE`。

## 4. 輸出與交付
- `research/outputs/v2/phase6_val.json`（所有指標、逐 seed 結果、選定超參數與 Train 內部驗證分數、可訓練參數量、樣本數、feature/graph SHA256、yaml SHA256、git commit）、`phase6_run.log`。
- 程式與測試：放 `research/pipeline/`、`research/tests/`。測試（小、不需 GPU 的部分必須在無 torch 時 skip 而非失敗，沿用既有風格）：Train 內部切分 purge 不重疊；sparse mask 外參數不更新；預算對齊裁切規則確定性；超參數選擇函式不接觸 Val（傳入 Val 資料時 raise）；合成資料上「含已知訊號的特徵」IC>0、「純噪音」IC≈0。
- 跑完先執行 `python -m pytest research/tests -q`（貼結果）。commit 並 push 到 `origin/remote/phase6`（不 commit 任何 `.npy`、decisions.parquet、graph cache）。commit 結尾加 `Co-Authored-By: Codex <noreply@openai.com>`。
- 最終回報：貼出 `phase6_val.json` 的逐項數值與 pass/fail、三個變體的可訓練參數量、耗時；**原樣回報不利結果，不美化**。

## 5. 立即停止並回報（不得自行繞過）
Phase 5 特徵 SHA256 不符；任何 NaN/Inf；上游來源活動全零或方差為 0 的比例 > 90%；可訓練參數預算無法對齊；發現任何程式路徑讀到 dev_test_v1 或 holdout；需要用 Val 選超參數才能繼續。
預估耗時（估計值）：6A 每模型數分鐘；6B 上游萃取與 Phase 5 同量級（三個圖各約數分鐘到十幾分鐘），訓練 5 seeds × 3–4 變體 × 4 組超參數，預計 1–2 小時。若預估超過 4 小時，先回報並提出「縮減 seeds 或網格」的選項，等待批准，不得自行縮減。
