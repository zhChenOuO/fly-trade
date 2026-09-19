# input_encoding_v2 預先註冊選擇規則(寫於執行 v2_health.py 之前;不使用任何 label / future_return / test)
候選: mean_sub(去 Train 平均圖) | zero_black(黑=0,x/255) | on_off(ON/OFF 雙通道,以 Train 平均圖為基準)
margin = z_buy - z_sell,z = (每神經元平均活動 - Train 基準均值) / Train 基準 std;基準只用 Train、凍結;門檻固定為 margin>0 -> BUY。
注意: 因 z 以 Train 校準,Train 上 BUY≈50% 是構造結果;collapse 檢查因此以 Val 為準。
Gate(全部在真實 Val 圖片,n=1000 取樣;重複測試用 200 輸入 x 10 次不同 noise):
 G1 minority_action_ratio >= 0.05   G2 margin 全有限且 std>0
 G3 between/within variance ratio >= 3
 G4 flip_price / shuffle_time / mask_recent25 至少一種 mean|Δmargin| / SD_train(margin) >= 0.2
 G5 mean-image 的 margin 分布與真實 Val margin 有差異(KS p<0.05)
選擇: 通過 G1-G5 者中,取「nuisance R²(margin 被 brightness/nonzero_ratio/energy/price_range/up/down/volatility 線性解釋的比例)」最小者;
差距 <0.02 視為平手,依 mean_sub > zero_black > on_off 順序取先者。無人通過 => 停止,不進入 Level 1-3。
