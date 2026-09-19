# FlyWire v783 資料(未納入版本控制)
公開資料,不需登入;下載後以 `SHA256SUMS.txt` 驗證。
```
B=https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main
A=https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files
curl -L -o Connectivity_783.parquet $B/Connectivity_783.parquet
curl -L -o Completeness_783.csv $B/Completeness_783.csv
curl -L -o Supplemental_file1_neuron_annotations.tsv $A/Supplemental_file1_neuron_annotations.tsv
shasum -a 256 -c SHA256SUMS.txt
```
`graph_cache.npz` 由 `research/pipeline/flywire_graph.py` 首次載入時自動產生。
市場資料 `research/data/images.npy` 由 `research/pipeline/build_dataset.py` 重建(hash 見 `research/data/data_hash.txt`)。
