mkdir -p "$DATADIR"
pushd "$DATADIR"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d 405b https://training.mlcommons-storage.org/metadata/mixtral-8x22b-preprocessed-c4-dataset.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d 405b/tokenizer https://training.mlcommons-storage.org/metadata/mixtral-8x22b-tokenizer.uri
popd
bash data_scripts/cleanup.sh
