pushd "$DATADIR"
pushd llama3_1_8b_tokenizer
rm config.json
rm generation_config.json
rm llama-3-1-8b-tokenizer.md5
rm model*.safetensors*
rm -rf original
popd
mv llama3_1_8b_preprocessed_c4_dataset 8b
mv llama3_1_8b_tokenizer 8b/tokenizer
popd