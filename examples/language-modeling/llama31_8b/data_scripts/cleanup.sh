pushd $PREPROCESSED_PATH
rm c4-train.en_0_text_document.bin
rm c4-train.en_0_text_document.idx
rm c4-train.en_1_text_document.bin
rm c4-train.en_1_text_document.idx
rm c4-train.en_2_text_document.bin
rm c4-train.en_2_text_document.idx
rm c4-train.en_3_text_document.bin
rm c4-train.en_3_text_document.idx
rm c4-train.en_4_text_document.bin
rm c4-train.en_4_text_document.idx
rm c4-train.en_5_text_document.bin
rm c4-train.en_5_text_document.idx
mv c4-validation-91205-samples.en_text_document.bin/c4-validationn-91205-samples.en_text_document.bin _c4-validationn-91205-samples.en_text_document.bin
mv c4-validation-91205-samples.en_text_document.idx/c4-validationn-91205-samples.en_text_document.idx _c4-validationn-91205-samples.en_text_document.idx
rm -r c4-validation-91205-samples.en_text_document.bin
rm -r c4-validation-91205-samples.en_text_document.idx
mv _c4-validationn-91205-samples.en_text_document.bin c4-validation-91205-samples.en_text_document.bin
mv _c4-validationn-91205-samples.en_text_document.idx c4-validation-91205-samples.en_text_document.idx
rm c4-validation-small.en_text_document.bin
rm c4-validation-small.en_text_document.idx
rm c4-validation.en_text_document.bin
rm c4-validation.en_text_document.idx
popd
