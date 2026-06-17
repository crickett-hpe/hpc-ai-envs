# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
