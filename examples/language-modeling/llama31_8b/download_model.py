# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_dir: str, max_workers: int):
    try:
        snapshot_download(
            "michal1270/llama31_405b_megatron_bridge",
            local_dir=model_dir,
            max_workers=max_workers
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise
    else:
        logger.info("Model download completed successfully")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Download model weights')
    parser.add_argument("--model_dir", default="/ckpt", type=str, help="Path to save the downloaded model")
    parser.add_argument("--max_workers", default=32, type=int, help="Maximum number of workers for downloading")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    download_model(args.model_dir, args.max_workers)
