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


import builtins
import logging
import os
import warnings

import torch

torch.cuda.set_device(int(os.getenv("SLURM_LOCALID", "0")))


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))


rank = get_rank()


class RankZeroFilter(logging.Filter):
    def filter(self, record):
        return rank == 0


class DeduplicateFilter(logging.Filter):
    """Filter that only allows each unique message once."""

    def __init__(self):
        super().__init__()
        self._seen = set()

    def filter(self, record):
        msg = record.getMessage()
        if msg in self._seen:
            return False
        self._seen.add(msg)
        return True


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("root").disabled = True
logging.getLogger("megatron.bridge.training.utils.log_utils").setLevel(logging.WARNING)
logging.getLogger("megatron.core.utils").disabled = True
logging.getLogger("megatron.core.optimizer_param_scheduler").setLevel(logging.WARNING)
logging.getLogger("megatron.core.distributed.param_and_grad_buffer").setLevel(logging.WARNING)
logging.getLogger("megatron.core.datasets").setLevel(logging.WARNING)
logging.getLogger("megatron.core.datasets.indexed_dataset").setLevel(logging.WARNING)
logging.getLogger("megatron.core.datasets.blended_megatron_dataset_builder").setLevel(logging.WARNING)
logging.getLogger("megatron.core.datasets.megatron_tokenizer").setLevel(logging.ERROR)
logging.getLogger("megatron.core.transformer.fsdp_dtensor_checkpoint").setLevel(logging.WARNING)


cg_logger = logging.getLogger("megatron.core.full_cuda_graph")
cg_logger.addFilter(RankZeroFilter())
llama_logger = logging.getLogger("megatron.bridge.models.llama.llama_provider")
llama_logger.disabled = True
llama_logger.propagate = False
ddp_logger = logging.getLogger("megatron.core.distributed.distributed_data_parallel")
ddp_logger.addFilter(RankZeroFilter())
ddp_logger.addFilter(DeduplicateFilter())

_original_print = builtins.print
_seen_messages = set()


def print_once_rank0(*args, **kwargs):
    # Rank check
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    elif rank != 0:
        return

    # Deduplicate
    msg = " ".join(str(a) for a in args)
    if msg not in _seen_messages:
        _seen_messages.add(msg)
        _original_print(*args, **kwargs)


builtins.print = print_once_rank0
