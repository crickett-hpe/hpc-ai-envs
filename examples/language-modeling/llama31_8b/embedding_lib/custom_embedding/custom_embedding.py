# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from typing import Any, Callable, List, Optional, Tuple, Union
from . import embedding_ops

global_embed_wgrad = None
class CustomEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        weight,
        indices,
        num_embeddings,
    ):
        ctx.num_embeddings = num_embeddings
        ctx.save_for_backward(indices)
        return embedding_ops.embedding_fwd(weight, indices).clone()

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (indices,) = ctx.saved_tensors
        global global_embed_wgrad
        if global_embed_wgrad is None:
            global_embed_wgrad = torch.empty((ctx.num_embeddings, grad_output.size(-1)), dtype=grad_output.dtype, device=grad_output.device)
        global_embed_wgrad.detach().zero_()
        global_embed_wgrad = embedding_ops.embedding_bwd(grad_output, indices, ctx.num_embeddings, global_embed_wgrad)
        return (global_embed_wgrad, None, None,)
