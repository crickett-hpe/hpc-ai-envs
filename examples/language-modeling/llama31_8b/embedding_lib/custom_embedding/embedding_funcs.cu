/* Copyright (c) 2019-2025 NVIDIA CORPORATION. All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include <vector>
#include <iostream>
#include <assert.h>

#include <ATen/AccumulateType.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <torch/extension.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/CUDAContext.h>  // for getCurrentCUDAStream
#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAStream.h>    // for CUDAStream type
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/thread_constants.h>

static const int BLOCKDIMY = 32;

at::Tensor embedding_fwd(const at::Tensor & weight, const at::Tensor & indices) {
  TORCH_CHECK(weight.dim() == 2,  "'weight' must be 2-D");
  auto indices_arg = at::TensorArg(indices, "indices", 1);
  at::checkScalarTypes("embedding", indices_arg, {at::kLong, at::kInt});

  // TODO: use tensor.index() after improving perf
  if (indices.dim() == 1) {
    return weight.index_select(0, indices);
  }

  auto size = indices.sym_sizes().vec();
  for (const auto& d : weight.sym_sizes().slice(1)) {
    size.push_back(d);
  }

  return weight.index_select(0, indices.reshape(-1)).view_symint(size);
}

template
  <typename scalar_t,
   typename accscalar_t,
   typename index_t>
__global__ void embedding_backward_feature_kernel
  (const index_t* indices,
   const scalar_t* __restrict__ grad,
   scalar_t* __restrict__ grad_weight,
   int n, // OK to pass as int, we don't expect 2 billion+ samples in one shot
   int64_t stride)
{
  int padding_idx = -1;
  extern __shared__ char buf[];
  accscalar_t* smem = (accscalar_t*)buf;
  accscalar_t* my_s = smem + C10_WARP_SIZE*threadIdx.y;
  int* indices_batch = (int*)(buf + sizeof(accscalar_t)*C10_WARP_SIZE*blockDim.y);

  const int s = (int)stride; // OK to make int, we don't expect 2 billion+ embedding row size

  const int f = threadIdx.x + blockIdx.x*blockDim.x; // feature_dim

  for(int batch_start = 0; batch_start < n; batch_start += blockDim.x*blockDim.y)
  {
    // Entire block cooperates to load a batch of 1024 indices to process
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    if(batch_start + tid < n)
      indices_batch[tid] = (int)indices[batch_start + tid];

    int batch_end = batch_start + blockDim.x*blockDim.y < n ?
                    batch_start + blockDim.x*blockDim.y : n;

    // Loop over the batch of <= 1024 loaded indices in chunks of blockDim.y = 32
    for(int chunk_start = batch_start; chunk_start < batch_end; chunk_start += blockDim.y)
    {
      // This does double duty:  it makes sure indices_batch is ready, and it makes sure match-group
      // leaders are done with their accumulates before other warps start loading again.
      __syncthreads();

      int n_this_chunk = (batch_end - chunk_start) < blockDim.y ?
                         (batch_end - chunk_start) : blockDim.y;

      int src_row = chunk_start + threadIdx.y;
      int dst_row = indices_batch[src_row - batch_start]; // This warp's target row in grad_weight

      // All warps load their smem segments with incoming grad data
      if(src_row < n && f < s && dst_row != padding_idx)
        my_s[threadIdx.x] = static_cast<accscalar_t>(grad[src_row*stride + f]);

      __syncthreads();

      // To ensure determinism, we can't just have each warp add its grad data to its dst_row.
      // We need to check if any other warps pulled grad data targeting dst_row.
      // If so, we elect the first warp in each matching group as the leader.
      // Each leader warp serializes the accumulates targeting dst_row in shared memory,
      // then finishes by adding the accumulated buffer to dst_row in grad_weight.
      if(dst_row != padding_idx && src_row < n) // Per-warp exit condition, safe with ballot_sync
      {
        int match_found_this_thread = 0;
        if(threadIdx.x < n_this_chunk)
          match_found_this_thread = (dst_row == indices_batch[chunk_start - batch_start + threadIdx.x]);
        unsigned int matchmask = WARP_BALLOT(match_found_this_thread);
        int first_remaining_peer = __ffs(matchmask) - 1;

        if(threadIdx.y == first_remaining_peer) // Nominate lowest-indexed warp as the leader
        {
          matchmask ^= (1 << first_remaining_peer);
          while(matchmask)
          {
            first_remaining_peer = __ffs(matchmask) - 1;
            my_s[threadIdx.x] += smem[threadIdx.x + C10_WARP_SIZE*first_remaining_peer];
            matchmask ^= (1 << first_remaining_peer);
          }
          if(f < s)
            grad_weight[dst_row*stride + f] += static_cast<scalar_t>(my_s[threadIdx.x]);
        }
      }
    }
  }
}
at::Tensor embedding_bwd(const at::Tensor & grad_, const at::Tensor & indices_, int64_t num_weights, at::Tensor grad_weight) {
  auto grad_arg = at::TensorArg(grad_, "grad", 1);
  auto indices_arg = at::TensorArg(indices_, "indices", 1);
  at::checkScalarTypes("embedding_backward", indices_arg, {at::kLong, at::kInt});
  at::checkSameGPU("embedding_backward", grad_arg, indices_arg);

  auto indices = indices_.contiguous();

  auto num_indices = indices.numel();
  auto grad = grad_.contiguous().view({num_indices, grad_.size(-1)});
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto indices_contig = indices.contiguous();
#if 0
  auto grad_weight = at::zeros({num_weights, grad_.size(-1)}, grad_.options());
  std::cout << "Alloc grad_weight: " << num_weights << "," << grad_.size(-1) << "\n";
#endif
  int64_t stride = grad_weight.stride(0);
  int warp_size = at::cuda::warp_size();
  dim3 grid(at::ceil_div(stride, (int64_t)warp_size));
  dim3 block(warp_size, BLOCKDIMY);

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    grad.scalar_type(),
     "embedding_backward",
     [&]
     {
        using accscalar_t = at::acc_type<scalar_t, true>;
        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_dense_backward_cuda", [&] () {
        embedding_backward_feature_kernel<scalar_t, accscalar_t, index_t>
          <<<grid,
              block,
              sizeof(accscalar_t)*warp_size*BLOCKDIMY + sizeof(int)*warp_size*BLOCKDIMY,
              stream>>>
          (indices_contig.const_data_ptr<index_t>(),
            grad.const_data_ptr<scalar_t>(),
            grad_weight.mutable_data_ptr<scalar_t>(),
            static_cast<int>(num_indices),
            static_cast<int64_t>(stride));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
     });
  return grad_weight;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("embedding_fwd", &embedding_fwd, "embedding_fwd");
  m.def("embedding_bwd", &embedding_bwd, "embedding_bwd");
}
