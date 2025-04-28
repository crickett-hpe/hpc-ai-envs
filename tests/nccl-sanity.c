#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>       // For uint64_t used in calculation
#include <mpi.h>          // For MPI Process Management
#include <cuda_runtime.h> // For CUDA Runtime API (needed for cudaGetDeviceCount)
#include <nccl.h>         // For NCCL API

// Error checking macros
typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testTimeout = 4,
  testNumResults = 5
} testResult_t;

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if( err != cudaSuccess ) {                        \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test CUDA failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    return testCudaError;                           \
  }                                                 \
} while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d "           \
           "'%s / %s'\n",                           \
           hostname,__FILE__,__LINE__,              \
           ncclGetErrorString(res),                 \
           ncclGetLastError(NULL));                 \
    return testNcclError;                           \
  }                                                 \
} while(0)
#else
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    char hostname[1024];                            \
    getHostName(hostname, 1024);                    \
    printf("%s: Test NCCL failure %s:%d '%s'\n",    \
         hostname,                                  \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    return testNcclError;                           \
  }                                                 \
} while(0)
#endif

#define MPICHECK(cmd) do {                          \
    int e = cmd;                                    \
    if( e != MPI_SUCCESS ) {                        \
        fprintf(stderr, "Failed: MPI error %s:%d code %d\n", \
               __FILE__,__LINE__, e);               \
        MPI_Abort(MPI_COMM_WORLD, 1);               \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)


int main(int argc, char* argv[]) {

    int my_total_rank, num_total_ranks;
    int num_gpus_available = 0; // Will be filled by cudaGetDeviceCount

    // --- MPI Initialization ---
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_total_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_total_ranks));

    // --- Query Number of GPUs (All Ranks) ---
    CUDACHECK(cudaGetDeviceCount(&num_gpus_available));

    if (my_total_rank == 0) {
        printf("Detected %d CUDA-capable GPUs.\n", num_gpus_available);
    }

    // --- Validation Checks ---
    if (num_gpus_available <= 0) {
        if (my_total_rank == 0) {
            fprintf(stderr, "Error: No CUDA-capable GPUs found or CUDA error.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (num_total_ranks % num_gpus_available != 0) {
        if (my_total_rank == 0) {
            fprintf(stderr, "Error: Number of MPI ranks (%d) must be divisible "
                            "by the number of available GPUs (%d) for this "
                            "program's grouping logic.\n",
                    num_total_ranks, num_gpus_available);
        }
        MPI_Finalize();
        return 1;
    }

    // --- Calculate Grouping based on Detected GPUs ---
    int ranks_per_gpu = num_total_ranks / num_gpus_available;
 
    // Which GPU group I belong to (0 to num_gpus_available-1)
    int my_gpu_index = my_total_rank / ranks_per_gpu;

    // Am I the leader for my GPU group?
    int is_leader = (my_total_rank % ranks_per_gpu == 0);
    int is_gpu_shared = 1; // default to false

    printf("[Rank %d] Total Ranks: %d, Detected GPUs: %d, Ranks/GPU: %d, "
           "My GPU Index: %d, Is Leader: %s\n",
           my_total_rank,
           num_total_ranks,
           num_gpus_available,
           ranks_per_gpu,
           my_gpu_index,
           is_leader ? "Yes" : "No");

    // --- Create Local MPI Communicator for each GPU group ---
    MPI_Comm local_comm;
    MPICHECK(MPI_Comm_split(MPI_COMM_WORLD, my_gpu_index,
                            my_total_rank, &local_comm));
    int my_local_rank;

    // My rank within the local group (0 to ranks_per_gpu-1)
    MPICHECK(MPI_Comm_rank(local_comm, &my_local_rank));


    // --- GPU Selection and Stream Creation (All Ranks) ---
    cudaStream_t stream;
    // Assign process to its designated GPU. Ensure GPU index is valid.
    // The modulo operator could handle cases where ranks > GPUs, mapping back
    // to available GPUs. However, `my_gpu_index` already ensures it's within
    // [0, num_gpus_available - 1] because of the divisibility check earlier.
    struct cudaDeviceProp deviceProp;
    CUDACHECK(cudaGetDeviceProperties(&deviceProp, my_gpu_index));
    is_gpu_shared = (deviceProp.computeMode == cudaComputeModeDefault) ? 1 : 0;

    printf("[Rank %d] My GPU Index: %d, Is Leader: %s GPU Shareable: %s\n",
           my_total_rank,
           my_gpu_index,
           is_leader ? "Yes" : "No",
           is_gpu_shared ? "Yes" : "No");
    fflush(stdout);

    if(is_leader || is_gpu_shared) {
        CUDACHECK(cudaSetDevice(my_gpu_index));
        CUDACHECK(cudaStreamCreate(&stream));
    }

    // --- NCCL Setup (Leaders Only) ---
    ncclUniqueId nccl_id;
    ncclComm_t nccl_comm = NULL; // Initialize to NULL

    // Get NCCL Unique ID (Rank 0 generates, broadcasts to all)
    if (my_total_rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&nccl_id));
        printf("[Rank 0] Generated NCCL Unique ID.\n");
    }
    MPICHECK(MPI_Bcast((void *)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));


    // Initialize NCCL Communicator (Only Leaders Participate)
    // The communicator size is the number of GPUs (leaders) participating.
    if (is_leader) {
        // Leaders use their GPU index as their rank within the NCCL communicator
        NCCLCHECK(ncclCommInitRank(&nccl_comm, num_gpus_available, nccl_id, my_gpu_index));
        printf("[Rank %d] (Leader for GPU %d) Initialized NCCL communicator (NCCL Rank %d of %d).\n",
               my_total_rank, my_gpu_index, my_gpu_index, num_gpus_available);
    }

    // --- Data Preparation (All Ranks) ---
    int my_data = my_total_rank; // Each rank contributes its own total rank number
    int local_sum = 0;           // Variable to hold the sum within the local group (on CPU)
    int global_sum = -1;         // Variable to hold the final global sum (on CPU)

    // --- Local Aggregation (MPI Reduce within each group) ---
    MPICHECK(MPI_Reduce(&my_data, &local_sum, 1, MPI_INT, MPI_SUM, 0, local_comm));

    // --- GPU Buffer Allocation & Transfer (Leaders Only) ---
    int* send_buff_gpu = NULL;
    int* recv_buff_gpu = NULL;
    size_t data_size = sizeof(int);

    if (is_leader) {
        CUDACHECK(cudaMalloc((void**)&send_buff_gpu, data_size));
        CUDACHECK(cudaMalloc((void**)&recv_buff_gpu, data_size));
        CUDACHECK(cudaMemcpyAsync(send_buff_gpu, &local_sum, data_size, cudaMemcpyHostToDevice, stream));
        // printf("[Rank %d] (Leader) Copied local sum %d to GPU buffer.\n", my_total_rank, local_sum);
    }

    // --- NCCL Collective Operation (Leaders Only) ---
    if (is_leader) {
        // printf("[Rank %d] (Leader) Calling ncclAllReduce...\n", my_total_rank);
        NCCLCHECK(ncclAllReduce((const void*)send_buff_gpu, (void*)recv_buff_gpu,
                                1,        // Number of elements
                                ncclInt,  // Data type
                                ncclSum,  // Operation
                                nccl_comm, stream));

        CUDACHECK(cudaStreamSynchronize(stream));
        // printf("[Rank %d] (Leader) Stream synchronized.\n", my_total_rank);

        CUDACHECK(cudaMemcpy(&global_sum, recv_buff_gpu, data_size, cudaMemcpyDeviceToHost));
        // printf("[Rank %d] (Leader) Copied final global sum %d from GPU.\n", my_total_rank, global_sum);
    }

    // --- Local Distribution (MPI Broadcast within each group) ---
    MPICHECK(MPI_Bcast(&global_sum, 1, MPI_INT, 0, local_comm));

    // --- Verification (All Ranks) ---
    int expected_sum = (num_total_ranks * (num_total_ranks - 1)) / 2;

    printf("[Rank %d] Final Result Check: Received = %d (Expected = %d)\n",
           my_total_rank, global_sum, expected_sum);

    if (global_sum != expected_sum) {
        fprintf(stderr, "[Rank %d] Verification FAILED!\n", my_total_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        // printf("[Rank %d] Verification PASSED!\n", my_total_rank);
    }

    // --- Cleanup ---
    // printf("[Rank %d] Cleaning up...\n", my_total_rank);
    if (is_leader) {
        CUDACHECK(cudaFree(send_buff_gpu));
        CUDACHECK(cudaFree(recv_buff_gpu));
        if (nccl_comm != NULL) {
             NCCLCHECK(ncclCommDestroy(nccl_comm));
        }
    }
    if (is_leader || is_gpu_shared) {
        CUDACHECK(cudaStreamDestroy(stream));
    }
    MPICHECK(MPI_Comm_free(&local_comm));

    // Finalize MPI
    MPICHECK(MPI_Finalize());
    // printf("[Rank %d] Finished.\n", my_total_rank);

    return 0;
}
