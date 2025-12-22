/*********************************************************************
 *
 * Description: C++/MPI proxy for Transformer-based models distributed training 
 *              with data parallelism
 * Author: Jacopo Raffi
 *
 *********************************************************************/

 //TODO: RCCLL, NCCL and CoCCL support
 #ifdef PROXY_ENABLE_NCLL
    #include <nccl.h>
    ncclComm_t world_comm;
#endif

#ifdef PROXY_ENABLE_RCCL
    #include <rccl.h>
    rcclComm_t world_comm;
#endif

#include <mpi.h>

#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdint>
#include <cstdlib> // for getenv

#include <filesystem>
namespace fs = std::filesystem;

#include <nlohmann/json.hpp>
using nlohmann::json;

#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/cuda/cuda_macros.hpp>
#include <ccutils/macros.hpp>

#include "../utils.hpp"
#include "../data_types.hpp"

// Determine device type based on compilation flags
#if defined(PROXY_ENABLE_CUDA) || defined(PROXY_ENABLE_HIP)
    constexpr Device device = Device::GPU;
#else
    constexpr Device device = Device::CPU;
#endif

// Default values
#define NUM_B 10
#define WARM_UP 8
#define RUNS 50

CCUTILS_MPI_TIMER_DEF(runtime)
CCUTILS_MPI_TIMER_DEF(barrier)

/**
 * @brief Simulates one iteration of data-parallel (using bucketing approach) training for a Transformer model.
 *
 * This function performs a mock forward pass and a backward pass with asynchronous
 * all-reduce operations for gradients over multiple parameter buckets.
 *
 * @param grad_ptrs Array of pointers to gradient buffers for each bucket.
 * @param sum_grad_ptrs Array of pointers to buffers storing the reduced gradients.
 * @param num_buckets Number of parameter buckets.
 * @param params_per_bucket Array containing the number of parameters in each bucket.
 * @param fwd_rt_whole_model Forward pass runtime in microseconds.
 * @param bwd_rt_per_B Backward pass runtime per bucket in microseconds.
 * @return int Always returns 0.
 */
int run_data_parallel(Tensor<_FLOAT, device>** grad_ptrs, Tensor<_FLOAT, device>** sum_grad_ptrs, 
                    int num_buckets, uint64_t* params_per_bucket,
                    uint64_t fwd_rt_whole_model, float bwd_rt_per_B){
    

    //forward compute
    usleep(fwd_rt_whole_model);
    //backward (idea is to overlap all-reduce with backward compute)

    #ifdef PROXY_ENABLE_NCCL //To overlap with *CCL use different streams
    _Stream stream[num_buckets];
    for(int i=0; i<num_buckets; i++){
    #if defined(PROXY_ENABLE_CUDA)
        CCUTILS_CUDA_CHECK(cudaStreamCreate(&stream[i]))
    #elif defined(PROXY_ENABLE_HIP)
        hipStreamCreate(&stream[i]);
    #endif
    }
    #endif

    #if !defined(PROXY_ENABLE_NCCL) || defined(PROXY_ENABLE_MPI_AWARE_CUDA) // MPI use case
    MPI_Request grad_allreduce_reqs[num_buckets];
    for (int i = 0; i < num_buckets; i++) {
        grad_allreduce_reqs[i] = MPI_REQUEST_NULL;
    }
    #endif

    int index, flag;
    for(int i=0; i<num_buckets; i++){
        usleep(bwd_rt_per_B); //compute backward of a bucket

        #if defined(PROXY_ENABLE_NCCL) || defined(PROXY_ENABLE_RCCL)
        // Launch NCCL all-reduce on the dedicated stream
        ncclAllReduce((const void*)grad_ptrs[i]->data, (void*)sum_grad_ptrs[i]->data,
                      params_per_bucket[i], NCCL_FLOAT_TYPE, ncclSum,
                      world_comm, stream[i]);
        #else     
	        MPI_Iallreduce(grad_ptrs[i]->data, sum_grad_ptrs[i]->data, params_per_bucket[i], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[i]);
        #endif
    }

    #if defined(PROXY_ENABLE_NCCL) || defined(PROXY_ENABLE_RCCL)
    // Synchronize all streams 
    //TODO: add timer
    for(int i=0; i<num_buckets; i++){
    #if defined(PROXY_ENABLE_CUDA)
        CCUTILS_CUDA_CHECK(cudaStreamSynchronize(stream[i]));
        CCUTILS_CUDA_CHECK(cudaStreamDestroy(stream[i]));
    #elif defined(PROXY_ENABLE_HIP) //TODO: add CCUTILS_HIP
        hipStreamSynchronize(stream[i]);
        hipStreamDestroy(stream[i]);
    #endif
    }
    #else
        CCUTILS_MPI_TIMER_START(barrier)
        MPI_Waitall(num_buckets, grad_allreduce_reqs, MPI_STATUSES_IGNORE);
        CCUTILS_MPI_TIMER_STOP(barrier) 
    #endif
    return 0;
}

int main(int argc, char* argv[]) {
    int rank, world_size;

    int num_buckets = NUM_B;
    if(argc < 4){
        std::cout << "Usage: mpirun -n <world_size> ./dp <model_name> <num_buckets> <base_path>\n";
        return -1;
    }

    std::string model_name = argv[1];
    if(argc > 2){
        num_buckets = std::stoi(argv[2]);
    }

    // --- Construct model stats file path ---
    fs::path repo_path = get_dnnproxy_base_path(argc, argv, rank);
    fs::path file_path = repo_path / "model_stats" / (model_name + ".txt");
    if (!fs::exists(file_path)) {
        std::cerr << "Error: model stats file does not exist: " << file_path << "\n";
        return -1;
    }

    std::map<std::string, uint64_t> model_stats = get_model_stats(file_path); // get model stats from file
    uint64_t fwd_rt_whole_model = model_stats["avgForwardTime"]; // in us
    float bwd_rt_per_B = (model_stats["avgBackwardTime"]) / num_buckets; // in us
    uint local_batch_size = model_stats["batchSize"];
    uint64_t total_model_size = model_stats["modelSize"]; // number of parameters
    
    uint64_t base_params_per_bucket = total_model_size / num_buckets;
    uint64_t remainder = total_model_size % num_buckets;
    uint64_t params_per_bucket[num_buckets];
    for (int i = 0; i < num_buckets; i++) {
        params_per_bucket[i] = base_params_per_bucket + (i < remainder ? 1 : 0); // distribute remainder across the buckets
    }
            
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    CCUTILS_MPI_INIT

    #ifdef PROXY_ENABLE_NCCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id); // Only rank 0 generates
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&world_comm, world_size, id, rank);
    #endif

    Tensor<_FLOAT, device>* grad_ptrs[num_buckets];
    Tensor<_FLOAT, device>* sum_grad_ptrs[num_buckets];
    for(int i=0; i<num_buckets; i++){
        grad_ptrs[i] = new Tensor<_FLOAT, device>(params_per_bucket[i]);
        sum_grad_ptrs[i] = new Tensor<_FLOAT, device>(params_per_bucket[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B);
    }

    for(int iter = 0; iter < RUNS; iter++){
        CCUTILS_MPI_TIMER_START(runtime)
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B);
        CCUTILS_MPI_TIMER_STOP(runtime)
    }

    char host_name[MPI_MAX_PROCESSOR_NAME];
	char (*host_names)[MPI_MAX_PROCESSOR_NAME];
	int namelen,bytes,n,color;
	MPI_Get_processor_name(host_name,&namelen);

    std::vector<uint64_t> bucket_sizes(params_per_bucket,
                                   params_per_bucket + num_buckets);
    std::pair<float, float> msg_stats = compute_msg_stats(bucket_sizes, 1);

    CCUTILS_MPI_SECTION_DEF(dp, "Data Parallelism")
    float msg_size_avg = msg_stats.first;
    float msg_size_std = msg_stats.second;
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "model_name", model_name)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "num_buckets", num_buckets)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "local_batch_size", local_batch_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "world_size", world_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "fwd_rt_whole_model_s", fwd_rt_whole_model)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "bwd_rt_per_bucket_s", bwd_rt_per_B)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "total_model_size_params", total_model_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "msg_size_avg_bytes", msg_size_avg*sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "msg_size_std_bytes", msg_size_std*sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "device", (device == Device::CPU) ? "CPU" : "GPU")
    
    CCUTILS_SECTION_JSON_PUT(dp, "runtimes", __timer_vals_runtime);
    CCUTILS_SECTION_JSON_PUT(dp, "barrier_time_us", __timer_vals_barrier);
    CCUTILS_SECTION_JSON_PUT(dp, "hostname", host_name);

    CCUTILS_MPI_SECTION_END(dp);

    #ifdef PROXY_ENABLE_NCLL
    ncclCommDestroy(world_comm);
    #endif

    MPI_Finalize();
}
