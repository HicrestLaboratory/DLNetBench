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
    CommType world_comm;
#endif

#ifdef PROXY_ENABLE_RCCL
    #include <rccl.h>
    CommType world_comm;
#endif

#include <mpi.h>

#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using nlohmann::json;

// CCUTILS headers
#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/macros.hpp>

#ifdef PROXY_ENABLE_CUDA
    #include <ccutils/cuda/cuda_macros.hpp>
#endif

#ifdef PROXY_ENABLE_HIP
    #include <ccutils/hip/hip_macros.hpp>
#endif

// Project headers
#include "../utils.hpp"
#include "../data_types.hpp"
#include "../proxy_classes.hpp"

// Device to use
#if defined(PROXY_ENABLE_CUDA) || defined(PROXY_ENABLE_HIP)
    constexpr Device device = Device::GPU;
#else
    constexpr Device device = Device::CPU;
#endif

// Default values
#define NUM_B 10
#define WARM_UP 8
#define RUNS 10

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
 * @param comm Pointer to the Communicator object for Collective operations.
 * @return int Always returns 0.
 */
int run_data_parallel(Tensor<_FLOAT, device>** grad_ptrs, Tensor<_FLOAT, device>** sum_grad_ptrs, 
                    int num_buckets, uint64_t* params_per_bucket,
                    uint64_t fwd_rt_whole_model, float bwd_rt_per_B, ProxyCommunicator* communicator) {
    

    //forward compute
    usleep(fwd_rt_whole_model);
    //backward (idea is to overlap all-reduce with backward compute)

    int index, flag;
    for(int i=0; i<num_buckets; i++){
        usleep(bwd_rt_per_B); //compute backward of a bucket 
        communicator->Iallreduce(grad_ptrs[i]->data, sum_grad_ptrs[i]->data, params_per_bucket[i], i); //start all-reduce for the bucket
    }

    CCUTILS_MPI_TIMER_START(barrier)
    communicator->WaitAll(num_buckets); //wait for all all-reduce to complete
    CCUTILS_MPI_TIMER_STOP(barrier) 
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

    Tensor<_FLOAT, device>* grad_ptrs[num_buckets];
    Tensor<_FLOAT, device>* sum_grad_ptrs[num_buckets];
    for(int i=0; i<num_buckets; i++){
        grad_ptrs[i] = new Tensor<_FLOAT, device>(params_per_bucket[i]);
        sum_grad_ptrs[i] = new Tensor<_FLOAT, device>(params_per_bucket[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    #ifdef PROXY_ENABLE_CCL
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&world_comm, world_size, id, rank);
    CCLCommunicator* communicator = new CCLCommunicator(world_comm, num_buckets);
    #else
    MPICommunicator* communicator = new MPICommunicator(MPI_COMM_WORLD, MPI_FLOAT, num_buckets);
    #endif

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B, communicator);
    }

    for(int iter = 0; iter < RUNS; iter++){
        CCUTILS_MPI_TIMER_START(runtime)
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B, communicator);
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
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp, "backend", communicator->get_name())

    //erase warm-up elemements

    
    CCUTILS_SECTION_JSON_PUT(dp, "runtimes", __timer_vals_runtime);
    __timer_vals_barrier.erase(__timer_vals_barrier.begin(), __timer_vals_barrier.begin() + WARM_UP); // remove the warm-up barriers
    CCUTILS_SECTION_JSON_PUT(dp, "barrier_time_us", __timer_vals_barrier);
    CCUTILS_SECTION_JSON_PUT(dp, "hostname", host_name);

    CCUTILS_MPI_SECTION_END(dp);

    #ifdef PROXY_ENABLE_CLL
    ncclCommDestroy(world_comm);
    #endif

    MPI_Finalize();
}
