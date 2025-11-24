/*********************************************************************
 *
 * Description: C++/MPI proxy for Transformer-based models distributed training 
 *              with data parallelism
 * Author: Jacopo Raffi
 *
 *********************************************************************/

 //TODO: add GPU support and NCCL support later
 #ifdef NCLL
    #include <nccl.h>
 #else
    #include <mpi.h>
#endif

#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

#include <ccutils/mpi/mpi_timers.h>
#include <ccutils/mpi/mpi_macros.h>

#include "../utils.hpp"

//TODO: how to handle mpi with float16? 
// Choose type based on compile-time macro
// #if defined(TYPE_FLOAT16)
//     #include <stdint.h>
//     typedef _Float16 real_t;
//     #define TYPE_NAME "float16"
// #elif defined(TYPE_FLOAT)
//     typedef float real_t;
//     #define TYPE_NAME "float"
// #elif defined(TYPE_DOUBLE)
//     typedef double real_t;
//     #define TYPE_NAME "double"
// #else
//     #error "Please define one of TYPE_FLOAT16, TYPE_FLOAT, TYPE_DOUBLE"
// #endif

// Default values
#define NUM_B 10
#define WARM_UP 8
#define RUNS 50

MPI_TIMER_DEF(runtime)
MPI_TIMER_DEF(barrier)

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
int run_data_parallel(Tensor<float>** grad_ptrs, Tensor<float>** sum_grad_ptrs, 
                    int num_buckets, uint64_t* params_per_bucket,
                    uint64_t fwd_rt_whole_model, float bwd_rt_per_B){
    

    //forward compute
    usleep(fwd_rt_whole_model);
    //backward (idea is to overlap all-reduce with backward compute)
    MPI_Request grad_allreduce_reqs[num_buckets];
    //must initialize with MPI_REQUEST_NULL
    for(int i=0; i<num_buckets; i++)
        grad_allreduce_reqs[i] = MPI_REQUEST_NULL;

    int index, flag;
    for(int i=0; i<num_buckets; i++){
        if(i > 1)
            MPI_Testany(num_buckets, grad_allreduce_reqs, &index, &flag, MPI_STATUSES_IGNORE); //Checks if any of the non-blocking allreduce requests have completed

        usleep(bwd_rt_per_B); //compute backward of a bucket

        MPI_Iallreduce(grad_ptrs[i]->data, sum_grad_ptrs[i]->data, params_per_bucket[i], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[i]);	
    }

    MPI_TIMER_START(barrier)
    MPI_Waitall(num_buckets, grad_allreduce_reqs, MPI_STATUSES_IGNORE);
    MPI_TIMER_STOP(barrier) 
    return 0;
}

int main(int argc, char* argv[]){
    int rank, world_size;

    int num_buckets = NUM_B;

    if(argc < 2){
        std::cout << "Usage: mpirun -n <world_size> ./dp <model_name> <num_buckets>\n";
        return -1;
    }

    std::string model_name = argv[1];
    if(argc > 2){
        num_buckets = std::stoi(argv[2]);
    }

    std::map<std::string, uint64_t> model_stats = get_model_stats("../../model_stats/" + model_name + ".txt"); // get model stats from file
    
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

    Tensor<float>* grad_ptrs[num_buckets];
    Tensor<float>* sum_grad_ptrs[num_buckets];
    for(int i=0; i<num_buckets; i++){
        grad_ptrs[i] = new Tensor<float>(params_per_bucket[i], Device::CPU); // switch to Device::GPU later
        sum_grad_ptrs[i] = new Tensor<float>(params_per_bucket[i], Device::CPU);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B);
    }

    for(int iter = 0; iter < RUNS; iter++){
        MPI_TIMER_START(runtime)
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B);
        MPI_TIMER_STOP(runtime)
    }

    ccutils_timers::TimerStats runtime_stats;
    ccutils_timers::TimerStats barrier_stats;

    runtime_stats = ccutils_timers::compute_stats(__timer_vals_runtime);
    barrier_stats = ccutils_timers::compute_stats(__timer_vals_barrier);

    float runtime_avg = runtime_stats.avg;
    float runtime_stddev = runtime_stats.stddev;
    float barrier_avg = barrier_stats.avg;
    float barrier_stddev = barrier_stats.stddev;

   std::vector<uint64_t> bucket_sizes(params_per_bucket,
                                   params_per_bucket + num_buckets);

    std::pair<float, float> msg_stats = compute_msg_stats(bucket_sizes, 1);
    float msg_size_avg = msg_stats.first;
    float msg_size_std = msg_stats.second;

    MPI_PRINT_ONCE(
        "Rank = %d\n"
        "world_size = %d\n"
        "total_params = %llu\n"
        "num_buckets = %d\n"
        "local_batch_size = %d\n"
        "global_batch_size = %llu\n"
        "msg_size_avg = %.2f\n"
        "msg_size_std = %.2f\n"
        "runtime_avg (us) = %.2f\n"
        "runtime_stddev (us) = %.2f\n"
        "barrier_avg (us) = %.2f\n"
        "barrier_stddev (us) = %.2f\n"
        "fwd_rt_whole_model (us) = %llu\n"
        "bwd_rt_per_B (us) = %.2f\n",
        rank,
        world_size,
        total_model_size,
        num_buckets,
        local_batch_size,
        static_cast<unsigned long long>(local_batch_size * world_size),
        msg_size_avg,
        msg_size_std,
        runtime_avg,
        runtime_stddev,
        barrier_avg,
        barrier_stddev,
        fwd_rt_whole_model,
        bwd_rt_per_B
    );

    MPI_Finalize();
}