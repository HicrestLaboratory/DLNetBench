/*********************************************************************
 *
 * Description: C++/MPI proxy for Transformer-based models distributed training 
 *              with data parallelism
 * Author: Jacopo Raffi
 *
 *********************************************************************/

#include <mpi.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <assert.h>

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
#define LOCAL_BATCH_SIZE 128
#define WARM_UP 8
#define RUNS 128

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
int run_data_parallel(float** grad_ptrs, float** sum_grad_ptrs, 
                      int num_buckets, uint64_t* params_per_bucket,
                      float fwd_rt_whole_model, float bwd_rt_per_B){
    

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
            MPI_Testany(num_buckets, grad_allreduce_reqs, &index, &flag, MPI_STATUSES_IGNORE); //Checks if any of the non-blocking all-reduce requests have completed

        usleep(bwd_rt_per_B); //compute backward of a bucket

        MPI_Iallreduce(grad_ptrs[i], sum_grad_ptrs[i], params_per_bucket[i], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &grad_allreduce_reqs[i]);	
    }

    MPI_Waitall(num_buckets, grad_allreduce_reqs, MPI_STATUSES_IGNORE); 
    return 0;
}

int main(int argc, char *argv[]){
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
    
    float fwd_rt_whole_model = model_stats["avgForwardTime"]; // in us
    float bwd_rt_per_B = (model_stats["avgBackwardTime"]) / num_buckets; // in us

    int local_batch_size = model_stats["batchSize"];

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

    float* grad_ptrs[num_buckets];
    float* sum_grad_ptrs[num_buckets];
    for(int i=0; i<num_buckets; i++){
        grad_ptrs[i] = (float *)calloc(params_per_bucket[i], sizeof(float)); //TODO: support also float16
        sum_grad_ptrs[i] = (float *)calloc(params_per_bucket[i], sizeof(float));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //warmup
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B);
    }

    double begin, elapse;
    begin = MPI_Wtime();
    for(int iter = 0; iter < RUNS; iter++){
        run_data_parallel(grad_ptrs, sum_grad_ptrs, num_buckets, params_per_bucket,
                         fwd_rt_whole_model, bwd_rt_per_B);
    }
    elapse = (MPI_Wtime()-begin)/RUNS;

    if(rank == 0){
        std::cout << "Rank = " << rank
                  << ", world_size = " << world_size
                  << ", data_shards = " << world_size
                  << ", total_params = " << total_model_size
                  << ", num_buckets = " << num_buckets
                  << ", global_batch_size = " << (local_batch_size * world_size)
                  << ".\n";
        //TODO: Add here all the stats related to data parallelism execution
    }

    MPI_Finalize();
}