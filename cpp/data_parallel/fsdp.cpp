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

#include <ccutils/mpi/mpi_timers.h>
#include <ccutils/mpi/mpi_macros.h>

#include "../utils.hpp"

/*
 * Parameters from users: n_units, save_parameters (avoid allgather during backward), sharding_factor F:
 * each layer is sharded across F processes, so each process holds 1/F of the parameters, if F=World_size, then each layer is fully sharded (no model replica)
 * if 1 < F < World_size, then each layer is partially sharded and we have more replicas of the model.
 * assert that (num_layers % num_units == 0) and (world_size % F == 0)
*/

//timers for each collective and the whole runtime
MPI_TIMER_DEF(allgather)
MPI_TIMER_DEF(reduce_scatter)
MPI_TIMER_DEF(barrier)
MPI_TIMER_DEF(runtime)

//default values
#define LOCAL_BATCH_SIZE 128
#define WARM_UP 8
#define RUNS 50


void run_fsdp(Tensor<float>** shard_params,
              Tensor<float>** layer_params,
              Tensor<float>** allreduce_params,
              float fwd_rt_whole_unit,
              float bwd_rt_whole_unit,
              uint num_units,
              uint sharding_factor,
              uint64_t* max_params_per_shard,
              uint num_replicas,
              bool save_parameters,
              MPI_Comm unit_comm,
              MPI_Comm allreduce_comm){
    MPI_Request request[num_units];
    int shard_rank;
    MPI_Comm_rank(unit_comm, &shard_rank);

    int allreduce_rank = 0;
    if (num_replicas > 1)
        MPI_Comm_rank(allreduce_comm, &allreduce_rank);

    // Forward pass
    for (uint u = 0; u < num_units; u++) {
        // 1. Allgather padded shards
        MPI_TIMER_START(allgather);
        MPI_Allgather(shard_params[u]->data,
                      static_cast<int>(max_params_per_shard[u]),
                      MPI_FLOAT,
                      layer_params[u]->data,
                      static_cast<int>(max_params_per_shard[u]),
                      MPI_FLOAT,
                      unit_comm);
        MPI_TIMER_STOP(allgather);

        // 2. Local forward computation (simulated)
        usleep(fwd_rt_whole_unit);
    }

    // Backward pass
    for (int u = num_units - 1; u >= 0; u--) {
        // 1. Optionally allgather saved parameters
        if (!save_parameters) {
            MPI_TIMER_START(allgather);
            MPI_Allgather(shard_params[u]->data,
                          static_cast<int>(max_params_per_shard[u]),
                          MPI_FLOAT,
                          layer_params[u]->data,
                          static_cast<int>(max_params_per_shard[u]),
                          MPI_FLOAT,
                          unit_comm);
            MPI_TIMER_STOP(allgather);    
        }

        // 2. Local backward computation (simulated)
        usleep(bwd_rt_whole_unit);

        // 3. Reduce-Scatter across shards (padded)
        MPI_TIMER_START(reduce_scatter);
        MPI_Reduce_scatter_block(layer_params[u]->data,
                                 shard_params[u]->data,
                                 static_cast<int>(max_params_per_shard[u]),
                                 MPI_FLOAT,
                                 MPI_SUM,
                                 allreduce_comm);
        MPI_TIMER_STOP(reduce_scatter);

        // 4. Optional allreduce across replicas
        if (num_replicas > 1) {
            MPI_Iallreduce(MPI_IN_PLACE,
                           shard_params[u]->data,
                           static_cast<int>(max_params_per_shard[u]),
                           MPI_FLOAT,
                           MPI_SUM,
                           allreduce_comm,
                           &request[u]);
        }
    }

    if (num_replicas > 1){
        MPI_TIMER_START(barrier);
        MPI_Waitall(num_units, request, MPI_STATUSES_IGNORE);
        MPI_TIMER_STOP(barrier);
    }
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 4) {
        if (rank == 0)
            std::cout << "Usage: mpirun -n <world_size> ./fsdp <model_name> <num_units> <sharding_factor> <save_parameters>\n";
        MPI_Finalize();
        return -1;
    }

    std::string model_name = argv[1];
    uint num_units = std::stoi(argv[2]);
    uint sharding_factor = std::stoi(argv[3]);
    bool save_parameters = (argc > 4) ? (std::string(argv[4]) == "true") : false;

    assert(world_size % sharding_factor == 0);

    // Model stats
    std::map<std::string,uint64_t> model_stats = get_model_stats("../../model_stats/" + model_name + ".txt");
    uint64_t total_model_size = model_stats["modelSize"];
    uint64_t fwd_rt_whole_model = model_stats["avgForwardTime"];
    uint64_t bwd_rt_whole_model = model_stats["avgBackwardTime"];

    // Compute per-unit parameter sizes
    uint64_t base_params_per_unit = total_model_size / num_units;
    uint64_t remainder = total_model_size % num_units;
    uint64_t params_per_unit[num_units];
    for (uint i = 0; i < num_units; i++)
        params_per_unit[i] = base_params_per_unit + (i < remainder ? 1 : 0);

    // Compute per-shard sizes (padded)
    uint64_t max_params_per_shard[num_units];
    for (uint u = 0; u < num_units; u++) {
        uint64_t base_shard = params_per_unit[u] / sharding_factor;
        max_params_per_shard[u] = base_shard + (params_per_unit[u] % sharding_factor ? 1 : 0);
    }

    // Communicators
    uint num_replicas = world_size / sharding_factor;
    int replica_color = rank / sharding_factor;
    MPI_Comm unit_comm;
    MPI_Comm_split(MPI_COMM_WORLD, replica_color, rank, &unit_comm);

    int shard_index_color = rank % sharding_factor;
    MPI_Comm allreduce_comm;
    MPI_Comm_split(MPI_COMM_WORLD, shard_index_color, rank, &allreduce_comm);

    // Allocate buffers (padded)
    Tensor<float>* shard_params[num_units];
    Tensor<float>* layer_params[num_units];
    Tensor<float>* allreduce_params[num_units];

    for (uint u = 0; u < num_units; u++) {
        shard_params[u] = new Tensor<float>(max_params_per_shard[u], Device::CPU);
        layer_params[u] = new Tensor<float>(max_params_per_shard[u] * sharding_factor, Device::CPU);
        if (num_replicas > 1)
            allreduce_params[u] = new Tensor<float>(max_params_per_shard[u], Device::CPU);
    }

    float fwd_rt_whole_unit = (float)fwd_rt_whole_model / num_units;
    float bwd_rt_whole_unit = (float)bwd_rt_whole_model / num_units;


    for(int i = 0; i < WARM_UP; i++)
        run_fsdp(shard_params, layer_params, allreduce_params,
                 fwd_rt_whole_unit, bwd_rt_whole_unit,
                 num_units, sharding_factor, max_params_per_shard,
                 num_replicas, save_parameters,
                 unit_comm, allreduce_comm);

    
    for(int i = 0; i < RUNS; i++){
        MPI_TIMER_START(runtime);
        run_fsdp(shard_params, layer_params, allreduce_params,
                 fwd_rt_whole_unit, bwd_rt_whole_unit,
                 num_units, sharding_factor, max_params_per_shard,
                 num_replicas, save_parameters,
                 unit_comm, allreduce_comm);
        MPI_TIMER_STOP(runtime);
    } 

    ccutils_timers::TimerStats runtime_stats;
    ccutils_timers::TimerStats allgather_stats;
    ccutils_timers::TimerStats reduce_scatter_stats;
    ccutils_timers::TimerStats barrier_stats;

    runtime_stats = ccutils_timers::compute_stats(__timer_vals_runtime);
    allgather_stats = ccutils_timers::compute_stats(__timer_vals_allgather);
    reduce_scatter_stats = ccutils_timers::compute_stats(__timer_vals_reduce_scatter);
    barrier_stats = ccutils_timers::compute_stats(__timer_vals_barrier);

    float runtime_avg = runtime_stats.avg;
    float runtime_stddev = runtime_stats.stddev;
    float allgather_avg = allgather_stats.avg;
    float allgather_stddev = allgather_stats.stddev;
    float reduce_scatter_avg = reduce_scatter_stats.avg;
    float reduce_scatter_stddev = reduce_scatter_stats.stddev;
    float barrier_avg = barrier_stats.avg;
    float barrier_stddev = barrier_stats.stddev;

    std::vector<uint64_t> shard_sizes(max_params_per_shard, max_params_per_shard + num_units);

    // Allgather message sizes
    std::pair<float, float> msg_size_allgather = compute_msg_stats(shard_sizes, sharding_factor);
    float msg_size_allgather_avg = msg_size_allgather.first;
    float msg_size_allgather_std = msg_size_allgather.second;

    // Reduce-Scatter message sizes
    std::pair<float, float> msg_size_reduce_scatter = compute_msg_stats(shard_sizes);
    float msg_size_reduce_scatter_avg = msg_size_reduce_scatter.first;
    float msg_size_reduce_scatter_std = msg_size_reduce_scatter.second;

    // Allreduce message sizes
    std::pair<float, float> msg_size_allreduce = compute_msg_stats(shard_sizes);
    float msg_size_allreduce_avg = msg_size_allreduce.first;
    float msg_size_allreduce_std = msg_size_allreduce.second;

    MPI_PRINT_ONCE(
        "Rank = %d\n"
        "world_size = %d\n"
        "total_params = %llu\n"
        "num_units = %d\n"
        "sharding_factor = %d\n"
        "save_parameters = %s\n"
        "msg_size_allgather_avg = %.2f\n"
        "msg_size_allgather_std = %.2f\n"
        "msg_size_reduce_scatter_avg = %.2f\n"
        "msg_size_reduce_scatter_std = %.2f\n"
        "msg_size_allreduce_avg = %.2f\n"
        "msg_size_allreduce_std = %.2f\n"
        "runtime_avg (us) = %.2f\n"
        "runtime_stddev (us) = %.2f\n"
        "allgather_avg (us) = %.2f\n"
        "allgather_stddev (us) = %.2f\n"
        "reduce_scatter_avg (us) = %.2f\n"
        "reduce_scatter_stddev (us) = %.2f\n"
        "barrier_avg (us) = %.2f\n"
        "barrier_stddev (us) = %.2f\n",
        rank,
        world_size,
        total_model_size,
        num_units,
        sharding_factor,
        save_parameters ? "true" : "false",
        msg_size_allgather_avg,
        msg_size_allgather_std,
        msg_size_reduce_scatter_avg,
        msg_size_reduce_scatter_std,
        msg_size_allreduce_avg,
        msg_size_allreduce_std,
        runtime_avg,
        runtime_stddev,
        allgather_avg,
        allgather_stddev,
        reduce_scatter_avg,
        reduce_scatter_stddev,
        barrier_avg,
        barrier_stddev
    );

    MPI_Finalize();
    return 0;
}