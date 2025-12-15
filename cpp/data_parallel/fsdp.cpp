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
#include <filesystem>
namespace fs = std::filesystem;

#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>

#include "../utils.hpp"

//TODO: integrate ccutils and make more fine-grained timers (follow dp.cpp structure)

/*
 * Parameters from users: n_units, save_parameters (avoid allgather during backward), sharding_factor F:
 * each layer is sharded across F processes, so each process holds 1/F of the parameters, if F=World_size, then each layer is fully sharded (no model replica)
 * if 1 < F < World_size, then each layer is partially sharded and we have more replicas of the model.
 * assert that (num_layers % num_units == 0) and (world_size % F == 0)
*/

//timers for each collective and the whole runtime
CCUTILS_MPI_TIMER_DEF(allgather)
CCUTILS_MPI_TIMER_DEF(reduce_scatter)
CCUTILS_MPI_TIMER_DEF(barrier)
CCUTILS_MPI_TIMER_DEF(runtime)

//default values
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
        CCUTILS_MPI_TIMER_START(allgather);
        MPI_Allgather(shard_params[u]->data,
                      static_cast<int>(max_params_per_shard[u]),
                      MPI_FLOAT,
                      layer_params[u]->data,
                      static_cast<int>(max_params_per_shard[u]),
                      MPI_FLOAT,
                      unit_comm);
        CCUTILS_MPI_TIMER_STOP(allgather);

        // 2. Local forward computation (simulated)
        usleep(fwd_rt_whole_unit);
    }

    // Backward pass
    for (int u = num_units - 1; u >= 0; u--) {
        // 1. Optionally allgather saved parameters
        if (!save_parameters) {
            CCUTILS_MPI_TIMER_START(allgather);
            MPI_Allgather(shard_params[u]->data,
                          static_cast<int>(max_params_per_shard[u]),
                          MPI_FLOAT,
                          layer_params[u]->data,
                          static_cast<int>(max_params_per_shard[u]),
                          MPI_FLOAT,
                          unit_comm);
            CCUTILS_MPI_TIMER_STOP(allgather);    
        }

        // 2. Local backward computation (simulated)
        usleep(bwd_rt_whole_unit);

        // 3. Reduce-Scatter across shards (padded)
        CCUTILS_MPI_TIMER_START(reduce_scatter);
        MPI_Reduce_scatter_block(layer_params[u]->data,
                                 shard_params[u]->data,
                                 static_cast<int>(max_params_per_shard[u]),
                                 MPI_FLOAT,
                                 MPI_SUM,
                                 allreduce_comm);
        CCUTILS_MPI_TIMER_STOP(reduce_scatter);

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
        CCUTILS_MPI_TIMER_START(barrier);
        MPI_Waitall(num_units, request, MPI_STATUSES_IGNORE);
        CCUTILS_MPI_TIMER_STOP(barrier);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 4) { //TODO: remove the save_parameters argument
        if (rank == 0)
            std::cout << "Usage: mpirun -n <world_size> ./fsdp <model_name> <num_units> <sharding_factor> <save_parameters> <base_path>\n";
        MPI_Finalize();
        return -1;
    }

    std::string model_name = argv[1];
    uint num_units = std::stoi(argv[2]);
    uint sharding_factor = std::stoi(argv[3]);
    bool save_parameters = (argc > 4) ? (std::string(argv[4]) == "true") : false;

    assert(world_size % sharding_factor == 0);
    //TODO: assert num_layers % num_units == 0 when num_layers info is available

     // --- Get DNNProxy base path ---
    fs::path repo_path = get_dnnproxy_base_path(argc, argv, rank);
    if (repo_path.empty()) {
        MPI_Finalize();
        return -1;  // DNNProxy not found
    }

    // --- Construct model stats file path ---
    fs::path file_path = repo_path / "model_stats" / (model_name + ".txt");
    if (!fs::exists(file_path)) {
        if (rank == 0)
            std::cerr << "Error: model stats file does not exist: " << file_path << "\n";
        MPI_Finalize();
        return -1;
    }

    std::map<std::string, uint64_t> model_stats = get_model_stats(file_path.string()); // get model stats from file

    uint64_t total_model_size = model_stats["modelSize"];
    uint local_batch_size = model_stats["batchSize"];
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
        CCUTILS_MPI_TIMER_START(runtime);
        run_fsdp(shard_params, layer_params, allreduce_params,
                 fwd_rt_whole_unit, bwd_rt_whole_unit,
                 num_units, sharding_factor, max_params_per_shard,
                 num_replicas, save_parameters,
                 unit_comm, allreduce_comm);
        CCUTILS_MPI_TIMER_STOP(runtime);
    } 

    // Use CCUTILS sections to print
    CCUTILS_MPI_SECTION_DEF(fsdp, "FSDP metrics")
    CCUTILS_SECTION_JSON_PUT(fsdp, "runtime", __timer_vals_runtime)
    CCUTILS_SECTION_JSON_PUT(fsdp, "allgather", __timer_vals_allgather)
    CCUTILS_SECTION_JSON_PUT(fsdp, "reduce_scatter", __timer_vals_reduce_scatter)
    CCUTILS_SECTION_JSON_PUT(fsdp, "barrier", __timer_vals_barrier)


    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "model_size_bytes", total_model_size*sizeof(float))
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "num_units", num_units)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "sharding_factor", sharding_factor)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "save_parameters", save_parameters)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "num_replicas", num_replicas)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "local_batch_size", local_batch_size)
    
    //fwd and bwd time per unit
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "fwd_time_per_unit_us", fwd_rt_whole_unit)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "bwd_time_per_unit_us", bwd_rt_whole_unit)

    // allgather and reducescatter msg_size
    // Since all units are equal
    uint64_t allgather_msg_size = max_params_per_shard[0] * sharding_factor * sizeof(float);
    uint64_t reducescatter_msg_size = max_params_per_shard[0] * sizeof(float);
    uint64_t allreduce_msg_size = 0;

    if (num_replicas > 1)
        allreduce_msg_size = max_params_per_shard[0] * sizeof(float);

    // Put in JSON
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "allgather_msg_size_bytes", allgather_msg_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "reducescatter_msg_size_bytes", reducescatter_msg_size)
    if (num_replicas > 1)
        CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "allreduce_msg_size_bytes", allreduce_msg_size)

    CCUTILS_MPI_SECTION_END(fsdp)

    
    MPI_Finalize();
    return 0;
}