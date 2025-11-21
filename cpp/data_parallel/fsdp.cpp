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
              uint64_t** params_per_shard,
              uint num_replicas,
              bool save_parameters,
              MPI_Comm unit_comm,
              MPI_Comm allreduce_comm,
              int** recvcounts_units,
              int** displs_units){

    int shard_rank, allreduce_rank;
    MPI_Comm_rank(unit_comm, &shard_rank);
    
    if(num_replicas > 1){
        MPI_Comm_rank(allreduce_comm, &allreduce_rank);
    }

    // forward pass
    for(uint u = 0; u < num_units; u++){
        int sendcount = static_cast<int>(params_per_shard[shard_rank][u]);

        // 1. Allgather parameters across the shard group using precomputed arrays
        MPI_Allgatherv(shard_params[u]->data, sendcount, MPI_FLOAT,
                       layer_params[u]->data,
                       recvcounts_units[u],
                       displs_units[u],
                       MPI_FLOAT,
                       unit_comm);

        // 2. Local forward computation
        usleep(fwd_rt_whole_unit); // simulate forward computation time
    }

    //backward pass
    for(int u = num_units - 1; u >= 0; u--){
        //1. allgather if save_parameters is false
        if(!save_parameters){
            int sendcount = static_cast<int>(params_per_shard[shard_rank][u]);
            MPI_Allgatherv(shard_params[u]->data, sendcount, MPI_FLOAT,
                           layer_params[u]->data,
                           recvcounts_units[u],
                           displs_units[u],
                           MPI_FLOAT,
                           unit_comm);
        }

        //2. local backward computation
        usleep(bwd_rt_whole_unit); // simulate backward computation time

        //3. Reduce-Scatter
        MPI_Reduce_scatter(layer_params[u]->data,   // full layer gradients
                    shard_params[u]->data,       // receive reduced gradient for this shard
                    recvcounts_units[u],         // counts per shard
                    MPI_FLOAT,
                    MPI_SUM,
                    allreduce_comm);

        if(num_replicas > 1){
            //4. Allreduce across replicas if num_replicas > 1
            MPI_Allreduce(MPI_IN_PLACE,
                          shard_params[u]->data,
                          static_cast<int>(params_per_shard[shard_rank][u]),
                          MPI_FLOAT,
                          MPI_SUM,
                          allreduce_comm);
        }
    }

}

int main(int argc, char* argv[]){
    int world_size, rank;

    bool save_parameters = false; // default DO NOT save_parameters during forward pass so it DOES NOT avoid allgather during backward

    if(argc < 3){
        std::cout << "Usage: mpirun -n <world_size> ./fsdp <model_name> <num_units> <sharding_factor> <save_parameters>\n";
        return -1;
    }

    uint num_units = std::stoi(argv[2]);
    uint sharding_factor = std::stoi(argv[3]);
    std::string model_name = argv[1];

    if (argc > 3) save_parameters = (std::string(argv[4]) == "true");
    
    std::map<std::string, uint64_t> model_stats = get_model_stats("../../model_stats/" + model_name + ".txt"); // get model stats from file
    std::string model_name_nobatch = model_name.substr(0, model_name.find_last_of("_"));
    uint num_layers = count_layers("../../models/" + model_name_nobatch + ".json");

    uint local_batch_size = model_stats["batchSize"];
    uint64_t total_model_size = model_stats["modelSize"]; // number of parameters
    uint64_t fwd_rt_whole_model = model_stats["avgForwardTime"]; // in us
    uint64_t bwd_rt_whole_model = model_stats["avgBackwardTime"]; // in us

    //TODO: compute per-process stats
    uint64_t base_params_per_unit = total_model_size / num_units;
    uint64_t remainder = total_model_size % num_units;
    uint64_t params_per_unit[num_units];

    for (int i = 0; i < num_units; i++) {
        params_per_unit[i] = base_params_per_unit + (i < remainder ? 1 : 0); // distribute remainder across the units
    }

    uint64_t** params_per_shard = new uint64_t*[sharding_factor];
    for(int s = 0; s < sharding_factor; s++){
        params_per_shard[s] = new uint64_t[num_units];
    }

    for (int u = 0; u < num_units; u++) {
        uint64_t base_params_per_shard = params_per_unit[u] / sharding_factor;
        uint64_t rem = params_per_unit[u] % sharding_factor;

        for (int s = 0; s < sharding_factor; s++) {
            params_per_shard[s][u] = base_params_per_shard + (s < rem ? 1 : 0);
        }
    }

    // Precompute recvcounts and displacements for each unit (useful for Allgatherv)
    int** recvcounts_units = new int*[num_units];
    int** displs_units = new int*[num_units];

    for(uint u = 0; u < num_units; u++){
        recvcounts_units[u] = new int[sharding_factor];
        displs_units[u] = new int[sharding_factor];
        displs_units[u][0] = 0;
        for(int s = 0; s < sharding_factor; s++){
            recvcounts_units[u][s] = static_cast<int>(params_per_shard[s][u]);
            if(s > 0)
                displs_units[u][s] = displs_units[u][s-1] + recvcounts_units[u][s-1];
        }
    }

    float fdw_rt_whole_unit = (float)fwd_rt_whole_model / num_units;
    float bwd_rt_whole_unit = (float)bwd_rt_whole_model / num_units;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    assert(("world_size must be divisible by sharding_factor", world_size % sharding_factor == 0));
    assert(("num_layers must be divisible by num_units", num_layers % num_units == 0));

    uint num_replicas = world_size / sharding_factor; // number of model replica
    
    // create communicator for each replica group
    int replica_color = rank / sharding_factor;
    MPI_Comm unit_comm;
    MPI_Comm_split(MPI_COMM_WORLD, replica_color, rank, &unit_comm);
    int group_rank;
    MPI_Comm_rank(unit_comm, &group_rank);

    int shard_index_color = rank % sharding_factor;
    MPI_Comm allreduce_comm;
    MPI_Comm_split(MPI_COMM_WORLD, shard_index_color, rank, &allreduce_comm); // each process need to call allreduce only with the other processes holding the same shard index
    int allreduce_rank;
    MPI_Comm_rank(allreduce_comm, &allreduce_rank); 

    // buffers used for Allgathers
    Tensor<float>* shard_params[num_units];
    Tensor<float>* layer_params[num_units];
    for(int i=0; i<num_units; i++){
        shard_params[i] = new Tensor<float>(params_per_shard[shard_index_color][i], Device::CPU); // switch to Device::GPU later
        layer_params[i] = new Tensor<float>(params_per_unit[i], Device::CPU);
    }

    //buffers used for Allreduce (optional)
    Tensor<float>* allreduce_params[num_units];
    if(num_replicas > 1){
        for(int i=0; i<num_units; i++){
            allreduce_params[i] = new Tensor<float>(params_per_unit[i], Device::CPU);
        }
    }
    run_fsdp(shard_params, layer_params, allreduce_params,
             fdw_rt_whole_unit, bwd_rt_whole_unit,
             num_units, sharding_factor, params_per_shard,
             num_replicas, save_parameters,
             unit_comm, allreduce_comm,
             recvcounts_units, displs_units);

    MPI_Finalize();
    return 0;
}