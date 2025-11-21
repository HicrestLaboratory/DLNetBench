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



void run_fsdp(Tensor<float>** grad_ptrs, Tensor<float>** sum_grad_ptrs, 
            uint64_t* params_per_unit, float fwd_rt_whole_unit, float bwd_rt_whole_unit, 
            uint num_replicas, bool save_parameters, MPI_Comm unit_comm){
   /*
    * For each unit:
    *
    * 1. Allgather parameters across the shard group (shard -> full layer)
    * 2. Local forward computation
    * 3. Free peer shards / release parameters (optional, memory optimization)
    * 4. Allgather parameters on-demand before backward 
    *      (only if they were released after forward)
    * 5. Local backward computation (compute gradients)
    * 6. ReduceScatter gradients within the shard group 
    *      (each rank keeps its shard)
    * 7. If F < world_size:
    *       Allreduce gradients across replicas 
    *       (synchronize model copies; each shard index talks to corresponding ranks)
    * 8. Free forward activations/buffers (optional, memory optimization)
    */

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
    uint num_layers = count_layers("../../models/" + model_name + ".json");

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

    uint64_t params_per_shard[sharding_factor][num_units]; // number of parameters held by each process for each unit
    
    for (int u = 0; u < num_units; u++) {
        uint64_t base_params_per_shard = params_per_unit[u] / sharding_factor;
        uint64_t rem = params_per_unit[u] % sharding_factor;
        for (int s = 0; s < sharding_factor; s++) {
            params_per_shard[s][u] = base_params_per_shard + (s < rem ? 1 : 0);
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
    int replica_color = rank / num_replicas;
    MPI_Comm unit_comm;
    MPI_Comm_split(MPI_COMM_WORLD, replica_color, rank, &unit_comm);
    int group_rank;
    MPI_Comm_rank(unit_comm, &group_rank);

    int shard_index_color = rank % sharding_factor;
    MPI_Comm allreduce_comm;
    MPI_Comm_split(MPI_COMM_WORLD, shard_index_color, rank, &allreduce_comm); // each process need to call allreduce only with the other processes holding the same shard index
    int allreduce_rank;
    MPI_Comm_rank(allreduce_comm, &allreduce_rank); 

    return 0;
}