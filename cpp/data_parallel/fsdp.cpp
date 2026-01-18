/*********************************************************************
 *
 * Description: C++/MPI proxy for Transformer-based models distributed training 
 *              with data parallelism
 * Author: Jacopo Raffi
 *
 * Proxy code inspired by: https://engineering.fb.com/2021/07/15/open-source/fsdp/
 * 
 * 
 *********************************************************************/

//TODO: add oneCCL support

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

#ifdef PROXY_ENERGY_PROFILING
#include <profiler/power_profiler.hpp>
#endif

#include "../utils.hpp"
#include "../data_types.hpp"
#include "../proxy_classes.hpp"

#ifdef PROXY_ENABLE_NCCL
    #include <nccl.h>
Proxy_CommType world_comm;
#endif

#ifdef PROXY_ENABLE_RCCL
    #include <rccl.h>
Proxy_CommType world_comm;
#endif

// Determine device type based on compilation flags
#if defined(PROXY_ENABLE_CUDA) || defined(PROXY_ENABLE_HIP)
    constexpr Device device = Device::GPU;
#else
    constexpr Device device = Device::CPU;
#endif

/*
 * Parameters from users: n_units, save_parameters (avoid allgather during backward), sharding_factor F:
 * each layer is sharded across F processes, so each process holds 1/F of the parameters, if F=World_size, then each layer is fully sharded (no model replica)
 * if 1 < F < World_size, then each layer is partially sharded and we have more replicas of the model.
 * assert that (num_layers % num_units == 0) and (world_size % F == 0)
*/

//timers for each collective and the whole runtime
CCUTILS_MPI_TIMER_DEF(allgather)
CCUTILS_MPI_TIMER_DEF(reduce_scatter)
CCUTILS_MPI_TIMER_DEF(allgather_wait_fwd)
CCUTILS_MPI_TIMER_DEF(allgather_wait_bwd)
CCUTILS_MPI_TIMER_DEF(barrier)
CCUTILS_MPI_TIMER_DEF(runtime)

//default values
#define WARM_UP 8
#define RUNS 10
#define POWER_SAMPLING_RATE_MS 5

void run_fsdp(Tensor<_FLOAT, device>** shard_params,
              Tensor<_FLOAT, device>* layer_params,
              Tensor<_FLOAT, device>** allreduce_params,
              float fwd_rt_whole_unit,
              float bwd_rt_whole_unit,
              uint num_units,
              uint sharding_factor,
              uint64_t* max_params_per_shard,
              uint num_replicas,
              ProxyCommunicator* unit_comm,
              ProxyCommunicator* allreduce_comm){

    //all-gather firs unit's parameters to form full layer parameters
    CCUTILS_MPI_TIMER_START(allgather);
    unit_comm->Allgather(shard_params[0]->data,
                            static_cast<int>(max_params_per_shard[0]),
                            layer_params->data,
                            static_cast<int>(max_params_per_shard[0]));
    CCUTILS_MPI_TIMER_STOP(allgather);


    // Forward pass
    for (uint u = 0; u < num_units-1; u++) {
        unit_comm->Iallgather(shard_params[u+1]->data,
                                static_cast<int>(max_params_per_shard[u+1]),
                                layer_params->data,
                                static_cast<int>(max_params_per_shard[u+1]),
                                u+1); // gather params next unit

        // Local forward computation current unit (simulated)
        usleep(fwd_rt_whole_unit);

        CCUTILS_MPI_TIMER_START(allgather_wait_fwd);
        unit_comm->Wait(u+1);
        CCUTILS_MPI_TIMER_STOP(allgather_wait_fwd);
    }

    // Backward pass
    for (int u = num_units - 1; u > 0; u--) {
        //1. Allgather for previous unit
        unit_comm->Iallgather(shard_params[u-1]->data,
                                static_cast<int>(max_params_per_shard[u-1]),
                                layer_params->data,
                                static_cast<int>(max_params_per_shard[u-1]),
                                u-1);

        // 2. Local backward computation (simulated)
        usleep(bwd_rt_whole_unit);

        // 3. Reduce-Scatter across shards
        CCUTILS_MPI_TIMER_START(reduce_scatter);
        unit_comm->Reduce_Scatter_block(layer_params->data,
                                  shard_params[u]->data,
                                  static_cast<int>(max_params_per_shard[u]));
        CCUTILS_MPI_TIMER_STOP(reduce_scatter);

        // 4. Optional allreduce across replicas
        if (num_replicas > 1) {
            allreduce_comm->Iallreduce(shard_params[u]->data,
                                       allreduce_params[u]->data,
                                       static_cast<int>(max_params_per_shard[u]),
                                       u);
        }

        CCUTILS_MPI_TIMER_START(allgather_wait_bwd);
        unit_comm->Wait(u-1);
        CCUTILS_MPI_TIMER_STOP(allgather_wait_bwd);
    }

    // Handle unit 0 backward
    int u = 0;
    // Local backward computation (simulated)
    usleep(bwd_rt_whole_unit);

    // Reduce-Scatter across shards
    CCUTILS_MPI_TIMER_START(reduce_scatter);
    unit_comm->Reduce_Scatter_block(layer_params->data,
                                    shard_params[u]->data,
                                    static_cast<int>(max_params_per_shard[u]));
    CCUTILS_MPI_TIMER_STOP(reduce_scatter);
    if (num_replicas > 1){
        allreduce_comm->Iallreduce(shard_params[u]->data,
                                       allreduce_params[u]->data,
                                       static_cast<int>(max_params_per_shard[u]),
                                       u); // handle last unit(unit 0) allreduce
        CCUTILS_MPI_TIMER_START(barrier);
        allreduce_comm->WaitAll(num_units);
        CCUTILS_MPI_TIMER_STOP(barrier);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    CCUTILS_MPI_INIT

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 5) {
        CCUTILS_MPI_PRINT_ONCE(std::cout << "Usage: mpirun -n <world_size> ./fsdp <model_name> <num_units> <sharding_factor> <base_path>\n";)
        MPI_Finalize();
        return -1;
    }

    std::string model_name = argv[1];
    uint num_units = std::stoi(argv[2]);
    uint sharding_factor = std::stoi(argv[3]);

    assert(world_size % sharding_factor == 0);
    //FIXME: assert num_layers % num_units == 0 when num_layers info is available

     // --- Get DNNProxy base path ---
    fs::path repo_path = get_dnnproxy_base_path(argc, argv, rank);
    if (repo_path.empty()) {
        MPI_Finalize();
        return -1;  // DNNProxy not found
    }

    // --- Construct model stats file path ---
    fs::path file_path = repo_path / "model_stats" / (model_name + ".txt");
    if (!fs::exists(file_path)) {
        CCUTILS_MPI_PRINT_ONCE(std::cerr << "Error: model stats file does not exist: " << file_path << "\n")
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
    Tensor<_FLOAT, device>* shard_params[num_units];
    Tensor<_FLOAT, device>* allreduce_params[num_units];

    uint64_t max_shard_size = 0;
    for (uint u = 0; u < num_units; u++)
        if (max_params_per_shard[u] > max_shard_size)
            max_shard_size = max_params_per_shard[u];

    Tensor<_FLOAT, device>* allgather_buf = new Tensor<_FLOAT, device>(max_shard_size * sharding_factor);

    for (uint u = 0; u < num_units; u++) {
        shard_params[u] = new Tensor<_FLOAT, device>(max_params_per_shard[u]);
        if (num_replicas > 1)
            allreduce_params[u] = new Tensor<_FLOAT, device>(max_params_per_shard[u]);
    }

    float fwd_rt_whole_unit = (float)fwd_rt_whole_model / num_units;
    float bwd_rt_whole_unit = (float)bwd_rt_whole_model / num_units;


    #ifdef PROXY_ENABLE_CCL 
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&world_comm, world_size, id, rank);
    // create NCCL communicators for unit_comm and allreduce_comm
    ncclComm_t unit_nccl_comm;
    ncclComm_t allreduce_nccl_comm;
    ncclCommSplit(world_comm, replica_color, rank, &unit_nccl_comm, NULL);
    ncclCommSplit(world_comm, shard_index_color, rank, &allreduce_nccl_comm, NULL);
    CCLCommunicator* unit_comm_proxy = new CCLCommunicator(unit_nccl_comm, num_units);
    CCLCommunicator* allreduce_comm_proxy = new CCLCommunicator(allreduce_nccl_comm, num_units);  
    #else
    MPICommunicator* unit_comm_proxy = new MPICommunicator(unit_comm, MPI_FLOAT, num_units);
    MPICommunicator* allreduce_comm_proxy = new MPICommunicator(allreduce_comm, MPI_FLOAT, num_units);
    #endif

    for(int i = 0; i < WARM_UP; i++)
        run_fsdp(shard_params, allgather_buf, allreduce_params,
                 fwd_rt_whole_unit, bwd_rt_whole_unit,
                 num_units, sharding_factor, max_params_per_shard,
                 num_replicas, unit_comm_proxy, allreduce_comm_proxy);

    
    // std::vector<float> energy_vals;
    for(int i = 0; i < RUNS; i++){ //TODO: add energy profiling
        CCUTILS_MPI_TIMER_START(runtime);
        run_fsdp(shard_params, allgather_buf, allreduce_params,
                 fwd_rt_whole_unit, bwd_rt_whole_unit,
                 num_units, sharding_factor, max_params_per_shard,
                 num_replicas, unit_comm_proxy, allreduce_comm_proxy);
        CCUTILS_MPI_TIMER_STOP(runtime);
    } 

    char host_name[MPI_MAX_PROCESSOR_NAME];
	char (*host_names)[MPI_MAX_PROCESSOR_NAME];
	int namelen,bytes,n,color;
	MPI_Get_processor_name(host_name,&namelen);

    //remove the warm-up timings
    __timer_vals_allgather.erase(__timer_vals_allgather.begin(), __timer_vals_allgather.begin() + WARM_UP);
    __timer_vals_allgather_wait_fwd.erase(__timer_vals_allgather_wait_fwd.begin(), __timer_vals_allgather_wait_fwd.begin() + WARM_UP * (num_units - 1));
    __timer_vals_allgather_wait_bwd.erase(__timer_vals_allgather_wait_bwd.begin(), __timer_vals_allgather_wait_bwd.begin() + WARM_UP * (num_units - 1));
    __timer_vals_reduce_scatter.erase(__timer_vals_reduce_scatter.begin(), __timer_vals_reduce_scatter.begin() + num_units * WARM_UP);

    if(num_replicas > 1){
        __timer_vals_barrier.erase(__timer_vals_barrier.begin(), __timer_vals_barrier.begin() + WARM_UP);
    }

    // Use CCUTILS sections to print
    CCUTILS_MPI_SECTION_DEF(fsdp, "FSDP metrics")
    CCUTILS_SECTION_JSON_PUT(fsdp, "runtime", __timer_vals_runtime)
    CCUTILS_SECTION_JSON_PUT(fsdp, "allgather", __timer_vals_allgather)
    CCUTILS_SECTION_JSON_PUT(fsdp, "allgather_wait_fwd", __timer_vals_allgather_wait_fwd)
    CCUTILS_SECTION_JSON_PUT(fsdp, "allgather_wait_bwd", __timer_vals_allgather_wait_bwd)
    CCUTILS_SECTION_JSON_PUT(fsdp, "reduce_scatter", __timer_vals_reduce_scatter)
    CCUTILS_SECTION_JSON_PUT(fsdp, "barrier", __timer_vals_barrier)
    // CCUTILS_SECTION_JSON_PUT(fsdp, "energy_consumed", energy_vals);
    CCUTILS_SECTION_JSON_PUT(fsdp, "hostname", host_name);
    CCUTILS_SECTION_JSON_PUT(fsdp, "rank", rank);


    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "model_size_bytes", total_model_size*sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "num_units", num_units)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "sharding_factor", sharding_factor)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "num_replicas", num_replicas)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "local_batch_size", local_batch_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "device", (device == Device::CPU) ? "CPU" : "GPU")
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "backend", unit_comm_proxy->get_name())
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "fwd_time_per_unit_us", fwd_rt_whole_unit)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "bwd_time_per_unit_us", bwd_rt_whole_unit)

    // allgather and reducescatter msg_size
    // Since all units are equal
    uint64_t allgather_msg_size = max_params_per_shard[0] * sharding_factor * sizeof(_FLOAT);
    uint64_t reducescatter_msg_size = max_params_per_shard[0] * sizeof(_FLOAT);
    uint64_t allreduce_msg_size = 0;

    if (num_replicas > 1)
        allreduce_msg_size = max_params_per_shard[0] * sizeof(_FLOAT);

    // Put in JSON
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "allgather_msg_size_bytes", allgather_msg_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "reducescatter_msg_size_bytes", reducescatter_msg_size)
    if (num_replicas > 1)
        CCUTILS_MPI_GLOBAL_JSON_PUT(fsdp, "allreduce_msg_size_bytes", allreduce_msg_size)

    CCUTILS_MPI_SECTION_END(fsdp)

    #ifdef PROXY_ENABLE_CLL
    ncclCommDestroy(world_comm);
    #endif

    MPI_Finalize();
    return 0;
}