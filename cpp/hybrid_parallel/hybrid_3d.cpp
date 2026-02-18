/********************************************************************* 
 * 
 * Description: C++/MPI proxy for Transformer-based models distributed training
 *              with hybrid data, pipeline, and tensor parallelism (DP+PP+TP)
 * 
 *********************************************************************/ 

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

#include "../netcommunicators.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

// CCUTILS headers
#include <ccutils/mpi/mpi_timers.hpp>
#include <ccutils/mpi/mpi_macros.hpp>
#include <ccutils/macros.hpp>

#ifdef PROXY_ENABLE_CUDA
#include <ccutils/cuda/cuda_macros.hpp>
#endif

#ifdef PROXY_ENERGY_PROFILING
#include <profiler/power_profiler.hpp>
#endif

#ifdef PROXY_ENABLE_ONECCL
#include <oneapi/ccl.hpp>
#include <CL/sycl.hpp>
#endif

// Project headers
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

// Device to use
#if defined(PROXY_ENABLE_CUDA) || defined(PROXY_ENABLE_HIP)
constexpr Device device = Device::GPU;
#else
constexpr Device device = Device::CPU;
#endif

// Default values
#define WARM_UP 8
#define RUNS 10
#define POWER_SAMPLING_RATE_MS 5

CCUTILS_MPI_TIMER_DEF(runtime)
CCUTILS_MPI_TIMER_DEF(pp_comm)
CCUTILS_MPI_TIMER_DEF(dp_comm)
CCUTILS_MPI_TIMER_DEF(tp_comm)

/**
 * @brief Simulates one iteration of hybrid DP+PP+TP training using GPipe schedule.
 * 
 * @param num_microbatches Number of micro-batches (gradient accumulation steps)
 * @param stage_id Pipeline stage ID for this process
 * @param num_stage Total number of pipeline stages
 * @param pipe_msg_size Size of activations/gradients passed between stages (per microbatch)
 * @param fwd_rt Forward pass runtime per micro-batch (in microseconds)
 * @param bwd_rt Backward pass runtime per micro-batch (in microseconds)
 * @param grad_ptr Pointer to gradient buffer for DP all-reduce
 * @param sum_grad_ptr Pointer to reduced gradient buffer
 * @param dp_allreduce_size Size of gradient buffer for DP all-reduce
 * @param fwd_send_buff Forward activation send buffer
 * @param fwd_recv_buff Forward activation receive buffer
 * @param bwd_send_buff Backward gradient send buffer
 * @param bwd_recv_buff Backward gradient receive buffer
 * @param tp_buffer Pointer to tensor parallel buffer
 * @param tp_result_buffer Pointer to tensor parallel result buffer
 * @param tp_allreduce_size Size of tensor parallel all-reduce (shard of one microbatch)
 * @param dp_communicator Communicator for data-parallel all-reduce
 * @param pp_communicator Communicator for pipeline-parallel p2p
 * @param tp_communicator Communicator for tensor-parallel all-reduce
 * @return int Always returns 0
 */
int run_data_pipe_tensor_parallel(
    int num_microbatches, 
    int stage_id, 
    int num_stage,
    uint64_t pipe_msg_size,
    uint64_t fwd_rt,
    uint64_t bwd_rt,
    Tensor<_FLOAT, device>* grad_ptr,
    Tensor<_FLOAT, device>* sum_grad_ptr,
    uint64_t dp_allreduce_size,
    Tensor<_FLOAT, device>* fwd_send_buff,
    Tensor<_FLOAT, device>* fwd_recv_buff,
    Tensor<_FLOAT, device>* bwd_send_buff,
    Tensor<_FLOAT, device>* bwd_recv_buff,
    Tensor<_FLOAT, device>* tp_buffer,
    Tensor<_FLOAT, device>* tp_result_buffer,
    uint64_t tp_allreduce_size,
    ProxyCommunicator* dp_communicator,
    ProxyCommunicator* pp_communicator,
    ProxyCommunicator* tp_communicator){
    
    // GPipe Pipeline Schedule
    // Forward pass for all micro-batches
    for(int i = 0; i < num_microbatches; i++){
        CCUTILS_MPI_TIMER_START(pp_comm)
        if(stage_id == 0){
            // First stage: compute then send
            usleep(fwd_rt);
            pp_communicator->send(fwd_send_buff->data, pipe_msg_size, stage_id+1);
        } 
        else if(stage_id == num_stage-1){
            // Last stage: receive then compute
            pp_communicator->recv(fwd_recv_buff->data, pipe_msg_size, stage_id-1);
            usleep(fwd_rt);
        } 
        else{
            // Middle stages: receive, compute, send
            pp_communicator->recv(fwd_recv_buff->data, pipe_msg_size, stage_id-1);
            usleep(fwd_rt);
            pp_communicator->send(fwd_send_buff->data, pipe_msg_size, stage_id+1);
        }
        CCUTILS_MPI_TIMER_STOP(pp_comm)

        // Tensor parallel communication during forward pass
        // 2 all-reduces per microbatch (column-parallel and row-parallel)
        for(int tp_iter = 0; tp_iter < 2; tp_iter++){
            CCUTILS_MPI_TIMER_START(tp_comm)
            tp_communicator->Allreduce(tp_buffer->data, tp_result_buffer->data, tp_allreduce_size);
            CCUTILS_MPI_TIMER_STOP(tp_comm)
        }
    }
    
    // Backward pass for all micro-batches
    for(int i = 0; i < num_microbatches; i++){
        CCUTILS_MPI_TIMER_START(pp_comm)
        if(stage_id == 0){
            // First stage: receive then compute
            pp_communicator->recv(bwd_recv_buff->data, pipe_msg_size, stage_id+1);
            usleep(bwd_rt);
        } 
        else if(stage_id == num_stage-1){
            // Last stage: compute then send
            usleep(bwd_rt);
            pp_communicator->send(bwd_send_buff->data, pipe_msg_size, stage_id-1);
        } 
        else{
            // Middle stages: receive, compute, send
            pp_communicator->recv(bwd_recv_buff->data, pipe_msg_size, stage_id+1);
            usleep(bwd_rt);
            pp_communicator->send(bwd_send_buff->data, pipe_msg_size, stage_id-1);
        }
        CCUTILS_MPI_TIMER_STOP(pp_comm)
        
        // Tensor parallel communication during backward pass
        // 2 all-reduces per microbatch
        for(int tp_iter = 0; tp_iter < 2; tp_iter++){
            CCUTILS_MPI_TIMER_START(tp_comm)
            tp_communicator->Allreduce(tp_buffer->data, tp_result_buffer->data, tp_allreduce_size);
            CCUTILS_MPI_TIMER_STOP(tp_comm)
        }
    }
    
    // Data-parallel all-reduce for accumulated gradients
    CCUTILS_MPI_TIMER_START(dp_comm)
    dp_communicator->Allreduce(grad_ptr->data, sum_grad_ptr->data, dp_allreduce_size);
    CCUTILS_MPI_TIMER_STOP(dp_comm)
    
    return 0;
}

int main(int argc, char* argv[]) {
    int rank, world_size;
    int num_stage;
    int num_microbatches;
    int num_tensor_shards;
    
    if(argc < 5){
        std::cout << "Usage: mpirun -n <world_size> ./hybrid_3d <model_name> <num_stages> <num_microbatches> <num_tensor_shards> <base_path>\n";
        return -1;
    }
    
    std::string model_name = argv[1];
    num_stage = std::stoi(argv[2]);
    num_microbatches = std::stoi(argv[3]);
    num_tensor_shards = std::stoi(argv[4]);
    
    // --- Construct model stats file path ---
    fs::path repo_path = get_dnnproxy_base_path(argc, argv, rank);
    fs::path file_path = repo_path / "model_stats" / (model_name + ".txt");
    std::string strip_model_name = model_name.substr(0, model_name.find_last_of('_'));
    fs::path model_architecture_path = repo_path / "models" / (strip_model_name + ".json");

    uint num_layers = count_layers(model_architecture_path);
    
    if (!fs::exists(file_path)) {
        std::cerr << "Error: model stats file does not exist: " << file_path << "\n";
        return -1;
    }
    
    std::map<std::string, uint64_t> model_stats = get_model_stats(file_path);

    print_topology_graph(MPI_COMM_WORLD);
    
    // Get model stats from file
    uint64_t fwd_rt_whole_model = model_stats["avgForwardTime"]; // in us
    uint64_t bwd_rt_whole_model = model_stats["avgBackwardTime"]; // in us
    uint local_batch_size = model_stats["batchSize"];
    uint64_t total_model_size = model_stats["modelSize"]; // number of parameters
    uint sequence_length = model_stats["sequenceLength"]; // sequence length
    uint embedded_dim = model_stats["embeddedDim"]; // hidden dimension size

    uint64_t sample_size_bytes = sequence_length * embedded_dim * sizeof(_FLOAT);
    
    assert(num_layers % num_stage == 0);
    assert(local_batch_size % num_microbatches == 0);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Check that world_size = num_stages * num_tensor_shards * dp_size
    assert(world_size % (num_stage * num_tensor_shards) == 0);
    int dp_size = world_size / (num_stage * num_tensor_shards);
    
    CCUTILS_MPI_INIT
    
    // Create DP, PP, and TP communicators
    // Hierarchy: world_size = num_stages * num_tensor_shards * dp_size
    // Layout: [DP replicas] x [Pipeline stages] x [Tensor shards]
    
    // Calculate position in 3D grid
    int tp_id = rank % num_tensor_shards;
    int stage_id = (rank / num_tensor_shards) % num_stage;
    int dp_id = rank / (num_tensor_shards * num_stage);
    
    // Create TP communicator: groups GPUs with same (stage_id, dp_id)
    int tp_color = dp_id * num_stage + stage_id;
    MPI_Comm tp_comm;
    MPI_Comm_split(MPI_COMM_WORLD, tp_color, rank, &tp_comm);
    
    // Create PP communicator: groups GPUs with same (dp_id, tp_id)
    int pp_color = dp_id * num_tensor_shards + tp_id;
    MPI_Comm pp_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pp_color, rank, &pp_comm);
    
    // Create DP communicator: groups GPUs with same (stage_id, tp_id)
    int dp_color = stage_id * num_tensor_shards + tp_id;
    MPI_Comm dp_comm;
    MPI_Comm_split(MPI_COMM_WORLD, dp_color, rank, &dp_comm);
    
    int dp_rank, pp_rank, tp_rank;
    MPI_Comm_rank(dp_comm, &dp_rank);
    MPI_Comm_rank(pp_comm, &pp_rank);
    MPI_Comm_rank(tp_comm, &tp_rank);
    
    // Verify stage_id matches pp_rank
    assert(stage_id == pp_rank);

    // Compute per-stage and per-microbatch runtimes
    uint64_t fwd_rt_per_stage = fwd_rt_whole_model / num_stage;
    uint64_t bwd_rt_per_stage = bwd_rt_whole_model / num_stage;
    
    uint64_t fwd_rt_per_microbatch = fwd_rt_per_stage / (num_microbatches * num_tensor_shards);
    uint64_t bwd_rt_per_microbatch = bwd_rt_per_stage / (num_microbatches * num_tensor_shards);
    
    // Pipeline message size: activations for batch_size/num_microbatches samples
    uint64_t samples_per_microbatch = local_batch_size / num_microbatches;
    uint64_t pipe_msg_size = (uint64_t)(sequence_length * embedded_dim * samples_per_microbatch);
    
    // TP all-reduce size: one microbatch split across tensor shards
    uint64_t tp_allreduce_size = pipe_msg_size / num_tensor_shards;
    
    // DP all-reduce size (gradients for parameters in this stage, divided by TP shards)
    uint64_t dp_allreduce_size = total_model_size / (num_stage * num_tensor_shards);
    
#if defined(PROXY_ENABLE_CUDA)
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    CCUTILS_CUDA_CHECK(cudaSetDevice(rank % num_gpus));
#elif defined(PROXY_ENABLE_HIP)
    int num_gpus;
    hipGetDeviceCount(&num_gpus);
    CCUTILS_HIP_CHECK(hipSetDevice(rank % num_gpus));
#endif

#ifdef PROXY_ENABLE_CCL
    // Initialize CCL for DP communicator
    ncclUniqueId dp_id_nccl;
    if (dp_rank == 0) {
        ncclGetUniqueId(&dp_id_nccl);
    }
    MPI_Bcast(&dp_id_nccl, sizeof(dp_id_nccl), MPI_BYTE, 0, dp_comm);
    
    Proxy_CommType dp_world_comm;
    ncclCommInitRank(&dp_world_comm, dp_size, dp_id_nccl, dp_rank);
    CCLCommunicator* dp_communicator = new CCLCommunicator(dp_world_comm, 1);
    
    // Initialize CCL for PP communicator
    ncclUniqueId pp_id_nccl;
    int pp_size;
    MPI_Comm_size(pp_comm, &pp_size);
    if (pp_rank == 0) {
        ncclGetUniqueId(&pp_id_nccl);
    }
    MPI_Bcast(&pp_id_nccl, sizeof(pp_id_nccl), MPI_BYTE, 0, pp_comm);
    
    Proxy_CommType pp_world_comm;
    ncclCommInitRank(&pp_world_comm, pp_size, pp_id_nccl, pp_rank);
    CCLCommunicator* pp_communicator = new CCLCommunicator(pp_world_comm, 1);
    
    // Initialize CCL for TP communicator
    ncclUniqueId tp_id_nccl;
    int tp_size;
    MPI_Comm_size(tp_comm, &tp_size);
    if (tp_rank == 0) {
        ncclGetUniqueId(&tp_id_nccl);
    }
    MPI_Bcast(&tp_id_nccl, sizeof(tp_id_nccl), MPI_BYTE, 0, tp_comm);
    
    Proxy_CommType tp_world_comm;
    ncclCommInitRank(&tp_world_comm, tp_size, tp_id_nccl, tp_rank);
    CCLCommunicator* tp_communicator = new CCLCommunicator(tp_world_comm, 1);
    
#elif defined(PROXY_ENABLE_ONECCL)
    // Select GPU device
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    int num_gpus = gpus.size();

    // DP communicator
    sycl::device dp_dev = gpus[dp_rank % num_gpus];
    sycl::context dp_ctx(dp_dev);
    sycl::queue dp_queue(dp_ctx, dp_dev);

    ccl::shared_ptr_class<ccl::kvs> dp_kvs;
    if (dp_rank == 0) dp_kvs = ccl::create_main_kvs();

    std::vector<char> dp_addr;
    if (dp_rank == 0) dp_addr = dp_kvs->get_address();

    size_t dp_addr_size = dp_addr.size();
    MPI_Bcast(&dp_addr_size, 1, MPI_UNSIGNED_LONG, 0, dp_comm);
    if (dp_rank != 0) dp_addr.resize(dp_addr_size);
    MPI_Bcast(dp_addr.data(), dp_addr_size, MPI_BYTE, 0, dp_comm);

    if (dp_rank != 0) dp_kvs = ccl::create_kvs(dp_addr);

    auto dp_world_comm = ccl::create_communicator(dp_size, dp_rank, dp_kvs, dp_ctx);
    OneCCLCommunicator* dp_communicator = new OneCCLCommunicator(dp_world_comm, 1);

    // PP communicator
    int pp_size;
    MPI_Comm_size(pp_comm, &pp_size);

    sycl::device pp_dev = gpus[pp_rank % num_gpus];
    sycl::context pp_ctx(pp_dev);
    sycl::queue pp_queue(pp_ctx, pp_dev);

    ccl::shared_ptr_class<ccl::kvs> pp_kvs;
    if (pp_rank == 0) pp_kvs = ccl::create_main_kvs();

    std::vector<char> pp_addr;
    if (pp_rank == 0) pp_addr = pp_kvs->get_address();

    size_t pp_addr_size = pp_addr.size();
    MPI_Bcast(&pp_addr_size, 1, MPI_UNSIGNED_LONG, 0, pp_comm);
    if (pp_rank != 0) pp_addr.resize(pp_addr_size);
    MPI_Bcast(pp_addr.data(), pp_addr_size, MPI_BYTE, 0, pp_comm);

    if (pp_rank != 0) pp_kvs = ccl::create_kvs(pp_addr);

    auto pp_world_comm = ccl::create_communicator(pp_size, pp_rank, pp_kvs, pp_ctx);
    OneCCLCommunicator* pp_communicator = new OneCCLCommunicator(pp_world_comm, 1);

    // TP communicator
    int tp_size;
    MPI_Comm_size(tp_comm, &tp_size);

    sycl::device tp_dev = gpus[tp_rank % num_gpus];
    sycl::context tp_ctx(tp_dev);
    sycl::queue tp_queue(tp_ctx, tp_dev);

    ccl::shared_ptr_class<ccl::kvs> tp_kvs;
    if (tp_rank == 0) tp_kvs = ccl::create_main_kvs();

    std::vector<char> tp_addr;
    if (tp_rank == 0) tp_addr = tp_kvs->get_address();

    size_t tp_addr_size = tp_addr.size();
    MPI_Bcast(&tp_addr_size, 1, MPI_UNSIGNED_LONG, 0, tp_comm);
    if (tp_rank != 0) tp_addr.resize(tp_addr_size);
    MPI_Bcast(tp_addr.data(), tp_addr_size, MPI_BYTE, 0, tp_comm);

    if (tp_rank != 0) tp_kvs = ccl::create_kvs(tp_addr);

    auto tp_world_comm = ccl::create_communicator(tp_size, tp_rank, tp_kvs, tp_ctx);
    OneCCLCommunicator* tp_communicator = new OneCCLCommunicator(tp_world_comm, 1);

#else
    MPICommunicator* dp_communicator = new MPICommunicator(dp_comm, MPI_FLOAT, 1);
    MPICommunicator* pp_communicator = new MPICommunicator(pp_comm, MPI_FLOAT, 1);
    MPICommunicator* tp_communicator = new MPICommunicator(tp_comm, MPI_FLOAT, 1);
#endif
    
    // Allocate buffers
    Tensor<_FLOAT, device>* grad_ptr = new Tensor<_FLOAT, device>(dp_allreduce_size);
    Tensor<_FLOAT, device>* sum_grad_ptr = new Tensor<_FLOAT, device>(dp_allreduce_size);
    
    Tensor<_FLOAT, device>* fwd_send_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    Tensor<_FLOAT, device>* fwd_recv_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    Tensor<_FLOAT, device>* bwd_send_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    Tensor<_FLOAT, device>* bwd_recv_buff = new Tensor<_FLOAT, device>(pipe_msg_size);
    
    Tensor<_FLOAT, device>* tp_buffer = new Tensor<_FLOAT, device>(tp_allreduce_size);
    Tensor<_FLOAT, device>* tp_result_buffer = new Tensor<_FLOAT, device>(tp_allreduce_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Warmup
    std::vector<float> energy_vals;
    for(int wmp = 0; wmp < WARM_UP; wmp++){
        run_data_pipe_tensor_parallel(num_microbatches, stage_id, num_stage, pipe_msg_size,
                              fwd_rt_per_microbatch, bwd_rt_per_microbatch,
                              grad_ptr, sum_grad_ptr, dp_allreduce_size,
                              fwd_send_buff, fwd_recv_buff, bwd_send_buff, bwd_recv_buff,
                              tp_buffer, tp_result_buffer, tp_allreduce_size,
                              dp_communicator, pp_communicator, tp_communicator);
    }
    
    #ifdef PROXY_ENERGY_PROFILING
    std::string sub_folder = model_name + "_dp_pp_tp_stages_" + std::to_string(num_stage) + 
                            "_tp_" + std::to_string(num_tensor_shards) + "/";
    std::string base_folder_path = "logs_" + std::to_string(world_size) + "/";
    #endif

    #ifdef PROXY_LOOP
    while(true){
        run_data_pipe_tensor_parallel(num_microbatches, stage_id, num_stage, pipe_msg_size,
                            fwd_rt_per_microbatch, bwd_rt_per_microbatch,
                            grad_ptr, sum_grad_ptr, dp_allreduce_size,
                            fwd_send_buff, fwd_recv_buff, bwd_send_buff, bwd_recv_buff,
                            tp_buffer, tp_result_buffer, tp_allreduce_size,
                            dp_communicator, pp_communicator, tp_communicator);
    }
    #else
    for(int iter = 0; iter < RUNS; iter++){
        #ifdef PROXY_ENERGY_PROFILING
        std::string power_file = base_folder_path + sub_folder + "power_dp_pp_tp_rank_" + 
                                std::to_string(rank) + "_run_" + std::to_string(iter) + ".csv";
        PowerProfiler powerProf(rank % num_gpus, POWER_SAMPLING_RATE_MS, power_file);
        #endif
        CCUTILS_MPI_TIMER_START(runtime)
        #ifdef PROXY_ENERGY_PROFILING
        powerProf.start();
        #endif
        
        run_data_pipe_tensor_parallel(num_microbatches, stage_id, num_stage, pipe_msg_size,
                              fwd_rt_per_microbatch, bwd_rt_per_microbatch,
                              grad_ptr, sum_grad_ptr, dp_allreduce_size,
                              fwd_send_buff, fwd_recv_buff, bwd_send_buff, bwd_recv_buff,
                              tp_buffer, tp_result_buffer, tp_allreduce_size,
                              dp_communicator, pp_communicator, tp_communicator);
        
        #ifdef PROXY_ENERGY_PROFILING
        powerProf.stop();
        float energy_consumed = powerProf.get_device_energy();
        energy_vals.push_back(energy_consumed);
        #endif
        CCUTILS_MPI_TIMER_STOP(runtime)
    }
    
    char host_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI_Get_processor_name(host_name, &namelen);
    
    CCUTILS_MPI_SECTION_DEF(dp_pp_tp, "Data + Pipeline + Tensor Parallelism")
    
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "model_name", model_name)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "num_stages", num_stage)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "num_microbatches", num_microbatches)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "num_tensor_shards", num_tensor_shards)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "samples_per_microbatch", samples_per_microbatch)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "local_batch_size", local_batch_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "global_batch_size", dp_size * local_batch_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "world_size", world_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "dp_size", dp_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "fwd_rt_per_microbatch", fwd_rt_per_microbatch)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "bwd_rt_per_microbatch", bwd_rt_per_microbatch)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "total_model_size_params", total_model_size)
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "pipe_msg_size_bytes", pipe_msg_size * sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "tp_allreduce_size_bytes", tp_allreduce_size * sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "dp_allreduce_size_bytes", dp_allreduce_size * sizeof(_FLOAT))
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "device", (device == Device::CPU) ? "CPU" : "GPU")
    CCUTILS_MPI_GLOBAL_JSON_PUT(dp_pp_tp, "backend", dp_communicator->get_name())
    
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "runtimes", __timer_vals_runtime);
    
    __timer_vals_pp_comm.erase(__timer_vals_pp_comm.begin(), __timer_vals_pp_comm.begin() + WARM_UP);
    __timer_vals_dp_comm.erase(__timer_vals_dp_comm.begin(), __timer_vals_dp_comm.begin() + WARM_UP);
    __timer_vals_tp_comm.erase(__timer_vals_tp_comm.begin(), __timer_vals_tp_comm.begin() + WARM_UP);
    
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "pp_comm_time", __timer_vals_pp_comm);
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "dp_comm_time", __timer_vals_dp_comm);
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "tp_comm_time", __timer_vals_tp_comm);
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "hostname", host_name);
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "stage_id", stage_id);
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "tp_id", tp_id);
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "dp_id", dp_id);
    #ifdef PROXY_ENERGY_PROFILING
    CCUTILS_SECTION_JSON_PUT(dp_pp_tp, "energy_consumed", energy_vals);
    #endif
    CCUTILS_MPI_SECTION_END(dp_pp_tp);
    #endif
    
#ifdef PROXY_ENABLE_CCL
    ncclCommDestroy(dp_world_comm);
    ncclCommDestroy(pp_world_comm);
    ncclCommDestroy(tp_world_comm);
#endif
    
    MPI_Finalize();
    
    return 0;
}
