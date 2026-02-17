#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <mpi.h>

// Include NCCL if enabled
#ifdef PROXY_ENABLE_NCCL
    #include <nccl.h>
#endif

#ifdef PROXY_ENABLE_RCCL
    #include <rccl.h>
#endif

#ifdef PROXY_ENABLE_CUDA
    #include <cuda_runtime.h>
    #define CCUTILS_GPU_CHECK CCUTILS_CUDA_CHECK
#endif

#ifdef PROXY_ENABLE_HIP
    #define CCUTILS_GPU_CHECK CCUTILS_HIP_CHECK
#endif

#ifdef PROXY_ENABLE_ONECCL
    #include <oneapi/ccl.hpp>
    #include <CL/sycl.hpp>  // SYCL header for queues and device memory
#endif

#if defined(PROXY_ENABLE_NCCL) || defined(PROXY_ENABLE_RCCL)
#define PROXY_ENABLE_CCL 1 
#endif 

#ifdef PROXY_BFLOAT16

    #if defined(PROXY_CUDA)
        #include <cuda_bf16.h>
        using _FLOAT = __nv_bfloat16;

    #elif defined(PROXY_HIP)
        #include <hip/hip_bfloat16.h>
        using _FLOAT = hip_bfloat16;

    #else
        #error "BFLOAT16 is defined but the target platform does not support it."
    #endif

    #ifdef PROXY_ENABLE_NCCL
        #define NCCL_FLOAT_TYPE ncclBfloat16
    #endif

    #ifdef PROXY_ENABLE_ONECCL
        using ONECCL_FLOAT_TYPE = sycl::ext::oneapi::bfloat16;
    #endif

#else
    using _FLOAT = float;

    #ifdef PROXY_ENABLE_CCL
        #define NCCL_FLOAT_TYPE ncclFloat
    #endif

    #ifdef PROXY_ENABLE_ONECCL
        using ONECCL_FLOAT_TYPE = float;
    #endif

#endif

// Communicator type
#ifdef PROXY_ENABLE_CCL
    using Proxy_CommType = ncclComm_t;
#elif defined(PROXY_ENABLE_ONECCL)
    using Proxy_CommType = oneapi::ccl::communicator;
#else
    using Proxy_CommType = MPI_Comm;
#endif

// STREAMS
#ifdef PROXY_ENABLE_CUDA
    using _Stream = cudaStream_t;
    #define CREATE_STREAM(stream) \
        CCUTILS_GPU_CHECK(cudaStreamCreate(&(stream)))
    #define DESTROY_STREAM(stream) \
        CCUTILS_GPU_CHECK(cudaStreamDestroy(stream))
    #define SYNC_STREAM(stream) \
        CCUTILS_GPU_CHECK(cudaStreamSynchronize(stream))

#elif defined(PROXY_ENABLE_HIP)
    using _Stream = hipStream_t;
    #define CREATE_STREAM(stream) \
        CCUTILS_GPU_CHECK(hipStreamCreate(&(stream)))
    #define DESTROY_STREAM(stream) \
        CCUTILS_GPU_CHECK(hipStreamDestroy(stream))
    #define SYNC_STREAM(stream) \
        CCUTILS_GPU_CHECK(hipStreamSynchronize(stream))

#elif defined(PROXY_ENABLE_ONECCL)
    using _Stream = sycl::queue;  // no pointer

    #define CREATE_STREAM(stream)                               \
        try {                                                   \
            stream = sycl::queue(sycl::gpu_selector_v);        \
        } catch (sycl::exception &e) {                         \
            std::cerr << "SYCL queue creation failed: " << e.what() << std::endl; \
            std::terminate();                                   \
        }

    #define DESTROY_STREAM(stream)  /* nothing needed, RAII handles it */

    #define SYNC_STREAM(stream)                                \
        try {                                                  \
            stream.wait();                                     \
        } catch (sycl::exception &e) {                        \
            std::cerr << "SYCL stream synchronization failed: " << e.what() << std::endl; \
            std::terminate();                                  \
        }

#endif


#endif // DATA_TYPES_HPP
