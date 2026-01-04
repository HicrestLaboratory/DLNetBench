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

#if defined(PROXY_ENABLE_NCCL) || defined(PROXY_ENABLE_RCCL)
#define PROXY_ENABLE_CCL 1 
#endif 

// Determine the floating-point type
#ifdef HALF_PRECISION
    // Half precision supported only on GPU
    #if defined(PROXY_CUDA)
        #include <cuda_fp16.h>
        using _FLOAT = half;
        #ifdef PROXY_ENABLE_NCCL
            #define NCCL_FLOAT_TYPE ncclHalf
        #endif

    #elif defined(PROXY_HIP)
        #include <hip/hip_fp16.h>
        using _FLOAT = half;
        #ifdef PROXY_ENABLE_NCCL
            #define NCCL_FLOAT_TYPE ncclHalf
        #endif

    #else
        #error "HALF_PRECISION is defined but the target platform is not a supported GPU."
    #endif

#else
    // Default to float precision
    using _FLOAT = float;
    #ifdef PROXY_ENABLE_NCCL
        #define NCCL_FLOAT_TYPE ncclFloat
    #endif
#endif

// Communicator type
#ifdef PROXY_ENABLE_NCCL
    using Proxy_CommType = ncclComm_t;
#else
    using Proxy_CommType = MPI_Comm;
#endif

// STREAMS
#ifdef PROXY_ENABLE_CUDA
    using _Stream = cudaStream_t;
#elif defined(PROXY_ENABLE_HIP)
    using _Stream = hipStream_t;
#endif

#ifdef PROXY_ENABLE_CUDA
    #define CREATE_STREAM(stream) cudaStreamCreate(&(stream))
    #define DESTROY_STREAM(stream) cudaStreamDestroy(stream)
    #define SYNC_STREAM(stream) cudaStreamSynchronize(stream)
#elif defined(PROXY_ENABLE_HIP)
    #define CREATE_STREAM(stream) hipStreamCreate(&(stream))
    #define DESTROY_STREAM(stream) hipStreamDestroy(stream)
    #define SYNC_STREAM(stream) hipStreamSynchronize(stream)
#else
#endif

#endif // DATA_TYPES_HPP