#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

// Include NCCL if enabled
#ifdef PROXY_ENABLE_NCCL
    #include <nccl.h>
#endif

#ifdef PROXY_ENABLE_RCCL
    #include <rccl.h>
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

#ifdef PROXY_ENABLE_CUDA
    using _Stream = cudaStream_t;
#elif defined(PROXY_ENABLE_HIP)
    using _Stream = hipStream_t;
#endif

#endif // DATA_TYPES_HPP