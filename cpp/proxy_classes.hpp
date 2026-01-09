#ifndef PROXY_CLASSES_HPP
#define PROXY_CLASSES_HPP

#include <mpi.h>

#ifdef PROXY_ENABLE_CUDA   
    #include <cuda_runtime.h>
    #include <ccutils/cuda/cuda_macros.hpp>
#endif

#ifdef PROXY_ENABLE_HIP
    #include "tmp_hip_ccutils.hpp" 
    #include <hip/hip_runtime.h>
#endif

#ifdef PROXY_ENABLE_NCLL
    #include <nccl.h>
#endif

#ifdef PROXY_ENABLE_RCCL
    #include <rccl.h>
#endif

class ProxyCommunicator {
public:
    virtual void Iallreduce(const void* sendbuf, void* recvbuf, int count, int index) = 0;
    virtual void Iallgather(const void* sendbuf, int sendcount,
                            void* recvbuf, int recvcount, int index) = 0;
    virtual void Allgather(const void* sendbuf, int sendcount,
                           void* recvbuf, int recvcount) = 0;
    virtual void Reduce_Scatter_block(const void* sendbuf, void* recvbuf, int recvcount) = 0;
    virtual void Barrier() = 0;
    virtual void WaitAll(int num_waits) = 0;
    virtual void Wait(int index) = 0;
    virtual void finalize() = 0;
    virtual std::string get_name() = 0;
    virtual ~ProxyCommunicator() {}
};

class MPICommunicator : public ProxyCommunicator {
public:
    MPICommunicator(MPI_Comm comm, MPI_Datatype datatype, int num_requests=0) {
        this->comm = comm;
        this->datatype = datatype;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &comm_size);

        if(num_requests > 0){
            requests = new MPI_Request[num_requests];
        }
    };

    void Iallreduce(const void* sendbuf, void* recvbuf, int count, int index) override {
        MPI_Iallreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, comm, &(requests[index]));
    };

    void Barrier() override {
        MPI_Barrier(comm);
    };

    void WaitAll(int num_waits) override {
        MPI_Waitall(num_waits, requests, MPI_STATUSES_IGNORE);
    };

    void Wait(int index) override {
        MPI_Wait(&requests[index], MPI_STATUS_IGNORE);
    };

    void Iallgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, int index) override {
        MPI_Iallgather(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, comm, &requests[index]);
    }

    void Allgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount) override {
        MPI_Allgather(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, comm);
    };

    void Reduce_Scatter_block(const void* sendbuf, void* recvbuf, int recvcount) override {
        MPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, MPI_FLOAT, MPI_SUM, comm);
    };

    std::string get_name() override {
        return std::string("MPI");
    };

    void finalize() override {
        MPI_Finalize();
    };
private:
    MPI_Comm comm;
    int comm_size;
    int rank;
    MPI_Datatype datatype = MPI_FLOAT;
    MPI_Request* requests = nullptr;

};

//TODO: add oneCCL class

#ifdef PROXY_ENABLE_CCL //NCCL or RCCL
class CCLCommunicator : public ProxyCommunicator {
public:
    CCLCommunicator(ncclComm_t comm, int num_streams=1) {
        this->comm = comm;
        this->num_streams = num_streams;
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &comm_size);
        this->streams = new _Stream[num_streams];
        for(int i = 0; i < num_streams; i++) {
            CREATE_STREAM(this->streams[i]);
        }
    };

    void Iallreduce(const void* sendbuf, void* recvbuf, int count, int index) override{
        ncclAllReduce(sendbuf, recvbuf, count, NCCL_FLOAT_TYPE, ncclSum,
                      comm, streams[index]);
    }

    void WaitAll(int num_waits) override {
        for(int i = 0; i < num_waits; i++) 
            SYNC_STREAM(streams[i]);
    }

    void Barrier() override {
        WaitAll(num_streams);
    };

    void Wait(int index) override {
        SYNC_STREAM(streams[index]);
    };

    void Allgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount) override {
        ncclAllGather(sendbuf, recvbuf, sendcount, NCCL_FLOAT_TYPE, comm, streams[0]);
        Wait(0);
    };

    void Iallgather(const void* sendbuf, int sendcount, void* recvbuf, int recvcount, int index) override {
        ncclAllGather(sendbuf, recvbuf, sendcount, NCCL_FLOAT_TYPE, comm, streams[index]);
    };

    void Reduce_Scatter_block(const void* sendbuf, void* recvbuf, int recvcount) override {
        ncclReduceScatter(sendbuf, recvbuf, recvcount, NCCL_FLOAT_TYPE, ncclSum, comm, streams[0]);
        Wait(0);
    };

    void finalize() override {
        for(int i = 0; i < num_streams; i++) {
            DESTROY_STREAM(streams[i]);
        }
        delete[] streams;
    };

    std::string get_name() override {
        std::string name;
        #ifdef PROXY_ENABLE_NCCL
            name = "NCCL";
        #elif defined(PROXY_ENABLE_RCCL)
            name = "RCCL";
        #endif
        return name;
    };
private:
    ncclComm_t comm;
    int comm_size;
    int rank;
    int num_streams;
    _Stream* streams = nullptr;
    
};
#endif

#ifdef PROXY_ENABLE_ONECCL
#endif


/**
* @enum Device
* @brief Enum to specify the device type for a tensor.
*/
enum class Device { CPU, GPU };


/**
* @class Tensor
* @brief A lightweight wrapper for a contiguous buffer of data that can reside on CPU or GPU.
*
* This class manages memory allocation and deallocation automatically depending on the device.
* It supports both CPU (host) memory using calloc and GPU (device) memory using cudaMalloc.
*
* @tparam T The data type of the tensor elements (e.g., float, double, half).
*/
template<typename T, Device device = Device::CPU>
class Tensor {
public:
    T* data = nullptr;
    uint64_t size = 0;

    /**
    * @brief Constructs a tensor of given size on a specified device.
    *
    * Allocates memory on the CPU using calloc or on the GPU using cudaMalloc.
    *
    * @param size_ Number of elements in the tensor
    * @param dev Device type (CPU by default)
    */
    explicit Tensor(uint64_t size_) : size(size_) {
        if constexpr (device == Device::CPU) {
            data = static_cast<T*>(calloc(size, sizeof(T)));
            if (!data)
                throw std::runtime_error("Failed to allocate CPU memory");
        }
        else if constexpr (device == Device::GPU) {
    #if defined(PROXY_ENABLE_CUDA)
            CCUTILS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),
                        size * sizeof(T)));
            CCUTILS_CUDA_CHECK(cudaMemset(data, 0, size * sizeof(T)));
    #elif defined(PROXY_ENABLE_HIP)
            CCUTILS_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data),
                        size * sizeof(T)));
            CCUTILS_HIP_CHECK(hipMemset(data, 0, size * sizeof(T)););
    #endif
        }
        else {
            static_assert(device == Device::CPU || device == Device::GPU,
                        "Unsupported device type");
        }
    }

    /**
    * @brief Destructor that frees the allocated memory depending on the device.
    */
    ~Tensor() {
        if (data) {
            if constexpr (device == Device::CPU) {
                free(data);
            }
            else if constexpr (device == Device::GPU) {
    #if defined(PROXY_CUDA)
                CCUTILS_CUDA_FREE_SAFE(data);
    #elif defined(PROXY_HIP)
                CCUTILS_HIP_FREE_SAFE(data);
    #endif

            }
        }
    }
};

#endif // PROXY_CLASSES_HPP