WITH_MPI ?= 1 # enable mpi
PROXY_CUDA ?= 0 # enable cuda
PROXY_HIP ?= 0 # enable hip
HALF ?= 0 # enable half precision (requires PROXY_CUDA or PROXY_HIP)
PROXY_NCCL ?= 0 # enable NCCL (requires PROXY_CUDA)

CXX := mpicxx
CXXFLAGS := -O3 -std=c++17
LDFLAGS :=

ifndef CCUTILS_INCLUDE
$(warning CCUTILS_INCLUDE is not set!)
$(warning Please set CCUTILS_INCLUDE to the include directory of ccutils)
$(error Example: install by running the provided install.sh)
endif

ifeq ($(PROXY_NCCL),1)
    PROXY_CUDA := 1
    CXXFLAGS += -DPROXY_ENABLE_NCCL
endif

# Check that CUDA and HIP are not both enabled
ifeq ($(PROXY_CUDA),1)
    ifeq ($(PROXY_HIP),1)
        $(error Cannot enable both PROXY_CUDA and PROXY_HIP at the same time)
    endif
    ifndef CUDA_HOME
        NVCC_PATH := $(shell which nvcc 2>/dev/null)
        ifeq ($(NVCC_PATH),)
            $(error nvcc not found in PATH and CUDA_HOME not set!)
        endif
        CUDA_HOME := $(shell dirname $(shell dirname $(NVCC_PATH)))
        $(warning CUDA_HOME auto-detected as $(CUDA_HOME))
    endif
    CXXFLAGS += -DPROXY_ENABLE_CUDA -DCCUTILS_ENABLE_CUDA -I$(CUDA_HOME)/include
    LDFLAGS += -L$(CUDA_HOME)/lib64 -lcudart
endif

ifeq ($(PROXY_HIP),1)
    CXXFLAGS += -DPROXY_ENABLE_HIP
    CXXFLAGS += -DCCUTILS_ENABLE_HIP
endif

# HALF_PRECISION iff CUDA or HIP is enabled
ifeq ($(HALF),1)
    ifneq ($(PROXY_CUDA),1)
        ifneq ($(PROXY_HIP),1)
            $(error HALF_PRECISION requires either PROXY_CUDA or PROXY_HIP to be enabled)
        endif
    endif
    CXXFLAGS += -DHALF_PRECISION
endif


CXXFLAGS += -I$(CCUTILS_INCLUDE)

ifeq ($(WITH_MPI),1)
export OMPI_MCA_opal_cuda_support=1
MPI_CFLAGS := $(shell mpicxx --showme:compile)
MPI_LIBS   := $(shell mpicxx --showme:link)
    
ifeq ($(PROXY_CUDA),1)
# Filter out -pthread for nvcc
CXXFLAGS += $(filter-out -pthread, $(MPI_CFLAGS)) -DWITH_MPI -DCCUTILS_ENABLE_MPI
MPI_LIBS := $(filter-out -pthread,$(MPI_LIBS))
else
    CXXFLAGS  += $(MPI_CFLAGS) -DWITH_MPI -DCCUTILS_ENABLE_MPI
endif

endif

DP_PATH = ./data_parallel

all: data_parallel

data_parallel: dp fsdp

dp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o $(DP_PATH)/dp $(DP_PATH)/dp.cpp $(LDFLAGS)

fsdp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o $(DP_PATH)/fsdp $(DP_PATH)/fsdp.cpp

clean:
	rm -f $(DP_PATH)/dp $(DP_PATH)/fsdp
