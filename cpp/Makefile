WITH_MPI ?= 1 # enable mpi
PROXY_CUDA ?= 0 # enable cuda
PROXY_HIP ?= 0 # enable hip
HALF ?= 0 # enable half precision (requires NCCL or RCCL)
PROXY_NCCL ?= 0 # enable NCCL
PROXY_RCCL ?= 0 # enable RCCL
#TODO: add oneCCL support

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

	ifndef NCCL_HOME
		$(error PROXY_NCCL=1 but NCCL_HOME is not set! Load the NCCL module and set NCCL_HOME.)
	endif
	ENERGY_PROFILER_HOME := $(HOME)/.local/
	CXXFLAGS += -I$(NCCL_HOME)/include -I$(ENERGY_PROFILER_HOME)/include
	LDFLAGS  += -L$(NCCL_HOME)/lib -lnccl
	LDFLAGS += -L$(ENERGY_PROFILER_HOME)/lib -lpower_profiler
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
		$(warning CUDA_HOME auto-detected as $(CUDA_HOME). To change it set manually CUDA_HOME environment variable)
	endif
	CXXFLAGS += -DPROXY_ENABLE_CUDA -DCCUTILS_ENABLE_CUDA -I$(CUDA_HOME)/include
	LDFLAGS += -L$(CUDA_HOME)/lib64 -lcudart
endif

ifeq ($(PROXY_HIP),1) #TODO: add HIP_HOME detection
	CXXFLAGS += -DPROXY_ENABLE_HIP
	CXXFLAGS += -DCCUTILS_ENABLE_HIP
endif

# HALF_PRECISION iff CUDA or HIP is enabled
ifeq ($(HALF),1)
	ifneq ($(PROXY_NCCL),1)
		ifneq ($(PROXY_RCCL),1)
			$(error HALF_PRECISION requires NCCL/RCCL to be enabled)
		endif
	endif
	CXXFLAGS += -DPROXY_HALF
endif


CXXFLAGS += -I$(CCUTILS_INCLUDE)

ifeq ($(WITH_MPI),1)
MPI_CFLAGS := $(shell mpicxx --showme:compile)
MPI_LIBS   := $(shell mpicxx --showme:link)
	
ifeq ($(PROXY_CUDA),1)
export OMPI_MCA_opal_cuda_support=1
# Filter out -pthread for nvcc
CXXFLAGS += $(filter-out -pthread, $(MPI_CFLAGS)) -DWITH_MPI -DCCUTILS_ENABLE_MPI
MPI_LIBS := $(filter-out -pthread,$(MPI_LIBS))
else
	CXXFLAGS  += $(MPI_CFLAGS) -DWITH_MPI -DCCUTILS_ENABLE_MPI
endif

endif

DP_PATH = ./data_parallel
HYBRID_PATH = ./hybrid_parallel

all: data_parallel

data_parallel: dp fsdp

dp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o $(DP_PATH)/dp $(DP_PATH)/dp.cpp $(LDFLAGS)

fsdp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o $(DP_PATH)/fsdp $(DP_PATH)/fsdp.cpp $(LDFLAGS)

hybrid_2d:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o $(HYBRID_PATH)/hybrid_2d $(HYBRID_PATH)/hybrid_2d.cpp $(LDFLAGS)

clean:
	rm -f $(DP_PATH)/dp $(DP_PATH)/fsdp
