MPICXX = mpicxx
# NVCC   = nvcc  For future NCCL and CUDA support

DP_PATH = ./data_parallel/
CCUTILS = $(HOME)/ccutils/install/include

$(CCUTILS):
        cd $(HOME)/ccutils && \
        cmake -S . -B build -DCCUTILS_ENABLE_MPI=ON && \
        cmake --build build && \
        cmake --install build --prefix ./install

CXXFLAGS = -O3 -I$(CCUTILS) $(shell mpicxx --showme:compile) -DCCUTILS_ENABLE_MPI
LDLIBS   += $(shell mpicxx --showme:link)


all: data_parallel

data_parallel: dp fsdp

dp: $(CCUTILS)
        $(MPICXX) $(CXXFLAGS) -o $(DP_PATH)/$@ $(DP_PATH)/$@.cpp

fsdp:
        $(MPICXX) $(CXXFLAGS) -o $(DP_PATH)/$@ $(DP_PATH)/$@.cpp

clean:
        rm -f $(DP_PATH)/dp $(DP_PATH)/fsdp