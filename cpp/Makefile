WITH_MPI ?= 1

CXX := mpicxx
CXXFLAGS := -O3 -Wall -std=c++17

ifndef CCUTILS_INCLUDE
$(warning CCUTILS_INCLUDE is not set!)
$(warning Please set CCUTILS_INCLUDE to the include directory of ccutils)
$(error Example: install by running the provided install.sh)
endif

CXXFLAGS += -I$(CCUTILS_INCLUDE)

ifeq ($(WITH_MPI),1)
    MPI_CFLAGS := $(shell mpicxx --showme:compile)
    MPI_LIBS   := $(shell mpicxx --showme:link)
    CXXFLAGS  += $(MPI_CFLAGS) -DWITH_MPI -DCCUTILS_ENABLE_MPI
endif

DP_PATH = ./data_parallel/

all: data_parallel

data_parallel: dp fsdp

dp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o $(DP_PATH)/dp $(DP_PATH)/dp.cpp

fsdp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o $(DP_PATH)/fsdp $(DP_PATH)/fsdp.cpp

clean:
	rm -f $(DP_PATH)/dp $(DP_PATH)/fsdp
