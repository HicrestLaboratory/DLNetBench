MPICXX = mpicxx

DP_PATH = ./data_parallel/
CCUTILS_PATH = $(HOME)/ccutils/include

CXXFLAGS = -O3 -I$(CCUTILS_PATH) $(shell mpicxx --showme:compile) -DCCUTILS_ENABLE_MPI
LDLIBS   += $(shell mpicxx --showme:link)


all: data_parallel

data_parallel: dp fsdp

dp:
	$(MPICXX) $(CXXFLAGS) -o $(DP_PATH)/dp $(DP_PATH)/dp.cpp 

fsdp:
	$(MPICXX) $(CXXFLAGS) -o $(DP_PATH)/fsdp $(DP_PATH)/fsdp.cpp

clean:
	rm $(DP_PATH)/dp $(DP_PATH)/fsdp