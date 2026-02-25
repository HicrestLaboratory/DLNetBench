# -------------------------
# Makefile.flags.mk
# Translates declared paths and backend selection into compiler/linker flags.
#
# Input variables (set by system Makefile):
#   CONFIGS                       — list of config names
#   BACKEND_<config>              := nccl | rccl | oneccl | mpi_gpu_cuda | mpi_gpu_hip | mpi_cpu
#   WITH_NVML_<config>            := 1  (optional)
#   WITH_ENERGY_PROFILER_<config> := 1  (optional)
#   WITH_MPI                      := 1  (system-wide, inherited by all configs)
#   WITH_MPI_<config>             := 1  (per-config override)
#   HIP_ARCH                      := gfx90a (required for rccl / mpi_gpu_hip)
#
# Output (per config):
#   CXXFLAGS_<config>, LDFLAGS_<config>, LDLIBS_<config>
# -------------------------

# ------------------------------------------------------------------
# Path auto-detection (only if not already set)
# ------------------------------------------------------------------

CUDA_HOME ?= $(shell which nvcc 2>/dev/null | xargs -I{} dirname {} | xargs -I{} dirname {})

NCCL_HOME ?= $(shell \
    if [ -f "$(CUDA_HOME)/include/nccl.h" ]; then echo "$(CUDA_HOME)"; \
    elif [ -f "/usr/include/nccl.h" ]; then echo "/usr"; \
    fi)

ROCM_HOME ?= $(or $(ROCM_PATH),$(EBROOTROCM),\
    $(shell ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1),\
    /opt/rocm)

RCCL_HOME ?= $(ROCM_HOME)

ONECCL_HOME ?= $(or $(CCL_ROOT),$(ONECCL_ROOT))

MPI_HOME ?= $(shell which mpicc 2>/dev/null | xargs -I{} dirname {} | xargs -I{} dirname {})
WITH_MPI ?= 1

CCUTILS_INCLUDE ?= $(HOME)/.local/ccutils/install/include

ENERGY_PROFILER_HOME ?= $(HOME)/.local

# ------------------------------------------------------------------
# Flag assembly
#
# Key constraint: inside $(call), ALL variable references expand
# immediately before $(eval) sees the result. This means you cannot
# set a variable and then reference it in the same template —
# $($(1)_MPI) would look up "nccl_MPI" before that variable exists.
#
# Solution: reference the *source* variables directly (WITH_MPI,
# BACKEND_<cfg>, etc.) everywhere inside the template, never the
# intermediate ones. The intermediate vars (_BACK, _MPI, ...) are
# only used as local shorthand via simple := assignment which still
# expands immediately but from already-known values.
# ------------------------------------------------------------------

define _assemble_flags_tpl
CXXFLAGS_$(1) := -O3 -std=c++17
LDFLAGS_$(1)  :=
LDLIBS_$(1)   :=

CXXFLAGS_$(1) += $(if $(CCUTILS_INCLUDE),-I$(CCUTILS_INCLUDE))

CXXFLAGS_$(1) += $(if $(filter 1,$(or $(WITH_MPI_$(1)),$(WITH_MPI))),\
    -DWITH_MPI -DCCUTILS_ENABLE_MPI \
    $(if $(MPI_HOME),-I$(MPI_HOME)/include))
LDFLAGS_$(1)  += $(if $(filter 1,$(or $(WITH_MPI_$(1)),$(WITH_MPI))),\
    $(if $(MPI_HOME),-L$(MPI_HOME)/lib))
LDLIBS_$(1)   += $(if $(filter 1,$(or $(WITH_MPI_$(1)),$(WITH_MPI))),\
    $(if $(MPI_HOME),-lmpi))

CXXFLAGS_$(1) += $(if $(filter nccl,$(BACKEND_$(1))),\
    $(if $(CUDA_HOME),-I$(CUDA_HOME)/include) \
    -DPROXY_ENABLE_CUDA -DCCUTILS_ENABLE_CUDA \
    $(if $(NCCL_HOME),-I$(NCCL_HOME)/include -DPROXY_ENABLE_NCCL))
LDFLAGS_$(1)  += $(if $(filter nccl,$(BACKEND_$(1))),\
    $(if $(CUDA_HOME),-L$(CUDA_HOME)/lib64) \
    $(if $(NCCL_HOME),-L$(NCCL_HOME)/lib))
LDLIBS_$(1)   += $(if $(filter nccl,$(BACKEND_$(1))),\
    $(if $(CUDA_HOME),-lcudart) \
    $(if $(NCCL_HOME),-lnccl))

CXXFLAGS_$(1) += $(if $(filter mpi_gpu_cuda,$(BACKEND_$(1))),\
    $(if $(CUDA_HOME),-I$(CUDA_HOME)/include) \
    -DPROXY_ENABLE_CUDA -DCCUTILS_ENABLE_CUDA)
LDFLAGS_$(1)  += $(if $(filter mpi_gpu_cuda,$(BACKEND_$(1))),\
    $(if $(CUDA_HOME),-L$(CUDA_HOME)/lib64))
LDLIBS_$(1)   += $(if $(filter mpi_gpu_cuda,$(BACKEND_$(1))),\
    $(if $(CUDA_HOME),-lcudart))

CXXFLAGS_$(1) += $(if $(filter rccl,$(BACKEND_$(1))),\
    $(if $(ROCM_HOME),-I$(ROCM_HOME)/include) \
    -DPROXY_ENABLE_HIP -DCCUTILS_ENABLE_HIP -x hip --offload-arch=$(HIP_ARCH) \
    $(if $(RCCL_HOME),-I$(RCCL_HOME)/include -I$(RCCL_HOME)/include/rccl -DPROXY_ENABLE_RCCL))
LDFLAGS_$(1)  += $(if $(filter rccl,$(BACKEND_$(1))),\
    $(if $(ROCM_HOME),-L$(ROCM_HOME)/lib) \
    $(if $(RCCL_HOME),-L$(RCCL_HOME)/lib))
LDLIBS_$(1)   += $(if $(filter rccl,$(BACKEND_$(1))),\
    $(if $(RCCL_HOME),-lrccl))

CXXFLAGS_$(1) += $(if $(filter mpi_gpu_hip,$(BACKEND_$(1))),\
    $(if $(ROCM_HOME),-I$(ROCM_HOME)/include) \
    -DPROXY_ENABLE_HIP -DCCUTILS_ENABLE_HIP -x hip --offload-arch=$(HIP_ARCH))
LDFLAGS_$(1)  += $(if $(filter mpi_gpu_hip,$(BACKEND_$(1))),\
    $(if $(ROCM_HOME),-L$(ROCM_HOME)/lib))

CXXFLAGS_$(1) += $(if $(filter oneccl,$(BACKEND_$(1))),\
    $(if $(ONECCL_HOME),-I$(ONECCL_HOME)/include -DPROXY_ENABLE_ONECCL))
LDFLAGS_$(1)  += $(if $(filter oneccl,$(BACKEND_$(1))),\
    $(if $(ONECCL_HOME),-L$(ONECCL_HOME)/lib))
LDLIBS_$(1)   += $(if $(filter oneccl,$(BACKEND_$(1))),\
    $(if $(ONECCL_HOME),-lccl))

CXXFLAGS_$(1) += $(if $(filter 1,$(WITH_NVML_$(1))),-DNVML)
LDLIBS_$(1)   += $(if $(filter 1,$(WITH_NVML_$(1))),-lnvidia-ml)

CXXFLAGS_$(1) += $(if $(filter 1,$(WITH_ENERGY_PROFILER_$(1))),\
    $(if $(ENERGY_PROFILER_HOME),-I$(ENERGY_PROFILER_HOME)/include -DPROXY_ENERGY_PROFILING))
LDFLAGS_$(1)  += $(if $(filter 1,$(WITH_ENERGY_PROFILER_$(1))),\
    $(if $(ENERGY_PROFILER_HOME),-L$(ENERGY_PROFILER_HOME)/lib))
LDLIBS_$(1)   += $(if $(filter 1,$(WITH_ENERGY_PROFILER_$(1))),\
    $(if $(ENERGY_PROFILER_HOME),-lpower_profiler))

LDLIBS_$(1)   += -lpthread -ldl
endef

# Instantiate for every declared config
$(foreach cfg,$(CONFIGS),$(eval $(call _assemble_flags_tpl,$(cfg))))