# DLNetBench Overview

DLNetBench is a proxy framework for benchmarking distributed deep neural network (DNN) training. It is designed to simulate the performance characteristics of large-scale distributed training workloads without running full end-to-end models.

The framework supports transformer-based architectures, including:

* Encoder-only models
* Decoder-only models

DLNetBench focuses on evaluating communication, parallelization strategies, and backend behavior in distributed environments.

## Supported Features

### Parallelization Strategies

* Data Parallelism (DP)
* Fully Sharded Data Parallelism (FSDP)
* Data Parallelism + Pipeline Parallelism (DP + PP)
* Data Parallelism + Pipeline Parallelism + Tensor Parallelism (DP + PP + TP)
* Data Parallelism + Pipeline Parallelism + Mixture of Experts (DP + PP + MoE)

### Communication Backends

* MPI
* MPI (CUDA-aware)
* NCCL
* RCCL
* oneCCL

### Data Types

* float32
* float16 *(not supported with MPI and MPI CUDA-aware backends)*

# Adding a New System Makefile

## Overview

The build system is split into three layers:

```
Makefile.<SYSTEM>     ← you write this (paths, compiler, configs)
Makefile.flags.mk     ← translates declarations into compiler flags
Makefile.common       ← all build targets, never edited
```

A **config** is a named build variant (e.g. `nccl`, `rccl_profiler`). Each config selects a backend and optional features, and produces its own output directory under `build/<config>/`.

## Step-by-step: Adding a new system

### 1. Create `Makefile.<YOURSYSTEM>`

```makefile
# Makefile.MYSYSTEM

# Compiler to use
CXX := mpicxx

# ── Paths ────────────────────────────────────────────────
# Explicitly set any paths that cannot be auto-detected.
# Paths not set here are auto-detected (see "Path auto-detection" below).
CUDA_HOME := /opt/cuda/12.4
NCCL_HOME := /opt/nccl/2.21

# ── Configs ───────────────────────────────────────────────
CONFIGS += nccl
BACKEND_nccl := nccl

CONFIGS += nccl_nvml
BACKEND_nccl_nvml   := nccl
WITH_NVML_nccl_nvml := 1

include Makefile.flags.mk
include Makefile.common
```

### 2. Run it

```bash
make -f Makefile.MYSYSTEM          # build all configs
make -f Makefile.MYSYSTEM cuda     # build one config
make -f Makefile.MYSYSTEM clean    # wipe build/
```

## Backend reference

The backend name identifies the **communication library**, which determines both the collective backend and the accelerator runtime:

| `BACKEND_<config>` | Preprocessor defines | Collectives | Accelerator |
|---|---|---|---|
| `nccl` | `PROXY_ENABLE_CUDA` + `PROXY_ENABLE_NCCL` | NCCL | CUDA GPU |
| `rccl` | `PROXY_ENABLE_HIP` + `PROXY_ENABLE_RCCL` | RCCL | AMD GPU |
| `oneccl` | `PROXY_ENABLE_ONECCL` | oneCCL | SYCL / Intel GPU |
| `mpi_gpu_cuda` | `PROXY_ENABLE_CUDA` | MPI only | CUDA GPU |
| `mpi_gpu_hip` | `PROXY_ENABLE_HIP` | MPI only | AMD GPU |
| `mpi_cpu` | *(none)* | MPI only | CPU buffers |

`WITH_MPI := 1` additionally emits `-DWITH_MPI -DCCUTILS_ENABLE_MPI` for every config in the system file.

## Optional feature flags

Set per config:

| Variable | Effect |
|---|---|
| `WITH_NVML_<config> := 1` | Adds `-DNVML -lnvidia-ml` |
| `WITH_ENERGY_PROFILER_<config> := 1` | Adds `-DPROXY_ENERGY_PROFILING`, links `libpower_profiler` |

---

## Path auto-detection

If a path variable is not set, the build system attempts auto-detection:

| Variable | Detection method |
|---|---|
| `CUDA_HOME` | `dirname $(dirname $(which nvcc))` |
| `NCCL_HOME` | Checks `$(CUDA_HOME)/include/nccl.h`, then `/usr/include/nccl.h` |
| `ROCM_HOME` | `$ROCM_PATH` → `$EBROOTROCM` → newest `/opt/rocm-*` → `/opt/rocm` |
| `RCCL_HOME` | Defaults to `$(ROCM_HOME)` (RCCL is bundled in ROCm) |
| `ONECCL_HOME` | `$CCL_ROOT` → `$ONECCL_ROOT` (set by Intel's `setvars.sh`) |
| `MPI_HOME` | `dirname $(dirname $(which mpicc))` |
| `CCUTILS_INCLUDE` | `$(HOME)/.local/ccutils/install/include` |
| `ENERGY_PROFILER_HOME` | `$(HOME)/.local` |

Override any of these by setting the variable before the `include` lines.

## Output layout

```
bin/
  nccl/
    cpp/
      data_parallel/
        dp
        fsdp
        dp_loop
        fsdp_loop
      hybrid_parallel/
        hybrid_2d
        hybrid_3d
        hybrid_3d_moe
        hybrid_2d_loop
        hybrid_3d_loop
        hybrid_3d_moe_loop
  rccl/ ...
```

Each config is fully self-contained under its own directory. Configs do not share build artifacts.

# License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

# Acknowledgments

This project is inspired by and builds upon prior work in the area of distributed DNN proxies:

* **[DNN-cpp-proxies](https://github.com/spcl/DNN-cpp-proxies)** – C++ proxy implementations for distributed deep neural network training
* Special thanks to **[Shigang Li](https://github.com/Shigangli)** for the original implementation and foundational research contributions
