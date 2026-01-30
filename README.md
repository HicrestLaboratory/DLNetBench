## DLNetBench Overview

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

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by and builds upon prior work in the area of distributed DNN proxies:

* **[DNN-cpp-proxies](https://github.com/spcl/DNN-cpp-proxies)** – C++ proxy implementations for distributed deep neural network training
* Special thanks to **[Shigang Li](https://github.com/Shigangli)** for the original implementation and foundational research contributions
