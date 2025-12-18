## DNNProxy Overview

A proxy framework for benchmarking distributed deep neural network training. Users can create transformer-based models, including encoder-only, decoder-only, or encoder-decoder architectures.

## Usage
## Load Modules

### On Leonardo
```bash
# to compile and run the experiments
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
module load python/3.11.7  # Just need it once to create the env
```

## Build 
Inside the cpp folder execute:
```bash
# Compile all targets
make
#N.B: you can specify some of the target like dp, fsdp etc...
```

## Python Env
Create you Python env for Pytorch, Sbatchman and other python scripts.

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Install Pytorch
You need Pytorch to run the experiments: Follow the [installation guide](https://pytorch.org/get-started/locally/). 

### Install Sbatchman

You need Sbatchman to run the experiments. Follow the [development installation guide](https://sbatchman.readthedocs.io/en/latest/development/). 

IMPORTANT: you need to install SbatchMan as a developer! Do NOT follow the standard installation.

## Setup Sbatchman
```bash
# Assuming you’ve set up the SbatchMan aliases:
sbmi                                   # sbmi -> sbatchman init
sbmc -f configs.yaml -ow               # sbmc -> sbatchman configure
```

## Run

```bash
# Assuming you’ve set up the SbatchMan aliases:
sbl -f jobs.yaml                       # sbl -> sbatchman launch
```

## Run on Leonardo
The Python script **run_on_leonardo** ensures that the nodes for the experiments are on different L1 switches but under the same L2 switch (same cell, different switch). Nodes are divided into two equally sized groups each under a different L1 switch.
```bash
python run_on_leonardo.py --csv ../machines/Leonardo/leo_map.txt --jobs jobs.yaml
```

### Service Level (Leonardo)
On Leonardo, the default service level is 0. To use a different service level, set the `NCCL_IB_SL` and `UCX_IB_SL` environment variables for NCCL and MPI, respectively.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by and builds upon the work from:

- **[DNN-cpp-proxies](https://github.com/spcl/DNN-cpp-proxies)** - C++ proxy implementations for distributed deep neural network training
- Special thanks to **[Shigang Li](https://github.com/Shigangli)** for the original implementation and research contributions in this area
