# Transformer Roofline Simulator (NVIDIA A100)

This repository contains a theoretical performance simulator for Transformer-based models (Vision Transformers, LLMs, and MoE architectures). It utilizes the **Roofline Model** to estimate execution time based on hardware constraints and architectural complexity.

## 🚀 Overview

Predicting the performance of Large Language Models (LLMs) is complex due to the interplay between compute-heavy operations (Matrix Multiplications) and memory-heavy operations (Weight/Activation loading). 

This tool identifies whether a model is **Compute-Bound** or **Memory-Bound** on an **NVIDIA A100-SXM4-80GB**, providing theoretical bounds for:
* Total FLOPs (Forward & Backward)
* Memory Traffic (Bytes)
* Execution Latency (Microseconds)

## 📊 The Mathematical Model

The simulator calculates the "Speed Limit" of the GPU for a specific model configuration. The achievable performance is defined by the **Minimum** of the hardware's peak compute and its memory-bandwidth-limited throughput:

$$Achievable Performance = \min(\text{Peak FLOPS}, \text{Arithmetic Intensity} \times \text{Bandwidth})$$



### Time Calculation
Execution time for any component (Attention or MLP) is derived as:
$$Time = \frac{\text{Total FLOPs}}{\text{Achievable Performance}}$$

## 🛠️ Key Components Modeled

### 1. Attention Mechanism
Calculates operations for QKV projections, attention scoring ($QK^T$), and output projection.
* **FLOPs:** $L \times (8BNd^2 + 4BN^2d)$
* **Memory:** Captures the movement of weights and KV cache/activations.

### 2. Feed-Forward Network (MLP)
Includes support for standard MLPs and **Mixture-of-Experts (MoE)**.
* **FLOPs:** $L \times (4BNdH \times k)$
* **Experts ($E$):** Total parameters scale with $E$.
* **Top-k ($k$):** Computation only scales with active experts.

### 3. Precision & Dtype
The model uses BFLOAT16 as default:
| Precision | Peak Throughput (A100) |
| :--- | :--- |
| **BF16 / FP16** | 312 TFLOPS |
