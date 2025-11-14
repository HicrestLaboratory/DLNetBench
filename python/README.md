# Transformer FLOPs and Model Size Profiler

This Python script computes the **FLOPs**, **model size**, and **forward/backward pass timings** for a Transformer model using a JSON configuration. It supports **encoder-only models** (like ViT or BERT), **decoder-only models** (like GPT3) **encoder-decoder models** (like standard Transformer seq2seq).

## Features

* Compute **forward and backward FLOPs** using `torch.profiler`.
* Calculate **model size** in bytes based on parameter data type.
* Measure **average forward and backward time** on CPU or GPU.
* Supports **configurable batch size** and **data type precision** (`float16`, `float32`, `int8`).
* Accepts **JSON configuration files** for flexibility.

## Requirements

* Python 3.9+
* PyTorch 2.x

For PyTorch install follow the official [guide](https://pytorch.org/get-started/locally/).

## JSON Configuration

Example JSON files are located in `../models`. The configuration supports the following fields:

| Field                | Type | Default | Description                      |
| -------------------- | ---- | ------- | -------------------------------- |
| `embed_dim`          | int  | 768     | Embedding dimension              |
| `num_heads`          | int  | 12      | Number of attention heads        |
| `ff_dim`             | int  | 3072    | Feedforward hidden dimension     |
| `seq_len`            | int  | 197     | Input sequence length            |
| `memory_seq_len`     | int  | 0       | Memory sequence length (decoder) |
| `num_encoder_blocks` | int  | 12      | Number of encoder layers         |
| `num_decoder_blocks` | int  | 0       | Number of decoder layers         |

## Usage

Run the script with a JSON configuration file:

```bash
python model_stats.py ../models/vit-b-16.json --batch_size 2 --dtype float32
```

### Optional Arguments

* `--batch_size`: Batch size (default: 1)
* `--scale_bwd_flops`: Scale factor for backward FLOPs (default: 2.0)
* `--dtype`: Parameter precision (`float16`, `float32`, `int8`; default: `float32`)
- `--device`: Device to run the model on (`cpu` or `cuda`; default: `cpu`)

## Output

The script writes results to `../model_stats/<config_name>.txt`. Example output:

```
Forward Flops:1234567890
Backward Flops:2469135780
Model Size (Bytes):12345678
Average Forward Time (s):0.045
Average Backward Time (s):0.092
```

* **Forward/Backward Flops**: Total operations for the model.
* **Model Size**: Memory footprint in bytes.
* **Forward/Backward Time**: The average time per batch is first computed for a single block, and then multiplied by the number of blocks to obtain the total time for the entire model.

For example, the total forward time can be written as:

$$
\text{Total Forward Time} =
(\text{num\_encoder\_blocks} \cdot \text{avg\_encoder\_fwd\_time}) \\
+ (\text{num\_decoder\_blocks} \cdot \text{avg\_decoder\_fwd\_time})
$$

