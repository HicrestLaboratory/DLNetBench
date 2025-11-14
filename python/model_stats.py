"""
*********************************************************************
*
* Description: Python script for computing FLOPs and model size of a Transformer
*              model using a JSON configuration for encoder and
*              decoder blocks.
* Authors: Jacopo Raffi
*
*********************************************************************
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import os

def count_flops(block, seq_len:int, batch_size:int, embed_dim:int, num_blocks:int, memory_seq_len:int=None):
    '''
    Count the FLOPs for a given Transformer block.

    Args:
        block: The Transformer block (encoder or decoder) to profile.
        seq_len: Sequence length of the input.
        batch_size: Batch size for the input.
        embed_dim: Embedding dimension of the model.
        num_blocks: Number of blocks to multiply the FLOPs.
        memory_seq_len: Sequence length of the memory (for decoder blocks).
    
    Returns:
        Total FLOPs for the specified number of blocks.
    '''
    input_tensor = torch.randn(1, seq_len, embed_dim).cpu()

    # For decoder blocks, create a memory tensor
    memory_tensor = torch.randn(1, memory_seq_len, embed_dim).cpu() if memory_seq_len else None
        

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=False
    ) as prof:
        _ = block(input_tensor, memory_tensor) # if encoder-only memory_tensor will be ignored

    key_avg = prof.key_averages()
    block_flops = sum(e.flops for e in key_avg if e.flops is not None)
    
    return block_flops * num_blocks * batch_size

def count_parameters(block, num_blocks:int, bytes:int=4):
    '''
    Count the number of parameters in the model and return the size in bytes.

    Args:
        model: The PyTorch model to count parameters for.
        num_blocks: Number of blocks to multiply the parameter count.
        bytes: Number of bytes per parameter (default is 4 for float32).

    Returns:
        Total size of the model parameters in bytes.
    '''
    block_size = sum(p.numel() for p in block.parameters() if p.requires_grad) * bytes

    return block_size * num_blocks

def fwd_bwd_time(block, seq_len:int, batch_size:int, embed_dim:int, num_blocks:int, memory_seq_len:int=0, device:str='cpu'):
    """
    Compute the average forward and backward time of a single block on CPU or GPU.
    
    Args:
        block: The Transformer block (encoder or decoder) to profile.
        seq_len: Sequence length of the input.
        batch_size: Batch size for the input.
        embed_dim: Embedding dimension of the model.
        num_blocks: Number of blocks to multiply the time.
        memory_seq_len: Memory sequence length (for decoder blocks).
        device: Device to run on ('cpu' or 'cuda').
    
    Returns:
        Tuple: (average forward time, average backward time) in seconds for the model.
    """

    input_tensor = torch.randn(batch_size, seq_len, embed_dim, device=device)
    memory_tensor = torch.randn(batch_size, memory_seq_len, embed_dim, device=device) if memory_seq_len else None

    # Warm-up
    for _ in range(10):
        output = block(input_tensor, memory_tensor) if memory_tensor is not None else block(input_tensor)
        loss = output.sum()
        loss.backward()
        block.zero_grad()

    # Timing
    n_iters = 50
    total_fwd_time = 0.0
    total_bwd_time = 0.0

    for _ in range(n_iters):
        # Forward pass
        start_time = time.time()
        output = block(input_tensor, memory_tensor) if memory_tensor is not None else block(input_tensor)
        loss = output.sum()
        fwd_time = time.time() - start_time

        # Backward pass
        start_time = time.time()
        loss.backward()
        bwd_time = time.time() - start_time

        total_fwd_time += fwd_time
        total_bwd_time += bwd_time

        block.zero_grad()

    avg_fwd_time = (total_fwd_time / n_iters) * num_blocks
    avg_bwd_time = (total_bwd_time / n_iters) * num_blocks

    return avg_fwd_time, avg_bwd_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FLOPs for Transformer using JSON config")
    parser.add_argument("config_file", type=str, help="Path to JSON model configuration file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--scale_bwd_flops", type=float, default=2.0, help="Scale factor for backward FLOPs")
    parser.add_argument("--dtype", type=str, default="float32", help="DType for model parameters [float16, float32, int8]", choices=["float16", "float32", "int8"])
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the timing tests on ['cpu', 'cuda']")
    args = parser.parse_args()

    dtype_map = {
        "float16": 2, # bytes
        "float32": 4,
        "int8": 1
    }

    num_bytes = dtype_map[args.dtype]

    # Load JSON configuration
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # the default config is a vit-b-16
    embed_dim = config.get("embed_dim", 768)
    num_heads = config.get("num_heads", 12)
    ff_dim = config.get("ff_dim", 3072)
    seq_len = config.get("seq_len", 197)
    memory_seq_len = config.get("memory_seq_len", 0)
    num_encoder_blocks = config.get("num_encoder_blocks", 12)
    num_decoder_blocks = config.get("num_decoder_blocks", 0)

    total_flops = 0

    # Encoder FLOPs
    if num_encoder_blocks > 0:
        encoder_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).cpu()
        total_flops += count_flops(encoder_block, seq_len, args.batch_size, embed_dim, num_encoder_blocks)

    # Decoder FLOPs
    if num_decoder_blocks > 0:
        decoder_block = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).cpu()
        total_flops += count_flops(decoder_block, seq_len, args.batch_size, embed_dim, num_decoder_blocks, memory_seq_len)

    print(f"Forward Flops:{total_flops}")
    print(f"Backward Flops:{total_flops * args.scale_bwd_flops}")
    # Parameter count
    total_params = 0
    if num_encoder_blocks > 0:
        encoder_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).cpu()
        total_params += count_parameters(encoder_block, num_encoder_blocks, bytes=num_bytes)

    if num_decoder_blocks > 0:
        decoder_block = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).cpu()
        total_params += count_parameters(decoder_block, num_decoder_blocks, bytes=num_bytes)

    # Forward and Backward time measurement
    device = args.device
    total_fwd_time = 0.0
    total_bwd_time = 0.0

    if num_encoder_blocks > 0:
        encoder_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).to(device)
        fwd_time, bwd_time = fwd_bwd_time(encoder_block, seq_len, args.batch_size, embed_dim, num_encoder_blocks, device=device)
        total_fwd_time += fwd_time
        total_bwd_time += bwd_time
    
    if num_decoder_blocks > 0:
        decoder_block = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).to(device)
        fwd_time, bwd_time = fwd_bwd_time(decoder_block, seq_len, args.batch_size, embed_dim, num_decoder_blocks, memory_seq_len, device=device)
        total_fwd_time += fwd_time
        total_bwd_time += bwd_time

    base_name = os.path.splitext(os.path.basename(args.config_file))[0]
    out_dir = os.path.join("..", "model_stats")
    out_path = os.path.join(out_dir, base_name + ".txt")

    with open(out_path, "w+") as out_file:
        out_file.write(f"Forward Flops:{total_flops}\n")
        out_file.write(f"Backward Flops:{total_flops * args.scale_bwd_flops}\n")
        out_file.write(f"Model Size (Bytes):{total_params}\n")
        out_file.write(f"Average Forward Time (s):{total_fwd_time}\n")
        out_file.write(f"Average Backward Time (s):{total_bwd_time}\n")
