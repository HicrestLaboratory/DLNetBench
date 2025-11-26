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
from torch.profiler import profile, record_function, ProfilerActivity
import os

def count_flops(block, seq_len:int, batch_size:int, embed_dim:int, num_blocks:int, memory_seq_len:int=0):
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

def count_parameters(block, num_blocks:int):
    '''
    Count the number of parameters in the model.

    Args:
        model: The PyTorch model to count parameters for.
        num_blocks: Number of blocks to multiply the parameter count.

    Returns:
        Total number of parameters of the model.
    '''
    block_size = sum(p.numel() for p in block.parameters() if p.requires_grad)

    return block_size * num_blocks


def fwd_bwd_time_ffn(block, seq_len: int, batch_size: int, embed_dim: int, device='cpu'):
    """
    Compute forward/backward time ONLY for the FFN part of a Transformer block.
    Returns (avg_fwd_time, avg_bwd_time) in seconds.
    """
    # FFN: linear1 → activation → linear2
    linear1 = block.linear1.to(device)
    linear2 = block.linear2.to(device)
    activation = block.activation

    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Warm-up
    for _ in range(10):
        h = activation(linear1(x))
        y = linear2(h)
        loss = y.sum()
        loss.backward()
        linear1.zero_grad()
        linear2.zero_grad()

    n_iters = 50
    total_fwd = 0.0
    total_bwd = 0.0

    for _ in range(n_iters):
        # Forward
        start = time.perf_counter()
        h = activation(linear1(x))
        y = linear2(h)
        if device == 'cuda':
            torch.cuda.synchronize()
        fwd = time.perf_counter() - start

        # Backward
        start = time.perf_counter()
        loss = y.sum()
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        bwd = time.perf_counter() - start

        total_fwd += fwd
        total_bwd += bwd

        linear1.zero_grad()
        linear2.zero_grad()

    return total_fwd / n_iters, total_bwd / n_iters

def fwd_bwd_time(block, seq_len:int, batch_size:int, embed_dim:int,
                 num_blocks:int, memory_seq_len:int=0, device:str='cpu'):
    """
    Measure forward/backward time for a full Transformer block.
    Returns (avg_fwd_time, avg_bwd_time) in seconds, scaled by num_blocks.
    """

    input_tensor = torch.randn(batch_size, seq_len, embed_dim, device=device)
    memory_tensor = (
        torch.randn(batch_size, memory_seq_len, embed_dim, device=device)
        if memory_seq_len else None
    )

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Warm-up
    for _ in range(10):
        out = block(input_tensor, memory_tensor) if memory_tensor is not None else block(input_tensor)
        loss = out.sum()
        loss.backward()
        block.zero_grad()

    n_iters = 1
    total_fwd = 0.0
    total_bwd = 0.0

    for _ in range(n_iters):
        with profile(activities=activities) as prof:

            with record_function("FWD"):
                out = block(input_tensor, memory_tensor) if memory_tensor is not None else block(input_tensor)
                loss = out.sum()

            with record_function("BWD"):
                loss.backward()

        events = prof.key_averages()

        # get the events
        fwd_evt = next(e for e in events if e.key == "FWD")
        bwd_evt = next(e for e in events if e.key == "BWD")

        # auto-select CPU or CUDA times
        if hasattr(fwd_evt, "self_cuda_time_total"):
            total_fwd += fwd_evt.self_cuda_time_total / 1e6
            total_bwd += bwd_evt.self_cuda_time_total / 1e6
        else:
            total_fwd += fwd_evt.self_cpu_time_total / 1e6
            total_bwd += bwd_evt.self_cpu_time_total / 1e6

        block.zero_grad()

    avg_fwd = (total_fwd / n_iters) * num_blocks
    avg_bwd = (total_bwd / n_iters) * num_blocks

    return avg_fwd, avg_bwd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FLOPs for Transformer using JSON config")
    parser.add_argument("config_file", type=str, help="Path to JSON model configuration file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--scale_bwd_flops", type=float, default=2.0, help="Scale factor for backward FLOPs")
    parser.add_argument("--experts", type=int, default=1, help="Number of experts for MoE models")
    parser.add_argument("--dtype", type=str, default="float32", help="DType for model parameters [float16, float32]", choices=["float16", "float32"])
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the timing tests on ['cpu', 'cuda']")
    args = parser.parse_args()

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

    # print(f"Forward Flops:{total_flops}")
    # print(f"Backward Flops:{total_flops * args.scale_bwd_flops}")
    # Parameter count
    total_params = 0
    if num_encoder_blocks > 0:
        encoder_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).cpu()
        total_params += count_parameters(encoder_block, num_encoder_blocks)

    if num_decoder_blocks > 0:
        decoder_block = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        ).cpu()
        total_params += count_parameters(decoder_block, num_decoder_blocks)

    # Forward and Backward time measurement
    device = args.device
    total_fwd_time = 0.0
    total_bwd_time = 0.0

    total_ffn_fwd_time = 0.0
    total_ffn_bwd_time = 0.0

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
        ffn_fwd_time, ffn_bwd_time = 0, 0
        if args.experts > 1:
            ffn_fwd_time, ffn_bwd_time = fwd_bwd_time_ffn(encoder_block, seq_len, args.batch_size, embed_dim, device=device)
        total_ffn_fwd_time += ffn_fwd_time
        total_ffn_bwd_time += ffn_bwd_time
    
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
        ffn_fwd_time, ffn_bwd_time = 0, 0
        if args.experts > 1:
            ffn_fwd_time, ffn_bwd_time = fwd_bwd_time_ffn(decoder_block, seq_len, args.batch_size, embed_dim, device=device)
        total_ffn_fwd_time += ffn_fwd_time
        total_ffn_bwd_time += ffn_bwd_time

    base_name = os.path.splitext(os.path.basename(args.config_file))[0]
    out_dir = os.path.join("..", "model_stats")
    out_path = os.path.join(out_dir, base_name + f"_{args.batch_size}.txt")

    with open(out_path, "w+") as out_file:
        out_file.write(f"Forward_Flops:{total_flops}\n")
        out_file.write(f"Backward_Flops:{total_flops * args.scale_bwd_flops}\n")
        out_file.write(f"Model_Size:{total_params}\n")
        out_file.write(f"Average_Forward_Time (us):{int(total_fwd_time * 1e6)}\n")
        out_file.write(f"Average_Backward_Time (us):{total_bwd_time * 1e6}\n")
        out_file.write(f"Batch_size:{args.batch_size}\n")
        out_file.write(f"FFN_Average_Forward_Time (us):{int(total_ffn_fwd_time * 1e6)}\n")
        out_file.write(f"FFN_Average_Backward_Time (us):{int(total_ffn_bwd_time * 1e6)}\n")
        out_file.write(f"Experts:{args.experts}\n")
