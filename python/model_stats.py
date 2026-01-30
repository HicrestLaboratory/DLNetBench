"""
*********************************************************************
*
* Description: Python script for computing FLOPs and model size of a Transformer
*              model using a JSON configuration for encoder-only or
*              decoder-only models.
*
*********************************************************************
"""

import argparse
import json
import time
import os
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import platform

# Detect device name
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
else:
    device_name = platform.processor()

# ------------------------------
# FLOPs counting
# ------------------------------
def count_flops(block, seq_len:int, batch_size:int, embed_dim:int, num_blocks:int, src_mask=None):
    input_tensor = torch.randn(1, seq_len, embed_dim).cpu()
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=False
    ) as prof:
        if src_mask is not None:
            _ = block(input_tensor, src_mask=src_mask)
        else:
            _ = block(input_tensor)
    key_avg = prof.key_averages()
    block_flops = sum(e.flops for e in key_avg if e.flops is not None)
    return block_flops * num_blocks * batch_size

# ------------------------------
# Parameter counting
# ------------------------------
def count_parameters(block, num_blocks:int):
    block_size = sum(p.numel() for p in block.parameters() if p.requires_grad)
    return block_size * num_blocks

# ------------------------------
# FFN timing
# ------------------------------
def fwd_bwd_time_ffn(block, seq_len: int, batch_size: int, embed_dim: int, device='cpu'):
    linear1 = block.linear1.to(device)
    linear2 = block.linear2.to(device)
    activation = block.activation
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    for _ in range(10):
        h = activation(linear1(x))
        y = linear2(h)
        y.sum().backward()
        linear1.zero_grad()
        linear2.zero_grad()
        if device=="cuda": torch.cuda.synchronize()

    fwd_times, bwd_times = [], []
    for _ in range(50):
        if device=="cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        h = activation(linear1(x))
        y = linear2(h)
        if device=="cuda": torch.cuda.synchronize()
        fwd_times.append((time.perf_counter()-t0)*1e6)

        loss = y.sum()
        if device=="cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        if device=="cuda": torch.cuda.synchronize()
        bwd_times.append((time.perf_counter()-t1)*1e6)
        linear1.zero_grad()
        linear2.zero_grad()

    median_fwd = torch.tensor(fwd_times).median().item()
    median_bwd = torch.tensor(bwd_times).median().item()
    return median_fwd, median_bwd

# ------------------------------
# Full block timing
# ------------------------------
def fwd_bwd_time(block, seq_len:int, batch_size:int, embed_dim:int,
                 num_blocks:int=1, device:str='cpu', src_mask=None):
    input_tensor = torch.randn(batch_size, seq_len, embed_dim, device=device)

    for _ in range(10):
        out = block(input_tensor, src_mask=src_mask) if src_mask is not None else block(input_tensor)
        out.sum().backward()
        block.zero_grad()
        if device=="cuda": torch.cuda.synchronize()

    fwd_times, bwd_times = [], []
    for _ in range(50):
        if device=="cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = block(input_tensor, src_mask=src_mask) if src_mask is not None else block(input_tensor)
        if device=="cuda": torch.cuda.synchronize()
        fwd_times.append((time.perf_counter()-t0)*1e6)

        loss = out.sum()
        if device=="cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        if device=="cuda": torch.cuda.synchronize()
        bwd_times.append((time.perf_counter()-t1)*1e6)
        block.zero_grad()

    median_fwd = torch.tensor(fwd_times).median().item() * num_blocks
    median_bwd = torch.tensor(bwd_times).median().item() * num_blocks
    return median_fwd, median_bwd

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FLOPs for Transformer using JSON config")
    parser.add_argument("config_file", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--scale_bwd_flops", type=float, default=2.0)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16","float32"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    with open(args.config_file,"r") as f:
        config = json.load(f)

    embed_dim = config.get("embed_dim",768)
    num_heads = config.get("num_heads",12)
    ff_dim = config.get("ff_dim",3072)
    seq_len = config.get("seq_len",197)
    num_encoder_blocks = config.get("num_encoder_blocks",0)
    num_decoder_blocks = config.get("num_decoder_blocks",0)
    model_type = config.get("model_type","encoder_only")

    device = args.device
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() if model_type=="decoder_only" else None

    total_flops = 0
    total_params = 0
    total_fwd_time = 0
    total_bwd_time = 0
    total_ffn_fwd_time = 0
    total_ffn_bwd_time = 0

    # Encoder-only
    if num_encoder_blocks > 0:
        encoder_block = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True).to(device)
        total_flops += count_flops(encoder_block, seq_len, args.batch_size, embed_dim, num_encoder_blocks)
        total_params += count_parameters(encoder_block, num_encoder_blocks)
        fwd_time, bwd_time = fwd_bwd_time(encoder_block, seq_len, args.batch_size, embed_dim, num_encoder_blocks, device=device)
        total_fwd_time += fwd_time
        total_bwd_time += bwd_time
        if args.experts>1:
            ffn_fwd_time, ffn_bwd_time = fwd_bwd_time_ffn(encoder_block, seq_len, args.batch_size, embed_dim, device=device)
            total_ffn_fwd_time += ffn_fwd_time
            total_ffn_bwd_time += ffn_bwd_time

    # Decoder-only (causal)
    if num_decoder_blocks > 0:
        decoder_block = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True).to(device)
        total_flops += count_flops(decoder_block, seq_len, args.batch_size, embed_dim, num_decoder_blocks, src_mask=causal_mask)
        total_params += count_parameters(decoder_block, num_decoder_blocks)
        fwd_time, bwd_time = fwd_bwd_time(decoder_block, seq_len, args.batch_size, embed_dim, num_decoder_blocks, device=device, src_mask=causal_mask)
        total_fwd_time += fwd_time
        total_bwd_time += bwd_time
        if args.experts>1:
            ffn_fwd_time, ffn_bwd_time = fwd_bwd_time_ffn(decoder_block, seq_len, args.batch_size, embed_dim, device=device)
            total_ffn_fwd_time += ffn_fwd_time
            total_ffn_bwd_time += ffn_bwd_time

    # Output
    base_name = os.path.splitext(os.path.basename(args.config_file))[0]
    out_dir = os.path.expanduser("~/DNNProxy/model_stats")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_{args.batch_size}.txt")

    with open(out_path,"w+") as out_file:
        out_file.write(f"Forward_Flops:{total_flops}\n")
        out_file.write(f"Backward_Flops:{total_flops * args.scale_bwd_flops}\n")
        out_file.write(f"Model_Size:{total_params}\n")
        out_file.write(f"Average_Forward_Time (us):{int(total_fwd_time)}\n")
        out_file.write(f"Average_Backward_Time (us):{total_bwd_time}\n")
        out_file.write(f"Batch_size:{args.batch_size}\n")
        out_file.write(f"FFN_Average_Forward_Time (us):{int(total_ffn_fwd_time)}\n")
        out_file.write(f"FFN_Average_Backward_Time (us):{int(total_ffn_bwd_time)}\n")
        out_file.write(f"Experts:{args.experts}\n")
        out_file.write(f"Device:{device_name}\n")
        out_file.write(f"Sample_Size:{seq_len}\n")
        out_file.write(f"Embedded_dim:{embed_dim}\n")
