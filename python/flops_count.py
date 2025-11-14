"""
*********************************************************************
*
* Description: Python script for computing FLOPs of a Transformer
*              model using a JSON configuration for encoder and
*              decoder blocks.
*
*********************************************************************
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

def count_flops(block, seq_len, batch_size, embed_dim, num_blocks, memory_seq_len=None):
    input_tensor = torch.randn(1, seq_len, embed_dim).cpu()

    # For decoder blocks, create a memory tensor
    memory_tensor = torch.randn(1, memory_seq_len, embed_dim).cpu() if memory_seq_len else None
        

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=False
    ) as prof:
        _ = block(input_tensor, memory_tensor) # if only encoder memory_tensor will be ignored

    key_avg = prof.key_averages()
    block_flops = sum(e.flops for e in key_avg if e.flops is not None)
    
    return block_flops * num_blocks * batch_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FLOPs for Transformer using JSON config")
    parser.add_argument("config_file", type=str, help="Path to JSON model configuration file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
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

    print(f"Total GFLOPs: {total_flops / 1e9:.2f} GFLOPs")
