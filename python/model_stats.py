#!/usr/bin/env python3
"""
*********************************************************************
*
* Python script to compute model size and forward/backward timing
* of Hugging Face Transformer models using a single-layer proxy.
* Backward is approximated as 2x forward.
*
*********************************************************************
"""

import argparse
import os
import platform
import time
import torch
from transformers import (
    AutoModelForCausalLM, AutoConfig,
    ViTModel, ViTConfig,
    GPT2LMHeadModel, GPT2Config
)

# Optional Mixtral
try:
    from transformers import MixtralForCausalLM, MixtralConfig
except ImportError:
    MixtralForCausalLM = None
    MixtralConfig = None

# Detect device
DEVICE_NAME = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else platform.processor()

# ------------------------------
# Fixed Model Configs
# ------------------------------
MODEL_CONFIGS = {
    "vit-b":  {"hf_name":"google/vit-base-patch16-224",      "model_class":ViTModel, "config_class":ViTConfig, "type":"encoder_only", "seq_len":197, "description":"ViT-Base: 86M params, 12 layers"},
    "vit-l":  {"hf_name":"google/vit-large-patch16-224",     "model_class":ViTModel, "config_class":ViTConfig, "type":"encoder_only", "seq_len":197, "description":"ViT-Large: 304M params, 24 layers"},
    "vit-h":  {"hf_name":"google/vit-huge-patch14-224-in21k","model_class":ViTModel, "config_class":ViTConfig, "type":"encoder_only", "seq_len":257, "description":"ViT-Huge: 632M params, 32 layers"},
    "gpt2-large": {"hf_name":"gpt2-large", "model_class":GPT2LMHeadModel, "config_class":GPT2Config, "type":"decoder_only", "seq_len":1024, "description":"GPT2-Large: 774M params, 36 layers"},
    "gpt2-xl":    {"hf_name":"gpt2-xl",    "model_class":GPT2LMHeadModel, "config_class":GPT2Config, "type":"decoder_only", "seq_len":1024, "description":"GPT2-XL: 1.5B params, 48 layers"},
    "minerva-7b": {"hf_name":"sapienzanlp/Minerva-7B-instruct-v1.0", "model_class":AutoModelForCausalLM, "config_class":AutoConfig, "type":"decoder_only", "seq_len":2048, "description":"Minerva-7B: 7B params, 32 layers"},
    "llama3-8b":  {"hf_name":"meta-llama/Meta-Llama-3-8B", "model_class":AutoModelForCausalLM, "config_class":AutoConfig, "type":"decoder_only", "seq_len":8192, "description":"LLaMA3-8B: 8B params, 32 layers"},
    "llama3-70b": {"hf_name":"meta-llama/Meta-Llama-3-70B","model_class":AutoModelForCausalLM, "config_class":AutoConfig, "type":"decoder_only", "seq_len":8192, "description":"LLaMA3-70B: 70B params, 80 layers"},
    "mixtral-8x7b":{"hf_name":"mistralai/Mixtral-8x7B-v0.1","model_class":MixtralForCausalLM if MixtralForCausalLM else AutoModelForCausalLM, "config_class":MixtralConfig if MixtralConfig else AutoConfig, "type":"decoder_only", "seq_len":32768, "description":"Mixtral-8x7B MoE: 47B params, 32 layers, 8 experts"}
}

# ------------------------------
# Helpers
# ------------------------------
def get_layer(model, model_type):
    if model_type=="encoder_only":
        if hasattr(model,"encoder") and hasattr(model.encoder,"layer"):
            return model.encoder.layer[0]
        elif hasattr(model,"layers"):
            return model.layers[0]
    else:
        if hasattr(model,"transformer") and hasattr(model.transformer,"h"):
            return model.transformer.h[0]
        elif hasattr(model,"model") and hasattr(model.model,"layers"):
            return model.model.layers[0]
    raise ValueError("Cannot find layer")

def get_num_blocks(config):
    return getattr(config,"num_hidden_layers", getattr(config,"n_layer", getattr(config,"num_layers",1)))

def get_embed_dim(config):
    return getattr(config,"hidden_size", getattr(config,"n_embd",0))

def get_num_experts(config):
    return getattr(config,"num_local_experts",1)

def count_params(layer):
    return sum(p.numel() for p in layer.parameters())

def fwd_time(layer, seq_len, batch_size, embed_dim, model_type, device="cpu"):
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device) if model_type=="decoder_only" else None
    # Run in no_grad to avoid storing activations for backward
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = layer(x, attention_mask=mask) if mask is not None else layer(x)
        # Timing
        times=[]
        for _ in range(20):
            t0=time.perf_counter()
            _ = layer(x, attention_mask=mask) if mask is not None else layer(x)
            if device=="cuda": torch.cuda.synchronize()
            t1=time.perf_counter()
            times.append((t1-t0)*1e6)
    return torch.tensor(times).median().item()

# ------------------------------
# Main
# ------------------------------
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Compute model stats proxy")
    parser.add_argument("model_name", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    args=parser.parse_args()

    cfg=MODEL_CONFIGS[args.model_name]
    print(f"Loading config for {args.model_name}...")
    config = cfg["config_class"].from_pretrained(cfg["hf_name"], trust_remote_code=True)
    num_blocks=get_num_blocks(config)
    embed_dim=get_embed_dim(config)
    experts=get_num_experts(config)
    print(f"✓ Config loaded: {num_blocks} layers, {embed_dim}D, {experts} expert(s)")

    print("Loading model weights...")
    model=cfg["model_class"].from_pretrained(cfg["hf_name"], trust_remote_code=True)
    model.eval()
    layer=get_layer(model, cfg["type"]).to(args.device)

    print("Timing single layer (proxy)...")
    fwd = fwd_time(layer, cfg["seq_len"], args.batch_size, embed_dim, cfg["type"], device=args.device)
    bwd = 2*fwd  # approximate backward

    total_params = count_params(layer)*num_blocks
    total_fwd_time = fwd*num_blocks
    total_bwd_time = bwd*num_blocks

    # ------------------------------
    # Write stats like original
    # ------------------------------
    out_dir = os.path.expanduser("~/DNNProxy/model_stats")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.model_name}_{args.batch_size}.txt")
    with open(out_path,"w") as f:
        f.write(f"Forward_Flops:0\n")
        f.write(f"Backward_Flops:0\n")
        f.write(f"Model_Size:{total_params}\n")
        f.write(f"Average_Forward_Time (us):{int(total_fwd_time)}\n")
        f.write(f"Average_Backward_Time (us):{int(total_bwd_time)}\n")
        f.write(f"Batch_size:{args.batch_size}\n")
        f.write(f"Experts:{experts}\n")
        f.write(f"Sample_Size:{cfg['seq_len']}\n")
        f.write(f"Embedded_dim:{embed_dim}\n")
        f.write(f"Device:{DEVICE_NAME}\n")

    print(f"✓ Stats written to {out_path}")
    print(f"Forward time (μs): {total_fwd_time:.2f}")
    print(f"Backward time approx (μs): {total_bwd_time:.2f}")

