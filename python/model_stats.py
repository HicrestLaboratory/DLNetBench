import argparse
import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoConfig,
    ViTModel, ViTConfig,
)

# Optional Mixtral
try:
    from transformers import MixtralForCausalLM, MixtralConfig
except ImportError:
    MixtralForCausalLM = None
    MixtralConfig = None

# ------------------------------
# Hardware Specs (NVIDIA B200 192GB - Single GPU, Dense Compute)
# ------------------------------
BASE_PEAK = {
    "bfloat16": 2.25e15,  # 2.25 PFLOPS
    "float8": 4.5e15,     # 4.50 PFLOPS 
    "nvfp4": 9.0e15       # 9.00 PFLOPS
}
BANDWIDTH = 8.0e12        # 8.0 TB/s Memory Bandwidth
DEVICE_NAME = "NVIDIA B200-192GB (Single)"

# ------------------------------
# Models
# ------------------------------
MODELS = {
    "vit-b": ("google/vit-base-patch16-224", ViTModel, ViTConfig),
    "vit-l": ("google/vit-large-patch16-224", ViTModel, ViTConfig),
    "vit-h": ("google/vit-huge-patch14-224-in21k", ViTModel, ViTConfig),
    "gpt2-l": ("gpt2-large", AutoModelForCausalLM, AutoConfig),
    "gpt2-xl": ("gpt2-xl", AutoModelForCausalLM, AutoConfig),
    "minerva-7b": ("sapienzanlp/Minerva-7B-instruct-v1.0", AutoModelForCausalLM, AutoConfig),
    "llama3-8b": ("meta-llama/Meta-Llama-3-8B", AutoModelForCausalLM, AutoConfig),
    "llama3-70b": ("meta-llama/Meta-Llama-3-70B", AutoModelForCausalLM, AutoConfig),
    "mixtral-8x7b": ("mistralai/Mixtral-8x7B-v0.1",
                      MixtralForCausalLM if MixtralForCausalLM else AutoModelForCausalLM,
                      MixtralConfig if MixtralConfig else AutoConfig),
}

# ------------------------------
# Roofline helpers
# ------------------------------
def roofline_time(flops, bytes_accessed, peak_flops, peak_bw):
    """Calculate time using roofline model."""
    ai = flops / bytes_accessed if bytes_accessed > 0 else float('inf')
    return flops / min(peak_flops, ai * peak_bw)

def bytes_per_element(dtype_str):
    """Return bytes per element for given dtype string."""
    if dtype_str == "bfloat16":
        return 2.0
    elif dtype_str == "float8":
        return 1.0
    elif dtype_str == "nvfp4":
        return 0.5
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theoretical Roofline Simulator for Single B200")
    parser.add_argument("model_name", choices=list(MODELS.keys()))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float8", "nvfp4"], default="bfloat16", 
                        help="Data type for the model calculations (default: bfloat16)")
    args = parser.parse_args()

    hf_name, model_class, config_class = MODELS[args.model_name]
    dtype_str = args.dtype

    # Load config
    print(f"Loading config for {args.model_name}...")
    config = config_class.from_pretrained(hf_name, trust_remote_code=True)

    # Determine sequence length
    if hasattr(config, "seq_len"):
        N = config.seq_len
    elif hasattr(config, "n_positions"):
        N = config.n_positions
    elif hasattr(config, "max_position_embeddings"):
        N = config.max_position_embeddings
    elif hasattr(config, "image_size") and hasattr(config, "patch_size"):
        N = (config.image_size // config.patch_size) ** 2 + 1
    else:
        N = 1024
        
    # Architecture parameters
    L = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    d = getattr(config, "hidden_size", getattr(config, "n_embd", 768))
    H = getattr(config, "intermediate_size", 4 * d)
    E = getattr(config, "num_local_experts", 1)
    k = 2 if "mixtral" in args.model_name else 1
    B = args.batch_size

    # Precision and peak limits
    s = bytes_per_element(dtype_str)
    peak_flops = BASE_PEAK[dtype_str]

    # FLOPs (Full Attention - Window size removed)
    attn_f = (8 * B * N * d**2 + 4 * B * N**2 * d) * L
    mlp_f = (4 * B * N * d * H * k) * L
    total_fwd_flops = attn_f + mlp_f

    # Memory bytes accessed
    attn_b = (4 * d**2 * s + 2 * B * N * d * s) * L
    mlp_b  = (2 * d * H * s * E + 2 * B * N * d * s) * L

    # Roofline timing calculations
    t_attn = roofline_time(attn_f, attn_b, peak_flops, BANDWIDTH)
    t_mlp  = roofline_time(mlp_f, mlp_b, peak_flops, BANDWIDTH)
    t_fwd  = t_attn + t_mlp
    t_bwd  = 2 * t_fwd

    # Model parameter count
    print("Loading model to count parameters...")
    model = model_class.from_pretrained(hf_name, trust_remote_code=True)
    model_size = sum(p.numel() for p in model.parameters())

    # Output writing
    out_dir = os.path.expanduser("~/DLNetBench/model_stats")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.model_name}_{B}_{dtype_str}.txt")

    with open(out_file, "w+") as f:
        f.write(f"Forward_Flops:{int(total_fwd_flops)}\n")
        f.write(f"Backward_Flops:{int(2*total_fwd_flops)}\n")
        f.write(f"Model_Size:{model_size}\n")
        f.write(f"Average_Forward_Time (us):{t_fwd*1e6:.2f}\n")
        f.write(f"Average_Backward_Time (us):{t_bwd*1e6:.2f}\n")
        f.write(f"Batch_size:{B}\n")
        f.write(f"FFN_Average_Forward_Time (us):{t_mlp*1e6:.2f}\n")
        f.write(f"FFN_Average_Backward_Time (us):{2*t_mlp*1e6:.2f}\n")
        f.write(f"Experts:{E}\n")
        f.write(f"Seq_len:{N}\n")
        f.write(f"Embedded_dim:{d}\n")
        f.write(f"Device:{DEVICE_NAME}\n")
        f.write(f"Dtype:{dtype_str}\n")
        f.write(f"Bytes_per_element:{s}\n")

    # Console output summary
    print(f"\n{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Device: {DEVICE_NAME}")
    print(f"Precision: {dtype_str} ({s} bytes/element)")
    print(f"Batch Size: {B}, Seq Len: {N}")
    print(f"Forward Pass Estimate: {t_fwd*1e6:.2f} us")
    print(f"{'='*60}\n")
