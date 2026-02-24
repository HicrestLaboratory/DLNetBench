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
# Hardware Specs (A100 80GB SXM4)
# ------------------------------
BASE_PEAK = {torch.float32: 19.5e12, torch.float16: 312e12, torch.bfloat16: 312e12}
TF32_PEAK = {torch.float32: 156e12}  # Approx TF32 matmul peak
BANDWIDTH = 2.0e12  # 2 TB/s
DEVICE_NAME = "NVIDIA A100-SXM4-80GB"

# ------------------------------
# Models
# ------------------------------
MODELS = {
    # Public
    "vit-b": ("google/vit-base-patch16-224", ViTModel, ViTConfig),
    "vit-l": ("google/vit-large-patch16-224", ViTModel, ViTConfig),
    "vit-h": ("google/vit-huge-patch14-224-in21k", ViTModel, ViTConfig),
    "gpt2-l": ("gpt2-large", AutoModelForCausalLM, AutoConfig),
    "gpt2-xl": ("gpt2-xl", AutoModelForCausalLM, AutoConfig),

    # Public Minerva 7B
    "minerva-7b": ("sapienzanlp/Minerva-7B-instruct-v1.0", AutoModelForCausalLM, AutoConfig),

    # Gated
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
    ai = flops / bytes_accessed
    return flops / min(peak_flops, ai * peak_bw)

def tf32_enabled():
    """Check if TF32 is enabled for CUDA operations."""
    return torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32

def bytes_per_element(dtype):
    """Return bytes per element for given dtype."""
    if dtype in [torch.float16, torch.bfloat16]:
        return 2
    elif dtype == torch.float32:
        return 4
    elif dtype == torch.float64:
        return 8
    else:
        return 4

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theoretical Roofline Simulator")
    parser.add_argument("model_name", choices=list(MODELS.keys()))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=None, help="Sliding window size (e.g. 4096)")
    args = parser.parse_args()

    hf_name, model_class, config_class = MODELS[args.model_name]
    config_window = getattr(config, "sliding_window", None)
    if config_window is not None:
        W = config_window
    elif args.window_size is not None:
        W = args.window_size
    else:
        W = N
    # Load config
    print(f"Loading config for {args.model_name}...")
    config = config_class.from_pretrained(hf_name, trust_remote_code=True)
    effective_window = min(W, N)
    # Architecture parameters
    L = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    d = getattr(config, "hidden_size", getattr(config, "n_embd", 768))
    H = getattr(config, "intermediate_size", 4 * d)
    E = getattr(config, "num_local_experts", 1)
    k = 2 if "mixtral" in args.model_name else 1
    B = args.batch_size

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

    # Precision and peak
    # FIXED: Added "minerva" to the list of models that use bfloat16
    dtype = torch.bfloat16
    s = bytes_per_element(dtype)
    peak_flops = BASE_PEAK[dtype]
    if dtype == torch.float32 and tf32_enabled():
        peak_flops = TF32_PEAK[torch.float32]

    # FLOPs
    attn_f = (8 * B * N * d**2 + 4 * B * N*effective_window * d) * L
    mlp_f = (4 * B * N * d * H * k) * L
    total_fwd_flops = attn_f + mlp_f

    # Memory bytes
    attn_b = (4*d**2*s + 2*B*N*d*s) * L
    mlp_b  = (2*d*H*s*E + 2*B*N*d*s) * L

    # Roofline timing
    t_attn = roofline_time(attn_f, attn_b, peak_flops, BANDWIDTH)
    t_mlp  = roofline_time(mlp_f, mlp_b, peak_flops, BANDWIDTH)
    t_fwd  = t_attn + t_mlp
    t_bwd  = 2 * t_fwd

    # Model size
    print(f"Loading model to count parameters...")
    model = model_class.from_pretrained(hf_name, trust_remote_code=True)
    model_size = sum(p.numel() for p in model.parameters())

    # Output
    out_dir = os.path.expanduser("~/DLNetBench/model_stats")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.model_name}_{B}.txt")

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
        f.write(f"Dtype:{dtype}\n")
        f.write(f"Bytes_per_element:{s}\n")
        f.write(f"TF32_Enabled:{tf32_enabled()}\n")

    # Console output
    print(f"\n{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"{'='*60}")
    print(f"Architecture:")
    print(f"  Layers (L): {L}")
    print(f"  Hidden dim (d): {d}")
    print(f"  FFN dim (H): {H}")
    print(f"  Sequence length (N): {N}")
    print(f"  Batch size (B): {B}")
    print(f"  Experts (E): {E}")
    print(f"  Top-k (k): {k}")
    print(f"\nPrecision:")
    print(f"  Dtype: {dtype}")
    print(f"  Bytes per element: {s}")
    print(f"  TF32 enabled: {tf32_enabled()}")
    print(f"\nCompute:")
    print(f"  Total parameters: {model_size:,}")
    print(f"  Forward FLOPs: {total_fwd_flops:.2e}")
    print(f"  Backward FLOPs: {2*total_fwd_flops:.2e}")
    print(f"  Peak FLOPS: {peak_flops:.2e}")
    print(f"\nTheoretical Times:")
    print(f"  Forward: {t_fwd*1e6:.2f} us ({t_fwd*1e3:.2f} ms)")
    print(f"  Backward: {t_bwd*1e6:.2f} us ({t_bwd*1e3:.2f} ms)")
    print(f"  Attention forward: {t_attn*1e6:.2f} us")
    print(f"  FFN forward: {t_mlp*1e6:.2f} us")
    print(f"\n✓ Stats written to {out_file}")
    print(f"{'='*60}\n")
