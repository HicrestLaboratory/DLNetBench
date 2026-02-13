#!/usr/bin/env python3
"""
Download HuggingFace models to local cache (config and/or weights).
Uses local HuggingFace login (huggingface-cli login).
No tokens passed in code.
"""

import argparse
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    ViTModel,
    ViTConfig,
)

# =========================
# Model registry
# =========================

MODELS = {
    # Public
    "vit-b": ("google/vit-base-patch16-224", ViTModel, ViTConfig),
    "vit-l": ("google/vit-large-patch16-224", ViTModel, ViTConfig),
    "vit-h": ("google/vit-huge-patch14-224-in21k", ViTModel, ViTConfig),
    "gpt2-l": ("gpt2-large", AutoModelForCausalLM, AutoConfig),
    "gpt2-xl": ("gpt2-xl", AutoModelForCausalLM, AutoConfig),

    # Public Minerva 7B
    "minerva-7b": ("sapienzanlp/Minerva-7B-instruct-v1.0", AutoModelForCausalLM, AutoConfig),

    # Gated (still possible if you're logged in + have access)
    "llama3-8b": ("meta-llama/Meta-Llama-3-8B", AutoModelForCausalLM, AutoConfig),
    "llama3-70b": ("meta-llama/Meta-Llama-3-70B", AutoModelForCausalLM, AutoConfig),
    "mixtral-8x7b": ("mistralai/Mixtral-8x7B-v0.1", AutoModelForCausalLM, AutoConfig),
}

# =========================
# Download logic
# =========================

def download_model(model_name, config_only=False):
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        return False

    hf_name, model_cls, config_cls = MODELS[model_name]

    print(f"\n{'=' * 70}")
    print(f"Model:   {model_name}")
    print(f"HF repo: {hf_name}")
    print(f"{'=' * 70}")

    try:
        print("📥 Downloading config...")
        config_cls.from_pretrained(
            hf_name,
            trust_remote_code=True,
        )
        print("✅ Config downloaded")

        if not config_only:
            print("📥 Downloading model weights (this may take a while)...")
            model_cls.from_pretrained(
                hf_name,
                trust_remote_code=True,
            )
            print("✅ Weights downloaded")
        else:
            print("⏭️  Config-only mode")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models to local cache (uses HF login)"
    )

    parser.add_argument("models", nargs="*", help="Models to download")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument(
        "--config_only", action="store_true", help="Download only configs"
    )
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:\n")
        for name, (hf, _, _) in MODELS.items():
            print(f"  {name:15s} → {hf}")
        sys.exit(0)

    if args.all:
        models = list(MODELS.keys())
    elif args.models:
        models = args.models
    else:
        parser.print_help()
        sys.exit(1)

    ok, fail = 0, 0
    for m in models:
        if download_model(m, args.config_only):
            ok += 1
        else:
            fail += 1

    print(f"\n✅ Success: {ok} | ❌ Failed: {fail}")
    print("📦 Cache location: ~/.cache/huggingface/hub/")

