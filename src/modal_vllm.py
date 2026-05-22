"""
vLLM inference server on Modal.

To swap models, change MODEL_ID and GPU_TYPE at the top of this file,
then re-run `modal deploy modal_vllm.py`.

Tier 1: microsoft/Phi-3.5-mini-instruct  + L4   — parity with local, cheapest
Tier 2: google/gemma-3-4b-it             + L4   — gated model, needs HF_TOKEN
Tier 3: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 + H100 — the real target
"""
import os
import subprocess

import modal

# ── Model & GPU selection (change both together when moving tiers) ─────────────
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
GPU_TYPE = "L4"

# ── Per-model vLLM flags ──────────────────────────────────────────────────────
# Construct from this dict so swapping MODEL_ID automatically picks the right flags.
MODEL_CONFIGS: dict[str, list[str]] = {
    "microsoft/Phi-3.5-mini-instruct": [
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
    ],
    "google/gemma-3-4b-it": [
        # Vision tower disabled to save VRAM; enable by removing limit-mm-per-prompt.
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        "--limit-mm-per-prompt", "image=0",
    ],
    # Requires vLLM >= 0.20.x for Mamba-2 kernels and the nano_v3 reasoning parser.
    # FP8 weights ~30 GB; H100 (80 GB) is the minimum viable GPU.
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8": [
        "--trust-remote-code",
        "--tensor-parallel-size", "1",
        "--max-model-len", "65536",
        "--max-num-seqs", "8",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "nano_v3",
        "--gpu-memory-utilization", "0.90",
    ],
}

# ── Modal infrastructure ──────────────────────────────────────────────────────
app = modal.App("vllm-inference")

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        # Pin vLLM. Nemotron (Tier 3) requires >= 0.20.x; bump the pin before that deploy.
        # flashinfer is bundled by vLLM >= 0.4; do not install separately.
        "vllm>=0.9.0",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    secrets=[
        modal.Secret.from_name("huggingface"),   # provides HF_TOKEN
        modal.Secret.from_name("vllm-auth"),      # provides VLLM_API_KEY
    ],
    scaledown_window=300,   # scale to zero 5 min after last request
    timeout=3600,
)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """Start vLLM's OpenAI-compatible server. Modal keeps the container alive."""
    import sys
    api_key = os.environ["VLLM_API_KEY"]
    cmd = [
        "vllm", "serve", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--api-key", api_key,
        *MODEL_CONFIGS[MODEL_ID],
    ]
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
