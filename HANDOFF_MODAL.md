# Handoff: Migrate Local vLLM Setup to Modal

## Context

I have a working local LLM setup using **vllm-metal** on a 16GB MacBook Air. It's serving an OpenAI-compatible API on `localhost:8000`, but the Air is memory-constrained and thermally throttled — sustained inference drops to ~2 tok/s on anything 7B+. I want to keep the same API surface (so existing client code doesn't change) but move the actual inference to a **Modal**-hosted NVIDIA GPU.

End goal: I run `python test_vllm.py` on my laptop, it hits a Modal endpoint instead of `localhost`, and I get fast inference on a real GPU.

## Current State

Two files describe the existing setup (both in this handoff):

- **`test_vllm.py`** — minimal OpenAI SDK client hitting `http://localhost:8000/v1`, calling `microsoft/phi-3.5-mini-instruct`.
- **`README.md`** — installs `vllm-metal`, runs `vllm serve microsoft/phi-3.5-mini-instruct --gpu-memory-utilization 0.6 --max-model-len 512`, documents troubleshooting.

The client code is the contract I want preserved. The serving layer is what's being replaced.

## Target State

- A Modal app (`modal_vllm.py`) that:
  - Boots a vLLM server inside a Modal container with an NVIDIA GPU attached
  - Exposes vLLM's OpenAI-compatible API as a Modal web endpoint
  - Stays warm long enough to be usable interactively (idle timeout in the 5–10 min range), then scales to zero
  - Uses a Modal Volume to cache model weights so cold starts after the first download are fast
  - Has `@modal.concurrent` set so a single container can batch many concurrent requests (this is what makes vLLM's continuous batching actually work — without it Modal serializes requests one-per-container)
- An updated `test_vllm.py` that points `base_url` at the Modal endpoint
- A short README section explaining `modal deploy` vs `modal serve` and how to retrieve the endpoint URL

## Framework Choice

Sticking with **vLLM** for parity with the existing setup. Worth knowing for later: **SGLang** is the main alternative and is ~29% faster on small models with prefix-heavy workloads (chatbots, RAG, agents with system prompts). Both speak the OpenAI API, so swapping is mostly a matter of changing the serving command. Not doing it now, but don't bury vLLM-specific code in a way that makes a future switch painful.

## Key Decisions (please make these explicit in the implementation)

### 1. Model progression (three tiers)

The single MODEL_ID constant should be easy to swap. Plan to ship in this order:

| Tier | Model ID | Purpose |
|------|----------|---------|
| 1 | `microsoft/Phi-3.5-mini-instruct` | Verify the Modal scaffolding end-to-end with minimum risk (parity with local) |
| 2 | `google/gemma-3-4b-it` | Validate the gated-model flow (HF token) and a slightly larger model |
| 3 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | The actual reason for migrating — needs real GPU |

**Heads-up on Nemotron**: in the current (Nemotron 3) family, "Nano" is **30B total / 3B active MoE** — not a small dense model. FP8 weights are ~30 GB. Don't size for a 4B model and get surprised.

### 2. GPU per tier

| Tier | GPU | Why |
|------|-----|-----|
| Phi-3.5-mini (3.8B) | `L4` (24 GB) | Cheapest viable option; `A10G` also fine if L4 is unavailable |
| Gemma 3 4B | `L4` (text-only) or `L40S` (48 GB, if enabling vision) | Vision tower needs headroom |
| Nemotron-3-Nano-30B-A3B-FP8 | `H100` (80 GB), TP=1 | FP8 weights ~30 GB; A100-40GB is too tight once KV cache is added. H200 if available is even better. |

GPU type goes alongside MODEL_ID as a constant at the top of the file. Both should be swappable in one place.

### 3. Per-model vLLM flags

These differ per tier and matter. The `serve()` function should construct the args from per-model config — don't hardcode flags that only work for one model.

**Phi-3.5-mini**:
```
--max-model-len 16384          # cap from 128K default or KV cache is destroyed
--gpu-memory-utilization 0.90
--enable-prefix-caching
```

**Gemma 3 4B-it**:
```
--max-model-len 8192
--gpu-memory-utilization 0.90
--enable-prefix-caching
--limit-mm-per-prompt image=0   # disable vision unless explicitly needed (saves VRAM)
```
Plus: requires accepting the license on HuggingFace and an `HF_TOKEN` in the Modal Secret. Set up the secret with `HF_TOKEN` from the start so the Tier 2 swap is just a constant change.

**Nemotron-3-Nano-30B-A3B-FP8**:
```
--trust-remote-code
--tensor-parallel-size 1
--max-model-len 65536           # not 256K; saves KV
--max-num-seqs 8                # NVIDIA's recipe starting point; tune up after profiling
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--reasoning-parser nano_v3
--gpu-memory-utilization 0.90
```
Requires vLLM ≥ 0.20.x. Pin the vLLM version in the image — pre-0.20 doesn't have the Mamba-2 kernels or the `nano_v3` parser.

### 4. Implementation pattern

**Updated from the previous draft**: use `@modal.web_server(port=8000)` with `subprocess.Popen("vllm serve ...")` rather than `@modal.asgi_app()` importing the FastAPI app directly. This is what Modal's own vLLM example uses and it sidesteps a class of init-order bugs around CUDA graph capture and engine warm-up. The OpenAI-compatible endpoints look identical from the client's perspective.

Pair it with `@modal.concurrent(max_inputs=100)` so a single warm container batches many requests through vLLM. Without this, Modal sends one request per container and continuous batching can't engage.

### 5. Auth

vLLM's `--api-key` flag should be set to a value pulled from a **Modal Secret** (not hardcoded). The existing `test_vllm.py` passes `api_key="token-abc123"` — replace that with reading from an env var locally too.

## Implementation Tasks

1. Write `modal_vllm.py`:
   - `modal.Image` from a CUDA base (`nvidia/cuda:12.4.1-devel-ubuntu22.04`), Python 3.12, pip-installing pinned `vllm`, `huggingface_hub[hf_transfer]`, `flashinfer-python`
   - `modal.Volume` mounted at `/root/.cache/huggingface` for weight caching
   - Two `modal.Secret` references: one for `HF_TOKEN`, one for `VLLM_API_KEY`
   - A `@app.function` decorated with `gpu=GPU_TYPE`, `volumes=...`, `secrets=...`, `scaledown_window=300`, `timeout=3600`
   - `@modal.concurrent(max_inputs=100)` on the function
   - `@modal.web_server(port=8000, startup_timeout=600)` (600s because Nemotron cold-starts can run 2–4 min on first boot)
   - Inside, `subprocess.Popen(["vllm", "serve", MODEL_ID, ...per-model-flags...])`
   - Model ID, GPU type, and the per-model flag dict at the top of the file
2. Update `test_vllm.py`:
   - Read `MODAL_VLLM_URL` and `VLLM_API_KEY` from env (with sensible fallbacks for local testing)
   - Read `MODEL_ID` from env too, so it stays in sync with the server
   - Keep everything else identical
3. Add a `.env.example` showing what env vars are needed (`MODAL_VLLM_URL`, `VLLM_API_KEY`, `MODEL_ID`, plus `HF_TOKEN` for setup)
4. Add a "Modal Deployment" section to the README covering: `modal setup`, creating the `huggingface` and `vllm-auth` secrets, `modal deploy modal_vllm.py`, retrieving the URL, setting env vars locally

## Acceptance Criteria

- [ ] `modal deploy modal_vllm.py` succeeds and prints a stable `https://*.modal.run` URL
- [ ] First request after cold deploy completes within ~2 min for Phi-3.5, ~4 min for Nemotron (model download + warm-up); subsequent cold starts within ~30–90s for Phi/Gemma and 1–2 min for Nemotron thanks to the Volume cache
- [ ] Running `python test_vllm.py` with `MODAL_VLLM_URL` set returns a sensible completion in under 5 seconds (warm) on Phi-3.5
- [ ] Swapping `MODEL_ID` from Phi-3.5-mini to Gemma 3 4B and re-deploying works without code changes (assuming `HF_TOKEN` secret is already set)
- [ ] Swapping to Nemotron-3-Nano-30B-A3B-FP8 requires only changing `MODEL_ID` and `GPU_TYPE` (to `H100`) at the top of the file
- [ ] No API keys, model IDs, or URLs are hardcoded in the client script

## Cost Notes

- L4 ≈ $1/hr, L40S ≈ $2/hr, H100 ≈ $4–5/hr on Modal
- `scaledown_window=300` means an idle container costs ~$0.08 (L4) or ~$0.42 (H100) per idle window after the last request — fine for interactive dev, not what you want if you forget about it overnight
- Do not set `min_containers=1` (always-warm) on the H100 tier without thinking about the monthly bill

## Reference Docs

- Modal vLLM example: https://modal.com/docs/examples/vllm_inference
- Modal GPU reference: https://modal.com/docs/guide/gpu
- Modal `@modal.concurrent`: https://modal.com/docs/guide/concurrent-inputs
- vLLM OpenAI server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- vLLM Nemotron-3-Nano recipe: https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html

## Out of Scope

- Authentication beyond a static API key (no JWT, no per-user tokens)
- Streaming responses (the current client doesn't use them; can add later)
- Anything related to `vllm-metal` — that setup stays on my laptop for offline experimentation but is not part of this migration
- Multi-model serving on one endpoint (one model per deployed app; redeploy to swap)
- Memory snapshots / cold-start optimization for Nemotron — useful once the basic setup works, but adds complexity; treat as a follow-up
- SGLang migration — noted as a future option, not part of this work
