# small-language-model

Using small language models on local and Modal GPU instances.

## Local setup (vllm-metal, Mac)

See `README_mac.md` for the local vllm-metal setup on Apple Silicon.

## Modal Deployment

Run vLLM on a real NVIDIA GPU via [Modal](https://modal.com), keeping the same OpenAI-compatible API surface as the local setup.

### Prerequisites

```bash
pip install modal
modal setup          # authenticates your Modal account
```

### Create secrets

Modal needs two secrets before deploying:

**`huggingface`** — provides `HF_TOKEN` for downloading gated models (required for Tier 2+):
```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

**`vllm-auth`** — provides `VLLM_API_KEY` for the vLLM API endpoint:
```bash
modal secret create vllm-auth VLLM_API_KEY=your-api-key-here
```

### Deploy vs serve

| Command | Use when |
|---------|----------|
| `modal deploy modal_vllm.py` | Production / persistent endpoint — survives after your terminal closes, scales to zero when idle |
| `modal serve modal_vllm.py` | Development — live-reloads on file save, tears down when you Ctrl-C |

### Deploy

```bash
cd src
modal deploy modal_vllm.py
```

The command prints a stable URL like:
```
https://<workspace>--vllm-inference-serve.modal.run
```

Copy that URL — it's your `MODAL_VLLM_URL`.

### Set local env vars

**bash (Mac/Linux):**
```bash
cp .env.example .env
# Edit .env: set MODAL_VLLM_URL, VLLM_API_KEY, MODEL_ID
export $(cat .env | xargs)
python src/test_vllm.py
```

**PowerShell (Windows):**
```powershell
copy .env.example .env
# Edit .env: set MODAL_VLLM_URL, VLLM_API_KEY, MODEL_ID
Get-Content .env | Where-Object { $_ -match '^\s*[^#]' } | ForEach-Object { $parts = $_ -split '=', 2; [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process') }
python src/test_vllm.py
```

Or set variables individually in PowerShell:
```powershell
$env:MODAL_VLLM_URL = "https://your-url.modal.run"
$env:VLLM_API_KEY   = "your-api-key"
$env:MODEL_ID       = "microsoft/Phi-3.5-mini-instruct"
python src/test_vllm.py
```

### Swapping models

Change `MODEL_ID` and `GPU_TYPE` at the top of `src/modal_vllm.py`, then redeploy:

| Tier | MODEL_ID | GPU_TYPE | Notes |
|------|----------|----------|-------|
| 1 | `microsoft/Phi-3.5-mini-instruct` | `L4` | Default; parity with local setup |
| 2 | `google/gemma-3-4b-it` | `L4` | Requires `huggingface` secret with valid `HF_TOKEN` |
| 3 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `H100` | ~30 GB FP8 weights; requires vLLM ≥ 0.20.x |

After changing constants, redeploy — the URL stays the same:
```bash
modal deploy src/modal_vllm.py
```

Also update `MODEL_ID` in your `.env` to match.

### Cold start times

| Model | First deploy (weight download) | Subsequent cold starts |
|-------|-------------------------------|------------------------|
| Phi-3.5-mini | ~2 min | ~30–60 s |
| Gemma 3 4B | ~3 min | ~30–60 s |
| Nemotron-3-Nano-30B | ~8–15 min | ~1–2 min |

Subsequent cold starts are faster because weights are cached in the `huggingface-cache` Modal Volume.

### Cost notes

- L4 ≈ \$1/hr, H100 ≈ \$4–5/hr
- `scaledown_window=300` means an idle container lingers up to 5 min after the last request before scaling to zero — roughly \$0.08 (L4) or \$0.42 (H100) per idle window
- Do **not** set `min_containers=1` on the H100 tier without accounting for the always-on cost
