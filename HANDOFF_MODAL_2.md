# Handoff: Modal vLLM Deployment — Debug Session

## Where We Are

The Modal scaffolding is built and deploying successfully, but requests return
`modal-http: invalid function call` (HTTP 404) before reaching vLLM. The
infrastructure is ~90% done; this is a Modal routing/startup issue, not an
architecture problem.

## What's Already Done

- `src/modal_vllm.py` — Modal app, fully written and deploying cleanly
- `src/test_vllm.py` — updated to read `MODAL_VLLM_URL`, `VLLM_API_KEY`,
  `MODEL_ID` from env (no hardcoded values)
- `.env.example` — documents all four env vars
- `README.md` — Modal deployment section with both bash and PowerShell instructions
- Modal account set up (`modal setup` completed)
- Two Modal secrets created:
  - `vllm-auth` with key `VLLM_API_KEY`
  - `huggingface` with key `HF_TOKEN`
- Modal Volume `huggingface-cache` will be created automatically on first deploy
- Deployed URL: `https://espickle1--vllm-inference-serve.modal.run`

## The Blocker

**Error:** `openai.NotFoundError: modal-http: invalid function call` (HTTP 404)

Modal's routing layer is returning this before the request ever reaches vLLM.
This error means Modal can't find or route to the `serve` function as a web
endpoint.

### What We've Ruled Out

- `@modal.concurrent(max_inputs=100)` — removed; it conflicts with
  `@modal.web_server` (designed for `.remote()` functions, not HTTP proxying)
- `flashinfer-python` — removed from pip install; newer vLLM bundles it and
  the PyPI package installs incorrectly against specific CUDA/PyTorch versions
- `subprocess.Popen` stdout/stderr — added `stdout=sys.stdout, stderr=sys.stderr`
  so vLLM output now appears in `modal app logs vllm-inference`
- The URL is correct — Modal printed `https://espickle1--vllm-inference-serve.modal.run`
  after deploy, matching the app name (`vllm-inference`) and function name (`serve`)

### Most Likely Remaining Causes (try in this order)

**1. `startup_timeout` not recognized by installed Modal version**

If Modal's version doesn't support `startup_timeout` as a keyword arg to
`@modal.web_server`, the decorator silently fails to register the function as
a web endpoint, causing the 404.

Check:
```powershell
pip show modal
python -c "import modal; help(modal.web_server)"
```

Fix — remove `startup_timeout` and redeploy:
```python
@modal.web_server(port=8000)
```
If this fixes the routing, add `startup_timeout` back using whatever the
correct parameter name is for the installed version.

**2. vLLM crashing before port 8000 opens**

The logs showed `Task's current input(s) cancelled because:` with no reason.
This can mean vLLM starts but crashes before binding to port 8000, so Modal's
startup check never passes. After adding stdout/stderr to Popen, run:

```powershell
modal app logs vllm-inference
```

immediately after triggering a request (`python src/test_vllm.py`) and look
for a vLLM error. Common causes:
- `vllm>=0.9.0` resolving to a version that doesn't exist → pip falls back to
  an older version with incompatible flags. Try pinning to a specific version:
  `"vllm==0.6.3"` (or whatever `pip show vllm` shows inside the container).
- GPU driver/CUDA version mismatch with the image base.

**3. Modal version-specific `@modal.web_server` behavior**

In some Modal versions, `@modal.web_server` requires the decorated function to
block (not return immediately after `Popen`). Try adding a wait loop:

```python
def serve():
    import sys, time, socket
    api_key = os.environ["VLLM_API_KEY"]
    cmd = ["vllm", "serve", MODEL_ID, "--host", "0.0.0.0", "--port", "8000",
           "--api-key", api_key, *MODEL_CONFIGS[MODEL_ID]]
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    # Block until vLLM is listening, then keep the function alive
    while True:
        try:
            with socket.create_connection(("localhost", 8000), timeout=1):
                break
        except OSError:
            time.sleep(1)
    proc.wait()
```

## Current State of `src/modal_vllm.py`

```python
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
GPU_TYPE = "L4"
# image: nvidia/cuda:12.4.1-devel-ubuntu22.04, Python 3.12
# pip: vllm>=0.9.0, huggingface_hub[hf_transfer]
# decorators: @app.function(...) then @modal.web_server(port=8000, startup_timeout=600)
# serve(): subprocess.Popen(vllm cmd, stdout=sys.stdout, stderr=sys.stderr)
```

## Windows-Specific Notes

- `modal shell` is not supported on Windows — can't SSH into the container
- `modal app logs --follow` flag doesn't exist on Windows — re-run
  `modal app logs vllm-inference` manually to poll for new output
- Use PowerShell for all Modal/Python commands; the bash `export $(cat .env | xargs)` 
  idiom doesn't work — use the one-liner in the README or set `$env:VAR` individually
- `modal setup` must be run via `python -m modal setup` (the `modal` CLI isn't on PATH
  via bash, but works in PowerShell if the Scripts directory is on the PATH)

## Next Steps After Unblocking

Once requests are reaching vLLM and returning completions:
- [ ] Verify `python src/test_vllm.py` returns a sensible response (Tier 1 acceptance)
- [ ] Swap to `google/gemma-3-4b-it` + `L4` to test the gated-model flow (change
      `MODEL_ID` and `GPU_TYPE` at top of `modal_vllm.py`, redeploy, update `.env`)
- [ ] Swap to `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` + `H100` — bump vLLM
      pin to `>=0.20.x` first; FP8 weights are ~30 GB so first cold start is slow

## Reference

- Modal vLLM example: https://modal.com/docs/examples/vllm_inference
- Modal web_server docs: https://modal.com/docs/guide/webhooks
- vLLM Nemotron-3-Nano recipe: https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html
