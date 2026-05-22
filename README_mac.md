# Small Language Model Project

Local LLM inference on Apple Silicon using [vllm-metal](https://github.com/vllm-project/vllm-metal) — a vLLM plugin that runs models via Metal GPU acceleration.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13+ (tested on macOS 26 Tahoe)
- Python 3.12 or 3.13

## Installation

Run the official install script (takes 10–30 min — vLLM core is built from source):

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

This creates a dedicated virtual environment at `~/.venv-vllm-metal`.

## Starting the Server

Activate the venv and serve a model:

```bash
source ~/.venv-vllm-metal/bin/activate
vllm serve microsoft/phi-3.5-mini-instruct --gpu-memory-utilization 0.6 --max-model-len 512
```

Wait for this line before making requests:

```
INFO:     Application startup complete.
```

## Making Requests

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-3.5-mini-instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

### Python

The server exposes an OpenAI-compatible API. Use the `openai` package (already installed in the venv):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

response = client.chat.completions.create(
    model="microsoft/phi-3.5-mini-instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in 2 sentences"}],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
```

Run with:

```bash
source ~/.venv-vllm-metal/bin/activate
python test_vllm.py
```

### VS Code

1. Open the project in VS Code
2. Press `Cmd+Shift+P` → **Python: Select Interpreter**
3. Enter path: `/Users/espickle/.venv-vllm-metal/bin/python`
4. Open `test_vllm.py` and run it with `Ctrl+F5`

The vLLM server must be running in a separate terminal while you run scripts.

## Stopping the Server

Press `Ctrl+C` in the terminal running the vLLM server.

## Supported Models

These models are verified to work with vllm-metal on ~16GB unified memory Macs. For 8GB Macs, stick to models under 4B parameters.

| Model | Size | Notes |
|---|---|---|
| `microsoft/phi-3.5-mini-instruct` | 3.8B | Recommended starting point |
| `HuggingFaceTB/SmolLM3-3B-Instruct` | 3B | Small and fast |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Requires 16GB+ RAM |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Smallest viable option |

For the full supported model list see the [vllm-metal docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/).

## Troubleshooting

**`Not enough Metal memory for KV cache`** — The model is too large. Use a smaller model or add `--gpu-memory-utilization 0.6 --max-model-len 512`.

**`kIOGPUCommandBufferCallbackErrorOutOfMemory`** — OOM during inference. Switch to a smaller model.

**`No module named 'openai'` in VS Code** — VS Code is using the wrong Python. Follow the VS Code setup steps above to point it at the vllm-metal venv.

**Slow first response** — Normal. Metal JIT-compiles kernels on first use. Subsequent requests are faster.