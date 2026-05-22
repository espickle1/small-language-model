## Running small language model with vLLM (Modal or local)
## Set MODAL_VLLM_URL to point at the Modal endpoint; falls back to localhost.

import os

from openai import OpenAI

base_url = os.environ.get("MODAL_VLLM_URL", "http://localhost:8000").rstrip("/") + "/v1"
api_key = os.environ.get("VLLM_API_KEY", "token-abc123")
model_id = os.environ.get("MODEL_ID", "microsoft/Phi-3.5-mini-instruct")

client = OpenAI(base_url=base_url, api_key=api_key)

response = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "What do you see when you look outside the window at Microsoft's Redmond office?"}],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
