## Running small language model locally with vLLM
## Test on vLLM-metal

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")                    

response = client.chat.completions.create(
    model="microsoft/phi-3.5-mini-instruct",
    messages=[{"role": "user", "content": "What do you see when you look outside the window at Microsoft's Redmond office?"}],
    temperature=0.7,
    max_tokens=100                                                                              
)

print(response.choices[0].message.content)