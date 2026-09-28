from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("AI_API_KEY")
base_url = os.getenv("AI_BASE_URL")

if not api_key:
    raise ValueError("AI_API_KEY is not set")
if not base_url:
    raise ValueError("AI_BASE_URL is not set")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

response = client.chat.completions.create(
    model="kimi/kimi-k3",
    messages=[
        {
            "role": "user",
            "content": "Hello! Tell me in one sentence what an AI agent is."
        }
    ]
)

print(response.choices[0].message.content)