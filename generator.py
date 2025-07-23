import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Create OpenAI client for OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def generate_answer(query, retrieved_chunks, chat_history=None):
    context = "\n".join(retrieved_chunks)
    system_prompt = "Answer the following question using the given context using exact value. Use specific keywords. Support Bangla and English."

    messages = [
        {"role": "system", "content": system_prompt},
        *(chat_history if chat_history else []),
        {"role": "user", "content": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}"}
    ]

    try:
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku:beta",  # ✅ Free OpenRouter model
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {e}"
