from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, MAX_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str):
    """
    Generate embedding vector for a given text input.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding


def chat_completion(prompt: str):
    """
    Send a chat prompt to OpenAI and return (answer, usage_dict)
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )

    content = resp.choices[0].message.content.strip()

    usage = {
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "total_tokens": resp.usage.total_tokens if resp.usage else None,
    }

    return content, usage
