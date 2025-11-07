from openai import OpenAI
from .config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, MAX_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding


def chat_completion(prompt: str):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    content = resp.choices[0].message.content
    usage = resp.usage  # contains prompt_tokens, completion_tokens, total_tokens
    return content, usage
