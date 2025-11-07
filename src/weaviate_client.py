import weaviate
from weaviate.classes.init import AuthApiKey
from .config import WEAVIATE_URL, WEAVIATE_API_KEY, CLASS_NAME

def get_weaviate_client():
    if WEAVIATE_API_KEY:
        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
        )
    else:
        client = weaviate.Client(url=WEAVIATE_URL)
    return client


def ensure_schema():
    client = get_weaviate_client()
    schema = client.schema.get()
    existing_classes = {c["class"] for c in schema.get("classes", [])}

    if CLASS_NAME not in existing_classes:
        client.schema.create_class({
            "class": CLASS_NAME,
            "vectorizer": "none",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"]
                }
            ]
        })
    return client
