import weaviate
from weaviate.auth import AuthApiKey
from src.config import WEAVIATE_URL, WEAVIATE_API_KEY, CLASS_NAME


def get_weaviate_client():
    """
    Creates a Weaviate client connection.
    Works with both Cloud (API key) and Local Docker instances.
    """
    if WEAVIATE_API_KEY:
        # ✅ Cloud connection (for console.weaviate.cloud)
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        )
    else:
        # ✅ Local connection (e.g., Docker at localhost:8080)
        client = weaviate.connect_to_local()
    return client


def ensure_schema():
    """
    Ensures that the specified collection exists in Weaviate (v4.17+ syntax).
    """
    client = get_weaviate_client()

    collections = client.collections.list_all()  

    if CLASS_NAME not in collections:
        client.collections.create(
            name=CLASS_NAME,
            vectorizer_config=None,  
            properties=[
                weaviate.classes.config.Property(
                    name="text",
                    data_type=weaviate.classes.config.DataType.TEXT,
                )
            ],
        )
        print(f"Created collection: {CLASS_NAME}")
    else:
        print(f"Collection already exists: {CLASS_NAME}")

    return client