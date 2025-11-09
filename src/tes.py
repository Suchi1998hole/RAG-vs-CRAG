from src.cache_layer import get_redis
from src.weaviate_client import get_weaviate_client

r = get_redis()
print("Redis connected:", r.ping())

client = get_weaviate_client()
print("Weaviate ready:", client.is_ready())

client.close()
print("Connection closed cleanly.")
