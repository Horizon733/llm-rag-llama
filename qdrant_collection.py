from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="llm_database",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)