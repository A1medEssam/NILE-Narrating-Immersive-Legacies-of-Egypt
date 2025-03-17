'''
[1.0]

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("http://localhost:6333")

client.create_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

collections = client.get_collections()
print(f"Available collections: {collections}")
'''



'''
[2.0]


from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

collection_info = client.get_collection("test_collection")
print(collection_info)
'''

'''
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Connect to Qdrant
client = QdrantClient("http://localhost:6333")

# Insert a test vector
client.upsert(
    collection_name="test_collection",
    points=[
        PointStruct(
            id=1, 
            vector=[0.1] * 768,  # A dummy vector with 768 dimensions
            payload={"text": "Hello, Ramesses!"}
        )
    ]
)

print("Inserted test vector successfully!")

[4.0]

from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

collection_info = client.get_collection("test_collection")
print(collection_info)


client.update_collection(
    collection_name="test_collection",
    optimizer_config={
        "indexing_threshold": 1  # Forces Qdrant to index immediately
    }
)

collection_info = client.get_collection("test_collection")
print(collection_info)
'''

'''
import qdrant_client

# Connect to Qdrant running on localhost
client = qdrant_client.QdrantClient("localhost", port=6333)

# Get collection info
collection_info = client.get_collection("test_collection")
print(collection_info)
'''

'''
import qdrant_client

client = qdrant_client.QdrantClient("localhost", port=6333)

query_vector = [0.1] * 768  # Replace with your real query vector

results = client.search(
    collection_name="test_collection",
    query_vector=query_vector,
    limit=1
)

print("Search Results:", results)
'''


import qdrant_client

client = qdrant_client.QdrantClient("localhost", port=6333)

query_vector = [0.1] * 768  # Replace with your real query vector

results = client.query_points(
    collection_name="test_collection",
    query_vector=query_vector,
    limit=1
)

print("Search Results:", results)



