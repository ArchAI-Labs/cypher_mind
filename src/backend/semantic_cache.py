import os
import uuid

from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class SemanticCache:
    def __init__(
        self, collection_name: str, mode: str, embedder: str, vector_size: int
    ):
        self.collection_name = collection_name
        self.mode = mode
        self.embedder = embedder
        self.vector_size = vector_size

        if os.getenv("QDRANT_MODE") == "memory":
            self.qdrant_client = QdrantClient(":memory:")
        elif os.getenv("QDRANT_MODE") == "cloud":
            self.qdrant_client = QdrantClient(
                host=os.getenv("QDRANT_HOST"),
                api_key=os.getenv("QDRANT_API_KEY"),
            )
        elif os.getenv("QDRANT_MODE") == "docker":
            self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"))
        else:
            raise ValueError("Qdrant has 3 mode: memory, cloud or docker")
        if not self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
        print("Semantic Cache Initialized")

        supported_models = [x["model"] for x in TextEmbedding.list_supported_models()]
        embedder_model = os.environ.get("EMBEDDER")
        if embedder_model in supported_models:
            print(f"The model {embedder_model} is already registred.")
            self.model = TextEmbedding(model_name=self.embedder)
        else:
            print(f"The model {embedder_model} isn't already registred.")
            TextEmbedding.add_custom_model(
                model=self.embedder,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=self.embedder),
                dim=self.vector_size,
                model_file="onnx/model.onnx",
            )
            self.model = TextEmbedding(model_name=self.embedder)

    def get_embedding(self, text):
        try:
            result = list(self.model.embed(text))
            return result[0]
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return None

    def store_query_and_response(
        self,
        question,
        cypher_query,
        response,
    ):
        """
        It stores a question, the Cypher query, and the answer in Qdrant.
        """
        try:
            question_embedding = self.get_embedding(question)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=question_embedding,
                payload={
                    "question": question,
                    "cypher_query": cypher_query,
                    "response": response,
                },
            )
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[point],
            )
        except Exception as e:
            print(f"Error during embedding generation: {e}")

    def search_similar_question(self, question, threshold):
        """
        Search for a similar question in Qdrant.
        """
        try:
            question_embedding = self.get_embedding(question)

            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=question_embedding,
                limit=1,
                score_threshold=float(threshold),
            ).points
            if search_result:
                return search_result[0].payload
            else:
                return None
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return None
