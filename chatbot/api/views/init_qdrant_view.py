import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from qdrant_client.http.models import VectorParams, Distance, OptimizersConfigDiff, HnswConfigDiff
from backend.utils.qdrant_client import get_qdrant_client
from backend.utils.embeddings import HuggingfaceEmbeddingsModel


def _embedding_dim() -> int:
    model_name = os.getenv("EMBEDDING_MODEL", "dangvantuan/vietnamese-embedding")
    model = HuggingfaceEmbeddingsModel(model_name)
    vec = model.embed_texts(["probe"])[0]
    return len(vec)


class InitQdrantView(APIView):
    """
    Idempotently ensure the collection exists and matches current embedding dim.
    If exists but mismatched, recreate.
    """
    def get(self, request):
        client = get_qdrant_client()
        col = settings.QDRANT_COLLECTION
        dim = _embedding_dim()

        try:
            info = client.get_collection(col)
            data = info.dict()
            try:
                current_dim = data["result"]["config"]["params"]["vectors"]["configs"]["default"]["size"]
            except Exception:
                current_dim = None

            if current_dim == dim:
                return Response(
                    {"message": f"Collection '{col}' already exists with correct dim={dim}."},
                    status=status.HTTP_200_OK
                )

            client.recreate_collection(
                collection_name=col,
                vectors_config={"default": VectorParams(size=dim, distance=Distance.COSINE, on_disk=True)},
                optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            )
            return Response(
                {"message": f"Collection '{col}' recreated with dim={dim} (was {current_dim})."},
                status=status.HTTP_200_OK
            )

        except Exception:
            client.recreate_collection(
                collection_name=col,
                vectors_config={"default": VectorParams(size=dim, distance=Distance.COSINE, on_disk=True)},
                optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            )
            return Response(
                {"message": f"Collection '{col}' created with dim={dim}."},
                status=status.HTTP_201_CREATED
            )
