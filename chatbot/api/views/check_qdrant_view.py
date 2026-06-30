from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from backend.utils.qdrant_client import get_qdrant_client


class CheckQdrantCollectionView(APIView):
    """
    Checks if a Qdrant collection exists without creating it.
    Returns collection status and metadata if it exists.
    """
    def get(self, request):
        client = get_qdrant_client()
        col = settings.QDRANT_COLLECTION
        
        try:
            info = client.get_collection(col)
            data = info.dict()
            return Response({
                "exists": True,
                "name": col,
                "info": data["result"]
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "exists": False,
                "name": col,
                "error": str(e)
            }, status=status.HTTP_404_NOT_FOUND)