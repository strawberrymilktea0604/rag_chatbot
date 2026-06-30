from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from backend.utils.qdrant_client import get_qdrant_client


class DeleteQdrantCollectionView(APIView):
    """
    Irreversibly deletes the collection.
    Use HTTP DELETE and require ?confirm=yes to avoid accidents.
    """
    def delete(self, request):
        confirm = request.query_params.get("confirm")
        if confirm != "yes":
            return Response(
                {"error": "Add ?confirm=yes to proceed with deletion."},
                status=status.HTTP_400_BAD_REQUEST
            )

        client = get_qdrant_client()
        col = settings.QDRANT_COLLECTION
        try:
            client.delete_collection(col)
            return Response({"message": f"Collection '{col}' deleted."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
