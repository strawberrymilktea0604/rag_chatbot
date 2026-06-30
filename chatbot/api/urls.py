from django.urls import path
from .views.ask_view import AskView
from .views.file_upload_view import FileUploadView
from .views.init_qdrant_view import InitQdrantView
from .views.delete_qdrant_collection_view import DeleteQdrantCollectionView
from .views.check_qdrant_view import CheckQdrantCollectionView  # Import the new view

urlpatterns = [
    path('ask/', AskView.as_view(), name='ask'),   # http://localhost:8000/api/ask/
    path('upload/', FileUploadView.as_view(), name='file-upload'),  # New upload endpoint
    path('init-qdrant/', InitQdrantView.as_view(), name='init-qdrant'),
    path('delete-qdrant/', DeleteQdrantCollectionView.as_view(), name='delete-qdrant'),
    path('check-qdrant/', CheckQdrantCollectionView.as_view(), name='check-qdrant'),  # New check endpoint
]
