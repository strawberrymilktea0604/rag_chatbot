from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from backend.utils.qdrant_client import get_qdrant_client
from backend.utils.mongo_client import check_mongo_health
import logging
import time

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """
    Health check endpoint that validates system components
    """
    
    def get(self, request):
        """
        Check health of all system components
        
        Returns:
            JSON with status of each component
        """
        start_time = time.time()
        health_status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "components": {}
        }
        
        try:
            # Check Django settings
            health_status["components"]["django"] = {
                "status": "healthy",
                "debug_mode": settings.DEBUG,
                "database_engine": settings.DATABASES['default']['ENGINE']
            }
            
            # Check Qdrant connection
            try:
                client = get_qdrant_client()
                collection_info = client.get_collection(settings.QDRANT_COLLECTION)
                
                health_status["components"]["qdrant"] = {
                    "status": "healthy",
                    "collection_name": settings.QDRANT_COLLECTION,
                    "collection_exists": True,
                    "vector_count": collection_info.points_count if collection_info else 0
                }
                
            except Exception as e:
                logger.warning(f"Qdrant health check failed: {e}")
                health_status["components"]["qdrant"] = {
                    "status": "unhealthy", 
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check MongoDB connection
            try:
                mongo_health = check_mongo_health()
                health_status["components"]["mongodb"] = mongo_health
                
                if mongo_health.get("status") != "healthy":
                    health_status["status"] = "degraded"
                    
            except Exception as e:
                logger.warning(f"MongoDB health check failed: {e}")
                health_status["components"]["mongodb"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check environment variables
            required_env_vars = ["GEMINI_API_KEY", "QDRANT_URL", "QDRANT_COLLECTION", "MONGO_URI"]
            missing_vars = []
            
            for var in required_env_vars:
                if not getattr(settings, var, None):
                    missing_vars.append(var)
            
            health_status["components"]["environment"] = {
                "status": "healthy" if not missing_vars else "unhealthy",
                "missing_variables": missing_vars
            }
            
            if missing_vars:
                health_status["status"] = "unhealthy"
            
            # Add response time
            health_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            
            return Response(health_status, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Health check error: {e}", exc_info=True)
            return Response({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": int(time.time())
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)