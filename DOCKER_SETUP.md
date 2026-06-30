# Docker Setup for RAG Chatbot Backend

This guide provides instructions for running the RAG Chatbot backend with Docker Compose.

## Prerequisites

- Docker Desktop (or Docker + Docker Compose)
- Your Google Gemini API key
- `.env` file with required environment variables

## Quick Start

### 1. Prepare Environment Variables

Create a `.env` file in the project root with your configuration:

```bash
# Copy the template
cp .env.docker .env

# Edit .env and replace with your actual values
# IMPORTANT: Set your GEMINI_API_KEY
```

### 2. Build and Start Services

```powershell
# Navigate to project root
cd d:\rag_chatbot

# Build and start containers
docker-compose up -d

# View logs
docker-compose logs -f backend
```

### 3. Initialize Vector Database

After backend is running, initialize the Qdrant collection:

```powershell
# Run the initialization script
docker-compose exec backend python recreate_collection.py
```

### 4. Access Services

- **Django Backend**: http://localhost:8000
- **Qdrant API**: http://localhost:6333
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Services

### Backend (Django)
- **Port**: 8000
- **Container Name**: rag-chatbot-backend
- **Health Check**: `GET /api/health/`
- **Auto-migrations**: Runs `python manage.py migrate` on startup

### Qdrant (Vector Database)
- **HTTP Port**: 6333
- **gRPC Port**: 6334
- **Container Name**: rag-chatbot-qdrant
- **Storage**: Docker volume `qdrant_storage`

## Common Commands

### View Logs
```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f qdrant
```

### Execute Commands in Container
```powershell
# Run Django management commands
docker-compose exec backend python manage.py shell

# Run migrations
docker-compose exec backend python manage.py migrate

# Create superuser
docker-compose exec backend python manage.py createsuperuser

# Run evaluation
docker-compose exec backend python run_eval.py
```

### Stop Services
```powershell
# Stop all containers
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v
```

### Rebuild Images
```powershell
# Rebuild without cache
docker-compose build --no-cache

# Rebuild and start
docker-compose up -d --build
```

## Troubleshooting

### Backend Container Won't Start
```powershell
# Check logs
docker-compose logs backend

# Common issues:
# 1. Missing GEMINI_API_KEY in .env
# 2. Port 8000 already in use
# 3. Qdrant not running yet (wait 30s for health check)
```

### Qdrant Connection Issues
```powershell
# Check Qdrant health
curl http://localhost:6333/health

# Verify collection exists
docker-compose exec backend python verify_qdrant.py
```

### Permission Denied Errors
```powershell
# On Windows, ensure Docker Desktop is running
# On Linux, you may need sudo or add user to docker group
```

### Database Issues
```powershell
# Reset database (CAUTION: Deletes all data)
docker-compose exec backend rm db.sqlite3
docker-compose restart backend
```

## Production Deployment

For production use:

1. **Security**: 
   - Set `DEBUG=0` in `.env`
   - Use strong `DJANGO_SECRET_KEY`
   - Configure `ALLOWED_HOSTS` appropriately
   - Use a production WSGI server (Gunicorn)

2. **Database**:
   - Replace SQLite with PostgreSQL
   - Update `DATABASES` in `settings.py`

3. **Environment**:
   - Use `.env.production` for production variables
   - Never commit `.env` files
   - Use Docker secrets for sensitive data

4. **Example Production Compose**:
   ```yaml
   # Add PostgreSQL service
   postgres:
     image: postgres:15
     environment:
       POSTGRES_DB: ragchatbot
       POSTGRES_USER: raguser
       POSTGRES_PASSWORD: securepassword
     volumes:
       - postgres_data:/var/lib/postgresql/data
   
   # Update backend to use PostgreSQL
   # And add Gunicorn/Nginx services
   ```

## File Structure

```
docker-compose.yml          # Main Docker Compose configuration
backend/
  ├── Dockerfile            # Backend container image
  ├── requirements.txt       # Python dependencies
  ├── .dockerignore         # Files to exclude from build
  ├── manage.py
  ├── settings.py
  └── media/               # Persistent volume mount
.env                        # Environment variables (create from .env.docker)
.env.docker                 # Template for Docker configuration
```

## Health Checks

Services include health checks that automatically restart if they fail:

```powershell
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Performance Notes

- **First Run**: Initial build takes ~5-10 minutes (downloads models)
- **Memory**: Ensure 4GB+ available RAM
- **Storage**: Allocate 10GB+ for volumes
- **Network**: Services communicate via `rag-network` bridge

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Verify `.env` configuration
3. Ensure ports 8000 and 6333 are available
4. Check Docker Desktop resource limits
