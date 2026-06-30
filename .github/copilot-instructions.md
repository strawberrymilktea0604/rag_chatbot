# RAG Chatbot - Coding Agent Instructions

## Project Overview

This is a production-ready **Retrieval-Augmented Generation (RAG) chatbot** that allows users to upload documents (PDF, DOCX, TXT) and ask questions about their content. The system combines document processing, vector similarity search, reranking, and LLM-based answer generation with proper citations.

**Key Features:**
- Document upload and processing with chunking
- Vector embeddings using Sentence Transformers
- Qdrant vector database for semantic search  
- CrossEncoder reranking for improved relevance
- Google Gemini API for answer generation
- Comprehensive evaluation metrics and testing framework
- Vietnamese language support

## Agent Workflow Guidelines

**Focus on Implementation:** Complete tasks directly through code changes, terminal commands, and testing. Do not create markdown documentation files unless specifically requested by the user.

**Task Completion Approach:**
1. Analyze the request and gather necessary context
2. Implement changes directly in the relevant code files
3. Test the implementation immediately
4. Provide brief status updates via console output or logs
5. Move to the next task without generating summary documents

## Tech Stack

### Backend (Django 5.2.5)
- **Framework**: Django with Django REST Framework
- **Database**: SQLite (development), configurable for PostgreSQL  
- **Vector DB**: Qdrant (local server at `localhost:6333`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Reranking**: CrossEncoder `ms-marco-MiniLM-L-6-v2`  
- **LLM**: Google Gemini (`gemini-2.5-flash` or `gemini-1.5-pro`)
- **File Processing**: PyMuPDF (PDF), python-docx (DOCX)
- **Environment**: python-dotenv for configuration

### Frontend (Next.js 15.5.0)
- **Framework**: Next.js with TypeScript
- **Styling**: Tailwind CSS v4
- **React**: v19.1.0
- **Linting**: ESLint with Next.js config

## Project Structure

```
├── backend/                    # Django backend
│   ├── .env                   # Environment configuration (API keys, DB config)
│   ├── manage.py              # Django management
│   ├── db.sqlite3            # Development database
│   ├── recreate_collection.py # Qdrant collection setup script
│   ├── run_eval.py           # Evaluation runner script
│   ├── verify_qdrant.py      # Qdrant health check
│   │
│   ├── backend/              # Django project settings
│   │   ├── settings.py       # Main configuration
│   │   └── utils/            # Core utilities
│   │       ├── embeddings.py           # Embedding models (HF + Gemini)
│   │       ├── qdrant_client.py        # Vector DB client
│   │       ├── retrieval_metrics.py    # Evaluation metrics
│   │       └── query_log.py            # Request logging
│   │
│   ├── api/                  # Main API app
│   │   ├── views/            # API endpoints
│   │   │   ├── file_upload_view.py     # Document upload
│   │   │   ├── ask_view.py             # Question answering
│   │   │   ├── init_qdrant_view.py     # Collection initialization
│   │   │   └── delete_qdrant_collection_view.py
│   │   ├── tools/            # Processing utilities
│   │   │   ├── chunking.py             # Text chunking logic
│   │   │   └── chunk_eval.py           # Quality assessment
│   │   └── services/         # Business logic
│   │
│   ├── documents/            # Document model Django app
│   ├── search/              # Search-related Django app  
│   └── media/               # File uploads and evaluation data
│
└── frontend/               # Next.js frontend
    ├── src/pages/         # Next.js pages
    ├── public/           # Static assets
    └── package.json      # Dependencies and scripts
```

## Development Setup

### Prerequisites
- **Python 3.8+** with pip
- **Node.js 18+** with npm
- **Qdrant server** running on `localhost:6333`
- **Google Gemini API key**

### Backend Setup
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install django djangorestframework python-dotenv qdrant-client sentence-transformers google-generativeai langdetect transformers torch PyMuPDF python-docx numpy scikit-learn
```

**Dependencies**: No requirements.txt exists. Install packages as needed based on import errors.

**Critical**: Ensure `.env` exists with proper configuration:
```properties
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=test_collection
DEBUG=1
```

**Database migration**: Always run after model changes:
```bash
python manage.py migrate
```

**Qdrant setup**: Initialize vector collection:
```bash
python recreate_collection.py
```

### Frontend Setup  
```bash
cd frontend
npm install
npm run dev  # Starts on http://localhost:3000
```

### Running Services
1. **Qdrant**: Start local server (see Qdrant docs)
2. **Backend**: `python manage.py runserver` (port 8000)
3. **Frontend**: `npm run dev` (port 3000)

## API Endpoints

**Base URL**: `http://localhost:8000/api/`

- `POST /ask/` - Question answering
  - Body: `{"question": "text", "document_id": "optional", "top_k": 5}`
  - Returns: answer with citations and performance metrics

- `POST /upload/` - Document upload  
  - Body: multipart form with file
  - Returns: document metadata and processing status

- `POST /init-qdrant/` - Initialize vector collection
- `POST /delete-qdrant/` - Delete collections  
- `GET /check-qdrant/` - Collection health check
- `GET /health/` - System health check (all components)

## Coding Guidelines

### Django Backend
- **Views**: Use class-based views, always add proper error handling
- **Logging**: Use `logging.getLogger(__name__)` for structured logs
- **Environment**: All secrets in `.env`, never hardcode API keys
- **Database**: Use Django ORM, add migrations for schema changes
- **Validation**: Validate all inputs, use DRF serializers where appropriate

### Code Style
- **Imports**: Group Django, third-party, local imports separately  
- **Naming**: Use descriptive variable names (`question_embedding` not `emb`)
- **Documentation**: Add docstrings for complex functions
- **Error Handling**: Catch specific exceptions, provide meaningful error messages

### Vector Operations
- **Embeddings**: Always use 384-dimension vectors (sentence-transformers)
- **Search**: Include `score_threshold=0.5` to filter low-quality matches
- **Collection**: Named vector field is `"default"`, distance is `COSINE`

## Common Operations

### Adding New Document Type Support
1. Update `file_upload_view.py` file processing logic
2. Add text extraction method in chunking utilities  
3. Update file validation in upload view
4. Test with sample files

### Modifying Chunking Strategy
1. Edit `api/tools/chunking.py`
2. Run `recreate_collection.py` to reset vector DB
3. Re-upload test documents
4. Validate with `chunk_eval.py`

### Evaluation Workflow
1. Create gold standard: `qa_mapping.py` 
2. Generate test queries through API calls
3. Run evaluation: `python run_eval.py`
4. Review metrics in `backend/media/evaluation_results.json`
5. Make improvements directly based on results

### Debugging Vector Search
1. Check Qdrant health: `python verify_qdrant.py`
2. Inspect collection: `GET /check-qdrant/`
3. Verify embeddings: Test with `embeddings.py` directly
4. Review query logs in `media/` directory

## Important Notes

### Performance Considerations
- **Chunking**: 400-500 tokens per chunk with 50-token overlap
- **Search**: Use `top_k * 2` for initial retrieval, then rerank to final `top_k`
- **Caching**: Embeddings are computed on-the-fly (consider caching for production)

### File Handling
- **Uploads**: Stored in `media/uploads/`, deduplicated by file hash
- **Processing**: Files are processed synchronously (add async for large files)
- **Cleanup**: Temporary files are cleaned up automatically

### Security
- **CSRF**: Views use `@csrf_exempt` for API calls
- **Validation**: All file uploads are validated by type and size
- **API Keys**: Never log API keys, use environment variables

### Troubleshooting
- **Import Errors**: Ensure Django setup with `os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")`
- **Qdrant Connection**: Verify server is running and accessible
- **Missing Dependencies**: Check if all packages from import statements are installed
- **File Processing**: Check file permissions and available disk space

### Testing
- **Unit Tests**: Use Django's TestCase for model/view testing
- **Integration**: Test full RAG pipeline with sample documents  
- **Evaluation**: Run `run_eval.py` regularly to monitor performance
- **Manual Testing**: Use provided test documents in media/ directory

This codebase prioritizes accuracy, performance monitoring, and maintainability. Always test changes with the evaluation framework and check Qdrant collection health after modifications.