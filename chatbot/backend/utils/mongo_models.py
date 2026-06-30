"""
MongoDB data models using dataclasses.
Defines schemas for all collections in the RAG chatbot.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
import uuid


@dataclass
class MongoDocument:
    """
    Schema for documents collection.
    Stores metadata about uploaded files.
    """
    document_id: str
    filename: str
    content_hash: str
    mimetype: str
    size_bytes: int
    uploaded_by: Optional[str] = None
    uploaded_at: Optional[datetime] = None
    status: str = "processed"  # processed, processing, failed
    token_count: int = 0
    vector_dim: Optional[int] = None
    num_chunks: int = 0
    
    def __post_init__(self):
        if self.uploaded_at is None:
            self.uploaded_at = datetime.utcnow()
    
    def to_dict(self):
        """Convert to MongoDB document format with _id."""
        data = asdict(self)
        data["_id"] = self.document_id
        return data


@dataclass
class MongoChunk:
    """
    Schema for chunks collection.
    Stores text chunks with embeddings for vector search.
    
    Hierarchy fields:
    - section_title: Current section title (from document structure)
    - section_path: Full path from root to current section
    - hierarchy_level: Depth in document hierarchy (0=root, 1=chapter, etc.)
    - parent_section: Title of parent section
    - document_title: Document's main title
    """
    chunk_id: str
    document_id: str
    text: str
    embedding: List[float]  # 768-dim vector for Vietnamese embedding
    page: Optional[int] = None
    end_page: Optional[int] = None
    section: Optional[str] = None  # Legacy field, prefer section_title
    order_index: int = 0
    text_len: int = 0
    # Document hierarchy fields (new)
    section_title: Optional[str] = None
    section_path: Optional[List[str]] = None
    hierarchy_level: int = 0
    parent_section: Optional[str] = None
    document_title: Optional[str] = None
    # General metadata
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if self.section_path is None:
            self.section_path = []
        if not self.text_len:
            self.text_len = len(self.text)
        # Extract hierarchy from metadata if provided there
        if self.metadata:
            if self.section_title is None:
                self.section_title = self.metadata.get("section_title")
            if not self.section_path:
                self.section_path = self.metadata.get("section_path", [])
            if self.hierarchy_level == 0:
                self.hierarchy_level = self.metadata.get("hierarchy_level", 0)
            if self.parent_section is None:
                self.parent_section = self.metadata.get("parent_section")
            if self.document_title is None:
                self.document_title = self.metadata.get("document_title")
    
    def to_dict(self):
        """Convert to MongoDB document format with _id."""
        data = asdict(self)
        data["_id"] = self.chunk_id
        return data
    
    def get_full_context(self) -> str:
        """
        Get a context string including hierarchy information.
        Useful for providing richer context to LLM.
        """
        context_parts = []
        if self.document_title:
            context_parts.append(f"Document: {self.document_title}")
        if self.section_path:
            context_parts.append(f"Section: {' > '.join(self.section_path)}")
        elif self.section_title:
            context_parts.append(f"Section: {self.section_title}")
        if self.page is not None:
            context_parts.append(f"Page: {self.page}")
        
        if context_parts:
            return f"[{', '.join(context_parts)}]\n{self.text}"
        return self.text


@dataclass
class MongoQueryLog:
    """
    Schema for query_logs collection.
    Logs all user questions and system responses.
    """
    question_id: str
    question: str
    correlation_id: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    reranked_chunk_ids: Optional[List[str]] = None
    final_answer: Optional[str] = None
    latency_ms: Optional[int] = None
    ts: Optional[datetime] = None
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    top_k: int = 5
    
    def __post_init__(self):
        if self.ts is None:
            self.ts = datetime.utcnow()
    
    def to_dict(self):
        """Convert to MongoDB document format with _id."""
        data = asdict(self)
        data["_id"] = self.question_id
        return data


@dataclass
class MongoRetrievalRun:
    """
    Schema for retrieval_runs collection.
    Stores per-query retrieval evaluation metrics.
    """
    run_id: str
    query_id: str
    gold_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    k: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    ts: Optional[datetime] = None
    
    def __post_init__(self):
        if self.ts is None:
            self.ts = datetime.utcnow()
    
    def to_dict(self):
        """Convert to MongoDB document format."""
        data = asdict(self)
        # Use combination of run_id and query_id for unique _id
        data["_id"] = f"{self.run_id}::{self.query_id}::k{self.k}"
        return data


@dataclass
class MongoGenerationRun:
    """
    Schema for generation_runs collection.
    Stores per-query answer generation evaluation metrics.
    """
    run_id: str
    query_id: str
    final_answer: str
    gold_answer: Optional[str] = None
    exact_match: float = 0.0
    rouge_l: float = 0.0
    bleu: float = 0.0
    citation_ok: bool = False
    ts: Optional[datetime] = None
    
    def __post_init__(self):
        if self.ts is None:
            self.ts = datetime.utcnow()
    
    def to_dict(self):
        """Convert to MongoDB document format."""
        data = asdict(self)
        data["_id"] = f"{self.run_id}::{self.query_id}"
        return data


@dataclass
class MongoEvalRun:
    """
    Schema for eval_runs collection (aggregate results).
    Stores summary statistics for entire evaluation runs.
    """
    run_id: str
    dataset_hash: str
    model: str
    backend: str  # "mongodb" or "qdrant"
    created_at: Optional[datetime] = None
    aggregates: Optional[Dict[str, Any]] = None
    notes: str = ""
    num_queries: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.aggregates is None:
            self.aggregates = {}
    
    def to_dict(self):
        """Convert to MongoDB document format with _id."""
        data = asdict(self)
        data["_id"] = self.run_id
        return data


# Helper functions for generating IDs
def generate_document_id(content_hash: str) -> str:
    """Generate document ID from content hash."""
    return content_hash


def generate_chunk_id(document_id: str, order_index: int) -> str:
    """Generate chunk ID from document ID and order."""
    return f"{document_id}:{order_index}"


def generate_question_id() -> str:
    """Generate unique question ID."""
    return str(uuid.uuid4())


def generate_run_id() -> str:
    """Generate unique run ID for evaluations."""
    return f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
