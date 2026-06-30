import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

def normalize_text(text):
    """
    Normalize text for consistent embedding generation.
    IMPORTANT: Do NOT lowercase - the embedding model is case-sensitive!
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text with consistent unicode and whitespace
    """
    if not text:
        return ""
    
    # Normalize unicode to NFC form (important for Vietnamese)
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace but preserve case
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_query(question):
    """
    Preprocess query consistently with document chunking.
    Does NOT lowercase because the embedding model is case-sensitive.
    
    Args:
        question: Raw question string from user
        
    Returns:
        Preprocessed question ready for embedding
    """
    if not question:
        return ""
    
    # Basic normalization (NO lowercasing - model is case-sensitive!)
    question = normalize_text(question)
    
    # Remove common Vietnamese question prefixes that don't add semantic value
    # But keep the core question content
    prefixes_to_remove = [
        r'^xin hỏi\s+',
        r'^cho biết\s+',
        r'^cho tôi biết\s+',
        r'^giúp tôi\s+',
        r'^hãy cho biết\s+',
        r'^có thể cho biết\s+',
    ]
    
    for prefix_pattern in prefixes_to_remove:
        question = re.sub(prefix_pattern, '', question, flags=re.IGNORECASE).strip()
    
    # Remove trailing punctuation (but keep internal punctuation)
    question = re.sub(r'[?!.]+$', '', question).strip()
    
    # Remove leading/trailing quotes if present
    question = question.strip('"\'""''')
    
    return question


def preprocess_document_text(text):
    """
    Preprocess document text before chunking.
    Ensures consistency with query preprocessing.
    
    Args:
        text: Raw document text
        
    Returns:
        Preprocessed text ready for chunking
    """
    if not text:
        return ""
    
    # Apply same normalization as queries
    text = normalize_text(text)
    
    # Remove excessive newlines but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove page numbers and headers (common patterns)
    text = re.sub(r'^\s*Page\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    return text
