"""
Document extraction utilities for different file formats.
"""
import logging
import fitz  # PyMuPDF
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


def extract_from_pdf(file_path, document_id):
    """
    Extract text and metadata from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        document_id: Unique identifier for the document
        
    Returns:
        List of raw text blocks with metadata
    """
    raw_blocks = []
    try:
        with fitz.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text()
                if text.strip():
                    raw_blocks.append({
                        "text": text,
                        "page": page_number,
                        "document_id": document_id
                    })
        logger.info(f"Extracted {len(raw_blocks)} blocks from PDF: {document_id}")
    except Exception as e:
        logger.error(f"Error extracting PDF {document_id}: {str(e)}", exc_info=True)
        raise
    
    return raw_blocks


def extract_from_docx(file_path, document_id):
    """
    Extract text and metadata from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        document_id: Unique identifier for the document
        
    Returns:
        List of raw text blocks with metadata
    """
    raw_blocks = []
    try:
        doc = DocxDocument(file_path)
        for i, paragraph in enumerate(doc.paragraphs, start=1):
            if paragraph.text.strip():
                raw_blocks.append({
                    "text": paragraph.text,
                    "section": f"Paragraph {i}",
                    "document_id": document_id
                })
        logger.info(f"Extracted {len(raw_blocks)} paragraphs from DOCX: {document_id}")
    except Exception as e:
        logger.error(f"Error extracting DOCX {document_id}: {str(e)}", exc_info=True)
        raise
    
    return raw_blocks


def extract_from_txt(file_path, document_id):
    """
    Extract text and metadata from a TXT file.
    
    Args:
        file_path: Path to the TXT file
        document_id: Unique identifier for the document
        
    Returns:
        List of raw text blocks with metadata
    """
    raw_blocks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, start=1):
                if line.strip():
                    raw_blocks.append({
                        "text": line.strip(),
                        "section": f"Line {i}",
                        "document_id": document_id
                    })
        logger.info(f"Extracted {len(raw_blocks)} lines from TXT: {document_id}")
    except Exception as e:
        logger.error(f"Error extracting TXT {document_id}: {str(e)}", exc_info=True)
        raise
    
    return raw_blocks
