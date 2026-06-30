"""
Document chunking with semantic boundary detection and hierarchy support.

Features:
- Semantic boundary detection (chapters, sections, paragraphs)
- Document structure/hierarchy propagation to chunks
- Vietnamese language support with structural markers
- Smart overlap for context preservation
"""

import nltk
import re
import os
import logging
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# HIERARCHY DATA STRUCTURES
# ============================================================================

def build_page_hierarchy_map(
    sections: List[Dict[str, Any]],
    total_pages: int
) -> Dict[int, Dict[str, Any]]:
    """
    Build a mapping from page numbers to hierarchy information.
    
    Args:
        sections: List of section dicts from extract_*_hierarchy() functions.
                 Each section has: title, level, page (0-indexed), page_end (optional)
        total_pages: Total number of pages in document
        
    Returns:
        Dict mapping page number (0-indexed) to hierarchy info:
        {
            page: {
                "section_title": "Current section title",
                "section_path": ["Part 1", "Chapter 1", "Section 1.1"],
                "hierarchy_level": 2,
                "parent_section": "Chapter 1",
                "document_structure_type": "hierarchical" | "flat"
            }
        }
    """
    if not sections:
        # No structure detected - return flat structure
        return {
            page: {
                "section_title": None,
                "section_path": [],
                "hierarchy_level": 0,
                "parent_section": None,
                "document_structure_type": "flat"
            }
            for page in range(total_pages)
        }
    
    # Sort sections by page number
    sorted_sections = sorted(sections, key=lambda x: (x.get("page", 0), x.get("level", 0)))
    
    # Build hierarchy stack for each page
    page_hierarchy: Dict[int, Dict[str, Any]] = {}
    
    # Track current hierarchy state (stack of sections at each level)
    hierarchy_stack: List[Dict[str, Any]] = []
    current_section_idx = 0
    
    for page in range(total_pages):
        # Update hierarchy stack with any sections that start on this page
        while current_section_idx < len(sorted_sections):
            section = sorted_sections[current_section_idx]
            section_page = section.get("page", 0)
            
            if section_page > page:
                break  # No more sections on this page
            
            if section_page == page:
                level = section.get("level", 1)
                
                # Pop sections from stack that are at same or lower level
                while hierarchy_stack and hierarchy_stack[-1].get("level", 0) >= level:
                    hierarchy_stack.pop()
                
                # Push this section
                hierarchy_stack.append(section)
            
            current_section_idx += 1
        
        # Build path from stack
        if hierarchy_stack:
            section_path = [s.get("title", "") for s in hierarchy_stack]
            current_section = hierarchy_stack[-1]
            parent_section = hierarchy_stack[-2].get("title") if len(hierarchy_stack) > 1 else None
            
            page_hierarchy[page] = {
                "section_title": current_section.get("title"),
                "section_path": section_path,
                "hierarchy_level": current_section.get("level", 1),
                "parent_section": parent_section,
                "document_structure_type": "hierarchical"
            }
        else:
            page_hierarchy[page] = {
                "section_title": None,
                "section_path": [],
                "hierarchy_level": 0,
                "parent_section": None,
                "document_structure_type": "hierarchical" if sections else "flat"
            }
    
    return page_hierarchy


def get_hierarchy_for_page(
    page: int,
    structure_map: Optional[Dict[int, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Get hierarchy information for a specific page.
    
    Args:
        page: Page number (0-indexed)
        structure_map: Pre-computed page-to-hierarchy mapping
        
    Returns:
        Hierarchy info dict
    """
    if structure_map is None or page not in structure_map:
        return {
            "section_title": None,
            "section_path": [],
            "hierarchy_level": 0,
            "parent_section": None,
            "document_structure_type": "flat"
        }
    
    return structure_map[page]


# ============================================================================
# SEMANTIC BOUNDARY DETECTION
# ============================================================================

def detect_semantic_boundaries(text):
    """
    Detect semantic boundaries in text based on structural markers.
    Returns a list of (position, boundary_type, weight) tuples.
    Higher weight = stronger boundary (better place to split).
    """
    boundaries = []
    
    # Vietnamese and English section markers with weights
    patterns = [
        # Strong boundaries (weight 1.0) - Major sections
        (r'\n\s*(?:Chương|CHƯƠNG|Chapter|CHAPTER)\s+\d+', 'chapter', 1.0),
        (r'\n\s*(?:Phần|PHẦN|Part|PART)\s+\d+', 'part', 1.0),
        (r'\n\s*(?:Mục|MỤC|Section|SECTION)\s+\d+', 'section_num', 0.95),
        
        # Medium-strong boundaries (weight 0.8) - Subsections
        (r'\n\s*\d+\.\d+\.?\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'numbered_subsection', 0.8),
        (r'\n\s*[IVX]+\.\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'roman_section', 0.8),
        (r'\n\s*[a-z]\)\s+', 'lettered_item', 0.6),
        
        # Medium boundaries (weight 0.6) - Paragraph breaks
        (r'\n\s*\n', 'paragraph_break', 0.6),
        (r'\n\s*[-•●○]\s+', 'bullet_point', 0.5),
        (r'\n\s*\d+\)\s+', 'numbered_list', 0.5),
        
        # Weak boundaries (weight 0.3) - Sentence-like breaks
        (r'[.!?]\s+(?=[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỬỮỰ])', 'sentence_end', 0.3),
    ]
    
    for pattern, boundary_type, weight in patterns:
        for match in re.finditer(pattern, text):
            boundaries.append((match.start(), boundary_type, weight))
    
    # Sort by position
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def find_best_split_point(text, target_pos, window=200, boundaries=None):
    """
    Find the best split point near target_pos considering semantic boundaries.
    Returns the best position to split the text.
    """
    if boundaries is None:
        boundaries = detect_semantic_boundaries(text)
    
    # Look for boundaries within the window
    best_pos = target_pos
    best_score = 0
    
    window_start = max(0, target_pos - window)
    window_end = min(len(text), target_pos + window)
    
    for pos, boundary_type, weight in boundaries:
        if window_start <= pos <= window_end:
            # Score based on weight and distance from target
            distance_penalty = abs(pos - target_pos) / window
            score = weight * (1 - distance_penalty * 0.5)
            
            if score > best_score:
                best_score = score
                best_pos = pos
    
    # If no good boundary found, fall back to sentence boundary
    if best_score < 0.2:
        # Look for sentence end near target
        sentence_end = text.rfind('. ', window_start, target_pos)
        if sentence_end > window_start:
            best_pos = sentence_end + 2
    
    return best_pos


# ============================================================================
# CHUNKING FUNCTIONS
# ============================================================================

def semantic_chunk_text(
    raw_blocks,
    tokenizer,
    target_tokens=400,
    max_tokens=512,
    overlap_tokens=100,
    structure_map: Optional[Dict[int, Dict[str, Any]]] = None,
    document_title: Optional[str] = None
):
    """
    Enhanced semantic chunking that respects document structure.
    - Detects section boundaries (chapters, paragraphs, lists)
    - Prefers splitting at semantic boundaries over arbitrary token limits
    - Preserves context with smart overlap
    - Propagates document hierarchy to chunks
    - Better for Vietnamese and multilingual documents
    
    Args:
        raw_blocks: List of raw text blocks with page info
        tokenizer: HuggingFace tokenizer for token counting
        target_tokens: Target tokens per chunk (soft limit)
        max_tokens: Maximum tokens per chunk (hard limit)
        overlap_tokens: Token overlap between consecutive chunks
        structure_map: Optional page-to-hierarchy mapping from build_page_hierarchy_map()
        document_title: Optional document title to include in chunk metadata
        
    Returns:
        List of chunk dictionaries with hierarchy metadata
    """
    def token_len(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=False))
    
    def make_chunk_id(document_id: str, order_index: int) -> str:
        return f"{document_id}:{order_index}"
    
    def get_hierarchy_info(page: Optional[int]) -> Dict[str, Any]:
        if page is None or structure_map is None:
            return {
                "section_title": None,
                "section_path": [],
                "hierarchy_level": 0,
                "parent_section": None,
            }
        return get_hierarchy_for_page(page, structure_map)
    
    chunks = []
    current_doc = None
    order_index = 0
    
    for block in raw_blocks:
        doc_id = block["document_id"]
        
        if doc_id != current_doc:
            current_doc = doc_id
            order_index = 0
        
        text = " ".join((block.get("text") or "").split())
        if not text:
            continue
        
        page = block.get("page")
        hierarchy = get_hierarchy_info(page)
        
        # Detect semantic boundaries in the block
        boundaries = detect_semantic_boundaries(text)
        
        # Base chunk metadata
        base_metadata = {
            "document_id": doc_id,
            "page": page,
            "end_page": block.get("end_page"),
            "section": block.get("section") or hierarchy.get("section_title"),
            "section_title": hierarchy.get("section_title"),
            "section_path": hierarchy.get("section_path", []),
            "hierarchy_level": hierarchy.get("hierarchy_level", 0),
            "parent_section": hierarchy.get("parent_section"),
            "document_title": document_title,
        }
        
        # If block is small enough, keep as single chunk
        block_tokens = token_len(text)
        if block_tokens <= max_tokens:
            chunks.append({
                "chunk_id": make_chunk_id(doc_id, order_index),
                "text": text,
                "order_index": order_index,
                "text_len": block_tokens,
                **base_metadata,
            })
            order_index += 1
            continue
        
        # Need to split - use semantic boundaries
        current_start = 0
        current_tokens = 0
        
        # Use sentences as base units
        sentences = nltk.sent_tokenize(text)
        sentence_positions = []
        pos = 0
        for sent in sentences:
            sent_start = text.find(sent, pos)
            sentence_positions.append((sent_start, sent_start + len(sent), sent))
            pos = sent_start + len(sent)
        
        current_sentences = []
        
        for sent_start, sent_end, sent in sentence_positions:
            sent_tokens = token_len(sent)
            
            # Check if adding this sentence exceeds target
            if current_tokens + sent_tokens > target_tokens and current_sentences:
                # Check if we're near a semantic boundary
                should_split = False
                
                for bpos, btype, bweight in boundaries:
                    if current_start <= bpos <= sent_start and bweight >= 0.5:
                        should_split = True
                        break
                
                # Split if we hit target or found a good boundary
                if should_split or current_tokens >= target_tokens:
                    chunk_text = " ".join(current_sentences)
                    chunks.append({
                        "chunk_id": make_chunk_id(doc_id, order_index),
                        "text": chunk_text,
                        "order_index": order_index,
                        "text_len": token_len(chunk_text),
                        **base_metadata,
                    })
                    order_index += 1
                    
                    # Calculate overlap - keep last few sentences for context
                    overlap_sents = []
                    overlap_tok = 0
                    for s in reversed(current_sentences):
                        s_tok = token_len(s)
                        if overlap_tok + s_tok > overlap_tokens:
                            break
                        overlap_sents.insert(0, s)
                        overlap_tok += s_tok
                    
                    current_sentences = overlap_sents + [sent]
                    current_tokens = overlap_tok + sent_tokens
                    current_start = sent_start
                    continue
            
            # Check if sentence alone exceeds max tokens (needs hard split)
            if sent_tokens > max_tokens:
                # First, flush current chunk if exists
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append({
                        "chunk_id": make_chunk_id(doc_id, order_index),
                        "text": chunk_text,
                        "order_index": order_index,
                        "text_len": token_len(chunk_text),
                        **base_metadata,
                    })
                    order_index += 1
                    current_sentences = []
                    current_tokens = 0
                
                # Hard split the long sentence
                sent_toks = tokenizer.encode(sent, add_special_tokens=False)
                i = 0
                while i < len(sent_toks):
                    end_i = min(i + max_tokens, len(sent_toks))
                    part_ids = sent_toks[i:end_i]
                    part_text = tokenizer.decode(part_ids, clean_up_tokenization_spaces=True)
                    
                    chunks.append({
                        "chunk_id": make_chunk_id(doc_id, order_index),
                        "text": part_text,
                        "order_index": order_index,
                        "text_len": len(part_ids),
                        **base_metadata,
                    })
                    order_index += 1
                    
                    # Overlap for next part
                    overlap_start = max(0, end_i - overlap_tokens)
                    i = overlap_start if end_i < len(sent_toks) else len(sent_toks)
                
                continue
            
            # Normal case: add sentence to current chunk
            current_sentences.append(sent)
            current_tokens += sent_tokens
        
        # Flush remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_id": make_chunk_id(doc_id, order_index),
                "text": chunk_text,
                "order_index": order_index,
                "text_len": token_len(chunk_text),
                **base_metadata,
            })
            order_index += 1
    
    return chunks


def clean_and_chunk_text(
    raw_blocks,
    tokenizer,
    max_tokens=512,
    overlap_tokens=50,
    structure_map: Optional[Dict[int, Dict[str, Any]]] = None,
    document_title: Optional[str] = None
):
    """
    Clean raw text blocks and create semantic chunks.
    
    Args:
        raw_blocks: List of dicts with 'text', 'page', 'document_id', etc.
        tokenizer: HuggingFace tokenizer for token counting
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        structure_map: Optional page-to-hierarchy mapping
        document_title: Optional document title for metadata
    
    Returns:
        List of chunk dictionaries with hierarchy metadata
    """
    chunks = []
    chunk_idx = 0
    
    for block in raw_blocks:
        text = block.get('text', '')
        
        # Skip empty or whitespace-only blocks
        if not text or not text.strip():
            logger.debug("Skipping empty block")
            continue
        
        # Clean the text
        text = clean_text(text)
        
        # Skip if text is empty after cleaning
        if not text or not text.strip():
            logger.debug("Skipping block after cleaning - empty")
            continue
        
        document_id = block.get('document_id', '')
        page = block.get('page')
        section = block.get('section')
        
        # Get hierarchy info for this page
        hierarchy = get_hierarchy_for_page(page, structure_map) if structure_map else {}
        
        # Tokenize to check length
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Tokenization failed for block: {e}")
            continue
        
        # Base metadata for chunks from this block
        base_metadata = {
            'document_id': document_id,
            'page': page,
            'section': section or hierarchy.get('section_title'),
            'section_title': hierarchy.get('section_title'),
            'section_path': hierarchy.get('section_path', []),
            'hierarchy_level': hierarchy.get('hierarchy_level', 0),
            'parent_section': hierarchy.get('parent_section'),
            'document_title': document_title,
        }
        
        # If text fits in one chunk, add it directly
        if len(tokens) <= max_tokens:
            if text.strip():  # Final check
                chunks.append({
                    'chunk_id': f"{document_id}:{chunk_idx}",
                    'text': text.strip(),
                    'order_index': chunk_idx,
                    **base_metadata,
                })
                chunk_idx += 1
        else:
            # Split into smaller chunks with overlap
            sub_chunks = split_text_into_chunks(text, tokenizer, max_tokens, overlap_tokens)
            for sub_text in sub_chunks:
                if sub_text and sub_text.strip():  # Only add non-empty chunks
                    chunks.append({
                        'chunk_id': f"{document_id}:{chunk_idx}",
                        'text': sub_text.strip(),
                        'order_index': chunk_idx,
                        **base_metadata,
                    })
                    chunk_idx += 1
    
    logger.info(f"Generated {len(chunks)} chunks from {len(raw_blocks)} raw blocks")
    return chunks


def clean_text(text):
    """Clean text by removing extra whitespace and normalizing."""
    if not text:
        return ''
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def split_text_into_chunks(text, tokenizer, max_tokens, overlap_tokens):
    """
    Split text into chunks based on token count with overlap.
    
    Args:
        text: Text to split
        tokenizer: Tokenizer for encoding
        max_tokens: Max tokens per chunk
        overlap_tokens: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s and s.strip()]
    
    if not sentences:
        return [text.strip()] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        if not sentence or not sentence.strip():
            continue
            
        try:
            sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        except Exception:
            # If tokenization fails, estimate tokens
            sentence_tokens = len(sentence.split())
        
        # If single sentence exceeds max, split by words
        if sentence_tokens > max_tokens:
            # Save current chunk first
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_tokens = 0
            
            for word in words:
                word_token_count = len(tokenizer.encode(word, add_special_tokens=False)) if word else 1
                if word_tokens + word_token_count > max_tokens and word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    if chunk_text.strip():
                        chunks.append(chunk_text.strip())
                    word_chunk = []
                    word_tokens = 0
                word_chunk.append(word)
                word_tokens += word_token_count
            
            if word_chunk:
                chunk_text = ' '.join(word_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
            continue
        
        # Check if adding this sentence exceeds max
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Start new chunk with overlap (take last few sentences)
            overlap_sentences = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
                if overlap_count + s_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_count
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Don't forget last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    return chunks
