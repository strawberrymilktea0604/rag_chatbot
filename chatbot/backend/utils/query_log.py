import csv
import os
from datetime import datetime
from pathlib import Path
import json

class QueryLogger:
    def __init__(self, base_path="backend/media"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.retrieval_log = self.base_path / "retrieval_run.csv"
        self.generation_log = self.base_path / "generation_run.csv"
        self.citation_log = self.base_path / "citation_run.csv"
        
        # Initialize CSV files with headers if they don't exist
        self._init_retrieval_log()
        self._init_generation_log()
        self._init_citation_log()
    
    def _init_retrieval_log(self):
        if not self.retrieval_log.exists():
            with open(self.retrieval_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'question_id', 'document_id', 'retrieved_chunk_ids', 
                    'retrieved_scores', 'top_k', 'retrieval_time_ms'
                ])
                writer.writeheader()
    
    def _init_generation_log(self):
        if not self.generation_log.exists():
            with open(self.generation_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'question_id', 'document_id', 'question_text', 
                    'final_answer_text', 'num_tokens', 'generation_time_ms'
                ])
                writer.writeheader()
    
    def _init_citation_log(self):
        if not self.citation_log.exists():
            with open(self.citation_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'question_id', 'chunk_id', 'document_id', 
                    'page_number', 'cited_in_answer'
                ])
                writer.writeheader()
    
    def log_retrieval(self, question_id, document_id, retrieved_chunk_ids, 
                      retrieved_scores, top_k=5, retrieval_time_ms=None):
        """Log retrieval results with timing"""
        with open(self.retrieval_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'question_id', 'document_id', 'retrieved_chunk_ids', 
                'retrieved_scores', 'top_k', 'retrieval_time_ms'
            ])
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'question_id': question_id,
                'document_id': document_id or '',
                'retrieved_chunk_ids': '|'.join(str(cid) for cid in retrieved_chunk_ids),
                'retrieved_scores': '|'.join(f"{score:.4f}" for score in retrieved_scores),
                'top_k': top_k,
                'retrieval_time_ms': retrieval_time_ms or ''
            })
    
    def log_generation(self, question_id, document_id, question_text, final_answer_text, 
                      num_tokens=None, generation_time_ms=None):
        """Log generation results with timing and token count"""
        with open(self.generation_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'question_id', 'document_id', 'question_text', 
                'final_answer_text', 'num_tokens', 'generation_time_ms'
            ])
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'question_id': question_id,
                'document_id': document_id or '',
                'question_text': question_text,
                'final_answer_text': final_answer_text,
                'num_tokens': num_tokens or '',
                'generation_time_ms': generation_time_ms or ''
            })
    
    def log_citation(self, question_id, chunk_id, document_id, page_number=None, 
                    cited_in_answer=False):
        """Log citation references for faithfulness evaluation"""
        with open(self.citation_log, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'question_id', 'chunk_id', 'document_id', 
                'page_number', 'cited_in_answer'
            ])
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'question_id': question_id,
                'chunk_id': chunk_id,
                'document_id': document_id,
                'page_number': page_number or '',
                'cited_in_answer': int(cited_in_answer)
            })