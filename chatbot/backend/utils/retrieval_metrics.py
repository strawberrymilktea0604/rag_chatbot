import os
import csv
from collections import defaultdict
import math
import numpy as np
import pandas as pd


def _parse_list_field(s):
    """Parse a list-like field which may be pipe-separated or comma-separated."""
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return list(s)
    s = str(s).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
    else:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def recall_at_k(relevant_items, retrieved_items, k=5):
    """Compute Recall@k."""
    retrieved_at_k = retrieved_items[:k]
    relevant_retrieved = len(set(relevant_items) & set(retrieved_at_k))
    return relevant_retrieved / len(relevant_items) if relevant_items else 0.0


def mrr_at_k(relevant_items, retrieved_items, k=5):
    """Compute MRR@k."""
    for rank, item in enumerate(retrieved_items[:k], start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_items, retrieved_items, k=5):
    """Compute nDCG@k."""
    # relevance list for retrieved items
    rels = [1 if item in relevant_items else 0 for item in retrieved_items[:k]]

    def dcg(relevance_list):
        return sum((rel / math.log2(idx + 2)) for idx, rel in enumerate(relevance_list))

    ideal_rels = sorted([1] * len(relevant_items) + [0] * max(0, k - len(relevant_items)), reverse=True)[:k]
    idcg = dcg(ideal_rels)
    actual = dcg(rels)
    return actual / idcg if idcg > 0 else 0.0


class RetrievalEvaluator:
    @staticmethod
    def evaluate(qa_gold_path: str, retrieval_run_path: str):
        """
        Evaluate retrieval performance.
        
        Args:
            qa_gold_path: Path to gold standard CSV with columns:
                - question_id
                - document_id  
                - gold_support_chunk_ids (pipe-separated)
            retrieval_run_path: Path to retrieval results CSV
        """
        try:
            gold_df = pd.read_csv(qa_gold_path)
            run_df = pd.read_csv(retrieval_run_path)
            
            # FIX: Rename column to match expected format
            if 'gold_support_chunk_ids' in gold_df.columns and 'relevant_chunk_ids' not in gold_df.columns:
                gold_df['relevant_chunk_ids'] = gold_df['gold_support_chunk_ids']
            
            # Validate required columns
            required_gold = ['question_id', 'relevant_chunk_ids']
            required_run = ['question_id', 'retrieved_chunk_ids']
            
            for col in required_gold:
                if col not in gold_df.columns:
                    raise ValueError(f"Missing required column '{col}' in gold standard")
            
            for col in required_run:
                if col not in run_df.columns:
                    raise ValueError(f"Missing required column '{col}' in retrieval run")
            
            qid_column = 'question_id'
            rid_column = 'retrieved_chunk_ids'
            gold_relevant_column = 'relevant_chunk_ids'
            
            # Merge retrieval results with gold standard
            merged_df = run_df.merge(gold_df, on=qid_column, suffixes=('_run', '_gold'))
            
            # Calculate metrics
            metrics = {}
            for k in [5, 10]:
                # Top-k retrieved
                merged_df[f'top_{k}'] = merged_df[rid_column].apply(lambda x: _parse_list_field(x)[:k])
                
                # Recall@k
                merged_df[f'recall@{k}'] = merged_df.apply(
                    lambda row: recall_at_k(
                        _parse_list_field(row[gold_relevant_column]),
                        row[f'top_{k}'],
                        k
                    ),
                    axis=1
                )
                
                # MRR@k
                merged_df[f'mrr@{k}'] = merged_df.apply(
                    lambda row: mrr_at_k(
                        _parse_list_field(row[gold_relevant_column]),
                        row[f'top_{k}'],
                        k
                    ),
                    axis=1
                )
                
                # nDCG@k
                merged_df[f'ndcg@{k}'] = merged_df.apply(
                    lambda row: ndcg_at_k(
                        _parse_list_field(row[gold_relevant_column]),
                        row[f'top_{k}'],
                        k
                    ),
                    axis=1
                )
                
                # Aggregate metrics
                metrics[f'recall@{k}'] = {
                    'mean': float(merged_df[f'recall@{k}'].mean()),
                    'std': float(merged_df[f'recall@{k}'].std())
                }
                metrics[f'mrr@{k}'] = {
                    'mean': float(merged_df[f'mrr@{k}'].mean()),
                    'std': float(merged_df[f'mrr@{k}'].std())
                }
                metrics[f'ndcg@{k}'] = {
                    'mean': float(merged_df[f'ndcg@{k}'].mean()),
                    'std': float(merged_df[f'ndcg@{k}'].std())
                }
            
            # Return in expected format
            return {
                'summary': metrics,
                'per_question': merged_df.to_dict('records')
            }
        
        except Exception as e:
            print(f"Error evaluating retrieval: {e}")
            import traceback
            traceback.print_exc()
            return None
