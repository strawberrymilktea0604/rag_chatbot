import csv
import numpy as np
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Try to import bert-score; if unavailable, skip BERTScore calculations
try:
    from bert_score import score as bertscore_score
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False


class GenerationEvaluator:
    """Batch evaluation of generation quality"""

    @staticmethod
    def normalize_text(text):
        """Normalize text by removing citations and extra whitespace."""
        if not text:
            return ""
        
        # Remove citations in format [cite: N], [id tr.N], [id page N], etc.
        text = re.sub(r'\[cite:\s*\d+\]', '', text)
        text = re.sub(r'\[[\w\-]+\s+(tr\.|page|p\.)\s*\d+\]', '', text)
        text = re.sub(r'\[[\w\-]+\]', '', text)
        
        # Remove markdown formatting (bold, italic)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def load_qa_gold(filepath):
        """Load gold standard answers"""
        qa_gold = {}
        if not Path(filepath).exists():
            return qa_gold
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row.get('question_id', '')).strip()
                qa_gold[qid] = {
                    'gold_answer': row.get('gold_answer', '') or '',
                    'gold_support_chunk_ids': [cid.strip() for cid in (row.get('gold_support_chunk_ids') or '').split('|') if cid.strip()],
                    'question_text': row.get('question_text', '') or ''
                }
        return qa_gold

    @staticmethod
    def load_generation_runs(filepath):
        """Load generated answers"""
        runs = {}
        if not Path(filepath).exists():
            return runs
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row.get('question_id', '')).strip()
                runs[qid] = {
                    'generated_answer': row.get('generated_answer', '') or '',
                    'final_answer_text': row.get('final_answer_text', '') or '',
                    'num_tokens': int(row.get('num_tokens', 0)) if row.get('num_tokens') else 0,
                    'generation_time_ms': float(row.get('generation_time_ms', 0)) if row.get('generation_time_ms') else 0.0
                }
        return runs

    @staticmethod
    def exact_match(gold_answer, generated_answer):
        """EM: normalized exact match (after normalization)"""
        gold = GenerationEvaluator.normalize_text(gold_answer or '').lower().strip()
        gen = GenerationEvaluator.normalize_text(generated_answer or '').lower().strip()
        return float(gold == gen)

    @staticmethod
    def token_f1(gold_answer, generated_answer):
        """F1 based on token overlap (after normalization)"""
        gold = GenerationEvaluator.normalize_text(gold_answer or '')
        gen = GenerationEvaluator.normalize_text(generated_answer or '')
        
        gold_tokens = set(gold.lower().split())
        gen_tokens = set(gen.lower().split())

        if not gold_tokens and not gen_tokens:
            return 1.0
        if not gold_tokens or not gen_tokens:
            return 0.0

        common = gold_tokens & gen_tokens
        precision = len(common) / len(gen_tokens) if gen_tokens else 0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def rouge_1(gold_answer, generated_answer):
        """ROUGE-1: unigram overlap"""
        gold_tokens = (gold_answer or '').lower().split()
        gen_tokens = (generated_answer or '').lower().split()

        if not gold_tokens or not gen_tokens:
            return 0.0

        common = len([t for t in gen_tokens if t in gold_tokens])
        precision = common / len(gen_tokens) if gen_tokens else 0
        recall = common / len(gold_tokens) if gold_tokens else 0

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def lcs_len(a, b):
        """Longest common subsequence length for ROUGE-L."""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    @staticmethod
    def rouge_l_f1(gold_answer, generated_answer):
        """ROUGE-L F1 using LCS over tokenized text."""
        gold_tokens = (gold_answer or '').split()
        gen_tokens = (generated_answer or '').split()
        if not gold_tokens or not gen_tokens:
            return 0.0
        lcslen = GenerationEvaluator.lcs_len(gold_tokens, gen_tokens)
        prec = lcslen / len(gen_tokens) if gen_tokens else 0
        rec = lcslen / len(gold_tokens) if gold_tokens else 0
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    @staticmethod
    def answer_length(generated_answer):
        """Answer length in characters"""
        return float(len(generated_answer or ''))

    @staticmethod
    def evaluate(qa_gold_path, generation_run_path):
        """
        Compute generation metrics for all questions.
        
        Supports two formats:
        1. CSV with 'generated_answer' column (from generation_run.csv)
        2. CSV with 'final_answer_text' column (from MongoDB logs)
        
        :return: Dict with summary and per_question metrics
        """
        qa_gold = GenerationEvaluator.load_qa_gold(qa_gold_path)
        gen_runs = GenerationEvaluator.load_generation_runs(generation_run_path)

        # Initialize metrics
        metrics = {
            'em': [],
            'token_f1': [],
            'rouge1': [],
            'rouge_l': [],
            'answer_length': [],
            'generation_time_ms': []
        }

        question_scores = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        for question_id, gold_data in qa_gold.items():
            if question_id not in gen_runs:
                continue

            gen_data = gen_runs[question_id]
            gold_answer = gold_data.get('gold_answer', '') or ''
            
            # Try both column names
            gen_answer = gen_data.get('generated_answer', '') or gen_data.get('final_answer_text', '') or ''
            
            if not gen_answer:
                continue

            # Normalize both answers
            gold_norm = GenerationEvaluator.normalize_text(gold_answer)
            gen_norm = GenerationEvaluator.normalize_text(gen_answer)

            # Compute metrics
            em = GenerationEvaluator.exact_match(gold_answer, gen_answer)
            f1 = GenerationEvaluator.token_f1(gold_answer, gen_answer)
            
            # ROUGE scores
            rouge_scores = scorer.score(gold_norm, gen_norm)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rouge_l = rouge_scores['rougeL'].fmeasure

            ans_len = float(len(gen_norm))
            gen_time = gen_data.get('generation_time_ms', 0.0)

            # Append to metrics lists
            metrics['em'].append(em)
            metrics['token_f1'].append(f1)
            metrics['rouge1'].append(rouge1)
            metrics['rouge_l'].append(rouge_l)
            metrics['answer_length'].append(ans_len)
            metrics['generation_time_ms'].append(gen_time)

            # Per-question scores
            question_scores.append({
                'question_id': question_id,
                'em': em,
                'token_f1': f1,
                'rouge1': rouge1,
                'rouge_l': rouge_l,
                'answer_length': ans_len,
                'generation_time_ms': gen_time
            })

        # Compute summary statistics
        summary = {}
        for metric_name, values in metrics.items():
            if not values:
                continue
            vals_arr = np.array(values, dtype=float)
            summary[metric_name] = {
                'mean': float(np.mean(vals_arr)),
                'std': float(np.std(vals_arr)),
                'min': float(np.min(vals_arr)),
                'max': float(np.max(vals_arr)),
                'count': int(len(vals_arr))
            }

        return {
            'summary': summary,
            'per_question': question_scores
        }