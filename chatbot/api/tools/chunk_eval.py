from collections import Counter
from statistics import mean
from typing import List, Dict

def eval_chunks(chunks: List[Dict], tokenizer, max_tokens=512):
    """
    chunks: list of dicts returned by clean_and_chunk_text
            required keys: "text", "document_id", "page", "section", "order_index"
    tokenizer: the SAME tokenizer you use for chunking (transformers)
    """
    token_lens = []
    over_512 = 0
    by_doc = {}

    # group chunks by document to compute overlaps correctly
    for c in chunks:
        by_doc.setdefault(c["document_id"], []).append(c)

    # sort within doc by order_index
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda x: x["order_index"])

    # compute lengths & overlaps
    overlaps = []  # token overlaps between consecutive chunks within the same doc

    for doc_id, cs in by_doc.items():
        prev_tokens = None
        for i, c in enumerate(cs):
            t = tokenizer.encode(c["text"], add_special_tokens=False)
            L = len(t)
            token_lens.append(L)
            if L > max_tokens:
                over_512 += 1
            if i > 0 and prev_tokens is not None:
                # measure suffix-prefix overlap length (exact token match)
                # walk back up to max_tokens tokens from prev tail
                max_check = min(len(prev_tokens), len(t), max_tokens)
                ov = 0
                # find the longest suffix of prev == prefix of current
                for k in range(max_check, 0, -1):
                    if prev_tokens[-k:] == t[:k]:
                        ov = k
                        break
                overlaps.append(ov)
            prev_tokens = t

    # page/section coverage
    pages = set()
    sections = set()
    for c in chunks:
        if c.get("page") is not None:
            pages.add((c["document_id"], c["page"]))
        if c.get("section") is not None:
            sections.add((c["document_id"], c["section"]))

    report = {
        "num_docs": len(by_doc),
        "num_chunks": len(chunks),
        "avg_tokens_per_chunk": round(mean(token_lens), 2) if token_lens else 0,
        "p95_tokens_per_chunk": (sorted(token_lens)[max(0, int(0.95*len(token_lens))-1)] if token_lens else 0),
        "max_tokens_in_chunk": max(token_lens) if token_lens else 0,
        "percent_over_512": round(100*over_512/len(token_lens), 2) if token_lens else 0.0,
        "avg_token_overlap": round(mean(overlaps), 2) if overlaps else 0,
        "p95_token_overlap": (sorted(overlaps)[max(0, int(0.95*len(overlaps))-1)] if overlaps else 0),
        "distinct_pages": len(pages),
        "distinct_sections": len(sections),
    }
    return report