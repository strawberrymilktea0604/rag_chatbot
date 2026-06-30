# tools/export_chunks.py
import csv, os, sys
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION  = os.getenv("QDRANT_COLLECTION", "test_collection")

# add an explicit, stable field order
CSV_FIELDNAMES = ["chunk_id", "document_id", "order_index", "page", "end_page", "section", "text_len", "text"]

def scroll_all(client, flt: Optional[Filter]=None, limit=256):
    next_off = None
    while True:
        points, next_off = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=flt,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=next_off
        )
        for p in points:
            yield p
        if next_off is None:
            break

def export_csv(out_path: str, document_id: Optional[str]):
    client = QdrantClient(url=QDRANT_URL)
    flt = None
    if document_id:
        flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])

    rows = []
    for p in scroll_all(client, flt):
        d = p.payload or {}
        rows.append({
            "chunk_id": d.get("chunk_id") or f"{d.get('document_id')}:{d.get('order_index')}",
            "document_id": str(d.get("document_id") or ""),
            "order_index": str(d.get("order_index") if d.get("order_index") is not None else ""),
            "page": d.get("page") or d.get("start_page"),
            "end_page": d.get("end_page"),
            "section": d.get("section"),
            "text_len": len((d.get("text") or "")),
            "text": (d.get("text") or "").replace("\n"," ").strip()
        })

    # stable ordering
    rows.sort(key=lambda r: (str(r["document_id"]), int(r["order_index"] or 0)))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} rows)")

def write_chunks_csv(chunks, out_path: str):
    """Write a list of chunk dicts to CSV.

    Expected chunk shape is the output of clean_and_chunk_text:
    {"chunk_id","text","document_id","page","section","order_index","token_len"}
    """
    import csv
    import os

    rows = []
    for c in chunks or []:
        rows.append({
            "chunk_id": c.get("chunk_id"),
            "document_id": str(c.get("document_id") or ""),
            "order_index": str(c.get("order_index") if c.get("order_index") is not None else ""),
            "page": c.get("page"),
            "end_page": c.get("end_page"),
            "section": c.get("section"),
            "text_len": len((c.get("text") or "")),
            "text": (c.get("text") or "").replace("\n", " ").strip(),
        })

    # stable ordering
    try:
        rows.sort(key=lambda r: (str(r.get("document_id")), int(r.get("order_index") or 0)))
    except Exception:
        # fallback: no-op if ordering fails
        pass

    # ensure directory exists
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} rows)")

if __name__ == "__main__":
    # Usage:
    #   python tools/export_chunks.py out.csv                # all docs
    #   python tools/export_chunks.py out.csv DOC_ID         # single doc
    if len(sys.argv) < 2:
        print("Usage: python tools/export_chunks.py <out.csv> [document_id]")
        sys.exit(1)
    out = sys.argv[1]
    doc = sys.argv[2] if len(sys.argv) > 2 else None
    export_csv(out, doc)
