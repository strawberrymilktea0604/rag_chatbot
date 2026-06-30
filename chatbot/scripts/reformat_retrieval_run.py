"""Reformat retrieval_run.csv so it aligns with qa_gold chunk IDs."""
import argparse
import csv
import logging
import os
import shutil
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover - optional dependency at runtime
    QdrantClient = None  # type: ignore[assignment]

LOG = logging.getLogger(__name__)


def normalise(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def load_question_map(qa_gold_path: Path, generation_run_path: Path) -> Dict[str, str]:
    with qa_gold_path.open(encoding="utf-8") as fh:
        qa_rows = {normalise(row["question_text"]): row["question_id"] for row in csv.DictReader(fh)}

    uuid_to_gold: Dict[str, str] = {}
    with generation_run_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = normalise(row["question_text"])
            gold_id = qa_rows.get(key)
            if gold_id:
                uuid_to_gold[row["question_id"]] = gold_id
    return uuid_to_gold


def read_retrieval_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows


def collect_chunk_ids(rows: Sequence[Dict[str, str]]) -> Set[str]:
    ids: Set[str] = set()
    for row in rows:
        raw = row.get("retrieved_chunk_ids", "")
        if not raw:
            continue
        ids.update(filter(None, (cid.strip() for cid in raw.split("|"))))
    return ids


def build_chunk_map(
    chunks_all_path: Path,
    needed_ids: Set[str],
    collection_name: Optional[str],
    batch_size: int,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    if chunks_all_path.exists():
        with chunks_all_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = reader.fieldnames or []
            point_key = next(
                (
                    col
                    for col in columns
                    if any(token in col.lower() for token in ("uuid", "guid", "vector_id", "point_id"))
                ),
                None,
            )
            chunk_key = next((col for col in columns if col.lower() in {"chunk_id", "chunkid"}), None)
            if point_key and chunk_key:
                for row in reader:
                    point_id = row.get(point_key)
                    chunk_id = row.get(chunk_key)
                    if point_id and chunk_id:
                        mapping[point_id] = chunk_id

    missing = needed_ids - mapping.keys()
    if not missing:
        return mapping

    LOG.info("Found %d chunk IDs in CSV; fetching %d from Qdrant", len(mapping), len(missing))
    mapping.update(fetch_from_qdrant(missing, collection_name, batch_size))

    remaining = needed_ids - mapping.keys()
    if remaining:
        sample = ", ".join(islice((str(x) for x in remaining), 5))
        raise RuntimeError(
            f"Unable to map {len(remaining)} chunk IDs; first few: {sample}. "
            "Verify they exist in Qdrant or provide a mapping CSV."
        )
    return mapping


def extract_chunk_id(payload: Optional[Dict[str, object]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    queue: deque[Dict[str, object]] = deque([payload])
    while queue:
        current = queue.popleft()
        for key, value in current.items():
            key_norm = key.lower().replace("-", "_")
            if key_norm in {"chunk_id", "chunkid"} and isinstance(value, str):
                return value
            if isinstance(value, dict):
                queue.append(value)
    return None


def fetch_from_qdrant(chunk_ids: Set[str], collection_name: Optional[str], batch_size: int) -> Dict[str, str]:
    if not chunk_ids:
        return {}
    if QdrantClient is None:
        raise ImportError("qdrant-client is required to fetch chunk IDs; install it or provide a CSV mapping.")

    client = connect_qdrant()
    collection = resolve_collection(client, collection_name)

    missing = set(chunk_ids)
    mapping: Dict[str, str] = {}
    offset: Optional[str] = None

    while missing:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            limit=batch_size,
            offset=offset,
        )
        if not points:
            break

        for point in points:
            point_id = str(point.id)
            if point_id not in missing:
                continue
            chunk_id = extract_chunk_id(point.payload)
            if chunk_id:
                mapping[point_id] = chunk_id
                missing.remove(point_id)
            else:
                LOG.warning("Point %s payload lacks chunk_id field", point_id)

        if offset is None:
            break

    if missing:
        LOG.warning("No Qdrant points returned for IDs: %s", ", ".join(islice((str(x) for x in missing), 5)))
    return mapping


def connect_qdrant() -> QdrantClient:
    path = os.getenv("QDRANT_PATH")
    api_key = os.getenv("QDRANT_API_KEY")
    if path:
        return QdrantClient(path=path)
    url = os.getenv("QDRANT_URL") or "http://localhost:6333"
    return QdrantClient(url=url, api_key=api_key)


def resolve_collection(client: QdrantClient, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    collections = client.get_collections().collections
    if len(collections) == 1:
        return collections[0].name
    available = ", ".join(col.name for col in collections)
    raise ValueError(
        "Multiple Qdrant collections detected. Please rerun with --collection <name>. "
        f"Available collections: {available}"
    )


def remap_chunk_list(raw_ids: str, mapper: Dict[str, str]) -> str:
    if not raw_ids:
        return raw_ids
    outputs: List[str] = []
    for cid in raw_ids.split("|"):
        cid = cid.strip()
        outputs.append(mapper.get(cid, cid))
    return "|".join(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=Path("backend/media/retrieval_run.csv"))
    parser.add_argument("--qa-gold", type=Path, default=Path("backend/media/qa_gold.csv"))
    parser.add_argument("--generation-run", type=Path, default=Path("backend/media/generation_run.csv"))
    parser.add_argument("--chunks-all", type=Path, default=Path("chunks_all.csv"))
    parser.add_argument("--output", type=Path, default=Path("backend/media/retrieval_run_aligned.csv"))
    parser.add_argument("--collection", type=str, default=os.getenv("QDRANT_COLLECTION"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    if args.retrieval.exists():
        backup = args.retrieval.with_suffix(args.retrieval.suffix + ".bak")
        shutil.copy2(args.retrieval, backup)
        LOG.info("Backup created at %s", backup)

    question_map = load_question_map(args.qa_gold, args.generation_run)
    header, rows = read_retrieval_rows(args.retrieval)
    if not rows:
        raise ValueError(f"No rows found in {args.retrieval}")

    needed_chunk_ids = collect_chunk_ids(rows)
    chunk_map = build_chunk_map(args.chunks_all, needed_chunk_ids, args.collection, args.batch_size)

    with args.output.open("w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            row["question_id"] = question_map.get(row["question_id"], row["question_id"])
            row["retrieved_chunk_ids"] = remap_chunk_list(row.get("retrieved_chunk_ids", ""), chunk_map)
            writer.writerow(row)

    LOG.info("Aligned retrieval run written to %s", args.output)


if __name__ == "__main__":
    main()