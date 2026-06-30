"""
Split QA CSV into one CSV per document_id with only the question_text column.

Usage:
    python backend\scripts\split_questions_by_doc.py \
        --input backend\qa_gold.csv \
        --outdir backend\backend\media\questions_by_doc

If --input is not provided the script will try several common locations
inside the repository (cwd and the backend/media path).
"""
import argparse
import csv
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_qa_file(candidate: Path | None) -> Path:
    # If user provided a path, use it if it exists
    if candidate:
        candidate = candidate.expanduser()
        if candidate.is_file():
            return candidate
        # if a relative path was provided, try resolving relative to cwd
        cand_resolved = Path.cwd() / candidate
        if cand_resolved.is_file():
            return cand_resolved

    # common locations to try (running from backend/ or repo root)
    script_dir = Path(__file__).resolve().parent
    repo_backend = script_dir  # script is in backend/scripts
    repo_root = script_dir.parent

    candidates = [
        Path.cwd() / "qa_gold.csv",
        Path.cwd() / "backend" / "qa_gold.csv",
        Path.cwd() / "backend" / "backend" / "media" / "qa_gold.csv",
        repo_backend / "qa_gold.csv",
        repo_backend / "backend" / "media" / "qa_gold.csv",
        repo_root / "backend" / "qa_gold.csv",
        repo_root / "backend" / "backend" / "media" / "qa_gold.csv",
    ]

    for p in candidates:
        if p.is_file():
            return p

    raise FileNotFoundError(
        "qa_gold.csv not found. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def split_questions(input_csv: Path, outdir: Path):
    input_csv = input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, tuple[csv.DictWriter, object]] = {}
    counts: dict[str, int] = {}

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV is empty or malformed")
        if "document_id" not in reader.fieldnames or "question_text" not in reader.fieldnames:
            raise ValueError("Input CSV must contain 'document_id' and 'question_text' columns")

        for row in reader:
            doc_id = (row.get("document_id") or "").strip() or "unknown"
            question = (row.get("question_text") or "").strip()

            if not question:
                continue

            if doc_id not in writers:
                out_path = outdir / f"{doc_id}.csv"
                out_f = out_path.open("w", encoding="utf-8", newline="")
                writer = csv.DictWriter(out_f, fieldnames=["question_text"])
                writer.writeheader()
                writers[doc_id] = (writer, out_f)
                counts[doc_id] = 0

            writer, _ = writers[doc_id]
            writer.writerow({"question_text": question})
            counts[doc_id] += 1

    # Close files
    for writer, out_f in writers.values():
        try:
            out_f.close()
        except Exception:
            logger.exception("Error closing file for a document")

    logger.info("Split completed. Files written: %d", len(counts))
    for doc_id, cnt in sorted(counts.items()):
        logger.info("  %s -> %d questions", doc_id, cnt)


def main():
    p = argparse.ArgumentParser(description="Split qa_gold.csv into per-document CSVs with questions only")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Path to the input QA CSV (optional). If omitted the script will search common locations."
    )
    p.add_argument(
        "--outdir",
        "-o",
        type=Path,
        default=Path("backend/backend/media/questions_by_doc"),
        help="Directory to write per-document CSVs"
    )
    args = p.parse_args()

    try:
        input_path = find_qa_file(args.input)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(2)

    logger.info("Using input file: %s", input_path)
    split_questions(input_path, args.outdir)


if __name__ == "__main__":
    main()