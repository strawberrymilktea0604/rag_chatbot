"""Generate `qa_gold` MongoDB collection (final-exam difficulty) from `gold_candidates`.

This script:
- Reads candidate chunks from MongoDB `gold_candidates` by run_tag
- Picks a balanced subset across clusters (default: 1 per cluster, rank 1)
- Calls Gemini to generate one hard exam question per chunk
- Enforces strict JSON output schema and writes to MongoDB `qa_gold` collection

Run (PowerShell):
  cd chatbot
  python generate_qa_gold_from_candidates.py --run-tag final_exam_v1 --limit 50

Env:
  GEMINI_API_KEY (required)
  Optional: GEMINI_MODEL (default: gemini-2.5-flash)
  Mongo: MONGO_URI, MONGO_DB_NAME

Output schema per item (in qa_gold collection):
  {
    "_id": "<run_tag>::<chunk_id>",
    "run_tag": "final_exam_v1",
    "question": "...",
    "answers": ["..."],
    "type": "Fact|Scenario|Reasoning",
    "relevant_docs": ["<chunk_id>"],
    "chunk_id": "...",
    "created_at": "2025-...",
    "model": "gemini-2.5-flash"
  }
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from google import generativeai

from backend.utils.mongo_client import get_mongo_db


SYSTEM_INSTRUCTION = (
	"Bạn là một Giảng viên Công nghệ thông tin khó tính tại một trường đại học. "
	"Nhiệm vụ của bạn là tạo ra các câu hỏi cho \"Đề thi cuối kỳ\" dựa trên tài liệu được cung cấp.\n\n"
	"Mục tiêu: Đánh giá khả năng HIỂU và VẬN DỤNG của sinh viên, chứ không phải khả năng học vẹt.\n\n"
	"Quy tắc bắt buộc:\n"
	"1. Đa dạng hóa: Không bắt đầu câu hỏi bằng \"Theo đoạn văn...\", \"Văn bản đề cập...\". Hãy hỏi như một người thật.\n"
	"2. Độ khó: 30% Lý thuyết (Fact), 40% Tình huống (Scenario/Troubleshooting), 30% Suy luận/So sánh (Reasoning).\n"
	"   - Lý thuyết (Fact): Định nghĩa, khái niệm, nhưng dùng từ ngữ khác văn bản gốc. Không học vẹt.\n"
	"   - Tình huống (Scenario): \"Máy bị lỗi X, dùng lệnh gì?\", \"Trường hợp Y, cấu hình thế nào?\".\n"
	"   - Suy luận (Reasoning): Giải thích vì sao, so sánh ưu nhược, đánh đổi giữa lựa chọn.\n"
	"3. Ngôn ngữ: Tiếng Việt tự nhiên, chuyên ngành, ngắn gọn. Không máy móc.\n"
	"4. Độ dài: Câu hỏi < 150 chữ; đáp án < 200 chữ.\n"
	"5. Luôn chỉ sử dụng thông tin từ CHUNK_TEXT. Không bịa.\n"
	"Quy tắc định dạng (BẮT BUỘC):\n"
    "- Chỉ trả về RAW JSON Array hợp lệ.\n"
    "- KHÔNG dùng Markdown block (```json ... ```).\n"
    "- KHÔNG có lời dẫn hay kết thúc."
)

USER_PROMPT_TEMPLATE = (
	"Dưới đây là nội dung một slide bài giảng (Context):\n"
	"\"\"\"{CHUNK_TEXT}\"\"\"\n\n"
	"Hãy tạo ra 2-3 câu hỏi-đáp án (QA pairs) chất lượng cao từ nội dung trên.\n\n"
	"Định dạng JSON:\n"
	"[\n"
	"  {\n"
	"    \"question\": \"Câu hỏi ở đây?\",\n"
	"    \"answers\": [\"Câu trả lời ngắn gọn, súc tích (1-2 câu).\"],\n"
	"    \"type\": \"Fact\" hoặc \"Scenario\" hoặc \"Reasoning\",\n"
	"    \"relevant_docs\": [\"CHUNK_ID\"]\n"
	"  }\n"
	"]\n\n"
	"Hướng dẫn:\n"
	"- Tạo tối thiểu 1 câu hỏi, tối đa 3 câu hỏi.\n"
	"- Cân bằng các loại: ưu tiên không lặp loại trong 1 nhóm.\n"
	"- Ví dụ mẫu:\n"
	"  * KÉM: \"Lệnh ls dùng để làm gì?\" (Quá dễ, học vẹt)\n"
	"  * TỐT: \"Tôi muốn xem cả các file ẩn (.config) trong thư mục hiện tại, cần thêm tham số nào?\"\n\n"
	"QUAN TRỌNG: Chỉ trả về JSON array. KHÔNG viết lời chào, giới thiệu, hay bất kỳ text nào khác ngoài JSON.\n"
	"Lưu ý:\n"
    "- Trường 'relevant_docs' PHẢI chứa chính xác ID tài liệu đã cung cấp ở trên.\n"
    "- Tuyệt đối KHÔNG thêm bất kỳ ký tự nào ngoài JSON list."
)


def _load_env() -> None:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	load_dotenv(dotenv_path=os.path.join(script_dir, ".env"), override=False)
	load_dotenv(override=False)


def _clean_json_text(s: str) -> str:
	"""Clean markdown and language identifiers from JSON text."""
	s = s.strip()
	# Handle markdown fence anywhere in the text (model may add intro text before it)
	if "```json" in s:
		idx = s.find("```json")
		s = s[idx + 7:].strip()  # Skip past ```json
		if "```" in s:
			s = s[:s.find("```")].strip()  # Remove closing fence
	elif "```" in s:
		parts = s.split("```")
		if len(parts) >= 2:
			s = parts[1].strip()
	if s.startswith("json"):
		s = s[4:].strip()
	return s


def _try_parse_array(s: str) -> Dict[str, Any]:
	"""Try to extract first element from JSON array."""
	start_arr = s.find("[")
	if start_arr == -1:
		raise ValueError("No array found")
	end = s.rfind("]")  # Use rfind to get the LAST ] for nested structures
	if end == -1 or end <= start_arr:
		raise ValueError("Array not closed")
	try:
		arr = json.loads(s[start_arr : end + 1])
		if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
			return arr[0]
	except (json.JSONDecodeError, IndexError):
		pass
	raise ValueError("Failed to extract from array")


def _try_parse_object(s: str) -> Dict[str, Any]:
	"""Try to extract JSON object from string."""
	start_obj = s.find("{")
	if start_obj == -1:
		raise ValueError("No object found")
	end = s.rfind("}")
	if end <= start_obj:
		raise ValueError("Object not closed properly")
	try:
		return json.loads(s[start_obj : end + 1])
	except json.JSONDecodeError:
		pass
	raise ValueError("Failed to parse object")


def _extract_json_object(text: str) -> Dict[str, Any]:
	"""Extract and parse the first JSON object from model output."""
	if text is None:
		raise ValueError("Empty model output")

	s = _clean_json_text(text)

	start_arr = s.find("[")
	start_obj = s.find("{")

	# If array comes first, try extracting from it
	if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
		try:
			return _try_parse_array(s)
		except ValueError:
			pass

	# Otherwise try extracting object
	try:
		return _try_parse_object(s)
	except ValueError:
		pass

	raise ValueError(f"No valid JSON found in output: {s[:300]}")


def _build_prompt(chunk_text: str) -> str:
	"""Build a prompt combining system instruction + user prompt with chunk context."""
	user_prompt = USER_PROMPT_TEMPLATE.replace("{CHUNK_TEXT}", chunk_text)
	return SYSTEM_INSTRUCTION + "\n\n" + user_prompt


def _call_gemini_json(
	model,
	prompt: str,
	temperature: float,
	max_output_tokens: int,
) -> Dict[str, Any]:
	resp = model.generate_content(
		prompt,
		generation_config=generativeai.types.GenerationConfig(
			temperature=temperature,
			max_output_tokens=max_output_tokens,
			candidate_count=1,
		),
	)
	text = getattr(resp, "text", None) or str(resp)
	try:
		return _extract_json_object(text)
	except Exception as e:
		return {
			"question": "",
			"answers": [""],
			"_error": f"json_parse_failed: {e}",
			"_raw": text[:2000],
		}


def _normalize_item(obj: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
	question = str(obj.get("question") or "").strip()
	answers = obj.get("answers")
	if not isinstance(answers, list):
		answers = [str(answers)] if answers else []
	answers = [str(a).strip() for a in answers if str(a).strip()]
	if not answers:
		answers = ["(Cần chỉnh sửa thủ công: mô hình không trả lời đúng định dạng)"]

	q_type = str(obj.get("type") or "Fact").strip()
	if q_type not in ["Fact", "Scenario", "Reasoning"]:
		q_type = "Fact"

	obj_out = {
		"question": question,
		"answers": answers[:3],
		"type": q_type,
		"relevant_docs": [chunk_id],
	}
	for k in ["_error", "_raw"]:
		if k in obj:
			obj_out[k] = obj[k]
	return obj_out


def _pick_candidates(run_tag: str, per_cluster: int, limit: int) -> List[Dict[str, Any]]:
	db = get_mongo_db()
	coll = db["gold_candidates"]

	cursor = coll.find({"run_tag": run_tag})
	# Sort to ensure stable ordering: cluster then rank
	items = sorted(cursor, key=lambda d: (d.get("cluster_id", 0), d.get("rank_in_cluster", 999)))
	if not items:
		raise RuntimeError(f"No gold_candidates found for run_tag={run_tag!r}")

	by_cluster: Dict[int, List[Dict[str, Any]]] = {}
	for d in items:
		c = int(d.get("cluster_id", 0))
		by_cluster.setdefault(c, []).append(d)

	selected: List[Dict[str, Any]] = []
	for c in sorted(by_cluster.keys()):
		cluster_items = sorted(by_cluster[c], key=lambda x: x.get("rank_in_cluster", 999))
		selected.extend(cluster_items[:per_cluster])
		if len(selected) >= limit:
			break

	return selected[:limit]


def _pick_chunks_by_ids(chunk_ids: List[str]) -> List[Dict[str, Any]]:
	"""Load specific chunks from MongoDB chunks collection by chunk_id."""
	db = get_mongo_db()
	coll = db["chunks"]
	
	selected: List[Dict[str, Any]] = []
	for chunk_id in chunk_ids:
		doc = coll.find_one({"_id": chunk_id})
		if doc:
			selected.append(doc)
		else:
			print(f"Warning: chunk_id {chunk_id!r} not found in chunks collection")
	
	if not selected:
		raise RuntimeError(f"No chunks found for provided chunk_ids")
	
	return selected


def _load_existing(run_tag: str, resume: bool) -> Set[str]:
	"""Load already-generated chunk_ids from MongoDB qa_gold collection."""
	if not resume:
		return set()
	db = get_mongo_db()
	coll = db["qa_gold"]
	skip_chunk_ids: set = set()
	existing = coll.find({"run_tag": run_tag})
	for item in existing:
		for cid in item.get("relevant_docs", []) or []:
			skip_chunk_ids.add(cid)
	return skip_chunk_ids


def _generate_for_candidates(
	model,
	candidates: List[Dict[str, Any]],
	run_tag: str,
	model_name: str,
	skip_chunk_ids: set,
	delay_s: float,
	temperature: float,
	max_output_tokens: int,
) -> int:
	"""Generate and insert QA pairs into MongoDB qa_gold collection."""
	db = get_mongo_db()
	coll = db["qa_gold"]
	count_inserted = 0

	for idx, cand in enumerate(candidates, start=1):
		chunk_id = str(cand.get("chunk_id") or "")
		chunk_text = str(cand.get("text") or "")
		if not chunk_id or not chunk_text:
			continue
		if chunk_id in skip_chunk_ids:
			print(f"[{idx}/{len(candidates)}] skip (already exists): {chunk_id}")
			continue

		prompt = _build_prompt(chunk_text=chunk_text)
		print(f"[{idx}/{len(candidates)}] generating for chunk_id={chunk_id} ...")

		obj = _call_gemini_json(
			model=model,
			prompt=prompt,
			temperature=temperature,
			max_output_tokens=max_output_tokens,
		)
		normalized = _normalize_item(obj=obj, chunk_id=chunk_id)

		# Build MongoDB document
		doc = {
			"_id": f"{run_tag}::{chunk_id}",
			"run_tag": run_tag,
			"chunk_id": chunk_id,
			"question": normalized["question"],
			"answers": normalized["answers"],
			"type": normalized["type"],
			"relevant_docs": normalized["relevant_docs"],
			"created_at": datetime.now(timezone.utc),
			"model": model_name,
		}
		# Preserve error info if present
		if "_error" in normalized:
			doc["_error"] = normalized["_error"]
		if "_raw" in normalized:
			doc["_raw"] = normalized["_raw"]

		# Insert or replace in MongoDB
		coll.replace_one({"_id": doc["_id"]}, doc, upsert=True)
		count_inserted += 1
		print("  -> inserted to MongoDB qa_gold collection")

		time.sleep(max(0.0, float(delay_s)))

	return count_inserted


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate qa_gold MongoDB collection from gold_candidates or direct chunk_ids using Gemini")
	parser.add_argument("--run-tag", type=str, default=None, help="gold_candidates.run_tag to read from (mutually exclusive with --chunk-ids)")
	parser.add_argument("--chunk-ids", type=str, nargs="+", default=None, help="Specific chunk_ids to generate Q&A for (space-separated)")
	parser.add_argument("--limit", type=int, default=50, help="Number of Q/A items to generate (30-50 recommended)")
	parser.add_argument("--per-cluster", type=int, default=1, help="How many chunks to use per cluster (only with --run-tag)")
	# rank 1 is always preferred (we sort by rank_in_cluster)
	parser.add_argument("--delay", type=float, default=1.5, help="Delay (seconds) between Gemini calls")
	parser.add_argument("--temperature", type=float, default=0.6, help="Gemini temperature")
	parser.add_argument("--max-output-tokens", type=int, default=4096, help="Gemini max output tokens")
	parser.add_argument("--resume", action="store_true", help="Skip chunk_ids already in qa_gold collection")
	args = parser.parse_args()

	# Validate mutually exclusive arguments
	if args.run_tag and args.chunk_ids:
		parser.error("Cannot use both --run-tag and --chunk-ids. Choose one mode.")
	
	if not args.run_tag and not args.chunk_ids:
		parser.error("Must provide either --run-tag or --chunk-ids")

	_load_env()

	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		raise RuntimeError("GEMINI_API_KEY not found in environment")

	model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
	generativeai.configure(api_key=api_key)
	model = generativeai.GenerativeModel(model_name)

	# Determine run_tag for tracking
	if args.chunk_ids:
		run_tag = f"direct_chunks_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
		candidates = _pick_chunks_by_ids(args.chunk_ids)
		mode = "Direct chunk_ids"
	else:
		run_tag = args.run_tag
		skip_chunk_ids = _load_existing(run_tag=run_tag, resume=args.resume)
		candidates = _pick_candidates(
			run_tag=run_tag,
			per_cluster=max(1, args.per_cluster),
			limit=max(1, args.limit),
		)
		mode = f"gold_candidates (run_tag={run_tag})"

	started_at = datetime.now(timezone.utc).isoformat()

	print(f"Mode: {mode}")
	print(f"Model: {model_name}")
	print(f"Candidates selected: {len(candidates)}")
	print("Output: MongoDB qa_gold collection")

	count_inserted = _generate_for_candidates(
		model=model,
		candidates=candidates,
		run_tag=run_tag,
		model_name=model_name,
		skip_chunk_ids=set() if args.chunk_ids else skip_chunk_ids,
		delay_s=args.delay,
		temperature=args.temperature,
		max_output_tokens=args.max_output_tokens,
	)

	print(f"Done. Inserted {count_inserted} items into MongoDB. started_at={started_at}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
