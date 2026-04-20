"""
generate_training_data.py (Refactored)
Function: Call large model to generate training data for Segmenter and Linker
Optimization: Directly let large model output JSON with precise indices, removing complex string matching logic
"""
import os
import json
import re
import time
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

# ───────────────────────── Configuration ─────────────────────────
API_KEY       = os.environ.get("SILICONFLOW_API_KEY", "YOUR_API_KEY")
BASE_URL      = "https://api.siliconflow.cn/v1"
LLM_MODEL     = "Qwen/Qwen2.5-72B-Instruct"
INPUT_FILE    = "math500_qwen_ans_raw.jsonl"
SEG_OUT_FILE  = "segmenter_train.jsonl"
LINK_OUT_FILE = "linker_train.jsonl"
BASE_MODEL    = "microsoft/deberta-v3-small"
MAX_SAMPLES   = None
RETRY_LIMIT   = 3

client    = OpenAI(api_key=API_KEY, base_url=BASE_URL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ───────────────────────── Prompts ─────────────────────────
SEG_SYSTEM = """You are a mathematical reasoning parser. Split the input text into minimal logical steps.
RULES:
1. Preserve original text EXACTLY. Do NOT modify, rewrite, or normalize whitespace/LaTeX.
2. Return ONLY a valid JSON object.
3. Each segment MUST include exact character indices: "start" (inclusive) and "end" (exclusive).
4. Indices must be 0-based and strictly match the original string: original[start:end] == segment.text
5. No markdown, no explanations."""

SEG_USER = "Reasoning text:\n{text}"

LINK_SYSTEM = """You are a logical dependency analyzer. Given a list of reasoning steps (0-indexed), identify direct logical dependencies.
RULES:
1. Step B depends on Step A if B uses a result, value, or conclusion derived in A.
2. Return ONLY a valid JSON object.
3. Each dependency: {"source": int, "target": int} where target > source.
4. No markdown, no explanations."""

LINK_USER = "Segments (0 to {last_idx}):\n{segments_json}"

# ───────────────────────── Core utility functions ─────────────────────────
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call LLM with retry mechanism"""
    for attempt in range(RETRY_LIMIT):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0.0, max_tokens=4096
            )
            return resp.choices[0].message.content # type: ignore
        except Exception as e:
            print(f"  [API Retry {attempt+1}] {e}")
            time.sleep(5)
    return ""

def extract_json(raw: str):
    """Safely extract JSON, automatically remove Markdown code blocks and surrounding impurities"""
    if not raw: return None
    raw = re.sub(r"^`{1,3}json?\s*", "", raw.strip())
    raw = re.sub(r"\s*`{1,3}$", "", raw.strip())
    # Try to match first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except json.JSONDecodeError: return None
    return None

def build_segmenter_record(record: dict) -> dict | None:
    """Build segmenter training record"""
    text = record.get("model_output", "").strip()
    if not text: return None

    raw = call_llm(SEG_SYSTEM, SEG_USER.format(text=text))
    data = extract_json(raw)
    if not data or "segments" not in data:
        print(f"  ⚠️ id={record['id']}: Invalid segmenter response")
        return None

    segments = data["segments"]
    # Sort by start index and validate
    segments.sort(key=lambda x: x.get("start", 0))
    valid_segs = []
    cursor = 0
    for seg in segments:
        txt, s, e = seg.get("text"), int(seg.get("start", -1)), int(seg.get("end", -1))
        if 0 <= s < e <= len(text) and text[s:e] == txt:
            valid_segs.append({"text": txt, "start": s, "end": e})
            cursor = e
            continue
        # Fallback: use find to correct index偏差
        idx = text.find(txt, cursor)
        if idx != -1:
            valid_segs.append({"text": txt, "start": idx, "end": idx + len(txt)})
            cursor = idx + len(txt)

    if not valid_segs: return None

    # Build token-level labels
    enc = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    cut_positions = {seg["start"] for seg in valid_segs if seg["start"] > 0}

    labels = []
    for tok_s, tok_e in offsets:
        if tok_s == tok_e: labels.append(-100)
        elif tok_s in cut_positions: labels.append(1)
        else: labels.append(0)

    return {
        "id": record["id"],
        "text": text,
        "segments": [s["text"] for s in valid_segs],
        "tokens": tokenizer.convert_ids_to_tokens(enc["input_ids"]),
        "labels": labels
    }

def build_linker_records(record: dict, segments: list[str]) -> list[dict]:
    """Build linker training records"""
    if len(segments) < 2: return []

    raw = call_llm(LINK_SYSTEM, LINK_USER.format(last_idx=len(segments)-1, segments_json=json.dumps(segments, ensure_ascii=False)))
    data = extract_json(raw)
    deps = set()
    if data and "dependencies" in data:
        for d in data["dependencies"]:
            s, t = d.get("source"), d.get("target")
            if isinstance(s, int) and isinstance(t, int) and 0 <= s < t < len(segments):
                deps.add((s, t))

    # Construct positive and negative sample pairs
    pairs = []
    for j in range(len(segments)):
        for i in range(j):
            pairs.append({
                "id": record["id"],
                "seg_a": segments[i],
                "seg_b": segments[j],
                "label": 1 if (i, j) in deps else 0,
                "idx_a": i, "idx_b": j
            })
    return pairs

# ───────────────────────── Main process ─────────────────────────
def main():
    """Main function to generate training data"""
    # Track completed IDs for resuming
    done_seg = set()
    done_link = set()
    for path, done_set in [(SEG_OUT_FILE, done_seg), (LINK_OUT_FILE, done_link)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try: done_set.add(json.loads(line)["id"])
                    except: pass

    # Load input records
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        records = [json.loads(line) for line in fin if line.strip()]
    if MAX_SAMPLES: records = records[:MAX_SAMPLES]

    # Process records
    with open(SEG_OUT_FILE, "a", encoding="utf-8") as f_seg, \
         open(LINK_OUT_FILE, "a", encoding="utf-8") as f_link:

        for rec in tqdm(records, desc="Generating Labels"):
            rid = rec["id"]
            print(f"\n--- Processing id={rid} ---")

            # 1. Segmenter
            seg_data = None
            if rid not in done_seg:
                seg_data = build_segmenter_record(rec)
                if seg_data:
                    f_seg.write(json.dumps(seg_data, ensure_ascii=False) + "\n")
                    f_seg.flush()
                    segments = seg_data["segments"]
                else:
                    segments = []
            else:
                segments = []  # Read from file if needed in Linker phase

            # 2. Linker
            if rid not in done_link:
                if not segments and rid not in done_seg:
                    # If just skipped Segmenter, try to read saved segments from file
                    with open(SEG_OUT_FILE, "r", encoding="utf-8") as tmp:
                        for line in tmp:
                            d = json.loads(line)
                            if d["id"] == rid:
                                segments = d["segments"]; break

                if segments:
                    pairs = build_linker_records(rec, segments)
                    for p in pairs:
                        f_link.write(json.dumps(p, ensure_ascii=False) + "\n")
                    f_link.flush()
                    print(f"  ✅ Linker: {len(pairs)} pairs generated")

    print("\n🎉 Done! Data saved to segmenter_train.jsonl & linker_train.jsonl")

if __name__ == "__main__":
    main()