"""
Step 1+2: 从 HuggingFace 下载 Math500，调用 SiliconFlow 上的
Qwen2.5-Math-7B-Instruct 逐题生成推理文本，保存为 JSONL。

依赖：
    pip install datasets openai tqdm
用法：
    export SILICONFLOW_API_KEY="your_key"
    python step1_run_math500.py
"""

import os
import json
import time
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

# ── 配置 ─────────────────────────────────────────────────────────────────────
API_KEY      = os.environ.get("SILICONFLOW_API_KEY", "YOUR_API_KEY")
BASE_URL     = "https://api.siliconflow.cn/v1"
MODEL        = "Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_FILE  = "qwen_ans_raw.jsonl"
MAX_SAMPLES  = None   # None 表示跑全量 500 题；调试时可设为 10
RETRY_LIMIT  = 3
RETRY_DELAY  = 5      # 秒

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Think step by step and show your full reasoning before giving the final answer."
)

# ── 客户端 ───────────────────────────────────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_model(problem: str) -> str:
    """调用模型，带简单重试逻辑。"""
    for attempt in range(RETRY_LIMIT):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": problem},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            return response.choices[0].message.content # type: ignore
        except Exception as e:
            print(f"  [Retry {attempt+1}/{RETRY_LIMIT}] Error: {e}")
            time.sleep(RETRY_DELAY)
    return ""   # 全部重试失败则返回空串


def main():
    print("Loading Math500 dataset ...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")['test']

    samples = list(dataset)
    if MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]

    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"  Resuming: {len(done_ids)} samples already done.")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for idx, sample in tqdm(enumerate(samples), total=len(samples), desc="Inference"):
            sample_id = str(idx)
            if sample_id in done_ids:
                continue

            problem = sample.get("problem", sample.get("question", "")) # type: ignore
            answer  = sample.get("solution", sample.get("answer", "")) # type: ignore

            model_output = call_model(problem)

            record = {
                "id":           sample_id,
                "problem":      problem,
                "gold_answer":  answer,
                "model_output": model_output,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nDone! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
