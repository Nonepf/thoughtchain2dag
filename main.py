"""
main.py —— GraphBuilderPipeline
读取 math500_qwen_ans_raw.jsonl，对每条推理文本：
  1. 用 ReasoningSegmenter  切分 → nodes
  2. 用 ReasoningLinker     批量推理所有 (i,j) 对 → edges
  3. 写出 ans_graphs.jsonl

用法：
    python main.py
    python main.py --input my_data.jsonl --output my_graphs.jsonl --batch_size 64
"""

import argparse
import json
import torch
from tqdm import tqdm

from model import ReasoningSegmenter, ReasoningLinker


class GraphBuilderPipeline:
    def __init__(
        self,
        seg_model:  str = "segmenter_finetuned",
        link_model: str = "linker_finetuned",
    ):
        print("Loading Segmenter ...")
        self.segmenter = ReasoningSegmenter(seg_model)
        print("Loading Linker ...")
        self.linker    = ReasoningLinker(link_model)

    def build_graph_from_text(
        self,
        text:       str,
        batch_size: int   = 32,
        threshold:  float = 0.5,
    ) -> dict:
        """
        输入推理原文，输出因果图 JSON。

        Args:
            text:       原始推理文本
            batch_size: Linker 批推理大小，8GB 显存建议 32-64
            threshold:  边的置信度阈值

        Returns:
            {
              "nodes": [{"id": int, "content": str}, ...],
              "edges": [{"source": int, "target": int, "weight": float}, ...]
            }
        """
        # 1. 分句 → 节点
        nodes_text = self.segmenter.segment(text)
        if not nodes_text:
            return {"nodes": [], "edges": []}

        n = len(nodes_text)

        # 2. 生成所有有序对 (i -> j, j > i)
        pairs        = []
        pair_indices = []
        for j in range(n):
            for i in range(j):
                pairs.append((nodes_text[i], nodes_text[j]))
                pair_indices.append((i, j))

        if not pairs:
            return {
                "nodes": [{"id": i, "content": c} for i, c in enumerate(nodes_text)],
                "edges": [],
            }

        # 3. 批量推理
        all_scores = []
        with torch.inference_mode():
            for k in range(0, len(pairs), batch_size):
                batch_pairs = pairs[k : k + batch_size]
                scores = self.linker.batch_check_dependency(batch_pairs)
                all_scores.extend(scores)

        # 4. 按阈值筛选边
        edges = []
        for (i, j), score in zip(pair_indices, all_scores):
            if score > threshold:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": round(score, 4),
                })

        # 5. 组装结果
        return {
            "nodes": [{"id": i, "content": c} for i, c in enumerate(nodes_text)],
            "edges": edges,
        }


# ── 主处理循环 ────────────────────────────────────────────────────────────────

def process_data(
    input_file:  str,
    output_file: str,
    batch_size:  int   = 32,
    threshold:   float = 0.5,
    seg_model:   str   = "segmenter_finetuned",
    link_model:  str   = "linker_finetuned",
    max_samples: int   = None,  # type: ignore
):
    pipeline = GraphBuilderPipeline(seg_model=seg_model, link_model=link_model)

    # 断点续跑：记录已处理的 id
    done_ids = set()
    import os
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} samples already done.")

    with open(input_file,  "r", encoding="utf-8") as fin, \
         open(output_file, "a", encoding="utf-8") as fout:

        lines = fin.readlines()
        if max_samples:
            lines = lines[:max_samples]

        for line in tqdm(lines, desc="Building Graphs"):
            try:
                data = json.loads(line)
            except Exception:
                continue

            sample_id = str(data.get("id", ""))
            if sample_id in done_ids:
                continue

            raw_text = data.get("model_output", "").strip()
            if not raw_text:
                continue

            # 构建图
            graph_data = pipeline.build_graph_from_text(
                raw_text, batch_size=batch_size, threshold=threshold
            )

            result = {
                "id":      data.get("id"),
                "problem": data.get("problem", ""),
                "graph":   graph_data,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nDone! Graphs saved to {output_file}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build reasoning graphs from JSONL")
    parser.add_argument("--input",      default="math500_qwen_ans_raw.jsonl")
    parser.add_argument("--output",     default="ans_graphs.jsonl")
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--seg_model",  default="segmenter_finetuned")
    parser.add_argument("--link_model", default="linker_finetuned")
    parser.add_argument("--max_samples",type=int,   default=None)
    args = parser.parse_args()

    process_data(
        input_file  = args.input,
        output_file = args.output,
        batch_size  = args.batch_size,
        threshold   = args.threshold,
        seg_model   = args.seg_model,
        link_model  = args.link_model,
        max_samples = args.max_samples,
    )
