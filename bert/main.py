"""
main.py - GraphBuilderPipeline
Reads math500_qwen_ans_raw.jsonl, for each reasoning text:
  1. Use ReasoningSegmenter to split → nodes
  2. Use ReasoningLinker to batch infer all (i,j) pairs → edges
  3. Write to ans_graphs.jsonl

Usage:
    python main.py
    python main.py --input my_data.jsonl --output my_graphs.jsonl --batch_size 64
"""

import argparse
import json
import os
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
        Input reasoning text, output causal graph JSON.

        Args:
            text:       Original reasoning text
            batch_size: Linker batch inference size, 32-64 recommended for 8GB GPU memory
            threshold:  Edge confidence threshold

        Returns:
            {
              "nodes": [{"id": int, "content": str}, ...],
              "edges": [{"source": int, "target": int, "weight": float}, ...]
            }
        """
        # 1. Segment text → nodes
        nodes_text = self.segmenter.segment(text)
        if not nodes_text:
            return {"nodes": [], "edges": []}

        n = len(nodes_text)

        # 2. Generate all ordered pairs (i -> j, j > i)
        pairs = []
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

        # 3. Batch inference
        all_scores = []
        with torch.inference_mode():
            for k in range(0, len(pairs), batch_size):
                batch_pairs = pairs[k : k + batch_size]
                scores = self.linker.batch_check_dependency(batch_pairs)
                all_scores.extend(scores)

        # 4. Filter edges by threshold
        edges = []
        for (i, j), score in zip(pair_indices, all_scores):
            if score > threshold:
                edges.append({
                    "source": i,
                    "target": j,
                    "weight": round(score, 4),
                })

        # 5. Assemble result
        return {
            "nodes": [{"id": i, "content": c} for i, c in enumerate(nodes_text)],
            "edges": edges,
        }


# ── Main processing loop ─────────────────────────────────────────────────────

def process_data(
    input_file:  str,
    output_file: str,
    batch_size:  int   = 32,
    threshold:   float = 0.5,
    seg_model:   str   = "segmenter_finetuned",
    link_model:  str   = "linker_finetuned",
    max_samples: int   = None,
):
    pipeline = GraphBuilderPipeline(seg_model=seg_model, link_model=link_model)

    # Resume processing: record completed IDs
    done_ids = set()
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

            # Build graph
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


# ── Entry point ─────────────────────────────────────────────────────────────

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
