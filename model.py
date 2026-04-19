"""
model.py  —— ReasoningSegmenter & ReasoningLinker
支持：
  - 直接传入微调后的模型路径（finetune 完成后使用）
  - 默认回退到 microsoft/deberta-v3-small（cold start / 调试）
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)


class ReasoningSegmenter:
    """
    将一段推理文本切分为逻辑语义单元（节点）。

    模型：Token 分类，每个 Token 预测 0（继续）或 1（新段起始）。
    """

    def __init__(self, model_name: str = "segmenter_finetuned"):
        # 如果微调模型目录不存在，回退到 base 模型（不带标注知识，仅结构对齐）
        import os
        if not os.path.isdir(model_name):
            print(f"[Segmenter] '{model_name}' not found, falling back to deberta-v3-small.")
            model_name = "microsoft/deberta-v3-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
        self.model.to("cuda")
        self.model.eval()

    def segment(self, text: str, max_length: int = 512) -> list[str]:
        """
        输入原始推理文本，返回分割后的 segment 列表。
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        input_ids = encoding["input_ids"][0]

        # 移到 GPU
        inputs_gpu = {k: v.to("cuda") for k, v in encoding.items()}

        with torch.inference_mode():
            logits = self.model(**inputs_gpu).logits[0]          # (seq_len, 2)
            preds  = torch.argmax(logits, dim=-1).cpu().tolist() # 0 or 1

        # 根据预测的切分点，将字符串切割成 segments
        # pred=1 的 token 所在字符位置就是新 segment 的起始字符偏移
        cut_char_positions = set()
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:   # 特殊 token，跳过
                continue
            if preds[tok_idx] == 1:
                cut_char_positions.add(tok_start)

        # 按切分位置切割原始文本
        segments = []
        prev = 0
        for pos in sorted(cut_char_positions):
            chunk = text[prev:pos].strip()
            if chunk:
                segments.append(chunk)
            prev = pos
        tail = text[prev:].strip()
        if tail:
            segments.append(tail)

        return segments if segments else [text]


class ReasoningLinker:
    """
    判断两个 segment 之间是否存在逻辑依赖关系。

    模型：句对分类（Cross-Encoder），输出 [无依赖, 有依赖] 的概率。
    """

    def __init__(self, model_name: str = "linker_finetuned"):
        import os
        if not os.path.isdir(model_name):
            print(f"[Linker] '{model_name}' not found, falling back to deberta-v3-small.")
            model_name = "microsoft/deberta-v3-small"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
        self.model.to("cuda")
        self.model.eval()

    def check_dependency(self, step_a: str, step_b: str, threshold: float = 0.5) -> bool:
        """判断 step_b 是否依赖 step_a，返回 bool。"""
        score = self.batch_check_dependency([(step_a, step_b)])[0]
        return score > threshold

    def batch_check_dependency(
        self, pairs: list[tuple[str, str]], max_length: int = 128
    ) -> list[float]:
        """
        批量推理，返回每对 (step_a, step_b) 中"依赖"的置信度列表（float）。
        """
        if not pairs:
            return []

        with torch.inference_mode():
            with torch.amp.autocast("cuda"):  # type: ignore
                inputs = self.tokenizer(
                    [p[0] for p in pairs],
                    [p[1] for p in pairs],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to("cuda")

                logits = self.model(**inputs).logits        # (N, 2)
                probs  = torch.softmax(logits, dim=-1)
                scores = probs[:, 1].cpu().tolist()         # 依赖概率

        del inputs
        return scores
