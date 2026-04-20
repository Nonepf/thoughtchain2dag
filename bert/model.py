"""
model.py - ReasoningSegmenter & ReasoningLinker
Supports:
  - Directly passing fine-tuned model paths (after finetuning)
  - Default fallback to microsoft/deberta-v3-small (cold start / debugging)
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)


def _load_model_with_fallback(model_name: str, default_model: str):
    """
    Load model with fallback to default if specified model doesn't exist
    """
    if not os.path.isdir(model_name):
        print(f"'{model_name}' not found, falling back to {default_model}.")
        return default_model
    return model_name


class ReasoningSegmenter:
    """
    Segment reasoning text into logical semantic units (nodes).

    Model: Token classification, each token predicts 0 (continue) or 1 (new segment start).
    """

    def __init__(self, model_name: str = "segmenter_finetuned"):
        # If fine-tuned model directory doesn't exist, fall back to base model
        model_name = _load_model_with_fallback(model_name, "microsoft/deberta-v3-small")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
        self.model.to("cuda")
        self.model.eval()

    def segment(self, text: str, max_length: int = 512) -> list[str]:
        """
        Input original reasoning text, return list of segmented units.
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        # Move to GPU
        inputs_gpu = {k: v.to("cuda") for k, v in encoding.items()}

        with torch.inference_mode():
            logits = self.model(**inputs_gpu).logits[0]          # (seq_len, 2)
            preds  = torch.argmax(logits, dim=-1).cpu().tolist() # 0 or 1

        # Identify cut positions based on predictions
        cut_char_positions = set()
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:   # Skip special tokens
                continue
            if preds[tok_idx] == 1:
                cut_char_positions.add(tok_start)

        # Segment text based on cut positions
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
    Determine if logical dependency exists between two segments.

    Model: Sentence pair classification (Cross-Encoder), output [no dependency, dependency] probabilities.
    """

    def __init__(self, model_name: str = "linker_finetuned"):
        # If fine-tuned model directory doesn't exist, fall back to base model
        model_name = _load_model_with_fallback(model_name, "microsoft/deberta-v3-small")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        )
        self.model.to("cuda")
        self.model.eval()

    def check_dependency(self, step_a: str, step_b: str, threshold: float = 0.5) -> bool:
        """Check if step_b depends on step_a, return bool."""
        score = self.batch_check_dependency([(step_a, step_b)])[0]
        return score > threshold

    def batch_check_dependency(
        self, pairs: list[tuple[str, str]], max_length: int = 128
    ) -> list[float]:
        """
        Batch inference, return list of "dependency" confidence scores (float) for each (step_a, step_b) pair.
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
                scores = probs[:, 1].cpu().tolist()         # Dependency probability

        return scores
