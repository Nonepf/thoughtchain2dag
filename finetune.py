"""
Step 3: 微调两个 DeBERTa-v3-small 模型
  - Segmenter : Token 分类，预测每个 token 是否是新 segment 的起始点
  - Linker    : 句对分类，预测两个 segment 之间是否存在依赖关系

关键：
  SegmenterDataset 使用与 step2 完全相同的 normalize_for_match / find_segment_in_text，
  保证训练数据的 Token 标签与数据生成阶段一致。
  每条推理文本单独 tokenize，不跨条混合。

依赖：
    pip install transformers torch accelerate scikit-learn
用法：
    python step3_finetune.py --task seg
    python step3_finetune.py --task link
    python step3_finetune.py --task both
"""

import argparse, json, os, random, re
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm


# ── 配置 ─────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    seg_data_file:  str   = "segmenter_train.jsonl"
    link_data_file: str   = "linker_train.jsonl"
    seg_model_out:  str   = "segmenter_finetuned"
    link_model_out: str   = "linker_finetuned"
    base_model:     str   = "microsoft/deberta-v3-small"
    seg_max_len:    int   = 512
    link_max_len:   int   = 128
    batch_size:     int   = 16
    grad_accum:     int   = 2
    epochs:         int   = 5
    lr:             float = 2e-5
    warmup_ratio:   float = 0.1
    val_ratio:      float = 0.1
    seed:           int   = 42

cfg = TrainConfig()

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── 归一化与匹配（与 step2 保持完全一致）──────────────────────────────────────
def normalize_for_match(s):
    s = s.replace('\\n', '\n').replace('\\t', ' ')
    s = re.sub(r'\\\(', '', s)
    s = re.sub(r'\\\)', '', s)
    s = re.sub(r'\\\[\s*', ' ', s)
    s = re.sub(r'\s*\\\]', ' ', s)
    s = re.sub(r'\\boxed\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.rstrip('.')
    return s


def find_segment_in_text(text, seg, start=0):
    idx = text.find(seg, start)
    if idx != -1:
        return idx, idx + len(seg)

    text_sub = text[start:]

    seg_ws = re.sub(r'\s+', ' ', seg).strip()
    for i in range(len(text_sub)):
        for win in range(max(1, len(seg_ws)),
                         min(len(seg_ws) * 2 + 30, len(text_sub) - i) + 1):
            if re.sub(r'\s+', ' ', text_sub[i:i+win]).strip() == seg_ws:
                return start + i, start + i + win

    seg_norm = normalize_for_match(seg)
    if not seg_norm:
        return None
    for i in range(len(text_sub)):
        for win in range(max(1, len(seg_norm)),
                         min(len(seg_norm) * 4 + 100, len(text_sub) - i) + 1):
            if normalize_for_match(text_sub[i:i+win]) == seg_norm:
                return start + i, start + i + win

    return None


# ── Segmenter Dataset ─────────────────────────────────────────────────────────
class SegmenterDataset(Dataset):
    def __init__(self, records, tokenizer, max_len):
        self.items = []
        skipped = 0
        for rec in records:
            text     = rec.get("text", "")
            segments = rec.get("segments", [])
            if not text or len(segments) < 2:
                skipped += 1
                continue

            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=True,
                padding="max_length",
            )
            offset_mapping = enc.pop("offset_mapping")

            seg_char_starts = set()
            cursor = 0
            for seg_idx, seg in enumerate(segments):
                result = find_segment_in_text(text, seg, cursor)
                if result is None:
                    continue
                char_start, char_end = result
                if seg_idx > 0:
                    seg_char_starts.add(char_start)
                cursor = char_end

            labels = []
            for tok_start, tok_end in offset_mapping:
                if tok_start == tok_end:
                    labels.append(-100)
                elif tok_start in seg_char_starts:
                    labels.append(1)
                else:
                    labels.append(0)

            self.items.append({
                "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"],  dtype=torch.long),
                "token_type_ids": torch.tensor(enc.get("token_type_ids", [0]*max_len), dtype=torch.long),
                "labels":         torch.tensor(labels,                 dtype=torch.long),
            })
        if skipped:
            print(f"  [Segmenter] Skipped {skipped} empty/short records")

    def __len__(self):          return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


# ── Linker Dataset ────────────────────────────────────────────────────────────
class LinkerDataset(Dataset):
    def __init__(self, records, tokenizer, max_len, upsample_pos=True):
        pos = [r for r in records if r.get("label") == 1]
        neg = [r for r in records if r.get("label") == 0]

        if upsample_pos and pos and len(neg) > len(pos):
            ratio  = min(3, len(neg) // max(len(pos), 1))
            pos_up = pos * ratio
            neg    = random.sample(neg, min(len(neg), len(pos_up) * 3))
            records = pos_up + neg
            random.shuffle(records)

        self.items = []
        for rec in records:
            seg_a = rec.get("seg_a", "")
            seg_b = rec.get("seg_b", "")
            label = int(rec.get("label", 0))
            if not seg_a or not seg_b:
                continue
            enc = tokenizer(seg_a, seg_b, truncation=True,
                            max_length=max_len, padding="max_length")
            self.items.append({
                "input_ids":      torch.tensor(enc["input_ids"],     dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(enc.get("token_type_ids", [0]*max_len), dtype=torch.long),
                "labels":         torch.tensor(label,                dtype=torch.long),
            })

    def __len__(self):          return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


# ── 训练引擎 ──────────────────────────────────────────────────────────────────
def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: recs.append(json.loads(line))
            except: pass
    return recs


def split_data(records, val_ratio):
    random.shuffle(records)
    cut = int(len(records) * (1 - val_ratio))
    return records[:cut], records[cut:]

def train_epoch(model, loader, optimizer, scheduler, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        # 纯 FP32，移除所有 autocast/scaler
        outputs = model(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        total_loss += outputs.loss.item()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 防爆炸
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
    return total_loss / len(loader)

@torch.inference_mode()
def eval_epoch(model, loader, task):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="  eval ", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        logits = outputs.logits
        
        if task == "seg":
            preds  = torch.argmax(logits, dim=-1).view(-1).cpu().numpy()
            labels = batch["labels"].view(-1).cpu().numpy()
            mask   = labels != -100
            all_preds.extend(preds[mask].tolist())
            all_labels.extend(labels[mask].tolist())
        else:
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
            
    # 二分类任务看 Class 1 的 F1 更合理
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(classification_report(all_labels, all_preds, zero_division=0))
    return total_loss / len(loader), f1


def train_model(model, train_loader, val_loader, output_dir, task, tokenizer):
    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * cfg.epochs // cfg.grad_accum
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # 已移除 GradScaler 初始化

    best_f1 = -1.0
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{cfg.epochs}")
        # 参数数量已对齐：5 个
        tl = train_epoch(model, train_loader, optimizer, scheduler, cfg.grad_accum)
        vl, f1 = eval_epoch(model, val_loader, task)
        print(f"  train_loss={tl:.4f}  val_loss={vl:.4f}  macro_f1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  ✓ Saved (f1={best_f1:.4f}) -> {output_dir}")
            
    print(f"\nBest F1: {best_f1:.4f}")


def finetune_segmenter():
    print("\n" + "="*60 + "\nFinetuning SEGMENTER\n" + "="*60)
    tok     = AutoTokenizer.from_pretrained(cfg.base_model)
    records = load_jsonl(cfg.seg_data_file)
    print(f"Loaded {len(records)} records")
    tr, va  = split_data(records, cfg.val_ratio)
    tds = SegmenterDataset(tr, tok, cfg.seg_max_len)
    vds = SegmenterDataset(va, tok, cfg.seg_max_len)
    print(f"  Train: {len(tds)} | Val: {len(vds)}")
    tl = DataLoader(tds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.base_model, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
    train_model(model, tl, vl, cfg.seg_model_out, "seg", tok)


def finetune_linker():
    print("\n" + "="*60 + "\nFinetuning LINKER\n" + "="*60)
    tok     = AutoTokenizer.from_pretrained(cfg.base_model)
    records = load_jsonl(cfg.link_data_file)
    print(f"Loaded {len(records)} records")
    tr, va  = split_data(records, cfg.val_ratio)
    tds = LinkerDataset(tr, tok, cfg.link_max_len, upsample_pos=True)
    vds = LinkerDataset(va, tok, cfg.link_max_len, upsample_pos=False)
    print(f"  Train (resampled): {len(tds)} | Val: {len(vds)}")
    tl = DataLoader(tds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    vl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
    train_model(model, tl, vl, cfg.link_model_out, "link", tok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["seg", "link", "both"], default="both")
    args = parser.parse_args()
    if args.task in ("seg",  "both"): finetune_segmenter()
    if args.task in ("link", "both"): finetune_linker()
