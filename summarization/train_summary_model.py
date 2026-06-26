import argparse
import os
import sys
from typing import Dict, List


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from common.text_utils import clean_text, project_path, read_csv_rows
from summarization.model_utils import load_summary_model, load_summary_tokenizer


def require_training_deps():
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("缺少摘要训练依赖，请安装: pip install transformers torch sentencepiece accelerate") from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Seq2SeqTrainer": Seq2SeqTrainer,
        "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
    }


def load_summary_rows(path: str, limit: int = 0) -> List[Dict[str, str]]:
    rows = []
    for row in read_csv_rows(path):
        content = clean_text(row.get("content", ""))
        summary = clean_text(row.get("summary", ""))
        if not content or not summary:
            continue
        rows.append({"content": content, "summary": summary})
        if limit and len(rows) >= limit:
            break
    return rows


def build_dataset_class(base_dataset, tokenizer, max_source_len: int, max_target_len: int):
    class LCSTSSummaryDataset(base_dataset):
        def __init__(self, rows: List[Dict[str, str]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> Dict:
            item = self.rows[index]
            model_inputs = tokenizer(
                item["content"],
                max_length=max_source_len,
                truncation=True,
            )

            try:
                labels = tokenizer(
                    text_target=item["summary"],
                    max_length=max_target_len,
                    truncation=True,
                )
            except TypeError:
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        item["summary"],
                        max_length=max_target_len,
                        truncation=True,
                    )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    return LCSTSSummaryDataset


def train(args: argparse.Namespace) -> None:
    deps = require_training_deps()

    train_rows = load_summary_rows(args.train_file, limit=args.limit_train)
    val_rows = load_summary_rows(args.val_file, limit=args.limit_val)
    if not train_rows:
        raise ValueError(f"训练集为空，请先运行 summarization/prepare_lcsts.py: {args.train_file}")
    if not val_rows:
        raise ValueError(f"验证集为空，请先运行 summarization/prepare_lcsts.py: {args.val_file}")

    tokenizer = load_summary_tokenizer(args.model_name)
    model = load_summary_model(args.model_name)

    dataset_cls = build_dataset_class(
        deps["Dataset"],
        tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
    )
    train_dataset = dataset_cls(train_rows)
    val_dataset = dataset_cls(val_rows)

    data_collator = deps["DataCollatorForSeq2Seq"](
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    training_args = deps["Seq2SeqTrainingArguments"](
        output_dir=project_path(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = deps["Seq2SeqTrainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(project_path(args.output_dir))
    tokenizer.save_pretrained(project_path(args.output_dir))
    print(f"摘要模型已保存到: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 LCSTS 微调中文 Pegasus 摘要模型")
    parser.add_argument("--model_name", type=str, default=cfg.pegasus_base_model)
    parser.add_argument("--train_file", type=str, default=cfg.lcsts_train_file)
    parser.add_argument("--val_file", type=str, default=cfg.lcsts_val_file)
    parser.add_argument("--output_dir", type=str, default=cfg.summary_model_dir)
    parser.add_argument("--max_source_len", type=int, default=cfg.summary_max_source_len)
    parser.add_argument("--max_target_len", type=int, default=cfg.summary_max_target_len)
    parser.add_argument("--epochs", type=float, default=cfg.summary_train_epochs)
    parser.add_argument("--train_batch_size", type=int, default=cfg.summary_train_batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=cfg.summary_eval_batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=cfg.summary_learning_rate)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--limit_train", type=int, default=0, help="调试用，仅读取前 N 条训练样本")
    parser.add_argument("--limit_val", type=int, default=0, help="调试用，仅读取前 N 条验证样本")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
