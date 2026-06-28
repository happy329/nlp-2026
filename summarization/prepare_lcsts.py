import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from common.text_utils import clean_text, project_path, write_csv_rows


TEXT_KEYS = ("content", "text", "article", "source", "document")
SUMMARY_KEYS = ("summary", "summarization", "target", "title")


def first_existing(record: Dict, keys: Iterable[str]) -> str:
    for key in keys:
        if key in record and record[key] is not None:
            return clean_text(record[key])
    return ""


def convert_jsonl(raw_file: str, split_name: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows = []
    raw_path = project_path(raw_file)
    with open(raw_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if limit is not None and len(rows) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = first_existing(record, TEXT_KEYS)
            summary = first_existing(record, SUMMARY_KEYS)
            if not content:
                continue

            rows.append(
                {
                    "sample_id": f"{split_name}_{len(rows) + 1:06d}",
                    "content": content,
                    "summary": summary,
                }
            )

    return rows


def normalize_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None or limit <= 0:
        return None
    return limit


def prepare_lcsts(
    limit_train: Optional[int] = cfg.lcsts_train_limit,
    limit_val: Optional[int] = cfg.lcsts_val_limit,
    limit_test: Optional[int] = cfg.lcsts_test_limit,
    use_full_output: bool = False,
) -> Dict[str, int]:
    train_file = cfg.lcsts_train_full_file if use_full_output else cfg.lcsts_train_file
    val_file = cfg.lcsts_val_full_file if use_full_output else cfg.lcsts_val_file
    test_file = cfg.lcsts_test_full_file if use_full_output else cfg.lcsts_test_file

    split_specs = [
        ("train", cfg.lcsts_train_raw_file, train_file, normalize_limit(limit_train)),
        ("valid", cfg.lcsts_val_raw_file, val_file, normalize_limit(limit_val)),
        ("test", cfg.lcsts_test_raw_file, test_file, normalize_limit(limit_test)),
    ]

    counts = {}
    for split_name, raw_file, out_file, limit in split_specs:
        if not os.path.exists(project_path(raw_file)):
            print(f"跳过 {split_name}: 原始文件不存在 {raw_file}")
            continue

        rows = convert_jsonl(raw_file, split_name, limit=limit)
        write_csv_rows(out_file, rows, ["sample_id", "content", "summary"])
        counts[split_name] = len(rows)
        print(f"{split_name} 保存到 {out_file}，数量: {len(rows)}")

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 LCSTS jsonl 为 sample_id/content/summary CSV")
    parser.add_argument("--limit_train", type=int, default=cfg.lcsts_train_limit, help="训练集保留前 N 条，0 表示全量")
    parser.add_argument("--limit_val", type=int, default=cfg.lcsts_val_limit, help="验证集保留前 N 条，0 表示全量")
    parser.add_argument("--limit_test", type=int, default=cfg.lcsts_test_limit, help="公开测试集保留前 N 条，0 表示全量")
    parser.add_argument("--full_output", action="store_true", help="输出到 train.csv/valid.csv/test_public.csv 全量文件名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = prepare_lcsts(
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        use_full_output=args.full_output,
    )
    print("LCSTS 整理完成:", counts)


if __name__ == "__main__":
    main()
