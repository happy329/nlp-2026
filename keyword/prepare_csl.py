import argparse
import csv
import os
import sys
from typing import Dict, List, Optional


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from common.text_utils import clean_text, parse_keywords, project_path, write_csv_rows


def prepare_csl(limit: Optional[int] = None) -> int:
    raw_path = project_path(cfg.csl_raw_file)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"CSL 原始文件不存在: {cfg.csl_raw_file}")

    rows: List[Dict[str, str]] = []
    with open(raw_path, "r", encoding="utf-8-sig", newline="") as f:
        header = f.readline().strip()
        fieldnames = header.split("\t") if "\t" in header else header.split()
        reader = csv.DictReader(f, delimiter="\t", fieldnames=fieldnames)
        for row in reader:
            title = clean_text(row.get("title", ""))
            abstract = clean_text(row.get("abstract", ""))
            keywords = parse_keywords(row.get("keywords", ""))
            if not title and not abstract:
                continue

            rows.append(
                {
                    "sample_id": f"csl_{len(rows) + 1:06d}",
                    "title": title,
                    "abstract": abstract,
                    "keywords": ";".join(keywords),
                }
            )
            if limit is not None and len(rows) >= limit:
                break

    write_csv_rows(cfg.csl_file, rows, ["sample_id", "title", "abstract", "keywords"])
    print(f"CSL 保存到 {cfg.csl_file}，数量: {len(rows)}")
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 CSL tsv 为 sample_id/title/abstract/keywords CSV")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，调试用")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_csl(limit=args.limit)


if __name__ == "__main__":
    main()
