import argparse
import csv
import os
import sys
from typing import Dict, List, Sequence, Tuple


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
KEYWORD_DIR = os.path.join(PROJECT_DIR, "keyword")
if KEYWORD_DIR not in sys.path:
    sys.path.insert(0, KEYWORD_DIR)

import config as cfg
from common.text_utils import clean_text, normalize_keyword, parse_keywords, project_path, safe_divide, write_csv_rows
from keybert_keywords import extract_keybert_keywords_batch
from textrank_keywords import extract_textrank_keywords
from tfidf_keywords import TfidfKeywordExtractor


KEYWORD_METHODS = {
    "tfidf": "TF-IDF",
    "textrank": "TextRank",
    "keybert": "KeyBERT",
}


def load_csl_rows(data_file: str = "", limit: int = 0) -> List[Dict]:
    path = data_file
    if not path:
        path = cfg.csl_file if os.path.exists(project_path(cfg.csl_file)) else cfg.csl_raw_file

    full_path = project_path(path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSL 数据不存在: {path}")

    rows = []
    with open(full_path, "r", encoding="utf-8-sig", newline="") as f:
        if path.endswith(".tsv"):
            header = f.readline().strip()
            fieldnames = header.split("\t") if "\t" in header else header.split()
            reader = csv.DictReader(f, delimiter="\t", fieldnames=fieldnames)
        else:
            reader = csv.DictReader(f)
        for row in reader:
            title = clean_text(row.get("title", ""))
            abstract = clean_text(row.get("abstract", ""))
            keywords = parse_keywords(row.get("keywords", ""))
            if not (title or abstract) or not keywords:
                continue
            rows.append(
                {
                    "sample_id": row.get("sample_id", f"csl_{len(rows) + 1:06d}"),
                    "title": title,
                    "abstract": abstract,
                    "text": f"{title}。{abstract}",
                    "keywords": keywords,
                }
            )
            if limit and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"CSL 数据为空: {path}")
    return rows


def expand_methods(methods: List[str]) -> List[str]:
    if "all" in methods:
        return list(KEYWORD_METHODS.keys())
    return methods


def normalized_set(words: Sequence[str]) -> set:
    return {normalize_keyword(word) for word in words if normalize_keyword(word)}


def score_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> Tuple[float, float, float]:
    pred_set = normalized_set(predicted[:k])
    gold_set = normalized_set(gold)
    if not gold_set:
        return 0.0, 0.0, 0.0

    hits = len(pred_set & gold_set)
    precision = safe_divide(hits, k)
    recall = safe_divide(hits, len(gold_set))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def evaluate_predictions(predictions: List[List[str]], golds: List[List[str]], top_ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    count = len(golds)
    for k in top_ks:
        p_total = 0.0
        r_total = 0.0
        f_total = 0.0
        for predicted, gold in zip(predictions, golds):
            precision, recall, f1 = score_at_k(predicted, gold, k)
            p_total += precision
            r_total += recall
            f_total += f1
        metrics[f"p@{k}"] = p_total / count
        metrics[f"r@{k}"] = r_total / count
        metrics[f"f1@{k}"] = f_total / count
    return metrics


def run_method(method: str, docs: List[str], top_k: int) -> List[List[str]]:
    if method == "tfidf":
        extractor = TfidfKeywordExtractor()
        extractor.fit(docs)
        return extractor.extract_batch(docs, top_k=top_k)
    if method == "textrank":
        return [extract_textrank_keywords(doc, top_k=top_k) for doc in docs]
    if method == "keybert":
        return extract_keybert_keywords_batch(docs, top_k=top_k)
    raise ValueError(f"未知关键词方法: {method}")


def plot_keyword_compare(score_rows: List[Dict[str, float]], top_ks: Sequence[int]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过关键词对比图")
        return

    if not score_rows:
        return

    os.makedirs(os.path.dirname(project_path(cfg.keyword_compare_fig)), exist_ok=True)
    methods = [row["method"] for row in score_rows]
    x = range(len(methods))
    width = 0.8 / len(top_ks)

    plt.figure(figsize=(9, 5))
    for offset, k in enumerate(top_ks):
        metric = f"f1@{k}"
        values = [float(row[metric]) for row in score_rows]
        positions = [item + (offset - (len(top_ks) - 1) / 2) * width for item in x]
        plt.bar(positions, values, width=width, label=metric.upper())

    plt.xticks(list(x), methods)
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(project_path(cfg.keyword_compare_fig), dpi=200)
    plt.close()


def evaluate(args: argparse.Namespace) -> None:
    rows = load_csl_rows(args.data_file, limit=args.limit)
    docs = [row["text"] for row in rows]
    golds = [row["keywords"] for row in rows]
    top_ks = sorted(set(args.top_k))
    max_k = max(top_ks)

    score_rows = []
    example_rows = []

    for method in expand_methods(args.methods):
        if method not in KEYWORD_METHODS:
            raise ValueError(f"未知关键词方法: {method}")

        print(f"开始评价 {KEYWORD_METHODS[method]}，样本数: {len(rows)}")
        try:
            predictions = run_method(method, docs, top_k=max_k)
        except Exception as exc:
            print(f"跳过 {method}: {exc}")
            continue

        metrics = evaluate_predictions(predictions, golds, top_ks=top_ks)
        score_rows.append({"method": KEYWORD_METHODS[method], **{key: round(value, 6) for key, value in metrics.items()}})

        for row, predicted in zip(rows[: args.example_count], predictions[: args.example_count]):
            example_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "method": KEYWORD_METHODS[method],
                    "title": row["title"],
                    "gold_keywords": ";".join(row["keywords"]),
                    "pred_keywords": ";".join(predicted),
                }
            )

    fieldnames = ["method"]
    for k in top_ks:
        fieldnames.extend([f"p@{k}", f"r@{k}", f"f1@{k}"])
    write_csv_rows(cfg.keyword_scores_file, score_rows, fieldnames)
    write_csv_rows(cfg.keyword_examples_file, example_rows, ["sample_id", "method", "title", "gold_keywords", "pred_keywords"])
    plot_keyword_compare(score_rows, top_ks)
    print("关键词评价完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="关键词提取评价：Precision/Recall/F1@K")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--methods", nargs="+", default=["tfidf", "textrank"], help="tfidf textrank keybert all")
    parser.add_argument("--top_k", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--limit", type=int, default=0, help="仅评价前 N 条，0 表示全部")
    parser.add_argument("--example_count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
