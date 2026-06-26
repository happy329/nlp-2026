import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Sequence, Tuple


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from summarization.pegasus_base_summary import pegasus_base_summarize_batch
from summarization.pegasus_finetuned_summary import pegasus_finetuned_summarize_batch
from summarization.randeng_pegasus_summary import randeng_pegasus_summarize_batch
from summarization.textrank_summary import textrank_summarize
from common.text_utils import clean_text, project_path, read_csv_rows, safe_divide, write_csv_rows


SUMMARY_METHODS = {
    "textrank": "TextRank",
    "pegasus_base": "Pegasus-base zero-shot",
    "pegasus_finetuned": "Pegasus-base + LCSTS fine-tuned",
    "randeng": "Randeng-Pegasus-238M-Summary-Chinese",
}


def char_tokens(text: str) -> List[str]:
    return [char for char in clean_text(text) if not char.isspace()]


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def f1_score(overlap: int, pred_total: int, ref_total: int) -> float:
    precision = safe_divide(overlap, pred_total)
    recall = safe_divide(overlap, ref_total)
    return safe_divide(2 * precision * recall, precision + recall)


def rouge_n(prediction: str, reference: str, n: int) -> float:
    pred_counts = ngrams(char_tokens(prediction), n)
    ref_counts = ngrams(char_tokens(reference), n)
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = sum((pred_counts & ref_counts).values())
    return f1_score(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = char_tokens(prediction)
    ref_tokens = char_tokens(reference)
    overlap = lcs_len(pred_tokens, ref_tokens)
    return f1_score(overlap, len(pred_tokens), len(ref_tokens))


def load_eval_rows(data_file: str, limit: int = 0) -> List[Dict[str, str]]:
    rows = []
    for row in read_csv_rows(data_file):
        content = clean_text(row.get("content", ""))
        summary = clean_text(row.get("summary", ""))
        if not content or not summary:
            continue
        rows.append(
            {
                "sample_id": row.get("sample_id", f"sample_{len(rows) + 1:06d}"),
                "content": content,
                "summary": summary,
            }
        )
        if limit and len(rows) >= limit:
            break

    if not rows:
        raise ValueError(f"评价数据为空或没有人工摘要: {data_file}")
    return rows


def expand_methods(methods: List[str]) -> List[str]:
    if "all" in methods:
        return list(SUMMARY_METHODS.keys())
    return methods


def generate_predictions(method: str, contents: List[str], batch_size: int) -> List[str]:
    if method == "textrank":
        return [textrank_summarize(text) for text in contents]
    if method == "pegasus_base":
        return pegasus_base_summarize_batch(contents, batch_size=batch_size)
    if method == "pegasus_finetuned":
        return pegasus_finetuned_summarize_batch(contents, batch_size=batch_size)
    if method == "randeng":
        return randeng_pegasus_summarize_batch(contents, batch_size=batch_size)
    raise ValueError(f"未知摘要方法: {method}")


def aggregate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    count = len(predictions)
    return {
        "rouge_1": sum(rouge_n(pred, ref, 1) for pred, ref in zip(predictions, references)) / count,
        "rouge_2": sum(rouge_n(pred, ref, 2) for pred, ref in zip(predictions, references)) / count,
        "rouge_l": sum(rouge_l(pred, ref) for pred, ref in zip(predictions, references)) / count,
    }


def aggregate_lengths(predictions: List[str], references: List[str], contents: List[str]) -> Dict[str, float]:
    count = len(predictions)
    pred_lens = [len(char_tokens(text)) for text in predictions]
    ref_lens = [len(char_tokens(text)) for text in references]
    source_lens = [len(char_tokens(text)) for text in contents]
    compression = [safe_divide(pred_len, source_len) for pred_len, source_len in zip(pred_lens, source_lens)]
    return {
        "avg_source_len": sum(source_lens) / count,
        "avg_reference_len": sum(ref_lens) / count,
        "avg_summary_len": sum(pred_lens) / count,
        "avg_compression_rate": sum(compression) / count,
    }


def compute_bertscore(predictions: List[str], references: List[str], model_type: str = "") -> float:
    try:
        from bert_score import score
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("缺少 BERTScore 依赖，请安装: pip install bert-score") from exc

    kwargs = {"lang": "zh", "verbose": False}
    if model_type:
        kwargs["model_type"] = model_type
    _, _, f1 = score(predictions, references, **kwargs)
    return float(f1.mean().item())


def plot_summary_compare(rouge_rows: List[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过摘要对比图")
        return

    if not rouge_rows:
        return

    methods = [row["method"] for row in rouge_rows]
    metrics = ["rouge_1", "rouge_2", "rouge_l"]
    x = range(len(methods))
    width = 0.24

    os.makedirs(os.path.dirname(project_path(cfg.summary_compare_fig)), exist_ok=True)
    plt.figure(figsize=(10, 5))
    for offset, metric in enumerate(metrics):
        values = [float(row[metric]) for row in rouge_rows]
        positions = [item + (offset - 1) * width for item in x]
        plt.bar(positions, values, width=width, label=metric.upper())

    plt.xticks(list(x), methods, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(project_path(cfg.summary_compare_fig), dpi=200)
    plt.close()


def evaluate(args: argparse.Namespace) -> None:
    data_rows = load_eval_rows(args.data_file, limit=args.limit)
    contents = [row["content"] for row in data_rows]
    references = [row["summary"] for row in data_rows]
    methods = expand_methods(args.methods)

    rouge_rows = []
    length_rows = []
    bert_rows = []
    example_rows = []

    for method in methods:
        if method not in SUMMARY_METHODS:
            raise ValueError(f"未知摘要方法: {method}")

        print(f"开始评价 {SUMMARY_METHODS[method]}，样本数: {len(data_rows)}")
        try:
            predictions = generate_predictions(method, contents, batch_size=args.batch_size)
        except Exception as exc:
            print(f"跳过 {method}: {exc}")
            continue

        rouge_metrics = aggregate_rouge(predictions, references)
        rouge_rows.append({"method": SUMMARY_METHODS[method], **{key: round(value, 6) for key, value in rouge_metrics.items()}})

        length_metrics = aggregate_lengths(predictions, references, contents)
        length_rows.append({"method": SUMMARY_METHODS[method], **{key: round(value, 6) for key, value in length_metrics.items()}})

        if args.bertscore:
            try:
                bert = compute_bertscore(predictions, references, model_type=args.bertscore_model)
                bert_rows.append({"method": SUMMARY_METHODS[method], "bertscore_f1": round(bert, 6)})
            except Exception as exc:
                print(f"{method} BERTScore 计算失败: {exc}")
                bert_rows.append({"method": SUMMARY_METHODS[method], "bertscore_f1": ""})
        else:
            bert_rows.append({"method": SUMMARY_METHODS[method], "bertscore_f1": ""})

        for row, prediction in zip(data_rows[: args.example_count], predictions[: args.example_count]):
            example_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "method": SUMMARY_METHODS[method],
                    "content": row["content"],
                    "reference": row["summary"],
                    "prediction": prediction,
                }
            )

    write_csv_rows(cfg.rouge_score_file, rouge_rows, ["method", "rouge_1", "rouge_2", "rouge_l"])
    write_csv_rows(
        cfg.summary_length_file,
        length_rows,
        ["method", "avg_source_len", "avg_reference_len", "avg_summary_len", "avg_compression_rate"],
    )
    write_csv_rows(cfg.summary_examples_file, example_rows, ["sample_id", "method", "content", "reference", "prediction"])

    write_csv_rows(cfg.bertscore_file, bert_rows, ["method", "bertscore_f1"])

    plot_summary_compare(rouge_rows)
    print("摘要评价完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动摘要评价：ROUGE / BERTScore / 长度与压缩率")
    parser.add_argument("--data_file", type=str, default=cfg.lcsts_eval_file)
    parser.add_argument("--methods", nargs="+", default=["textrank"], help="textrank pegasus_base pegasus_finetuned randeng all")
    parser.add_argument("--limit", type=int, default=0, help="仅评价前 N 条，0 表示全部")
    parser.add_argument("--batch_size", type=int, default=cfg.summary_eval_batch_size)
    parser.add_argument("--example_count", type=int, default=20)
    parser.add_argument("--bertscore", action="store_true", help="启用 BERTScore，首次运行会下载模型")
    parser.add_argument("--bertscore_model", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
