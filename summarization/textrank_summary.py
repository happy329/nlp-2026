import argparse
import math
import os
import sys
from typing import List


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from common.text_utils import clean_text, load_stopwords, split_sentences, tokenize


def sentence_similarity(words_a: List[str], words_b: List[str]) -> float:
    if not words_a or not words_b:
        return 0.0

    set_a = set(words_a)
    set_b = set(words_b)
    overlap = len(set_a & set_b)
    if overlap == 0:
        return 0.0

    denominator = math.log(len(set_a) + 1.0) + math.log(len(set_b) + 1.0)
    if denominator == 0:
        return 0.0
    return overlap / denominator


def pagerank(scores: List[List[float]], damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> List[float]:
    n = len(scores)
    if n == 0:
        return []
    ranks = [1.0 / n] * n
    out_sums = [sum(row) for row in scores]

    for _ in range(max_iter):
        new_ranks = [(1.0 - damping) / n for _ in range(n)]
        for source in range(n):
            if out_sums[source] == 0:
                continue
            for target in range(n):
                if source == target or scores[source][target] == 0:
                    continue
                new_ranks[target] += damping * scores[source][target] / out_sums[source] * ranks[source]

        diff = sum(abs(new_ranks[i] - ranks[i]) for i in range(n))
        ranks = new_ranks
        if diff < tol:
            break

    return ranks


def textrank_summarize(
    text: str,
    max_sentences: int = 2,
    max_chars: int = cfg.summary_max_target_len,
) -> str:
    text = clean_text(text)
    if not text:
        return ""

    sentences = split_sentences(text)
    if len(sentences) == 1:
        return sentences[0][:max_chars]

    stopwords = load_stopwords()
    sentence_words = [tokenize(sentence, stopwords=stopwords, keep_single_char=False) for sentence in sentences]

    n = len(sentences)
    graph = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            score = sentence_similarity(sentence_words[i], sentence_words[j])
            graph[i][j] = score
            graph[j][i] = score

    ranks = pagerank(graph)
    ranked_indices = sorted(range(n), key=lambda index: ranks[index], reverse=True)
    selected = sorted(ranked_indices[: max(1, max_sentences)])

    summary = "".join(sentences[index] for index in selected)
    if len(summary) > max_chars:
        summary = summary[:max_chars]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TextRank 抽取式中文摘要")
    parser.add_argument("--text", type=str, required=True, help="待摘要文本")
    parser.add_argument("--max_sentences", type=int, default=2)
    parser.add_argument("--max_chars", type=int, default=cfg.summary_max_target_len)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(textrank_summarize(args.text, args.max_sentences, args.max_chars))


if __name__ == "__main__":
    main()
