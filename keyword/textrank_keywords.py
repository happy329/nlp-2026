import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from common.text_utils import load_stopwords, tokenize, unique_keep_order


def build_word_graph(words: List[str], window_size: int) -> Dict[str, Dict[str, float]]:
    graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for index, word in enumerate(words):
        end = min(index + window_size, len(words))
        for next_index in range(index + 1, end):
            other = words[next_index]
            if word == other:
                continue
            graph[word][other] += 1.0
            graph[other][word] += 1.0
    return graph


def run_pagerank(graph: Dict[str, Dict[str, float]], damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
    nodes = list(graph.keys())
    if not nodes:
        return {}

    scores = {node: 1.0 for node in nodes}
    out_weights = {node: sum(neighbors.values()) for node, neighbors in graph.items()}

    for _ in range(max_iter):
        new_scores = {}
        max_diff = 0.0
        for node in nodes:
            rank = 1.0 - damping
            for neighbor, weight in graph[node].items():
                if out_weights.get(neighbor, 0.0) == 0:
                    continue
                rank += damping * weight / out_weights[neighbor] * scores[neighbor]
            new_scores[node] = rank
            max_diff = max(max_diff, abs(rank - scores[node]))
        scores = new_scores
        if max_diff < tol:
            break
    return scores


def extract_textrank_keywords(
    text: str,
    top_k: int = cfg.keyword_top_k,
    window_size: int = cfg.keyword_window_size,
) -> List[str]:
    words = tokenize(text, stopwords=load_stopwords(), keep_single_char=False)
    if not words:
        return []

    graph = build_word_graph(words, window_size=window_size)
    scores = run_pagerank(graph)
    ranked: List[Tuple[str, float]] = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return unique_keep_order([word for word, _ in ranked[: top_k * 2]])[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TextRank 关键词提取")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=cfg.keyword_top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(extract_textrank_keywords(args.text, top_k=args.top_k))


if __name__ == "__main__":
    main()
