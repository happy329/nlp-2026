import argparse
import os
import sys
from collections import Counter
from typing import Iterable, List, Optional


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from common.text_utils import load_stopwords, tokenize, unique_keep_order


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - fallback keeps API usable
    TfidfVectorizer = None


class TfidfKeywordExtractor:
    def __init__(self, max_features: int = cfg.keyword_max_features) -> None:
        self.max_features = max_features
        self.stopwords = load_stopwords()
        self.vectorizer = None
        self.feature_names = None

    def _tokenizer(self, text: str) -> List[str]:
        return tokenize(text, stopwords=self.stopwords, keep_single_char=False)

    def fit(self, documents: Iterable[str]) -> "TfidfKeywordExtractor":
        docs = list(documents)
        if TfidfVectorizer is None:
            return self

        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer,
            token_pattern=None,
            lowercase=False,
            max_features=self.max_features,
        )
        self.vectorizer.fit(docs)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def extract(self, text: str, top_k: int = cfg.keyword_top_k) -> List[str]:
        if self.vectorizer is None or self.feature_names is None:
            return fallback_term_frequency(text, top_k=top_k, stopwords=self.stopwords)

        vector = self.vectorizer.transform([text])
        if vector.nnz == 0:
            return []

        row = vector.tocoo()
        scored = sorted(zip(row.col, row.data), key=lambda item: item[1], reverse=True)
        return unique_keep_order([self.feature_names[index] for index, _ in scored[: top_k * 2]])[:top_k]

    def extract_batch(self, documents: Iterable[str], top_k: int = cfg.keyword_top_k) -> List[List[str]]:
        docs = list(documents)
        if self.vectorizer is None or self.feature_names is None:
            return [self.extract(doc, top_k=top_k) for doc in docs]

        matrix = self.vectorizer.transform(docs)
        results = []
        for row_index in range(matrix.shape[0]):
            row = matrix.getrow(row_index).tocoo()
            scored = sorted(zip(row.col, row.data), key=lambda item: item[1], reverse=True)
            keywords = unique_keep_order([self.feature_names[index] for index, _ in scored[: top_k * 2]])[:top_k]
            results.append(keywords)
        return results


def fallback_term_frequency(text: str, top_k: int, stopwords: Optional[set] = None) -> List[str]:
    words = tokenize(text, stopwords=stopwords or load_stopwords(), keep_single_char=False)
    counts = Counter(words)
    return unique_keep_order([word for word, _ in counts.most_common(top_k * 2)])[:top_k]


def extract_tfidf_keywords(text: str, top_k: int = cfg.keyword_top_k, corpus: Optional[Iterable[str]] = None) -> List[str]:
    extractor = TfidfKeywordExtractor()
    extractor.fit(list(corpus) if corpus is not None else [text])
    return extractor.extract(text, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF 关键词提取")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=cfg.keyword_top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(extract_tfidf_keywords(args.text, top_k=args.top_k))


if __name__ == "__main__":
    main()
