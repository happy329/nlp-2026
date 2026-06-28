import argparse
import os
import sys
from functools import lru_cache
from typing import List, Sequence


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

import config as cfg
from common.text_utils import clean_text, load_stopwords, tokenize, unique_keep_order


def require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("缺少 KeyBERT 语义向量依赖，请安装: pip install sentence-transformers") from exc
    return SentenceTransformer


def cosine_scores(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    vector_norm = np.linalg.norm(vector) + 1e-12
    return (matrix @ vector) / (matrix_norm[:, 0] * vector_norm)


def generate_candidates(text: str, max_candidates: int = 256) -> List[str]:
    words = tokenize(text, stopwords=load_stopwords(), keep_single_char=False)
    candidates = []
    candidates.extend(words)

    for index in range(len(words) - 1):
        first = words[index]
        second = words[index + 1]
        if first != second:
            candidates.append(first + second)

    return unique_keep_order(candidates)[:max_candidates]


class KeyBertKeywordExtractor:
    def __init__(self, model_name: str = cfg.keybert_model) -> None:
        SentenceTransformer = require_sentence_transformers()
        self.model = SentenceTransformer(model_name)

    def extract(self, text: str, top_k: int = cfg.keyword_top_k) -> List[str]:
        text = clean_text(text)
        candidates = generate_candidates(text)
        if not text or not candidates:
            return []

        embeddings = self.model.encode([text] + candidates, convert_to_numpy=True, normalize_embeddings=False)
        doc_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        scores = cosine_scores(candidate_embeddings, doc_embedding)
        ranked_indices = np.argsort(-scores)
        return [candidates[index] for index in ranked_indices[:top_k]]

    def extract_batch(
        self,
        texts: Sequence[str],
        top_k: int = cfg.keyword_top_k,
        doc_batch_size: int = 32,
        encode_batch_size: int = 128,
    ) -> List[List[str]]:
        results: List[List[str]] = []
        cleaned_texts = [clean_text(text) for text in texts]

        for start in range(0, len(cleaned_texts), doc_batch_size):
            text_batch = cleaned_texts[start : start + doc_batch_size]
            candidate_batch = [generate_candidates(text) if text else [] for text in text_batch]
            flat_candidates = [candidate for candidates in candidate_batch for candidate in candidates]

            if not flat_candidates:
                results.extend([[] for _ in text_batch])
                continue

            embeddings = self.model.encode(
                list(text_batch) + flat_candidates,
                batch_size=encode_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            doc_embeddings = embeddings[: len(text_batch)]
            candidate_embeddings = embeddings[len(text_batch) :]

            offset = 0
            for doc_embedding, candidates in zip(doc_embeddings, candidate_batch):
                if not candidates:
                    results.append([])
                    continue
                end = offset + len(candidates)
                scores = cosine_scores(candidate_embeddings[offset:end], doc_embedding)
                ranked_indices = np.argsort(-scores)
                results.append([candidates[index] for index in ranked_indices[:top_k]])
                offset = end

        return results


@lru_cache(maxsize=2)
def get_extractor(model_name: str = cfg.keybert_model) -> KeyBertKeywordExtractor:
    return KeyBertKeywordExtractor(model_name=model_name)


def extract_keybert_keywords(text: str, top_k: int = cfg.keyword_top_k, model_name: str = cfg.keybert_model) -> List[str]:
    return get_extractor(model_name).extract(text, top_k=top_k)


def extract_keybert_keywords_batch(
    texts: Sequence[str],
    top_k: int = cfg.keyword_top_k,
    model_name: str = cfg.keybert_model,
    doc_batch_size: int = 32,
    encode_batch_size: int = 128,
) -> List[List[str]]:
    return get_extractor(model_name).extract_batch(
        texts,
        top_k=top_k,
        doc_batch_size=doc_batch_size,
        encode_batch_size=encode_batch_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KeyBERT 关键词提取")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=cfg.keyword_top_k)
    parser.add_argument("--model_name", type=str, default=cfg.keybert_model)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(extract_keybert_keywords(args.text, top_k=args.top_k, model_name=args.model_name))


if __name__ == "__main__":
    main()
