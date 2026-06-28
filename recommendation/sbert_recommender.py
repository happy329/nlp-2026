"""
Sentence-BERT + 余弦相似度推荐
延迟加载模型，避免 import 阶段强制联网
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg


class SBERTRecommender:
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.news_df = None
        self.news_ids = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(cfg.sbert_model_name)

    def build(self, news_csv_path=None):
        news_csv_path = news_csv_path or cfg.news_file
        df = pd.read_csv(news_csv_path)
        self.news_df = df
        self.news_ids = df["news_id"].astype(str).tolist()
        texts = df["content"].astype(str).tolist()
        self._load_model()
        self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)

    def save(self):
        if self.embeddings is None:
            raise RuntimeError("embeddings is empty, call build() first")
        Path(cfg.sbert_index_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(cfg.sbert_index_file, self.embeddings)
        with open(cfg.news_ids_file, "w", encoding="utf-8") as f:
            json.dump(self.news_ids, f, ensure_ascii=False)

    def load(self):
        self.embeddings = np.load(cfg.sbert_index_file)
        with open(cfg.news_ids_file, "r", encoding="utf-8") as f:
            self.news_ids = json.load(f)
        self.news_df = pd.read_csv(cfg.news_file)

    def recommend(self, text, top_k=5, label_filter=None):
        self._load_model()
        q = self.model.encode([text])
        sims = cosine_similarity(q, self.embeddings).flatten()
        idx = np.argsort(sims)[::-1][: top_k * 2]
        results = []
        for i in idx:
            row = self.news_df.iloc[i]
            content = str(row["content"])
            results.append({
                "news_id": row["news_id"],
                "title": content[:50],
                "summary": content[:120] + ("..." if len(content) > 120 else ""),
                "content": content,
                "label": row["label"],
                "score": float(sims[i]),
            })
        if label_filter:
            same = [r for r in results if r["label"] == label_filter]
            others = [r for r in results if r["label"] != label_filter]
            results = same + others
        return results[:top_k]
