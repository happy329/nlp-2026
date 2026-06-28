"""
TF-IDF + 余弦相似度推荐（可选）
"""

import pickle
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config as cfg


class TFIDFRecommender:
    def __init__(self):
        self.vectorizer = None
        self.matrix = None
        self.news_df = None

    def build(self):
        df = pd.read_csv(cfg.news_file)
        self.news_df = df

        stopwords = set()
        if Path(cfg.stopwords_file).exists():
            stopwords = {line.strip() for line in open(cfg.stopwords_file, "r", encoding="utf-8")}

        def tokenize(text):
            return " ".join([w for w in jieba.lcut(str(text)) if w.strip() and w not in stopwords])

        texts = [tokenize(t) for t in df["content"].astype(str).tolist()]
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)

    def save(self):
        Path(cfg.tfidf_vectorizer_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.tfidf_vectorizer_file, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(cfg.tfidf_matrix_file, "wb") as f:
            pickle.dump({"matrix": self.matrix, "news_df": self.news_df}, f)

    def load(self):
        with open(cfg.tfidf_vectorizer_file, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(cfg.tfidf_matrix_file, "rb") as f:
            data = pickle.load(f)
            self.matrix = data["matrix"]
            self.news_df = data["news_df"]

    def recommend(self, text, top_k=5, label_filter=None):
        stopwords = set()
        if Path(cfg.stopwords_file).exists():
            stopwords = {line.strip() for line in open(cfg.stopwords_file, "r", encoding="utf-8")}
        vec = self.vectorizer.transform([" ".join([w for w in jieba.lcut(str(text)) if w.strip() and w not in stopwords])])
        sims = cosine_similarity(vec, self.matrix).flatten()
        idx = np.argsort(sims)[::-1][:top_k]

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
