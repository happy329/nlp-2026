"""
推荐统一接口 recommend_api.py
SBERT 失败时自动回退到 TF-IDF，避免离线环境报错
"""

from recommendation.tfidf_recommender import TFIDFRecommender
from recommendation.sbert_recommender import SBERTRecommender
import config as cfg


class RecommendAPI:
    def __init__(self):
        self.tfidf = TFIDFRecommender()
        self.sbert = SBERTRecommender()
        self._load()

    def _load(self):
        if cfg.TFIDF_VECTORIZER_PKL.exists() and cfg.TFIDF_MATRIX_PKL.exists():
            self.tfidf.load()
        if cfg.SBERT_EMBEDDINGS_NPY.exists() and cfg.NEWS_IDS_JSON.exists():
            try:
                self.sbert.load()
            except Exception as exc:
                print("[RecommendAPI] SBERT 索引加载失败，将回退到 TF-IDF:", exc)

    def recommend(self, text, label_filter=None, top_k=5, method="sbert"):
        if method == "tfidf":
            return self.tfidf.recommend(text, top_k=top_k, label_filter=label_filter)
        try:
            return self.sbert.recommend(text, top_k=top_k, label_filter=label_filter)
        except Exception as exc:
            print("[RecommendAPI] SBERT 推荐失败，回退到 TF-IDF:", exc)
            return self.tfidf.recommend(text, top_k=top_k, label_filter=label_filter)

    def recommend_with_class_filter(self, text, predicted_label, top_k=5, method="sbert"):
        return self.recommend(text=text, label_filter=predicted_label, top_k=top_k, method=method)


def recommend_with_class_filter(text, predicted_label, top_k=5, method="sbert"):
    api = RecommendAPI()
    return api.recommend_with_class_filter(text=text, predicted_label=predicted_label, top_k=top_k, method=method)


if __name__ == "__main__":
    api = RecommendAPI()
    print(api.recommend("谷歌投资游戏平台", top_k=3))
