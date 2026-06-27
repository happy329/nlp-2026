"""
提前构建 TF-IDF / SBERT 索引
SBERT 失败时仅跳过该部分，不中断整体流程
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommendation.tfidf_recommender import TFIDFRecommender
import config as cfg


def build_index():
    cfg.SAVED_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tfidf = TFIDFRecommender()
    tfidf.build()
    tfidf.save()
    print("TF-IDF 索引已保存:", cfg.TFIDF_VECTORIZER_PKL, cfg.TFIDF_MATRIX_PKL)

    try:
        from recommendation.sbert_recommender import SBERTRecommender
        sbert = SBERTRecommender()
        sbert.build()
        sbert.save()
        print("SBERT 索引已保存:", cfg.SBERT_EMBEDDINGS_NPY, cfg.NEWS_IDS_JSON)
    except Exception as exc:
        print("SBERT 索引构建失败，已跳过:", exc)


if __name__ == "__main__":
    build_index()
