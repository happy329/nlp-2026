import sys
from functools import lru_cache
from pathlib import Path

from classification.classifier_api import predict_class
from recommendation.recommend_api import RecommendAPI
from summarization.textrank_summary import textrank_summarize


PROJECT_ROOT = Path(__file__).resolve().parent
KEYWORD_DIR = PROJECT_ROOT / "keywords"
if str(KEYWORD_DIR) not in sys.path:
    sys.path.insert(0, str(KEYWORD_DIR))

from keyword_api import extract_keywords


@lru_cache(maxsize=1)
def get_recommend_api():
    return RecommendAPI()


def normalize_classification(class_result):
    result = dict(class_result)
    if "probs" not in result:
        result["probs"] = {
            item["label"]: item["prob"]
            for item in result.get("top_k", [])
        }
    return result


def run_pipeline(text, top_k=5, recommend_method="tfidf"):
    class_result = normalize_classification(
        predict_class(
            text,
            model_name="bilstm_attention",
            top_k=5,
        )
    )

    keywords = extract_keywords(text, top_k=5, method="tfidf")
    summary = {
        "textrank_summary": textrank_summarize(text),
    }
    recommendations = get_recommend_api().recommend(
        text,
        label_filter=class_result.get("label"),
        top_k=top_k,
        method=recommend_method,
    )

    return {
        "classification": class_result,
        "keywords": keywords,
        "summary": summary,
        "recommendations": recommendations,
    }
